"""Precompute the demo data from the Synthegra generator.

Writes data/grid.json (story 1) and data/resolution.json (story 2).

Run inside the `synthegra` conda env, with the synthegra repo checked out next to this one:

    OMP_NUM_THREADS=1 python tools/export_demo_data.py [--quick] [--procs 4]

Story 1 grid: planted synergy (cross-modal interaction strength) x shared latent fraction
(redundancy) x cohort size, several seeds. For every simulated cohort we fit, with 3-fold
stratified CV, gradient-boosted survival models (scikit-survival, Cox partial likelihood):
  best_single      best of the two modalities used alone (gradient-boosted survival model)
  late             one boosted model per modality, standardised risk scores averaged (a vote)
  early_linear     ridge-penalised Cox model on the concatenated features (cannot see interactions)
  early_nonlinear  gradient-boosted survival model on the concatenated features
plus the oracle C-index of the true linear predictor, and report the held-out C-index.
The generator's own quick PID estimate is NOT used: it reads synergy on data with none planted.

Story 2 reuses synthegra/results/pid_synergy_nsweep_resolution.csv (part B_resolution),
the per-seed measured synergy vs cohort size behind fig5 of the paper.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent
SYNTHEGRA = DEMO.parent / "synthegra"
sys.path.insert(0, str(SYNTHEGRA))
sys.path.insert(0, str(SYNTHEGRA / "experiments"))

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ design
SYNERGY = [0.0, 1.0, 2.0, 4.0, 8.0]          # interaction_strength on 2 cross-modal unique pairs
SYNERGY_LABELS = {0.0: "none", 1.0: "weak", 2.0: "moderate", 4.0: "strong", 8.0: "very strong"}
SHARED = [0.0, 0.25, 0.5]                    # column fraction of latent factors shared by both layers
SHARED_LABELS = {0.0: "0 % shared", 0.25: "25 % shared", 0.5: "50 % shared"}
NS = [150, 400, 1000]
SEEDS = list(range(1, 9))
K, P, N_PAIRS, ASYM, NOISE = 8, 40, 2, 0.5, 0.1
N_FOLDS, RIDGE_ALPHA = 3, 10.0
MODELS = [
    dict(key="best_single", label="Best single layer", color="#9a9a9a"),
    dict(key="late", label="Late fusion (vote)", color="#d9a066"),
    dict(key="early_linear", label="Early fusion, linear", color="#99AB9F"),
    dict(key="early_nonlinear", label="Early fusion, non-linear", color="#2F434A"),
]
MODEL_KEYS = [m["key"] for m in MODELS]
GB = dict(n_estimators=150, learning_rate=0.1, max_depth=2, subsample=0.8, random_state=0)


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ one cell
def _fit_cell(args):
    synergy, shared, n, seed = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from sklearn.preprocessing import StandardScaler
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv
    from doe_runner import generate_cv_splits
    from main_v2 import simulate_data_v2
    from pid_simplex_sweep import _build_v2_config

    cfg, _, _ = _build_v2_config(R_col=shared, asymmetry=ASYM, n_pairs=N_PAIRS,
                                 risk_form="additive", seed=seed, N=n, P=P, K=K,
                                 interaction_strength=synergy, noise_std=NOISE)
    if cfg is None:
        return dict(key=(synergy, shared, n, seed), skipped=True)
    res = simulate_data_v2(config=cfg)
    X, time, status = res.X, np.asarray(res.time, float), np.asarray(res.status, int)
    splits = generate_cv_splits(N=n, n_folds=N_FOLDS, seed=seed, stratify=status)

    def gb():
        return GradientBoostingSurvivalAnalysis(**GB)

    eta = np.asarray(res.eta, float)
    scores = {k: [] for k in ["mod0", "mod1", "late", "early_linear", "early_nonlinear", "oracle"]}
    for tr, _va, te in splits:
        y_tr = Surv.from_arrays(status[tr].astype(bool), time[tr])
        ev_te, t_te = status[te].astype(bool), time[te]
        Xs_tr, Xs_te = [], []
        for m in range(2):
            sc = StandardScaler().fit(X[m][tr])
            Xs_tr.append(sc.transform(X[m][tr])); Xs_te.append(sc.transform(X[m][te]))
        # unimodal + late
        preds = []
        for m in range(2):
            mdl = gb().fit(Xs_tr[m], y_tr)
            p = mdl.predict(Xs_te[m])
            scores[f"mod{m}"].append(concordance_index_censored(ev_te, t_te, p)[0])
            preds.append((p - p.mean()) / (p.std() + 1e-9))
        scores["late"].append(concordance_index_censored(ev_te, t_te, np.mean(preds, axis=0))[0])
        # early, non-linear
        mdl = gb().fit(np.hstack(Xs_tr), y_tr)
        scores["early_nonlinear"].append(concordance_index_censored(ev_te, t_te, mdl.predict(np.hstack(Xs_te)))[0])
        # early, linear (ridge Cox); increase the penalty if the solver fails
        for alpha in (RIDGE_ALPHA, 10 * RIDGE_ALPHA, 100 * RIDGE_ALPHA):
            try:
                cox = CoxPHSurvivalAnalysis(alpha=alpha, n_iter=200).fit(np.hstack(Xs_tr), y_tr)
                scores["early_linear"].append(concordance_index_censored(ev_te, t_te, cox.predict(np.hstack(Xs_te)))[0])
                break
            except Exception:
                continue
        else:
            scores["early_linear"].append(float("nan"))
        # oracle: the true linear predictor
        scores["oracle"].append(concordance_index_censored(ev_te, t_te, eta[te])[0])

    out = {k: float(np.nanmean(v)) for k, v in scores.items()}
    out["best_single"] = max(out["mod0"], out["mod1"])
    alloc = cfg.pid.to_latent_allocation(K, 2)
    planted = dict(n_shared=int(alloc["n_shared"]), n_unique_per_mod=int(alloc["n_unique_per_mod"]),
                   n_pairs=N_PAIRS if synergy > 0 else 0, strength=synergy)
    return dict(key=(synergy, shared, n, seed), cindex=out, planted=planted, event_rate=float(status.mean()))


def _summ(vals):
    v = np.asarray(vals, float)
    m, sd = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return [round(m, 4), round(m - 1.96 * sd, 4), round(m + 1.96 * sd, 4)]


def build_grid(quick: bool, procs: int) -> dict:
    syn, sh, ns, seeds = (SYNERGY[::2], SHARED[::2], NS[1:2], SEEDS[:2]) if quick else (SYNERGY, SHARED, NS, SEEDS)
    jobs = [(s, r, n, seed) for s in syn for r in sh for n in ns for seed in seeds]
    print(f"{len(jobs)} simulations on {procs} processes", flush=True)
    rows = []
    with Pool(procs) as pool:
        for i, r in enumerate(pool.imap_unordered(_fit_cell, jobs), 1):
            rows.append(r)
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)
    cells = []
    for s in syn:
        for r in sh:
            for n in ns:
                got = [x for x in rows if x["key"][:3] == (s, r, n) and not x.get("skipped")]
                if not got:
                    continue
                cind = {k: _summ([g["cindex"][k] for g in got]) for k in MODEL_KEYS + ["oracle"]}
                cells.append(dict(synergy=s, shared=r, n=n, seeds=len(got), cindex=cind, planted=got[0]["planted"],
                                  event_rate=round(float(np.mean([g["event_rate"] for g in got])), 3)))
    commit = _git_commit(SYNTHEGRA)
    prov = (
        f"<p>Generator: Synthegra <code>main_v2.simulate_data_v2</code>, mode <code>pid_controlled</code> "
        f"(commit {commit}), two modalities with {P} features each from K = {K} latent factors, "
        f"noise σ = {NOISE}, asymmetric unique weights (β = 1.5 / 0.5), censoring 30 %. "
        f"Synergy = strength of {N_PAIRS} planted cross-modal interactions between unique latent factors "
        f"(<code>InteractionConfig.cross_modal_unique_pairs</code>); shared signal = fraction of latent columns "
        f"loaded by both modalities (<code>PIDConfig.redundancy</code>).</p>"
        f"<p>Models (scikit-survival), {N_FOLDS}-fold stratified CV, held-out C-index: best single layer and "
        f"early non-linear fusion = gradient-boosted survival models (Cox partial likelihood, {GB['n_estimators']} trees, "
        f"depth {GB['max_depth']}); late fusion = the per-layer boosted models with standardised risk scores averaged; "
        f"early linear fusion = ridge-penalised Cox model (α = {RIDGE_ALPHA:g}) on the concatenated features. "
        f"Oracle = C-index of the true linear predictor used to generate the event times. "
        f"Bars: mean over {len(seeds)} seeds; whiskers: mean ± 1.96 SD across seeds (seed-to-seed spread, not a standard error).</p>"
        f"<p>Why no redundancy / uniqueness / synergy read-out: the generator's quick built-in PID estimate reports "
        f"substantial synergy on cohorts with no planted interaction, so it is not shown. The planted structure is exact by construction.</p>"
    )
    return dict(
        meta=dict(models=MODELS, seeds=len(seeds),
                  generated=dt.date.today().isoformat(), synthegra_commit=commit, provenance_html=prov),
        axes=dict(synergy=syn, shared=sh, n=ns),
        labels=dict(synergy={f"{k:g}": v for k, v in SYNERGY_LABELS.items() if k in syn},
                    shared={f"{k:g}": v for k, v in SHARED_LABELS.items() if k in sh}),
        defaults=dict(synergy=2.0 if 2.0 in syn else syn[0], shared=0.25 if 0.25 in sh else sh[0], n=400 if 400 in ns else ns[0]),
        cells=cells,
    )


def build_resolution() -> dict:
    import pandas as pd
    csv = SYNTHEGRA / "results" / "pid_synergy_nsweep_resolution.csv"
    df = pd.read_csv(csv)
    df = df[(df["part"] == "B_resolution") & (df["source"] == "Z_oracle")]
    ns = sorted(int(n) for n in df["N"].unique())
    per = {str(n): [round(float(x), 5) for x in df.loc[df["N"] == n, "S_bits"]] for n in ns}
    share = {str(n): [round(float(x), 4) for x in df.loc[df["N"] == n, "S_share"]] for n in ns}
    mi = {str(n): [round(float(x), 4) for x in df.loc[df["N"] == n, "MI"]] for n in ns}
    prov = (
        f"<p>Source: <code>synthegra/results/pid_synergy_nsweep_resolution.csv</code> "
        f"(script <code>experiments/pid_synergy_nsweep.py</code>, commit {_git_commit(SYNTHEGRA)}), "
        f"the numbers behind figure 5 of the manuscript. Planted structure: two modalities, K = 16 latent factors, "
        f"R/S/U = 0.1/0.3/0.6, two cross-modal interaction pairs with strength λ = {float(df['lam'].iloc[0]):g}. "
        f"Synergy is estimated with the robust BROJA estimator on the oracle latent blocks "
        f"(<code>compute_broja_pid_multidim_robust</code>, occupancy binning, null-corrected), "
        f"{len(per[str(ns[0])])} seeds per cohort size. Points are single seeds; the band is the 2.5–97.5 % range across seeds. "
        f"Synergy estimates are lower bounds.</p>"
    )
    return dict(n=ns, per_seed=per, share=share, mi=mi, planted=None,
                title="Same planted synergy at every n: only the measurement changes", provenance_html=prov)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny grid for smoke testing")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--grid-only", action="store_true")
    ap.add_argument("--resolution-only", action="store_true")
    a = ap.parse_args()
    (DEMO / "data").mkdir(exist_ok=True)
    if not a.grid_only:
        json.dump(build_resolution(), open(DEMO / "data" / "resolution.json", "w"), indent=1)
        print("wrote data/resolution.json")
    if a.resolution_only:
        sys.exit(0)
    g = build_grid(a.quick, a.procs)
    out = DEMO / "data" / ("grid_quick.json" if a.quick else "grid.json")
    json.dump(g, open(out, "w"), indent=1)
    print(f"wrote {out.relative_to(DEMO)} with {len(g['cells'])} cells")
