"""Tab 2: one planted structure from the tab-1 grid, read at many cohort sizes by two instruments.

Structure (identical to the tab-1 cell "very strong synergy, 25 % shared"): K = 8 latent factors,
2 shared + 3 unique per layer, two cross-layer interaction pairs at strength 8, additive risk,
40 features per layer, noise 0.1, asymmetric unique weights, censoring 30 %.

For each cohort size and seed:
  read-out 1  synergy by robust BROJA PID on the oracle latent blocks (same estimator and
              settings as the paper's resolution sweep: occupancy binning, 2 KMeans seeds,
              60 permutations, target = risk cut into 5 quantile classes)
  read-out 2  held-out C-index and IBS of best single layer / late / early linear /
              early non-linear / best neural intermediate fusion (demolib, as in tab 1)

Run inside the `synthegra` conda env:
    OMP_NUM_THREADS=1 python tools/export_tab2.py [--quick] [--procs 4]
Writes data/tab2.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent
SYNTHEGRA = DEMO.parent / "synthegra"
for p in (SYNTHEGRA, SYNTHEGRA / "experiments", HERE):
    sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")

NS = [100, 150, 250, 400, 700, 1000, 2000, 3000]
SEEDS = list(range(10))
SYNERGY, SHARED = 8.0, 0.25
K, P, N_PAIRS, ASYM, NOISE = 8, 40, 2, 0.5, 0.1
N_FOLDS = 3
NN_STRATEGIES = ["intermediate_simple_sum", "intermediate_gmu", "intermediate_grouped_kronecker"]
PID = dict(kmeans_seeds=(0, 1), n_perm=60, n_bins="occupancy")
MODEL_KEYS = ["best_single", "late", "early_linear", "early_nonlinear", "intermediate"]


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _qbin(v, k=5):
    v = np.asarray(v, float)
    return np.digitize(v, np.quantile(v, np.linspace(0, 1, k + 1)[1:-1]))


def _zblocks(pid, k, Z):
    al = pid.to_latent_allocation(k, M=2)
    ns, us, nu = al["n_shared"], al["unique_start"], al["n_unique_per_mod"]
    sh = list(range(ns))
    return [Z[:, sh + list(range(us, us + nu))], Z[:, sh + list(range(us + nu, us + 2 * nu))]]


def _cell(args):
    n, seed = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    from doe_runner import generate_cv_splits
    from main_v2 import simulate_data_v2
    from pid_simplex_sweep import _build_v2_config
    from utils import compute_broja_pid_multidim_robust
    import demolib

    cfg, _, _ = _build_v2_config(R_col=SHARED, asymmetry=ASYM, n_pairs=N_PAIRS, risk_form="additive",
                                 seed=seed, N=n, P=P, K=K, interaction_strength=SYNERGY, noise_std=NOISE)
    res = simulate_data_v2(config=cfg)
    X, time, status = res.X, np.asarray(res.time, float), np.asarray(res.status, int)
    eta, Z = np.asarray(res.eta, float), np.asarray(res.Z, float)

    # read-out 1: PID on oracle latent blocks
    o = compute_broja_pid_multidim_robust(_zblocks(cfg.pid, K, Z), None, target_labels=_qbin(eta), **PID)
    S = float(o["point"]["S"]); mi = float(o["total_mi_bits"])
    resolved = int(np.isfinite(S) and np.isfinite(mi))
    pid = dict(share=S if resolved else 0.0, bits=S * mi if resolved else 0.0, mi=mi if np.isfinite(mi) else 0.0,
               resolved=resolved, has_signal=int(o.get("has_signal", 0)), regime=str(o.get("regime", "")))

    # read-out 2: models
    splits = generate_cv_splits(N=n, n_folds=N_FOLDS, seed=seed, stratify=status)
    sk = demolib.score_sksurv(X, time, status, eta, splits)
    nn_uni = demolib.score_nn_unimodal(X, time, status, splits, seed)
    nn_int = demolib.score_nn_intermediate(X, time, status, splits, seed, strategies=NN_STRATEGIES)
    scores = demolib.combine(sk, nn_uni, nn_int)
    return dict(n=n, seed=seed, pid=pid, scores=scores, best=nn_int["strategy"] if nn_int else None,
                event_rate=float(status.mean()))


def main(quick: bool, procs: int):
    ns, seeds = ([150, 1000], SEEDS[:2]) if quick else (NS, SEEDS)
    jobs = [(n, s) for n in ns for s in seeds]
    print(f"{len(jobs)} cohorts on {procs} processes", flush=True)
    rows = []
    with get_context("spawn").Pool(procs) as pool:
        for i, r in enumerate(pool.imap_unordered(_cell, jobs), 1):
            rows.append(r)
            sc = r["scores"]
            print(f"  {i}/{len(jobs)} n={r['n']} seed={r['seed']} S={r['pid']['bits']:.3f}b share={r['pid']['share']:.2f} "
                  f"MI={r['pid']['mi']:.2f} | C single={sc['best_single']['c']:.3f} nonlin={sc['early_nonlinear']['c']:.3f} "
                  f"inter={sc['intermediate']['c']:.3f} ({r['best']})", flush=True)
    rnd = lambda x: (round(float(x), 4) if x is not None and np.isfinite(x) else None)
    by_n = lambda n: [r for r in rows if r["n"] == n]
    out = dict(
        n=ns, seeds=len(seeds), generated=dt.date.today().isoformat(), synthegra_commit=_git_commit(SYNTHEGRA),
        structure=dict(label="Tab-1 structure: very strong synergy, 25 % shared", K=K, n_shared=2, n_unique_per_mod=3,
                       n_pairs=N_PAIRS, strength=SYNERGY, P=P, noise=NOISE),
        pid={str(n): {k: [rnd(r["pid"][k]) for r in by_n(n)] for k in ["bits", "share", "mi", "resolved"]} for n in ns},
        per_seed={str(n): {k: [rnd(r["scores"][k]["c"]) for r in by_n(n)] for k in MODEL_KEYS + ["oracle"]} for n in ns},
        per_seed_ibs={str(n): {k: [rnd(r["scores"][k]["ibs"]) for r in by_n(n)] for k in MODEL_KEYS} for n in ns},
        winners={str(n): {} for n in ns},
        event_rate={str(n): rnd(np.mean([r["event_rate"] for r in by_n(n)])) for n in ns},
    )
    for r in rows:
        if r["best"]:
            w = out["winners"][str(r["n"])]; w[r["best"]] = w.get(r["best"], 0) + 1
    out["provenance_html"] = (
        f"<p>Same generator and planted structure as the tab-1 cell “very strong synergy, 25 % shared” "
        f"(Synthegra <code>pid_controlled</code> mode, commit {out['synthegra_commit']}): K = {K} latent factors, 2 shared and "
        f"3 unique per layer, {N_PAIRS} cross-layer interaction pairs at strength {SYNERGY:g}, additive risk, {P} features per layer, "
        f"noise σ = {NOISE}, asymmetric unique weights, censoring 30 %. Simulated at each cohort size with {len(seeds)} seeds.</p>"
        f"<p>Read-out 1: synergy estimated on the oracle latent blocks of each layer with the robust BROJA estimator "
        f"(<code>compute_broja_pid_multidim_robust</code>: occupancy-capped clustering, {len(PID['kmeans_seeds'])} KMeans seeds, "
        f"{PID['n_perm']} permutations for null correction, Miller-Madow correction), target = true risk cut into 5 quantile classes. "
        f"Synergy estimates are lower bounds; cohorts where the estimator finds no usable signal are recorded as 0 bits "
        f"(unresolved). Points are single seeds, the band is the 2.5–97.5 % range.</p>"
        f"<p>Read-out 2: {N_FOLDS}-fold stratified CV, held-out C-index and IBS, models as in tab 1 (best single layer, late fusion, "
        f"early linear ridge Cox, early non-linear boosted survival model, best of three neural intermediate-fusion architectures × two "
        f"hidden sizes selected on validation C-index). Gains are relative to the best single layer of the same cohort.</p>"
    )
    json.dump(out, open(DEMO / "data" / ("tab2_quick.json" if quick else "tab2.json"), "w"), indent=1)
    print("wrote data/tab2" + ("_quick" if quick else "") + ".json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    main(a.quick, a.procs)
