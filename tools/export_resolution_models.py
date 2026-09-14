"""Model-based read-out of synergy vs cohort size (tab 2, right panel).

Same planted structure as the paper's PID resolution sweep (experiments/pid_synergy_nsweep.py:
K = 16, R/S/U = 0.1/0.3/0.6, two cross-modal interaction pairs, lambda = 10, P = 64 + 64),
simulated at several cohort sizes. For each cohort (3-fold stratified CV, held-out C-index):

  best_single      best of the two layers alone (boosted survival model or unimodal neural net)
  early_linear     ridge-penalised Cox on the concatenated features
  early_nonlinear  boosted survival model on the concatenated features
  intermediate     best neural intermediate-fusion model in synthegra, chosen per cohort on the
                   VALIDATION C-index over strategies x hidden size x learning rate
  oracle           true linear predictor

Run inside the `synthegra` conda env:
    OMP_NUM_THREADS=1 python tools/export_resolution_models.py [--quick] [--procs 4]
Writes data/resolution_models.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent
SYNTHEGRA = DEMO.parent / "synthegra"
sys.path.insert(0, str(SYNTHEGRA))
sys.path.insert(0, str(SYNTHEGRA / "experiments"))
warnings.filterwarnings("ignore")

NS = [150, 300, 600, 1500, 3000]
SEEDS = list(range(10))
LAM, K, P = 10.0, 16, [64, 64]
N_FOLDS, RIDGE_ALPHA = 3, 10.0
GB = dict(n_estimators=150, learning_rate=0.1, max_depth=2, subsample=0.8, random_state=0)
STRATEGIES = ["intermediate_simple_sum", "intermediate_temp_attention", "intermediate_cross_attention",
              "intermediate_gmu", "intermediate_grouped_kronecker"]
HIDDEN = [16, 64]
LRS = [1e-3, 3e-3]
NN = dict(n_epochs=200, patience=40, batch_size=64, l1_lambda=1e-5)


def _git_commit(repo: Path) -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _sim(n, seed):
    from main_v2 import InteractionConfig, PIDConfig, SimulationConfig, simulate_data_v2
    pid = PIDConfig(redundancy=0.1, synergy=0.3, uniqueness=0.6, risk_form="additive")
    inter = InteractionConfig.cross_modal_unique_pairs(pid, K=K, M=2, n_pairs=2, interaction_strength=LAM)
    return simulate_data_v2(config=SimulationConfig(N=n, M=2, K=K, P=P, seed=seed, pid=pid, interaction=inter))


def _cell(args):
    n, seed = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    from sklearn.preprocessing import StandardScaler
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv
    from doe_runner import generate_cv_splits
    from utils_nn import fit_integrative_model_nn, fit_unimodal_models_nn

    res = _sim(n, seed)
    X, time, status = res.X, np.asarray(res.time, float), np.asarray(res.status, int)
    eta = np.asarray(res.eta, float)
    splits = generate_cv_splits(N=n, n_folds=N_FOLDS, seed=seed, stratify=status)

    sc = {k: [] for k in ["gb0", "gb1", "early_linear", "early_nonlinear", "oracle"]}
    for tr, _va, te in splits:
        y_tr = Surv.from_arrays(status[tr].astype(bool), time[tr])
        ev, tt = status[te].astype(bool), time[te]
        Xs_tr, Xs_te = [], []
        for m in range(2):
            s = StandardScaler().fit(X[m][tr]); Xs_tr.append(s.transform(X[m][tr])); Xs_te.append(s.transform(X[m][te]))
            mdl = GradientBoostingSurvivalAnalysis(**GB).fit(Xs_tr[m], y_tr)
            sc[f"gb{m}"].append(concordance_index_censored(ev, tt, mdl.predict(Xs_te[m]))[0])
        mdl = GradientBoostingSurvivalAnalysis(**GB).fit(np.hstack(Xs_tr), y_tr)
        sc["early_nonlinear"].append(concordance_index_censored(ev, tt, mdl.predict(np.hstack(Xs_te)))[0])
        for alpha in (RIDGE_ALPHA, 10 * RIDGE_ALPHA, 100 * RIDGE_ALPHA):
            try:
                cox = CoxPHSurvivalAnalysis(alpha=alpha, n_iter=200).fit(np.hstack(Xs_tr), y_tr)
                sc["early_linear"].append(concordance_index_censored(ev, tt, cox.predict(np.hstack(Xs_te)))[0]); break
            except Exception:
                continue
        else:
            sc["early_linear"].append(float("nan"))
        sc["oracle"].append(concordance_index_censored(ev, tt, eta[te])[0])
    out = {k: float(np.nanmean(v)) for k, v in sc.items()}

    # unimodal neural nets (default HPs)
    uni, _ = fit_unimodal_models_nn(X=X, time=time, status=status, splits=splits, seed=seed,
                                    hidden_dim=16, lr=1e-3, **NN)
    nn_uni = [float(uni[f"Modality_{m}"]["avg_c_index"]) for m in range(2)]
    out["best_single"] = max(out["gb0"], out["gb1"], *nn_uni)
    out["best_single_nn"] = max(nn_uni)

    # intermediate fusion: HP search selected on validation C-index
    best = None
    trials = []
    for strat in STRATEGIES:
        for hd in HIDDEN:
            for lr in LRS:
                try:
                    m, _ = fit_integrative_model_nn(X=X, time=time, status=status, splits=splits, seed=seed,
                                                    integration_strategy=strat, hidden_dim=hd, lr=lr, **NN)
                    val, test = float(m["avg_val_c_index"]), float(m["avg_c_index"])
                except Exception:
                    val, test = float("nan"), float("nan")
                trials.append(dict(strategy=strat, hidden=hd, lr=lr, val=val, test=test))
                if np.isfinite(val) and (best is None or val > best["val"]):
                    best = trials[-1]
    out["intermediate"] = best["test"] if best else float("nan")
    return dict(n=n, seed=seed, cindex=out, best=best, event_rate=float(status.mean()))


def main(quick: bool, procs: int):
    ns, seeds = (NS[1:3], SEEDS[:2]) if quick else (NS, SEEDS)
    jobs = [(n, s) for n in ns for s in seeds]
    print(f"{len(jobs)} cohorts on {procs} processes", flush=True)
    rows = []
    with get_context("spawn").Pool(procs) as pool:
        for i, r in enumerate(pool.imap_unordered(_cell, jobs), 1):
            rows.append(r)
            print(f"  {i}/{len(jobs)}  n={r['n']} seed={r['seed']} best={r['best']['strategy'] if r['best'] else None} "
                  f"single={r['cindex']['best_single']:.3f} lin={r['cindex']['early_linear']:.3f} "
                  f"nonlin={r['cindex']['early_nonlinear']:.3f} inter={r['cindex']['intermediate']:.3f}", flush=True)
    keys = ["best_single", "early_linear", "early_nonlinear", "intermediate", "oracle", "best_single_nn"]
    per_seed = {str(n): {k: [round(r["cindex"][k], 4) for r in rows if r["n"] == n] for k in keys} for n in ns}
    winners = {str(n): {} for n in ns}
    for r in rows:
        if r["best"]:
            w = winners[str(r["n"])]; w[r["best"]["strategy"]] = w.get(r["best"]["strategy"], 0) + 1
    prov = (
        f"<p>Same generator and planted structure as the PID sweep (K = {K} latent factors, R/S/U = 0.1/0.3/0.6, "
        f"two cross-modal interaction pairs at λ = {LAM:g}, {P[0]} + {P[1]} features), simulated at each cohort size with "
        f"{len(seeds)} seeds (Synthegra commit {_git_commit(SYNTHEGRA)}). {N_FOLDS}-fold stratified CV, held-out C-index.</p>"
        f"<p>Early linear = ridge Cox (α = {RIDGE_ALPHA:g}); early non-linear = boosted survival model "
        f"({GB['n_estimators']} trees, depth {GB['max_depth']}); intermediate = the best of Synthegra's neural "
        f"intermediate-fusion architectures ({', '.join(s.replace('intermediate_', '') for s in STRATEGIES)}) × hidden size "
        f"{HIDDEN} × learning rate {LRS}, selected per cohort on the validation C-index and reported on the test folds. "
        f"Best single layer = best of the two layers alone (boosted model or unimodal neural net). "
        f"Lines: mean gain over the best single layer; bands: 2.5–97.5 % across seeds.</p>"
    )
    json.dump(dict(n=ns, per_seed=per_seed, winners=winners, seeds=len(seeds),
                   generated=dt.date.today().isoformat(), provenance_html=prov),
              open(DEMO / "data" / ("resolution_models_quick.json" if quick else "resolution_models.json"), "w"), indent=1)
    print("wrote data/resolution_models" + ("_quick" if quick else "") + ".json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    main(a.quick, a.procs)
