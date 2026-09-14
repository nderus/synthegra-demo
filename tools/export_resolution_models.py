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
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

NS = [150, 300, 600, 1500, 3000]
SEEDS = list(range(10))
LAM, K, P = 10.0, 16, [64, 64]
N_FOLDS = 3
HIDDEN = (16, 64)
LRS = (1e-3, 3e-3)


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
    from doe_runner import generate_cv_splits
    import demolib
    res = _sim(n, seed)
    X, time, status = res.X, np.asarray(res.time, float), np.asarray(res.status, int)
    eta = np.asarray(res.eta, float)
    splits = generate_cv_splits(N=n, n_folds=N_FOLDS, seed=seed, stratify=status)
    sk = demolib.score_sksurv(X, time, status, eta, splits)
    nn_uni = demolib.score_nn_unimodal(X, time, status, splits, seed)
    nn_int = demolib.score_nn_intermediate(X, time, status, splits, seed, hidden=HIDDEN, lrs=LRS)
    out = demolib.combine(sk, nn_uni, nn_int)
    return dict(n=n, seed=seed, scores=out, best=nn_int, event_rate=float(status.mean()))


def main(quick: bool, procs: int):
    ns, seeds = (NS[1:3], SEEDS[:2]) if quick else (NS, SEEDS)
    jobs = [(n, s) for n in ns for s in seeds]
    print(f"{len(jobs)} cohorts on {procs} processes", flush=True)
    rows = []
    with get_context("spawn").Pool(procs) as pool:
        for i, r in enumerate(pool.imap_unordered(_cell, jobs), 1):
            rows.append(r)
            sc = r["scores"]
            print(f"  {i}/{len(jobs)}  n={r['n']} seed={r['seed']} best={r['best']['strategy'] if r['best'] else None} "
                  f"C: single={sc['best_single']['c']:.3f} lin={sc['early_linear']['c']:.3f} "
                  f"nonlin={sc['early_nonlinear']['c']:.3f} inter={sc['intermediate']['c']:.3f} | "
                  f"IBS: single={sc['best_single']['ibs']:.3f} nonlin={sc['early_nonlinear']['ibs']:.3f}", flush=True)
    keys = ["best_single", "late", "early_linear", "early_nonlinear", "intermediate", "oracle"]
    rnd = lambda x: (round(float(x), 4) if x is not None and np.isfinite(x) else None)
    per_seed = {str(n): {k: [rnd(r["scores"][k]["c"]) for r in rows if r["n"] == n] for k in keys} for n in ns}
    per_seed_ibs = {str(n): {k: [rnd(r["scores"][k]["ibs"]) for r in rows if r["n"] == n] for k in keys if k != "oracle"} for n in ns}
    winners = {str(n): {} for n in ns}
    for r in rows:
        if r["best"]:
            w = winners[str(r["n"])]; w[r["best"]["strategy"]] = w.get(r["best"]["strategy"], 0) + 1
    prov = (
        f"<p>Same generator and planted structure as the PID sweep (K = {K} latent factors, R/S/U = 0.1/0.3/0.6, "
        f"two cross-modal interaction pairs at λ = {LAM:g}, {P[0]} + {P[1]} features), simulated at each cohort size with "
        f"{len(seeds)} seeds (Synthegra commit {_git_commit(SYNTHEGRA)}). {N_FOLDS}-fold stratified CV, held-out C-index.</p>"
        f"<p>Early linear = ridge Cox; early non-linear = boosted survival model (150 trees, depth 2); "
        f"intermediate = the best of Synthegra's neural intermediate-fusion architectures (simple sum, temporal attention, "
        f"cross-attention, GMU, grouped Kronecker) × hidden size {list(HIDDEN)} × learning rate {list(LRS)}, "
        f"selected per cohort on the validation C-index and reported on the test folds. "
        f"Best single layer = best of the two layers alone (boosted model or unimodal neural net). "
        f"IBS = integrated Brier score over 100 time points across the test fold's follow-up with Kaplan-Meier censoring "
        f"weights (pycox EvalSurv), lower is better; for IBS the gain is best-single minus model, so positive is still better. "
        f"Lines: mean gain over the best single layer; bands: 2.5–97.5 % across seeds.</p>"
    )
    json.dump(dict(n=ns, per_seed=per_seed, per_seed_ibs=per_seed_ibs, winners=winners, seeds=len(seeds),
                   generated=dt.date.today().isoformat(), provenance_html=prov),
              open(DEMO / "data" / ("resolution_models_quick.json" if quick else "resolution_models.json"), "w"), indent=1)
    print("wrote data/resolution_models" + ("_quick" if quick else "") + ".json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    main(a.quick, a.procs)
