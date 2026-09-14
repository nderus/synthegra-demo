"""Shared model fitting for the demo exports.

Every strategy is scored with 3-fold stratified CV on held-out folds with two metrics:
  C-index  (concordance_index_censored, higher is better)
  IBS      integrated Brier score over 100 time points spanning the test fold's follow-up,
           with Kaplan-Meier censoring weights (pycox EvalSurv), lower is better.
           The neural fitters in synthegra use the same EvalSurv call, so numbers are comparable.

Strategies:
  gb0, gb1         gradient-boosted survival model per layer (best_single = better of the two)
  late             per-layer boosted models; C-index from averaged standardised risk scores,
                   IBS from the averaged survival curves
  early_linear     ridge-penalised Cox on concatenated features
  early_nonlinear  boosted survival model on concatenated features
  intermediate     best neural intermediate-fusion model in synthegra, selected on validation C-index
  oracle           true linear predictor (C-index only)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

GB = dict(n_estimators=150, learning_rate=0.1, max_depth=2, subsample=0.8, random_state=0)
RIDGE_ALPHAS = (10.0, 100.0, 1000.0)
STRATEGIES = ["intermediate_simple_sum", "intermediate_temp_attention", "intermediate_cross_attention",
              "intermediate_gmu", "intermediate_grouped_kronecker"]
NN = dict(n_epochs=200, patience=40, batch_size=64, l1_lambda=1e-5)


def _surv_matrix(model, X, grid):
    fns = model.predict_survival_function(X)
    out = np.empty((len(fns), len(grid)))
    for i, fn in enumerate(fns):
        g = np.clip(grid, fn.x[0], fn.x[-1])
        out[i] = fn(g)
    return out


def _ibs(S, grid, t_te, e_te):
    from pycox.evaluation import EvalSurv
    surv = pd.DataFrame(S.T, index=grid)
    ev = EvalSurv(surv, np.asarray(t_te, float), np.asarray(e_te).astype(bool), censor_surv="km")
    return float(ev.integrated_brier_score(grid))


def score_sksurv(X, time, status, eta, splits):
    """Boosted / Cox / late strategies + oracle. Returns {model: {"c": mean, "ibs": mean}}."""
    from sklearn.preprocessing import StandardScaler
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv

    keys = ["gb0", "gb1", "late", "early_linear", "early_nonlinear", "oracle"]
    acc = {k: {"c": [], "ibs": []} for k in keys}
    for tr, _va, te in splits:
        y_tr = Surv.from_arrays(status[tr].astype(bool), time[tr])
        t_te, e_te = time[te], status[te].astype(bool)
        grid = np.linspace(t_te.min(), t_te.max(), 100)
        cidx = lambda p: concordance_index_censored(e_te, t_te, p)[0]
        Xs_tr, Xs_te = [], []
        for m in range(2):
            sc = StandardScaler().fit(X[m][tr])
            Xs_tr.append(sc.transform(X[m][tr])); Xs_te.append(sc.transform(X[m][te]))
        risks, survs = [], []
        for m in range(2):
            mdl = GradientBoostingSurvivalAnalysis(**GB).fit(Xs_tr[m], y_tr)
            p = mdl.predict(Xs_te[m]); S = _surv_matrix(mdl, Xs_te[m], grid)
            acc[f"gb{m}"]["c"].append(cidx(p)); acc[f"gb{m}"]["ibs"].append(_ibs(S, grid, t_te, e_te))
            risks.append((p - p.mean()) / (p.std() + 1e-9)); survs.append(S)
        acc["late"]["c"].append(cidx(np.mean(risks, axis=0)))
        acc["late"]["ibs"].append(_ibs(np.mean(survs, axis=0), grid, t_te, e_te))
        A_tr, A_te = np.hstack(Xs_tr), np.hstack(Xs_te)
        mdl = GradientBoostingSurvivalAnalysis(**GB).fit(A_tr, y_tr)
        acc["early_nonlinear"]["c"].append(cidx(mdl.predict(A_te)))
        acc["early_nonlinear"]["ibs"].append(_ibs(_surv_matrix(mdl, A_te, grid), grid, t_te, e_te))
        for alpha in RIDGE_ALPHAS:
            try:
                cox = CoxPHSurvivalAnalysis(alpha=alpha, n_iter=200).fit(A_tr, y_tr)
                acc["early_linear"]["c"].append(cidx(cox.predict(A_te)))
                acc["early_linear"]["ibs"].append(_ibs(_surv_matrix(cox, A_te, grid), grid, t_te, e_te))
                break
            except Exception:
                continue
        else:
            acc["early_linear"]["c"].append(np.nan); acc["early_linear"]["ibs"].append(np.nan)
        acc["oracle"]["c"].append(cidx(eta[te])); acc["oracle"]["ibs"].append(np.nan)
    return {k: {"c": float(np.nanmean(v["c"])), "ibs": float(np.nanmean(v["ibs"])) if np.isfinite(v["ibs"]).any() else None}
            for k, v in acc.items()}


def score_nn_unimodal(X, time, status, splits, seed):
    from utils_nn import fit_unimodal_models_nn
    uni, _ = fit_unimodal_models_nn(X=X, time=time, status=status, splits=splits, seed=seed,
                                    hidden_dim=16, lr=1e-3, **NN)
    return [{"c": float(uni[f"Modality_{m}"]["avg_c_index"]), "ibs": float(uni[f"Modality_{m}"].get("avg_ibs", np.nan))}
            for m in range(2)]


def score_nn_intermediate(X, time, status, splits, seed, strategies=STRATEGIES, hidden=(16, 64), lrs=(1e-3,)):
    """Search architectures x hidden x lr; pick by validation C-index; report test C and IBS."""
    from utils_nn import fit_integrative_model_nn
    best = None
    for strat in strategies:
        for hd in hidden:
            for lr in lrs:
                try:
                    m, _ = fit_integrative_model_nn(X=X, time=time, status=status, splits=splits, seed=seed,
                                                    integration_strategy=strat, hidden_dim=hd, lr=lr, **NN)
                    val, c, ibs = float(m["avg_val_c_index"]), float(m["avg_c_index"]), float(m.get("avg_ibs", np.nan))
                except Exception:
                    continue
                if np.isfinite(val) and (best is None or val > best["val"]):
                    best = dict(strategy=strat, hidden=hd, lr=lr, val=val, c=c, ibs=ibs)
    return best


def combine(sk, nn_uni, nn_int):
    """Assemble per-model {c, ibs} dicts with best_single = best over boosted and neural unimodal."""
    out = {k: dict(v) for k, v in sk.items() if k not in ("gb0", "gb1")}
    singles = [sk["gb0"], sk["gb1"]] + nn_uni
    out["best_single"] = {"c": max(s["c"] for s in singles),
                          "ibs": min(s["ibs"] for s in singles if s["ibs"] is not None and np.isfinite(s["ibs"]))}
    out["intermediate"] = {"c": nn_int["c"], "ibs": nn_int["ibs"]} if nn_int else {"c": np.nan, "ibs": np.nan}
    return out
