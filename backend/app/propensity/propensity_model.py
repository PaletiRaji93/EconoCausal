"""Propensity-score model + overlap diagnostics for BINARY treatment.

- Out-of-fold scores (cross_val_predict) are used for diagnostics to avoid optimistic overfit.
- In randomised data (Criteo) the true propensity is constant; a good model gives AUC ~ 0.5.
- Weighting (stabilised, trimmed IPW) is provided for balance checks and weighted uplift metrics.
- NOT valid for continuous treatment; use a treatment-density approach (not implemented here).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from backend.app.preprocessing.feature_selection import build_preprocessor

logger = logging.getLogger(__name__)


def fit_propensity(X: pd.DataFrame, t: np.ndarray, seed: int = 42, cv: int = 5,
                   model: str = "logistic") -> tuple[Pipeline, np.ndarray]:
    """Return (pipeline fitted on all rows, out-of-fold propensity scores)."""
    t = np.asarray(t).astype(int)
    if set(np.unique(t)) - {0, 1}:
        raise ValueError("Propensity model requires binary 0/1 treatment.")
    if model == "logistic":
        clf = LogisticRegression(max_iter=2000)
    elif model == "gbm":
        clf = HistGradientBoostingClassifier(max_iter=100, max_depth=3, random_state=seed)
    else:
        raise ValueError("model must be 'logistic' or 'gbm'")
    pipe = Pipeline([("prep", build_preprocessor(X)), ("clf", clf)])
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    oof = cross_val_predict(pipe, X, t, cv=skf, method="predict_proba")[:, 1]
    pipe.fit(X, t)
    return pipe, oof


def ipw_weights(e: np.ndarray, t: np.ndarray, trim: tuple[float, float] = (0.01, 0.99),
                stabilized: bool = True) -> np.ndarray:
    """Inverse-probability weights with trimming of extreme propensities."""
    t = np.asarray(t).astype(int)
    e = np.clip(np.asarray(e, float), *trim)
    w = t / e + (1 - t) / (1 - e)
    if stabilized:
        p = t.mean()
        w = np.where(t == 1, p, 1 - p) * w
    return w


def _ess(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / (np.square(w).sum() + 1e-12)) if len(w) else 0.0


def overlap_diagnostics(e: np.ndarray, t: np.ndarray, extreme: float = 0.05) -> dict:
    """Common-support and overlap summary with human-readable warnings."""
    e = np.asarray(e, float)
    t = np.asarray(t).astype(int)
    e1, e0 = e[t == 1], e[t == 0]
    lo, hi = max(e1.min(), e0.min()), min(e1.max(), e0.max())
    outside = float(np.mean((e < lo) | (e > hi))) if hi > lo else 1.0
    extreme_frac = float(np.mean((e < extreme) | (e > 1 - extreme)))
    w = ipw_weights(e, t)
    diag = {
        "auc_oof": float(roc_auc_score(t, e)),
        "treated": {"n": int(len(e1)), "min": float(e1.min()), "mean": float(e1.mean()), "max": float(e1.max())},
        "control": {"n": int(len(e0)), "min": float(e0.min()), "mean": float(e0.mean()), "max": float(e0.max())},
        "common_support": [float(lo), float(hi)],
        "frac_outside_common_support": outside,
        "frac_extreme_propensity": extreme_frac,
        "ess_treated": _ess(w[t == 1]), "ess_control": _ess(w[t == 0]),
    }
    warnings = []
    if hi <= lo:
        warnings.append("No common support: treated and control propensity ranges do not overlap.")
    if outside > 0.05:
        warnings.append(f"{outside:.1%} of rows lie outside common support - effects there are extrapolated.")
    if extreme_frac > 0.10:
        warnings.append(f"{extreme_frac:.1%} of rows have propensity <{extreme} or >{1-extreme}: weak overlap.")
    if diag["ess_treated"] < 0.5 * len(e1) or diag["ess_control"] < 0.5 * len(e0):
        warnings.append("Effective sample size after weighting is below 50% of group size.")
    diag["warnings"] = warnings
    diag["overlap_ok"] = not warnings
    return diag


def plot_propensity(e: np.ndarray, t: np.ndarray, path: str | Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.asarray(t).astype(int)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bins = np.linspace(0, 1, 41)
    ax.hist(e[t == 0], bins=bins, alpha=0.6, density=True, label="control")
    ax.hist(e[t == 1], bins=bins, alpha=0.6, density=True, label="treated")
    ax.set_xlabel("estimated propensity P(T=1|X) (out-of-fold)")
    ax.set_ylabel("density")
    ax.set_title("Propensity overlap (estimated)")
    ax.legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
