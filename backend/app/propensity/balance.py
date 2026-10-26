"""Covariate balance: standardised mean differences (SMD), before and after weighting."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def _wmean_var(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    m = np.average(x, weights=w)
    v = np.average((x - m) ** 2, weights=w)
    return float(m), float(v)


def standardized_mean_diff(x: np.ndarray, t: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    """(mean_treated - mean_control) / sqrt((var_t + var_c)/2). 0 if both variances are 0."""
    x = np.asarray(x, float)
    t = np.asarray(t).astype(int)
    w = np.ones(len(x)) if w is None else np.asarray(w, float)
    m1, v1 = _wmean_var(x[t == 1], w[t == 1])
    m0, v0 = _wmean_var(x[t == 0], w[t == 0])
    denom = np.sqrt((v1 + v0) / 2.0)
    return 0.0 if denom < 1e-12 else float((m1 - m0) / denom)


def balance_table(Xp: np.ndarray | pd.DataFrame, t: np.ndarray, feature_names: Sequence[str],
                  weights: Optional[np.ndarray] = None) -> pd.DataFrame:
    """SMD per (processed) feature before and, if weights given, after weighting."""
    arr = Xp.to_numpy(float) if isinstance(Xp, pd.DataFrame) else np.asarray(Xp, float)
    rows = []
    for j, name in enumerate(feature_names):
        before = standardized_mean_diff(arr[:, j], t)
        after = standardized_mean_diff(arr[:, j], t, weights) if weights is not None else np.nan
        rows.append({"feature": name, "smd_before": before, "smd_after": after})
    return pd.DataFrame(rows)


def balance_summary(table: pd.DataFrame, threshold: float = 0.1) -> dict:
    """Aggregate balance with warnings (|SMD| > 0.1 is the usual rule of thumb)."""
    out = {"threshold": threshold,
           "max_abs_smd_before": float(table["smd_before"].abs().max()),
           "n_imbalanced_before": int((table["smd_before"].abs() > threshold).sum())}
    warnings = []
    if out["n_imbalanced_before"]:
        warnings.append(f"{out['n_imbalanced_before']} features have |SMD|>{threshold} before adjustment "
                        "(expected in observational data; unexpected in a randomised experiment).")
    if table["smd_after"].notna().any():
        out["max_abs_smd_after"] = float(table["smd_after"].abs().max())
        out["n_imbalanced_after"] = int((table["smd_after"].abs() > threshold).sum())
        if out["n_imbalanced_after"]:
            warnings.append(f"{out['n_imbalanced_after']} features remain imbalanced after weighting.")
    out["warnings"] = warnings
    return out


def plot_balance(table: pd.DataFrame, path: str | Path, threshold: float = 0.1) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tb = table.reindex(table["smd_before"].abs().sort_values().index)
    fig, ax = plt.subplots(figsize=(6, max(3, 0.28 * len(tb) + 1)))
    y = np.arange(len(tb))
    ax.scatter(tb["smd_before"].abs(), y, label="before", marker="o")
    if tb["smd_after"].notna().any():
        ax.scatter(tb["smd_after"].abs(), y, label="after weighting", marker="x")
    ax.axvline(threshold, ls="--", c="grey")
    ax.set_yticks(y)
    ax.set_yticklabels(tb["feature"])
    ax.set_xlabel("|standardised mean difference|")
    ax.set_title("Covariate balance (love plot)")
    ax.legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
