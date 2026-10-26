"""Uplift evaluation for BINARY treatment: Qini curve/coefficient, AUUC, uplift@k, deciles.

Definitions (all computed on a HELD-OUT test set, ranked by predicted effect, highest first)
- Qini curve Q(k) = Y_T(k) - Y_C(k) * N_T(k) / N_C(k)   (Radcliffe), k = top-k rows.
  Y_T/Y_C: summed outcomes of treated/control among top-k; N_T/N_C: group counts (or weighted counts).
  x-axis is k / N (fraction targeted). Q(N) = estimated total incremental outcomes.
- Random-targeting line: straight line from (0,0) to (1, Q(N)).
- AUUC (this module) = area under the Qini curve over x in [0,1] (trapezoid).
- Qini coefficient = AUUC(model) - AUUC(random line)  (area between curve and random line).
- uplift@k = mean outcome(treated) - mean outcome(control) within the top k fraction.

Validity: the formulas assume treatment is randomised (as in Criteo). For observational data pass
`weights` = inverse-propensity weights; results then depend on the propensity model.
Not applicable to continuous treatment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _prep(y, t, score, weights):
    y = np.asarray(y, float)
    t = np.asarray(t).astype(int)
    s = np.asarray(score, float)
    if not (len(y) == len(t) == len(s)):
        raise ValueError("y, t, score must have the same length")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    if set(np.unique(t)) - {0, 1}:
        raise ValueError("treatment must be binary 0/1")
    if (t == 1).sum() == 0 or (t == 0).sum() == 0:
        raise ValueError("both treated and control rows are required")
    order = np.argsort(-s, kind="mergesort")  # stable -> deterministic ties
    return y[order], t[order], w[order]


def qini_curve(y, t, score, weights=None) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, q): fraction targeted (0..1) and cumulative incremental outcomes."""
    ys, ts, ws = _prep(y, t, score, weights)
    n_t = np.cumsum(ws * ts)
    n_c = np.cumsum(ws * (1 - ts))
    y_t = np.cumsum(ws * ts * ys)
    y_c = np.cumsum(ws * (1 - ts) * ys)
    ratio = np.divide(n_t, n_c, out=np.zeros_like(n_t), where=n_c > 0)
    q = y_t - y_c * ratio
    x = np.arange(1, len(ys) + 1) / len(ys)
    return np.concatenate([[0.0], x]), np.concatenate([[0.0], q])


def auuc(y, t, score, weights=None) -> float:
    x, q = qini_curve(y, t, score, weights)
    return float(np.trapz(q, x) if hasattr(np, "trapz") else np.trapezoid(q, x))


def qini_coefficient(y, t, score, weights=None) -> float:
    x, q = qini_curve(y, t, score, weights)
    rand = x * q[-1]
    diff = q - rand
    return float(np.trapz(diff, x) if hasattr(np, "trapz") else np.trapezoid(diff, x))


def uplift_at_k(y, t, score, k: float = 0.1, weights=None) -> float:
    """Treated-minus-control mean outcome within the top-k fraction (nan if a group is empty)."""
    if not 0 < k <= 1:
        raise ValueError("k must be in (0, 1]")
    ys, ts, ws = _prep(y, t, score, weights)
    m = max(1, int(np.ceil(k * len(ys))))
    ys, ts, ws = ys[:m], ts[:m], ws[:m]
    if ts.sum() == 0 or (1 - ts).sum() == 0:
        return float("nan")
    mt = np.average(ys[ts == 1], weights=ws[ts == 1])
    mc = np.average(ys[ts == 0], weights=ws[ts == 0])
    return float(mt - mc)


def uplift_by_decile(y, t, score, weights=None, n_bins: int = 10) -> pd.DataFrame:
    """Observed treated/control outcome rates per score bin (bin 1 = highest predicted effect)."""
    ys, ts, ws = _prep(y, t, score, weights)
    bins = np.array_split(np.arange(len(ys)), n_bins)
    rows = []
    for i, b in enumerate(bins, 1):
        yt, tt, wt = ys[b], ts[b], ws[b]
        has = tt.sum() > 0 and (1 - tt).sum() > 0
        mt = np.average(yt[tt == 1], weights=wt[tt == 1]) if tt.sum() else np.nan
        mc = np.average(yt[tt == 0], weights=wt[tt == 0]) if (1 - tt).sum() else np.nan
        rows.append({"bin": i, "n": len(b), "n_treated": int(tt.sum()), "n_control": int((1 - tt).sum()),
                     "rate_treated": mt, "rate_control": mc, "uplift": mt - mc if has else np.nan})
    return pd.DataFrame(rows)


def evaluate_scores(y, t, score, weights=None, ks=(0.1, 0.2, 0.3), seed: int = 42,
                    n_random: int = 20) -> dict:
    """All metrics plus a Monte-Carlo random-ranking reference for the same data."""
    out = {"qini_coefficient": qini_coefficient(y, t, score, weights),
           "auuc": auuc(y, t, score, weights),
           "uplift_at_k": {f"{int(k*100)}%": uplift_at_k(y, t, score, k, weights) for k in ks},
           "total_incremental_outcomes": float(qini_curve(y, t, score, weights)[1][-1])}
    rng = np.random.default_rng(seed)
    rnd = [qini_coefficient(y, t, rng.random(len(np.asarray(y))), weights) for _ in range(n_random)]
    out["random_ranking_qini_mean"] = float(np.mean(rnd))
    out["random_ranking_qini_sd"] = float(np.std(rnd))
    return out


def bootstrap_qini_ci(y, t, score, weights=None, n_boot: int = 200, seed: int = 42,
                      alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the Qini coefficient (resamples rows)."""
    y, t, s = np.asarray(y), np.asarray(t), np.asarray(score)
    w = None if weights is None else np.asarray(weights)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if t[i].sum() == 0 or (1 - t[i]).sum() == 0:
            continue
        vals.append(qini_coefficient(y[i], t[i], s[i], None if w is None else w[i]))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def plot_qini(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: str | Path) -> Path:
    """curves: name -> (x, q). Draws each curve plus the random-targeting line of the first."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    first = True
    for name, (x, q) in curves.items():
        ax.plot(x, q, label=name)
        if first:
            ax.plot([0, 1], [0, q[-1]], "k--", label="random targeting")
            first = False
    ax.set_xlabel("fraction of population targeted")
    ax.set_ylabel("cumulative incremental outcomes (estimated)")
    ax.set_title("Qini curves (held-out test set)")
    ax.legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
