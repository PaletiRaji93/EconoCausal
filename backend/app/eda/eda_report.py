"""Reproducible EDA: tables (JSON) + figures (PNG) under reports/<run>/eda/.

EDA describes OBSERVED data. Differences between treatment groups here are NOT causal effects
unless treatment was randomised.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from backend.app.preprocessing.dataset import CausalDataset
from backend.app.propensity.balance import standardized_mean_diff

logger = logging.getLogger(__name__)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    _plt().close(fig)
    return path.name


def run_eda(ds: CausalDataset, out_dir: str | Path, max_hist: int = 12) -> dict:
    """Compute EDA summary, save figures and eda_summary.json; return the summary dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    df, tcol, ycol = ds.df, ds.treatment_col, ds.outcome_col
    binary_t = set(df[tcol].dropna().unique()) <= {0, 1}
    group = df[tcol] if binary_t else (df[tcol] > 0).astype(int)
    warnings: list[str] = []
    figures: list[str] = []

    num_feats = [c for c in ds.feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_feats = [c for c in ds.feature_cols if c not in num_feats]

    summary: dict = {
        "dataset": ds.name,
        "shape": {"rows": int(len(df)), "columns": int(df.shape[1])},
        "feature_types": {"numeric": num_feats, "categorical": cat_feats},
        "excluded_columns": ds.excluded_cols,
        "missing_by_column": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "duplicate_rows": int(df.duplicated().sum()),
        "treatment": {"column": tcol, "kind": "binary" if binary_t else "continuous (dose)"},
    }

    # treatment distribution
    share = group.value_counts(normalize=True).sort_index()
    summary["treatment"]["group_share"] = {("treated" if k == 1 else "control"): round(float(v), 4)
                                           for k, v in share.items()}
    fig, ax = plt.subplots(figsize=(4, 3))
    group.map({0: "control", 1: "treated"}).value_counts().reindex(["control", "treated"]).plot.bar(ax=ax)
    ax.set_title("Treatment distribution")
    figures.append(_save(fig, out / "treatment_distribution.png"))

    # outcome by group (with normal-approx CI for binary outcomes)
    y = df[ycol]
    rates = {}
    for g, name in ((0, "control"), (1, "treated")):
        yy = y[group == g]
        m, n = float(yy.mean()), int(len(yy))
        se = float(np.sqrt(m * (1 - m) / n)) if set(y.unique()) <= {0, 1} else float(yy.std() / np.sqrt(n))
        rates[name] = {"n": n, "mean": m, "ci95": [m - 1.96 * se, m + 1.96 * se]}
    summary["outcome_by_group"] = {"outcome": ycol, **rates,
                                   "naive_difference": rates["treated"]["mean"] - rates["control"]["mean"]}
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["control", "treated"], [rates["control"]["mean"], rates["treated"]["mean"]],
           yerr=[1.96 * (rates[k]["ci95"][1] - rates[k]["mean"]) / 1.96 for k in ("control", "treated")])
    ax.set_title(f"{ycol} by group (observed)")
    figures.append(_save(fig, out / "outcome_by_group.png"))

    # spending / revenue only if the column exists
    for col in ("purchase_value", "revenue", "spend"):
        if col in df.columns:
            pos = df.loc[df[col] > 0, col]
            summary["spending"] = {"column": col, "share_positive": float((df[col] > 0).mean()),
                                   "mean_when_positive": float(pos.mean()) if len(pos) else None,
                                   "median_when_positive": float(pos.median()) if len(pos) else None}
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(pos, bins=40)
            ax.set_title(f"{col} (positive values, post-treatment)")
            figures.append(_save(fig, out / "spending_distribution.png"))
            break

    # covariate balance across groups (confounding / imbalance screen)
    smd = {c: standardized_mean_diff(df[c].fillna(df[c].median()).to_numpy(float), group.to_numpy())
           for c in num_feats}
    summary["covariate_smd"] = {k: round(v, 4) for k, v in smd.items()}
    imbalanced = [c for c, v in smd.items() if abs(v) > 0.1]
    summary["imbalanced_features"] = imbalanced
    if imbalanced:
        warnings.append(f"Features imbalanced across treatment groups (|SMD|>0.1): {imbalanced}. "
                        "Suggests non-random assignment / confounding; adjust (DML) and check overlap.")
    cat_rates = {}
    for c in cat_feats:
        cat_rates[c] = {str(k): round(float(v), 4) for k, v in df.groupby(c)[tcol].apply(
            lambda s: (s > 0).mean()).items()}
    if cat_rates:
        summary["treated_share_by_category"] = cat_rates

    # covariate distributions
    if num_feats:
        cols = min(4, len(num_feats[:max_hist]))
        rows = int(np.ceil(len(num_feats[:max_hist]) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.5 * rows), squeeze=False)
        for ax, c in zip(axes.ravel(), num_feats[:max_hist]):
            lo, hi = df[c].quantile([0.01, 0.99])
            for g, lab in ((0, "control"), (1, "treated")):
                ax.hist(df.loc[group == g, c].clip(lo, hi), bins=30, alpha=0.5, density=True, label=lab)
            ax.set_title(c, fontsize=9)
        axes.ravel()[0].legend(fontsize=7)
        figures.append(_save(fig, out / "covariate_distributions.png"))

    # correlations
    if len(num_feats) >= 2:
        corr = df[num_feats].corr()
        fig, ax = plt.subplots(figsize=(5, 4.2))
        im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(num_feats)))
        ax.set_xticklabels(num_feats, rotation=90, fontsize=7)
        ax.set_yticks(range(len(num_feats)))
        ax.set_yticklabels(num_feats, fontsize=7)
        fig.colorbar(im)
        ax.set_title("Feature correlations")
        figures.append(_save(fig, out / "correlations.png"))
        strong = [(a, b, round(float(corr.loc[a, b]), 3)) for i, a in enumerate(num_feats)
                  for b in num_feats[i + 1:] if abs(corr.loc[a, b]) > 0.9]
        if strong:
            warnings.append(f"Highly correlated feature pairs (|r|>0.9): {strong}")

    # segment summaries (categoricals, or quartiles of first numeric)
    seg_col = cat_feats[0] if cat_feats else (num_feats[0] if num_feats else None)
    if seg_col:
        seg = df[seg_col] if seg_col in cat_feats else pd.qcut(df[seg_col], 4, duplicates="drop").astype(str)
        tab = (df.assign(_seg=seg, _g=group).groupby(["_seg", "_g"])[ycol].agg(["mean", "size"])
               .unstack("_g"))
        summary["segment_summary"] = {"by": seg_col, "table": json.loads(tab.to_json())}

    if summary["duplicate_rows"]:
        warnings.append(f"{summary['duplicate_rows']} duplicated rows present.")
    if summary["missing_by_column"]:
        warnings.append(f"Missing values in columns: {list(summary['missing_by_column'])}")
    summary["warnings"] = warnings
    summary["figures"] = figures
    summary["caveat"] = ("Observed group differences are descriptive. They are causal only if treatment "
                         "assignment was randomised or fully explained by adjusted covariates.")
    (out / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("EDA written to %s (%d figures)", out, len(figures))
    return summary
