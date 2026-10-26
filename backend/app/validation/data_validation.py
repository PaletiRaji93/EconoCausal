"""Schema, missing-value, duplicate, treatment/outcome validation with helpful messages."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from backend.app.core.errors import DataValidationError

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.errors:
            raise DataValidationError("Data validation failed:\n- " + "\n- ".join(self.errors))

    def to_dict(self) -> dict:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings,
                "summary": self.summary}


def read_csv_checked(path: str | Path, **kwargs) -> pd.DataFrame:
    """pd.read_csv with friendly errors for missing / empty / unparsable files."""
    p = Path(path)
    if not p.exists():
        raise DataValidationError(f"File not found: {p}. Check the path (and that the download finished).")
    try:
        df = pd.read_csv(p, **kwargs)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"File is empty: {p}") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"Could not parse {p} as CSV: {exc}") from exc
    if df.empty:
        raise DataValidationError(f"File has a header but no rows: {p}")
    return df


def validate_dataset(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    feature_cols: list[str],
    id_col: Optional[str] = None,
    treatment_kind: str = "binary",      # "binary" or "continuous"
    min_group_size: int = 50,
    missing_warn_frac: float = 0.05,
) -> ValidationReport:
    """Validate a dataframe for causal modelling. Never mutates `df`."""
    rep = ValidationReport()
    needed = [treatment_col, outcome_col, *feature_cols] + ([id_col] if id_col else [])
    absent = [c for c in needed if c not in df.columns]
    if absent:
        rep.errors.append(f"Missing required columns: {absent}. Available: {list(df.columns)[:30]}")
        return rep
    if len(df) == 0:
        rep.errors.append("Dataset has zero rows.")
        return rep

    rep.summary.update(n_rows=int(len(df)), n_features=len(feature_cols))

    # --- treatment ---
    t = pd.to_numeric(df[treatment_col], errors="coerce")
    if t.isna().any():
        rep.errors.append(f"Treatment '{treatment_col}' has {int(t.isna().sum())} missing/non-numeric values.")
    elif not np.isfinite(t.to_numpy(float)).all():
        rep.errors.append(f"Treatment '{treatment_col}' contains infinite values.")
    elif treatment_kind == "binary":
        bad = sorted(set(t.unique()) - {0, 1})
        if bad:
            rep.errors.append(f"Binary treatment '{treatment_col}' must be 0/1; found values {bad[:5]}.")
        else:
            n1, n0 = int((t == 1).sum()), int((t == 0).sum())
            rep.summary.update(n_treated=n1, n_control=n0, treated_share=round(n1 / len(t), 4))
            if n1 == 0 or n0 == 0:
                rep.errors.append("Both treated and control rows are required (one group is empty).")
            elif min(n1, n0) < min_group_size:
                rep.warnings.append(f"Small treatment group ({min(n1, n0)} rows < {min_group_size}).")
    else:
        if (t < 0).any():
            rep.errors.append("Continuous treatment (dose) must be non-negative.")
        if not (t == 0).any():
            rep.warnings.append("No zero-dose (untreated) rows: effects vs. no treatment are not identified.")
        rep.summary.update(n_distinct_doses=int(t.nunique()), dose_min=float(t.min()), dose_max=float(t.max()))

    # --- outcome ---
    y = pd.to_numeric(df[outcome_col], errors="coerce")
    if y.isna().any():
        rep.errors.append(f"Outcome '{outcome_col}' has {int(y.isna().sum())} missing/non-numeric values.")
    elif not np.isfinite(y.to_numpy(float)).all():
        rep.errors.append(f"Outcome '{outcome_col}' contains infinite values.")
    else:
        binary = set(y.unique()) <= {0, 1}
        rep.summary.update(outcome_kind="binary" if binary else "continuous",
                           outcome_mean=round(float(y.mean()), 6))
        if y.nunique() < 2:
            rep.errors.append(f"Outcome '{outcome_col}' is constant; nothing to model.")
        if binary and y.sum() < 20:
            rep.warnings.append(f"Very few positive outcomes ({int(y.sum())}): estimates will be noisy.")

    # --- features ---
    miss = df[feature_cols].isna().mean()
    for c, frac in miss[miss > 0].items():
        (rep.warnings if frac <= 0.5 else rep.errors).append(
            f"Feature '{c}' has {frac:.1%} missing values" + ("" if frac <= 0.5 else " (>50%: drop it)"))
    rep.summary["missing_feature_frac_max"] = round(float(miss.max()), 4)
    for c in feature_cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not np.isfinite(s.dropna().to_numpy(float)).all():
            rep.errors.append(f"Feature '{c}' contains infinite values.")
        if s.nunique(dropna=True) <= 1:
            rep.warnings.append(f"Feature '{c}' is constant and carries no information.")

    # --- duplicates ---
    n_dup = int(df.duplicated().sum())
    rep.summary["duplicate_rows"] = n_dup
    if n_dup:
        frac = n_dup / len(df)
        # Criteo-style anonymised data legitimately repeats rows; only warn.
        rep.warnings.append(f"{n_dup} fully duplicated rows ({frac:.2%}).")
    if id_col:
        n_dup_id = int(df[id_col].duplicated().sum())
        rep.summary["duplicate_ids"] = n_dup_id
        if n_dup_id:
            rep.errors.append(f"Identifier '{id_col}' has {n_dup_id} duplicates (one row per customer expected).")
        if id_col in feature_cols:
            rep.errors.append(f"Identifier '{id_col}' must not be a feature.")

    if treatment_col in feature_cols or outcome_col in feature_cols:
        rep.errors.append("Treatment/outcome must not appear in the feature list.")

    for e in rep.errors:
        logger.error("validation: %s", e)
    for w in rep.warnings:
        logger.warning("validation: %s", w)
    return rep
