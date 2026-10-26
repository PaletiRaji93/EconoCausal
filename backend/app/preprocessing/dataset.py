"""CausalDataset: the canonical object adapters return (INTEGRATION_CONTRACT section 3).

Contract extension (documented in PART2_README): `eligible_col` field and `make_splits`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class CausalDataset:
    df: pd.DataFrame
    name: str                      # "criteo" or "synthetic_retail"
    treatment_col: str
    outcome_col: str
    feature_cols: list[str]        # pretreatment covariates ONLY
    id_col: Optional[str] = None   # never a feature
    excluded_cols: dict[str, str] = field(default_factory=dict)  # col -> reason
    metadata: dict = field(default_factory=dict)
    eligible_col: Optional[str] = None

    @property
    def X(self) -> pd.DataFrame:
        return self.df[self.feature_cols]

    @property
    def T(self) -> np.ndarray:
        return self.df[self.treatment_col].to_numpy()

    @property
    def Y(self) -> np.ndarray:
        return self.df[self.outcome_col].to_numpy()


def schema_hash(df: pd.DataFrame) -> str:
    """Short hash of column names + dtypes (detects schema changes between runs)."""
    text = "|".join(f"{c}:{df[c].dtype}" for c in df.columns)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def make_splits(
    treatment: np.ndarray,
    outcome: np.ndarray,
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> dict[str, np.ndarray]:
    """Positional train/validation/test indices, stratified by treatment x outcome.

    Train: fit models. Validation: model comparison/tuning (reserved).
    Test: final evaluation only.
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    n = len(treatment)
    idx = np.arange(n)
    t = (np.asarray(treatment) > 0).astype(int)
    y = (np.asarray(outcome) > 0).astype(int)
    strata = t * 2 + y
    counts = np.bincount(strata, minlength=4)
    if counts[counts > 0].min() < 3:  # tiny stratum -> stratify on treatment only
        strata = t
    train_idx, rest = train_test_split(
        idx, train_size=fractions[0], random_state=seed, stratify=strata
    )
    val_share = fractions[1] / (fractions[1] + fractions[2])
    val_idx, test_idx = train_test_split(
        rest, train_size=val_share, random_state=seed, stratify=strata[rest]
    )
    return {"train": np.sort(train_idx), "val": np.sort(val_idx), "test": np.sort(test_idx)}
