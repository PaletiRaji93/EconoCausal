"""Pretreatment-only feature selection, leakage guard and the sklearn preprocessor."""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.app.core.errors import LeakageError
from backend.app.preprocessing.column_mapping import ColumnMapping

logger = logging.getLogger(__name__)

# Column-name patterns that usually indicate post-treatment / target / identifier info.
LEAKAGE_NAME_PATTERNS = [
    r"^exposure$", r"^post_", r"^future_", r"^true_", r"^(cate|ite|tau)(_|$)",
    r"^purchase_value$", r"^revenue", r"^discount_amount$", r"^conversion$", r"^visit$",
    r"(^|_)(id|uuid)$",
]
_LEAK_RE = re.compile("|".join(LEAKAGE_NAME_PATTERNS), flags=re.IGNORECASE)


def _name_leak(col: str) -> bool:
    return bool(_LEAK_RE.search(col))


def select_features(df: pd.DataFrame, mapping: ColumnMapping) -> tuple[list[str], dict[str, str]]:
    """Return (feature_cols, excluded_cols). Raises LeakageError on unsafe explicit features."""
    excluded: dict[str, str] = dict(mapping.excluded)
    reserved = {mapping.treatment: "treatment", mapping.outcome: "outcome"}
    if mapping.id_col:
        reserved[mapping.id_col] = "identifier (never a feature)"
    if mapping.eligible_col:
        reserved[mapping.eligible_col] = "eligibility filter (not a model feature)"
    for c in mapping.other_outcomes:
        reserved.setdefault(c, "alternative outcome (post-treatment)")
    excluded.update({c: r for c, r in reserved.items() if c in df.columns or c in excluded})

    explicit = mapping.features is not None
    candidates = list(mapping.features) if explicit else [c for c in df.columns]
    missing = [c for c in candidates if c not in df.columns]
    if missing:
        raise LeakageError(f"Requested feature columns not found in data: {missing}")

    features: list[str] = []
    for c in candidates:
        if c in reserved or c in mapping.excluded:
            if explicit and c in mapping.features:
                raise LeakageError(
                    f"Column '{c}' was requested as a feature but is excluded: "
                    f"{excluded.get(c, 'reserved')}"
                )
            continue
        if _name_leak(c):
            if explicit:
                raise LeakageError(
                    f"Feature '{c}' looks post-treatment/target/identifier by name. "
                    "Remove it from the mapping or justify it in the causal design."
                )
            excluded[c] = "auto-excluded: name pattern suggests post-treatment/identifier"
            continue
        features.append(c)

    if not features:
        raise LeakageError("No eligible pretreatment features remain after leakage filtering.")
    check_target_copies(df, features, mapping)
    logger.info("Selected %d features; excluded %d columns", len(features), len(excluded))
    return features, excluded


def check_target_copies(df: pd.DataFrame, features: list[str], mapping: ColumnMapping,
                        threshold: float = 0.99) -> None:
    """Raise LeakageError if a feature is (almost) a copy of the outcome or treatment."""
    for target in (mapping.outcome, mapping.treatment):
        if target not in df.columns:
            continue
        y = pd.to_numeric(df[target], errors="coerce")
        if y.nunique() < 2:
            continue
        for c in features:
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
            x = df[c]
            if x.nunique() < 2:
                continue
            corr = np.corrcoef(x.fillna(x.median()).to_numpy(float), y.fillna(0).to_numpy(float))[0, 1]
            if abs(corr) >= threshold:
                raise LeakageError(
                    f"Feature '{c}' has |corr|={abs(corr):.3f} with '{target}' - likely a leaked copy."
                )


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Unfitted sklearn preprocessor: numeric -> impute+scale, categorical -> impute+one-hot."""
    numeric, categorical = [], []
    for c in X.columns:
        (numeric if pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])
         else categorical).append(c)
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore",sparse_output=False,drop="first"))]), categorical))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
