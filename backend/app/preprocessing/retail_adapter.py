"""Synthetic retail discount data: generator + adapter.

SYNTHETIC. Every number here is simulated; nothing represents real company performance.
`true_effect*` columns are ground truth for EDUCATIONAL evaluation only (never features).

Data-generating process (documented for dataset_card.md)
- Pretreatment covariates: age, income_k, tenure_months, past_purchases, avg_basket, region, is_member.
- eligible ~ Bernoulli(0.9). Only eligible customers can be treated.
- Baseline conversion p0(x) = sigmoid(...) of covariates.
- Treatment assignment is CONFOUNDED (observed covariates): marketing targets low-history customers,
  so propensity e(x) depends on past_purchases and income (clipped to [0.05, 0.95] -> overlap holds).
- Effect per discount dollar slope(x) >= 0 varies with income, past purchases, age, region.
- P(convert) = clip(p0 + slope * discount_amount, 0.001, 0.99).
- purchase_value (gross basket) is only non-zero after conversion. Cost/margin accounting is applied
  later by the optimizer, not here.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from backend.app.preprocessing.column_mapping import ColumnMapping, retail_mapping
from backend.app.preprocessing.dataset import CausalDataset, schema_hash
from backend.app.preprocessing.feature_selection import select_features
from backend.app.validation.data_validation import read_csv_checked, validate_dataset

logger = logging.getLogger(__name__)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def generate_retail_data(n: int = 20_000, seed: int = 42,
                         discount_levels: Sequence[float] = (10.0,),
                         confounded: bool = True, margin_rate: float = 0.30) -> pd.DataFrame:
    """Deterministic synthetic retail campaign data (same seed -> identical frame)."""
    if n < 100:
        raise ValueError("n must be >= 100")
    levels = np.asarray(sorted(set(float(x) for x in discount_levels)))
    if (levels <= 0).any():
        raise ValueError("discount_levels must be positive (0 means untreated)")
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(42, 12, n), 18, 80).round(0)
    income_k = np.clip(rng.lognormal(np.log(55), 0.45, n), 10, 400).round(1)
    tenure = np.clip(rng.exponential(24, n), 0, 120).round(0)
    past = rng.poisson(1.5 + tenure / 18.0)
    avg_basket = np.clip(rng.lognormal(np.log(40), 0.35, n), 8, 400).round(2)
    region = rng.choice(["urban", "suburban", "rural"], size=n, p=[0.45, 0.35, 0.20])
    member = rng.binomial(1, 0.4, n)
    eligible = rng.binomial(1, 0.9, n)

    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)  # noqa: E731
    zp, zi, za = z(past.astype(float)), z(income_k), z(age)

    p0 = _sigmoid(-2.0 + 0.30 * zp + 0.20 * zi + 0.25 * member - 0.10 * za)
    slope = (0.0060 - 0.0015 * zp - 0.0012 * zi - 0.0008 * za
             + 0.0010 * (region == "urban") - 0.0006 * member)
    slope = np.clip(slope, 0.0, None)

    e = _sigmoid(-0.2 - 0.6 * zp + 0.4 * zi) if confounded else np.full(n, 0.5)
    e = np.clip(e, 0.05, 0.95)
    treat = (eligible == 1) & (rng.random(n) < e)
    dose = np.where(treat, rng.choice(levels, size=n), 0.0)

    p = np.clip(p0 + slope * dose, 0.001, 0.99)
    conversion = rng.binomial(1, p)
    basket = avg_basket * rng.lognormal(0, 0.25, n)
    purchase_value = np.where(conversion == 1, basket, 0.0).round(2)

    ref = float(np.median(levels))
    true_effect = np.clip(p0 + slope * ref, 0.001, 0.99) - p0

    return pd.DataFrame({
        "customer_id": [f"C{i:07d}" for i in range(1, n + 1)],
        "age": age, "income_k": income_k, "tenure_months": tenure,
        "past_purchases": past, "avg_basket": avg_basket, "region": region,
        "is_member": member, "eligible": eligible,
        "treatment": treat.astype(int), "discount_amount": dose,
        "conversion": conversion, "purchase_value": purchase_value,
        "margin_rate": margin_rate,
        "true_effect": true_effect.round(6), "true_effect_per_dollar": slope.round(6),
    })


def load_retail(path: Optional[str | Path] = None, df: Optional[pd.DataFrame] = None,
                outcome: str = "conversion", treatment_kind: str = "binary",
                eligible_only: bool = True, mapping: Optional[ColumnMapping] = None,
                validate: bool = True) -> CausalDataset:
    """Wrap a retail CSV/DataFrame as a CausalDataset.

    If your own generator uses other column names, pass a `mapping` (ColumnMapping JSON).
    """
    if (path is None) == (df is None):
        raise ValueError("Provide exactly one of `path` or `df`.")
    raw = read_csv_checked(path) if path is not None else df.copy()
    treat_col = "treatment" if treatment_kind == "binary" else "discount_amount"
    mapping = mapping or retail_mapping(outcome=outcome, treatment=treat_col)
    n_before = len(raw)
    if eligible_only and mapping.eligible_col:
        if mapping.eligible_col not in raw.columns:
            raise ValueError(f"Eligibility column '{mapping.eligible_col}' not in data.")
        raw = raw[raw[mapping.eligible_col] == 1].reset_index(drop=True)
    features, excluded = select_features(raw, mapping)
    if validate:
        validate_dataset(raw, mapping.treatment, mapping.outcome, features, mapping.id_col,
                         treatment_kind=treatment_kind).raise_if_failed()
    meta = {
        "name": "synthetic_retail", "synthetic": True,
        "source": "generate_retail_data (simulated); not real company data",
        "rows_total": n_before, "rows_after_eligibility_filter": int(len(raw)),
        "eligible_only": eligible_only, "treatment_kind": treatment_kind,
        "schema_hash": schema_hash(raw),
        "design": "observational with confounded assignment on observed covariates",
        "ground_truth_columns": [c for c in ("true_effect", "true_effect_per_dollar") if c in raw.columns],
    }
    return CausalDataset(df=raw, name="synthetic_retail", treatment_col=mapping.treatment,
                         outcome_col=mapping.outcome, feature_cols=features, id_col=mapping.id_col,
                         excluded_cols=excluded, metadata=meta, eligible_col=mapping.eligible_col)
