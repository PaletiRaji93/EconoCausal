"""LinearDML for a CONTINUOUS treatment (discount amount / dose). Support-checked.

Interpretation
- theta(x) = estimated change in outcome per ONE UNIT of dose (e.g. per $1 discount).
- LinearDML assumes the effect is LINEAR in dose: effect(0 -> d) = theta(x) * d.
  A binary 'any discount' effect is NOT the effect of $5/$10/$20; do not mix them.
- Requires at least 3 distinct dose levels (incl. 0) to say anything beyond a binary contrast.
- Dose-effects are only reported inside the observed support; larger doses raise an error.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from backend.app.core.errors import InsufficientSupportError, require
from backend.app.dml.dml_estimator import make_nuisance, make_version_id
from backend.app.preprocessing.dataset import CausalDataset
from backend.app.preprocessing.feature_selection import build_preprocessor

logger = logging.getLogger(__name__)


def check_dose_support(dose: np.ndarray, min_levels: int = 3, min_per_level: int = 30) -> dict:
    """Describe the observed dose support; raise if it cannot support a dose-response analysis."""
    dose = np.asarray(dose, float)
    if (dose < 0).any():
        raise InsufficientSupportError("Dose must be non-negative.")
    levels, counts = np.unique(dose.round(6), return_counts=True)
    positive = levels[levels > 0]
    info = {"levels": levels.tolist(), "counts": counts.tolist(),
            "min_positive_dose": float(positive.min()) if len(positive) else None,
            "max_dose": float(levels.max()), "warnings": []}
    if 0 not in levels:
        raise InsufficientSupportError("No zero-dose rows: effect relative to no discount is not identified.")
    if len(levels) < min_levels:
        raise InsufficientSupportError(
            f"Only {len(levels)} distinct dose levels {levels.tolist()}; need >= {min_levels} "
            "(including 0) for a continuous dose-response analysis. Use the binary DML instead.")
    thin = [float(l) for l, c in zip(levels, counts) if c < min_per_level]
    if thin:
        info["warnings"].append(f"Thin support (<{min_per_level} rows) at dose levels {thin}.")
    return info


class ContinuousDML:
    def __init__(self, nuisance: str = "gbm", cv: int = 3, seed: int = 42):
        self.params = dict(nuisance=nuisance, cv=cv, seed=seed)
        self.preprocessor = None
        self.est = None
        self.support: dict = {}
        self.feature_cols: list[str] = []
        self.meta: dict = {}

    def fit(self, ds: CausalDataset, train_idx) -> "ContinuousDML":
        dml = require("econml.dml", "econml")
        X = ds.X.iloc[train_idx]
        dose = ds.df[ds.treatment_col].to_numpy(float)[train_idx]
        self.support = check_dose_support(dose)
        self.feature_cols = list(ds.feature_cols)
        self.preprocessor = build_preprocessor(X)
        Xp = self.preprocessor.fit_transform(X)
        my, mt = make_nuisance(self.params["nuisance"], self.params["seed"], discrete_treatment=False)
        self.est = dml.LinearDML(model_y=my, model_t=mt, discrete_treatment=False,
                                 cv=self.params["cv"], random_state=self.params["seed"])
        self.est.fit(ds.Y[train_idx].astype(float), dose, X=Xp)
        self.meta = {"version_id": make_version_id(ds.name, "continuous-linear", self.params["seed"]),
                     "dataset": ds.name, "treatment": ds.treatment_col, "outcome": ds.outcome_col,
                     "n_train": int(len(Xp)), "support": self.support, "params": self.params,
                     "assumption": "effect linear in dose within observed support"}
        return self

    def _xp(self, X: pd.DataFrame) -> np.ndarray:
        if self.est is None:
            raise RuntimeError("Model is not fitted.")
        return self.preprocessor.transform(X[self.feature_cols])

    def effect_per_unit(self, X: pd.DataFrame) -> np.ndarray:
        """theta(x): estimated outcome change per +1 dose unit, shape (n,)."""
        theta = np.asarray(self.est.const_marginal_effect(self._xp(X))).reshape(len(X), -1)[:, 0]
        if not np.isfinite(theta).all():
            raise ValueError("Non-finite marginal effects produced.")
        return theta

    def effect_at_dose(self, X: pd.DataFrame, dose: float) -> tuple[np.ndarray, list[str]]:
        """Estimated effect of moving from 0 to `dose`. Errors beyond observed support."""
        warnings: list[str] = []
        if dose < 0:
            raise ValueError("dose must be >= 0")
        if dose > self.support["max_dose"] + 1e-9:
            raise InsufficientSupportError(
                f"Dose {dose} exceeds the maximum observed dose {self.support['max_dose']}: refusing to extrapolate.")
        mind = self.support["min_positive_dose"]
        if 0 < dose < mind:
            warnings.append(f"Dose {dose} lies between 0 and the smallest observed positive dose {mind}: "
                            "relies on the linearity assumption.")
        warnings.extend(self.support.get("warnings", []))
        return self.effect_per_unit(X) * float(dose), warnings

    def summary(self) -> dict:
        return dict(self.meta)
