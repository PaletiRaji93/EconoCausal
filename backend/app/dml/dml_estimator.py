"""Double Machine Learning for a BINARY treatment (EconML).

Prediction vs causation (for the viva)
- A predictive model estimates P(Y=1 | X): who is likely to buy anyway.
- DML estimates tau(X) = E[Y(1) - Y(0) | X]: how much the TREATMENT changes the outcome.
- DML residualises Y and T on X with cross-fitted nuisance models, then regresses the outcome
  residual on the treatment residual, so nuisance-model overfitting does not bias tau.
- tau_hat(x) is a model-based CONDITIONAL estimate, not an observed individual fact.

Estimators
- "linear": EconML LinearDML  (CATE linear in the processed features; fast, interpretable)
- "forest": EconML CausalForestDML (non-parametric heterogeneity; needs more data)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from backend.app.core.errors import InsufficientSupportError, require
from backend.app.preprocessing.dataset import CausalDataset
from backend.app.preprocessing.feature_selection import build_preprocessor

logger = logging.getLogger(__name__)


def make_nuisance(kind: str, seed: int, discrete_treatment: bool = True):
    """Return (model_y, model_t). 'linear' = fast/robust, 'gbm' = flexible."""
    if kind == "linear":
        my = Ridge(alpha=1.0)
        mt = LogisticRegression(max_iter=2000) if discrete_treatment else Ridge(alpha=1.0)
    elif kind == "gbm":
        my = HistGradientBoostingRegressor(max_iter=100, max_depth=3, learning_rate=0.1, random_state=seed)
        mt = (HistGradientBoostingClassifier(max_iter=100, max_depth=3, random_state=seed)
              if discrete_treatment else
              HistGradientBoostingRegressor(max_iter=100, max_depth=3, random_state=seed))
    else:
        raise ValueError("nuisance must be 'linear' or 'gbm'")
    return my, mt


def make_version_id(dataset: str, estimator: str, seed: int) -> str:
    """<dataset>-<estimator>-<UTC yyyymmddHHMMSS>-<seed> (INTEGRATION_CONTRACT section 4)."""
    return f"{dataset}-{estimator}-{datetime.now(timezone.utc):%Y%m%d%H%M%S}-{seed}"


def build_binary_estimator(estimator: str, nuisance: str, cv: int, seed: int,
                           n_estimators: int = 200, min_samples_leaf: int = 50):
    """Construct an (unfitted) EconML estimator for binary treatment."""
    dml = require("econml.dml", "econml")
    my, mt = make_nuisance(nuisance, seed, discrete_treatment=True)
    if estimator == "linear":
        return dml.LinearDML(model_y=my, model_t=mt, discrete_treatment=True, cv=cv, random_state=seed)
    if estimator == "forest":
        return dml.CausalForestDML(model_y=my, model_t=mt, discrete_treatment=True, cv=cv,
                                   n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                   random_state=seed)
    raise ValueError("estimator must be 'linear' or 'forest'")


class DMLModel:
    """Binary-treatment DML with its own preprocessing. Implements the CausalModel contract."""

    def __init__(self, estimator: str = "linear", nuisance: str = "gbm", cv: int = 3,
                 seed: int = 42, n_estimators: int = 200, min_samples_leaf: int = 50):
        self.params = dict(estimator=estimator, nuisance=nuisance, cv=cv, seed=seed,
                           n_estimators=n_estimators, min_samples_leaf=min_samples_leaf)
        self.preprocessor = None
        self.est = None
        self.feature_cols: list[str] = []
        self.feature_names_out: list[str] = []
        self.meta: dict = {}

    # ---- contract ----
    def fit(self, ds: CausalDataset, train_idx) -> "DMLModel":
        X = ds.X.iloc[train_idx]
        t = (ds.T[train_idx] > 0).astype(int)
        y = ds.Y[train_idx].astype(float)
        if t.min() == t.max():
            raise InsufficientSupportError("Training split contains only one treatment group.")
        self.feature_cols = list(ds.feature_cols)
        self.preprocessor = build_preprocessor(X)
        Xp = self.preprocessor.fit_transform(X)
        self.feature_names_out = list(self.preprocessor.get_feature_names_out())
        self.est = build_binary_estimator(**{k: self.params[k] for k in
                                             ("estimator", "nuisance", "cv", "seed", "n_estimators",
                                              "min_samples_leaf")})
        logger.info("Fitting %s DML on %d rows, %d processed features",
                    self.params["estimator"], len(Xp), Xp.shape[1])
        self.est.fit(y, t, X=Xp)
        self.meta = {"version_id": make_version_id(ds.name, self.params["estimator"], self.params["seed"]),
                     "dataset": ds.name, "treatment": ds.treatment_col, "outcome": ds.outcome_col,
                     "n_train": int(len(Xp)), "n_treated_train": int(t.sum()),
                     "features": self.feature_cols, "params": self.params,
                     "trained_utc": datetime.now(timezone.utc).isoformat()}
        return self

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.est is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing:
            raise ValueError(f"Prediction data is missing feature columns: {missing}")
        return self.preprocessor.transform(X[self.feature_cols])

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        """Estimated conditional effect of treatment, shape (n,), finite."""
        Xp = self._transform(X)          # raises a clear error first if the model is not fitted
        tau = np.asarray(self.est.effect(Xp)).reshape(-1)
        if not np.isfinite(tau).all():
            raise ValueError("Non-finite treatment-effect estimates produced.")
        return tau

    def ate(self, X: Optional[pd.DataFrame] = None) -> dict:
        """Average effect over X (default: not provided -> uses supplied rows only)."""
        if X is None:
            raise ValueError("Provide the rows over which to average (e.g. test features).")
        Xp = self._transform(X)
        out = {"ate": float(np.mean(self.predict_cate(X))), "ci95": None, "ci_note": None}
        try:
            lo, hi = self.est.ate_interval(Xp, alpha=0.05)
            out["ci95"] = [float(np.ravel(lo)[0]), float(np.ravel(hi)[0])]
        except (AttributeError, NotImplementedError, ValueError) as exc:
            out["ci_note"] = f"Confidence interval unavailable for this estimator: {exc}"
            logger.warning(out["ci_note"])
        return out

    def summary(self) -> dict:
        return {**self.meta, "estimator_class": type(self.est).__name__ if self.est else None}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        path.with_suffix(".json").write_text(json.dumps(self.summary(), indent=2, default=str),
                                             encoding="utf-8")
        return path

    @staticmethod
    def load(path: str | Path) -> "DMLModel":
        return joblib.load(path)


def heterogeneity_summary(cate: np.ndarray, X: pd.DataFrame, n_bins: int = 4, max_features: int = 12) -> dict:
    """Distribution of estimated effects and mean effect per feature segment."""
    cate = np.asarray(cate, float)
    out = {"quantiles": {f"q{int(q*100):02d}": float(np.quantile(cate, q))
                         for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
           "mean": float(cate.mean()), "std": float(cate.std()),
           "share_positive": float((cate > 0).mean()), "segments": {}}
    for c in list(X.columns)[:max_features]:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > n_bins:
            seg = pd.qcut(s, n_bins, duplicates="drop").astype(str)
        else:
            seg = s.astype(str)
        g = pd.Series(cate).groupby(seg.to_numpy()).agg(["mean", "size"])
        out["segments"][c] = {str(k): {"mean_cate": float(v["mean"]), "n": int(v["size"])}
                              for k, v in g.iterrows()}
    return out


# ---- baselines for comparison -------------------------------------------------------------
def response_model_scores(ds: CausalDataset, train_idx, test_idx, seed: int = 42) -> np.ndarray:
    """PREDICTIVE baseline: ranks by P(Y=1|X) fitted ignoring treatment. Not a causal estimate."""
    from sklearn.pipeline import Pipeline
    X = ds.X
    pipe = Pipeline([("prep", build_preprocessor(X.iloc[train_idx])),
                     ("clf", HistGradientBoostingClassifier(max_iter=100, max_depth=3, random_state=seed))])
    pipe.fit(X.iloc[train_idx], (ds.Y[train_idx] > 0).astype(int))
    return pipe.predict_proba(X.iloc[test_idx])[:, 1]


def t_learner_scores(ds: CausalDataset, train_idx, test_idx, seed: int = 42) -> np.ndarray:
    """Simple causal baseline (sklearn only): mu1(x) - mu0(x) from separate outcome models."""
    from sklearn.pipeline import Pipeline
    X, t, y = ds.X, (ds.T > 0).astype(int), ds.Y.astype(float)
    preds = {}
    for g in (0, 1):
        idx = train_idx[t[train_idx] == g]
        pipe = Pipeline([("prep", build_preprocessor(X.iloc[idx])),
                         ("m", HistGradientBoostingRegressor(max_iter=100, max_depth=3, random_state=seed))])
        pipe.fit(X.iloc[idx], y[idx])
        preds[g] = pipe.predict(X.iloc[test_idx])
    return preds[1] - preds[0]
