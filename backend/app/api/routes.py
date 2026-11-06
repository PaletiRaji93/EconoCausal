
"""API endpoints for EconoCausal Criteo and retail DML models."""

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from backend.app.core.config import settings
from backend.app.dml.dml_estimator import DMLModel
from backend.app.dml.continuous_dml import ContinuousDML
from backend.app.preprocessing.retail_adapter import load_retail
from backend.app.preprocessing.feature_selection import build_preprocessor
from backend.app.causal_engine.budget_optimizer import (
    optimize_discount_allocation,
)

router = APIRouter()


# ---------------------------------------------------------
# Existing Criteo model
# ---------------------------------------------------------

def get_model_path() -> Path:
    configured = os.getenv("ECONO_MODEL_PATH")
    if configured:
        return Path(configured)
    return settings.reports_dir / "run_criteo" / "model.joblib"


@lru_cache(maxsize=1)
def load_model() -> DMLModel:
    path = get_model_path()

    if not path.is_file():
        raise FileNotFoundError(f"Saved model not found: {path}")

    model = DMLModel.load(path)

    if model.est is None or model.preprocessor is None:
        raise ValueError("Saved model is not fitted.")

    return model


class EffectRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    features: dict[str, float] = Field(
        ...,
        description="Numeric Criteo model features.",
    )


@router.get("/health")
def api_health() -> dict:
    return {"status": "ok", "api_version": "v1"}


@router.get("/model/status")
def model_status() -> dict:
    path = get_model_path()

    if not path.is_file():
        return {
            "loaded": False,
            "model_path": str(path),
            "message": "Model file not found. Train the model first.",
        }

    try:
        model = load_model()

        return {
            "loaded": True,
            "model_path": str(path),
            "version_id": model.meta.get("version_id"),
            "dataset": model.meta.get("dataset"),
            "treatment": model.meta.get("treatment"),
            "outcome": model.meta.get("outcome"),
            "features": model.feature_cols,
            "estimator": (
                model.meta.get("estimator_class")
                or model.meta.get("estimator")
                or type(model.est).__name__
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load saved model: {exc}",
        ) from exc


@router.post("/model/estimate")
def estimate_effect(request: EffectRequest) -> dict:
    try:
        model = load_model()

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Saved model is unavailable: {exc}",
        ) from exc

    expected = model.feature_cols
    supplied = request.features

    missing = [name for name in expected if name not in supplied]
    extra = [name for name in supplied if name not in expected]

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": missing,
                "unexpected_features": extra,
            },
        )

    try:
        X = pd.DataFrame(
            [[supplied[name] for name in expected]],
            columns=expected,
        )

        effect = float(model.predict_cate(X)[0])

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Effect estimation failed: {exc}",
        ) from exc

    return {
        "estimated_treatment_effect": effect,
        "dataset": model.meta.get("dataset"),
        "treatment": model.meta.get("treatment"),
        "outcome": model.meta.get("outcome"),
        "interpretation": (
            "Estimated effect of advertising treatment on visit outcome; "
            "this is not a price-discount effect."
        ),
    }


# ---------------------------------------------------------
# Retail continuous-treatment model
# ---------------------------------------------------------

def get_retail_model_path() -> Path:
    return settings.reports_dir / "run_retail_continuous" / "model.joblib"


def get_retail_data_path() -> Path:
    return settings.processed_dir / "retail_synthetic.csv"


@lru_cache(maxsize=1)
def load_retail_context():
    """Load the saved retail DML model and fit a baseline model once."""

    model_path = get_retail_model_path()
    data_path = get_retail_data_path()

    if not model_path.is_file():
        raise FileNotFoundError(f"Retail model not found: {model_path}")

    if not data_path.is_file():
        raise FileNotFoundError(f"Retail dataset not found: {data_path}")

    model = joblib.load(model_path)

    if not isinstance(model, ContinuousDML) or model.est is None:
        raise ValueError(
            "Saved retail model is not a fitted ContinuousDML model."
        )

    ds = load_retail(
        path=data_path,
        outcome="conversion",
        treatment_kind="continuous",
        eligible_only=True,
        validate=True,
    )

    if list(ds.feature_cols) != list(model.feature_cols):
        raise ValueError(
            "Retail dataset features do not match the saved model. "
            f"Dataset: {ds.feature_cols}; model: {model.feature_cols}"
        )

    untreated = ds.df[ds.df[ds.treatment_col] == 0]

    if untreated[ds.outcome_col].nunique() < 2:
        raise ValueError(
            "Untreated customers must contain both conversion classes."
        )

    X0 = untreated[ds.feature_cols]
    y0 = untreated[ds.outcome_col].astype(int)

    baseline_model = Pipeline([
        ("preprocessor", build_preprocessor(X0)),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])

    baseline_model.fit(X0, y0)

    return ds, model, baseline_model


# ---------------------------------------------------------
# Retail API request validation
# ---------------------------------------------------------

class RetailEffectRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    features: dict[str, Any] = Field(
        ...,
        description=(
            "Customer features: age, income_k, tenure_months, "
            "past_purchases, avg_basket, region, is_member."
        ),
    )

    dose: float = Field(
        ...,
        ge=0,
        description=(
            "Discount amount; must not exceed the model's "
            "maximum observed dose."
        ),
    )

    @field_validator("features")
    @classmethod
    def validate_customer_features(cls, features):

        expected_features = {
            "age",
            "income_k",
            "tenure_months",
            "past_purchases",
            "avg_basket",
            "region",
            "is_member",
        }

        missing = expected_features - set(features)
        extra = set(features) - expected_features

        if missing:
            raise ValueError(
                f"Missing customer features: {sorted(missing)}"
            )

        if extra:
            raise ValueError(
                f"Unexpected customer features: {sorted(extra)}"
            )

        # Validate region.
        if features["region"] not in {
            "urban",
            "suburban",
            "rural",
        }:
            raise ValueError(
                "region must be urban, suburban, or rural."
            )

        # Numeric feature ranges based on the data generator.
        ranges = {
            "age": (18, 80),
            "income_k": (10, 400),
            "tenure_months": (0, 120),
            "avg_basket": (8, 400),
            "past_purchases": (0, None),
            "is_member": (0, 1),
        }

        integer_features = {
            "age",
            "tenure_months",
            "past_purchases",
            "is_member",
        }

        for name, (minimum, maximum) in ranges.items():

            value = features[name]

            if isinstance(value, bool) or not isinstance(
                value, (int, float)
            ):
                raise ValueError(
                    f"{name} must be a numeric value."
                )

            if not math.isfinite(value):
                raise ValueError(
                    f"{name} must be finite."
                )

            if value < minimum:
                raise ValueError(
                    f"{name} must be at least {minimum}."
                )

            if maximum is not None and value > maximum:
                raise ValueError(
                    f"{name} must not exceed {maximum}."
                )

            if (
                name in integer_features
                and not float(value).is_integer()
            ):
                raise ValueError(
                    f"{name} must be a whole number."
                )

        return features


class RetailOptimizeRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    budget: float = Field(
        ...,
        ge=0,
        description="Total expected discount-spend budget.",
    )

    margin_rate: float = Field(
        default=0.30,
        gt=0,
        le=1,
        description="Business contribution margin assumption.",
    )

    customer_limit: int | None = Field(
        default=None,
        ge=1,
        le=20000,
        description="Maximum number of customers to optimize.",
    )


# ---------------------------------------------------------
# Retail model status
# ---------------------------------------------------------

@router.get("/retail/model/status")
def retail_model_status() -> dict:
    """Check availability of the continuous retail DML model."""

    try:
        ds, model, _ = load_retail_context()

        return {
            "loaded": True,
            "model_path": str(get_retail_model_path()),
            "dataset_path": str(get_retail_data_path()),
            "version_id": model.meta.get("version_id"),
            "dataset": ds.name,
            "treatment": ds.treatment_col,
            "outcome": ds.outcome_col,
            "features": model.feature_cols,
            "supported_discount_levels": model.support.get(
                "levels", []
            ),
            "synthetic_data": True,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Retail model unavailable: {exc}",
        ) from exc


# ---------------------------------------------------------
# Retail causal effect estimation
# ---------------------------------------------------------

@router.post("/retail/effect")
def retail_effect(request: RetailEffectRequest) -> dict:
    """Estimate conversion lift relative to zero discount."""

    try:
        _, model, _ = load_retail_context()

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Retail model unavailable: {exc}",
        ) from exc

    expected = model.feature_cols
    supplied = request.features

    missing = [
        name for name in expected if name not in supplied
    ]

    extra = [
        name for name in supplied if name not in expected
    ]

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": missing,
                "unexpected_features": extra,
            },
        )

    try:
        X = pd.DataFrame(
            [[supplied[name] for name in expected]],
            columns=expected,
        )

        effects, warnings = model.effect_at_dose(
            X,
            request.dose,
        )

        effect = float(effects[0])

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Retail effect estimation failed: {exc}",
        ) from exc

    return {
        "dose": request.dose,
        "estimated_conversion_lift": effect,
        "interpretation": (
            "Model-estimated change in conversion probability versus "
            "zero discount. This is an estimate, not a guaranteed outcome."
        ),
        "warnings": warnings,
        "synthetic_data": True,
    }


# ---------------------------------------------------------
# Retail budget optimization
# ---------------------------------------------------------

@router.post("/retail/optimize")
def retail_optimize(request: RetailOptimizeRequest) -> dict:
    """Allocate discounts under a total expected-discount-spend budget."""

    try:
        ds, model, baseline_model = load_retail_context()

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Retail optimization dependencies unavailable: "
                f"{exc}"
            ),
        ) from exc

    try:
        # Use the same eligible population as the offline workflow.
        customers = ds.df.reset_index(drop=True)

        if request.customer_limit is not None:
            customers = customers.head(
                request.customer_limit
            ).copy()

        X = customers[ds.feature_cols]

        baseline = baseline_model.predict_proba(X)[:, 1]

        effect_per_dollar = model.effect_per_unit(X)

        recommendations, summary = optimize_discount_allocation(
            customers=customers,
            baseline_conversion=baseline,
            effect_per_dollar=effect_per_dollar,
            budget=request.budget,
            discount_levels=tuple(
                float(d)
                for d in model.support["levels"]
            ),
            margin_rate=request.margin_rate,
        )

        # Return a manageable preview rather than thousands of rows.
        preview_cols = [
            "customer_id",
            "recommended_discount",
            "estimated_conversion_probability",
            "expected_discount_spend",
            "estimated_incremental_contribution",
        ]

        preview = recommendations[preview_cols].head(20)

        preview = preview.replace({
            np.nan: None,
        })

        return {
            "summary": summary,
            "recommendations_preview": preview.to_dict(
                orient="records"
            ),
            "total_recommendations": int(
                len(recommendations)
            ),
            "synthetic_data": True,
            "note": (
                "Estimated results depend on the fitted causal model, "
                "baseline conversion model, and business assumptions. "
                "Discount allocation uses SciPy Mixed-Integer Linear "
                "Programming (MILP) to maximize estimated incremental "
                "contribution under an expected discount-spend budget. "
                "Results are based on synthetic educational data and "
                "are not guaranteed real-world outcomes."
            ),
        }

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Retail optimization failed: {exc}",
        ) from exc
