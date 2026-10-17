
"""Run the EconoCausal retail discount allocator.

Example:
python scripts/optimize_retail_budget.py --budget 1000
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from backend.app.preprocessing.retail_adapter import load_retail
from backend.app.preprocessing.feature_selection import build_preprocessor
from backend.app.causal_engine.budget_optimizer import optimize_discount_allocation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        default="data/processed/retail_synthetic.csv",
    )
    parser.add_argument(
        "--model",
        default="reports/run_retail_continuous/model.joblib",
    )
    parser.add_argument("--budget", type=float, default=1000.0)
    parser.add_argument("--margin-rate", type=float, default=0.30)
    parser.add_argument(
        "--out",
        default="reports/run_retail_budget",
    )
    args = parser.parse_args()

    if args.budget < 0:
        parser.error("--budget must be non-negative")

    ds = load_retail(
        path=args.path,
        outcome="conversion",
        treatment_kind="continuous",
        eligible_only=True,
        validate=True,
    )

    model_path = Path(args.model)
    if not model_path.exists():
        parser.error(f"Model file not found: {model_path}")

    model = joblib.load(model_path)

    # Fit a baseline conversion model using only untreated customers.
    # This is a demonstration baseline, not a production-validated model.
    untreated = ds.df[ds.df[ds.treatment_col] == 0]
    if untreated[ds.outcome_col].nunique() < 2:
        parser.error("Untreated rows must contain both conversion classes")

    X0 = untreated[ds.feature_cols]
    y0 = untreated[ds.outcome_col].astype(int)

    baseline_model = Pipeline([
        ("preprocessor", build_preprocessor(X0)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    baseline_model.fit(X0, y0)

    X = ds.X
    p0 = baseline_model.predict_proba(X)[:, 1]
    effect_per_dollar = model.effect_per_unit(X)

    # These are the observed dose levels supported by the saved model.
    levels = tuple(float(d) for d in model.support["levels"])

    recommendations, summary = optimize_discount_allocation(
        customers=ds.df,
        baseline_conversion=p0,
        effect_per_dollar=effect_per_dollar,
        budget=args.budget,
        discount_levels=levels,
        margin_rate=args.margin_rate,
    )

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    recommendations.to_csv(
        output_dir / "recommendations.csv",
        index=False,
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nBudget optimization completed.")
    print(json.dumps(summary, indent=2))
    print("\nDiscount allocation:")
    print(
        recommendations["recommended_discount"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print("\nSaved files:")
    print(output_dir / "recommendations.csv")
    print(output_dir / "summary.json")
    print(
        "\nNOTE: Synthetic-data demonstration. Estimated values depend on "
        "the baseline model, causal model, and business assumptions."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
