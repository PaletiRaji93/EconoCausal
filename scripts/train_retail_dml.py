
"""Train continuous-treatment DML for synthetic retail discounts.

Run from the repository root:
python scripts/train_retail_dml.py --path data/processed/retail_synthetic.csv
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from backend.app.core.config import settings
from backend.app.core.logging_config import setup_logging
from backend.app.preprocessing.dataset import make_splits
from backend.app.preprocessing.retail_adapter import load_retail
from backend.app.dml.continuous_dml import ContinuousDML


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        default="data/processed/retail_synthetic.csv",
        help="Path to the retail dataset CSV",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nuisance", choices=["linear", "gbm"], default="gbm")
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument(
        "--out",
        default="reports/run_retail_continuous",
        help="Output directory for model and evaluation artifacts",
    )
    args = parser.parse_args()

    setup_logging(settings.log_level)

    # Use discount_amount as the continuous treatment dose.
    ds = load_retail(
        path=args.path,
        outcome="conversion",
        treatment_kind="continuous",
        eligible_only=True,
        validate=True,
    )

    # Reproducible train/validation/test split.
    splits = make_splits(ds.T, ds.Y, seed=args.seed)
    train_idx = splits["train"]
    test_idx = splits["test"]

    print("Dataset:", ds.name)
    print("Treatment:", ds.treatment_col)
    print("Outcome:", ds.outcome_col)
    print("Features:", ds.feature_cols)
    print(
        f"Eligible rows: {len(ds.df)} | "
        f"Train: {len(train_idx)} | "
        f"Validation: {len(splits['val'])} | "
        f"Test: {len(test_idx)}"
    )

    # Fit only on the training split.
    model = ContinuousDML(
        nuisance=args.nuisance,
        cv=args.cv,
        seed=args.seed,
    )
    model.fit(ds, train_idx)

    # Evaluate the estimated effect at $10 against synthetic ground truth.
    # The true_effect column is evaluation-only and is never a model feature.
    test_X = ds.X.iloc[test_idx]
    predicted_effect_10, warnings = model.effect_at_dose(test_X, 10.0)
    true_effect_10 = ds.df["true_effect"].to_numpy()[test_idx]

    metrics = {
        "evaluation_dose": 10.0,
        "test_rows": int(len(test_idx)),
        "mean_predicted_effect": float(np.mean(predicted_effect_10)),
        "mean_synthetic_true_effect": float(np.mean(true_effect_10)),
        "mae_vs_synthetic_truth": float(
            mean_absolute_error(true_effect_10, predicted_effect_10)
        ),
        "rmse_vs_synthetic_truth": float(
            np.sqrt(mean_squared_error(true_effect_10, predicted_effect_10))
        ),
    }

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep this model separate from the existing Criteo model.
    joblib.dump(model, output_dir / "model.joblib")

    metadata = {
        **model.summary(),
        "synthetic_data": True,
        "feature_cols": ds.feature_cols,
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(splits["val"])),
        "test_rows": int(len(test_idx)),
        "seed": args.seed,
        "nuisance": args.nuisance,
        "cv": args.cv,
        "metrics": metrics,
        "note": (
            "Synthetic educational evaluation only; results do not establish "
            "real-world discount effectiveness. true_effect is used only for "
            "evaluation, never as a model feature."
        ),
    }

    with (output_dir / "model.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"metrics": metrics, "warnings": warnings},
            f,
            indent=2,
        )

    print("\nModel training completed.")
    print("Model:", output_dir / "model.joblib")
    print("Metadata:", output_dir / "model.json")
    print("Evaluation:", json.dumps(metrics, indent=2))

    for warning in warnings:
        print("WARNING:", warning)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
