import sys
from pathlib import Path

# Add the project root directory to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import joblib
import numpy as np
import pandas as pd

from backend.app.preprocessing.retail_adapter import load_retail
from backend.app.preprocessing.dataset import make_splits
# --------------------------------------------------
# 1. Project paths and settings
# --------------------------------------------------

DATA_PATH = Path("data/processed/retail_synthetic.csv")
MODEL_PATH = Path("reports/run_retail_continuous/model.joblib")
OUTPUT_PATH = Path("reports/retail_dose_effect_validation.csv")

SEED = 42
DOSE_LEVELS = [5, 10, 20]


# --------------------------------------------------
# 2. Check required files
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"GBM model not found: {MODEL_PATH}")


# --------------------------------------------------
# 3. Load the retail dataset
# --------------------------------------------------

print("\nLoading retail dataset...")

ds = load_retail(
    path=str(DATA_PATH),
    outcome="conversion",
    treatment_kind="continuous",
    eligible_only=True,
    validate=True,
)

print("Eligible customers:", len(ds.X))


# --------------------------------------------------
# 4. Recreate the same train/validation/test split
# --------------------------------------------------

splits = make_splits(
    ds.T,
    ds.Y,
    seed=SEED,
)

test_idx = splits["test"]

X_test = ds.X.iloc[test_idx]

print("Test customers:", len(X_test))


# --------------------------------------------------
# 5. Load synthetic ground truth
# --------------------------------------------------

raw_df = pd.read_csv(DATA_PATH)

required_columns = ["eligible", "true_effect_per_dollar"]

for column in required_columns:
    if column not in raw_df.columns:
        raise ValueError(f"Required column missing: {column}")

# Match the eligible-only dataset used by load_retail().
eligible_df = raw_df.loc[
    raw_df["eligible"].astype(bool)
].reset_index(drop=True)

if len(eligible_df) != len(ds.X):
    raise ValueError(
        "Row count mismatch between eligible data and load_retail(). "
        "Check the eligibility filtering before continuing."
    )

true_effect_slope = (
    eligible_df["true_effect_per_dollar"]
    .to_numpy()[test_idx]
)


# --------------------------------------------------
# 6. Load the primary GBM DML model
# --------------------------------------------------

print("\nLoading GBM DML model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# --------------------------------------------------
# 7. Evaluate effects at each discount level
# --------------------------------------------------

results = []

print("\n========== EFFECT VALIDATION BY DISCOUNT ==========")

for dose in DOSE_LEVELS:

    predicted_effect, warnings = model.effect_at_dose(
        X_test,
        dose,
    )

    predicted_effect = np.asarray(
        predicted_effect
    ).reshape(-1)

    # Synthetic ground truth:
    # true effect at dose = effect per dollar * discount amount
    actual_effect = true_effect_slope * dose

    mae = np.mean(
        np.abs(predicted_effect - actual_effect)
    )

    rmse = np.sqrt(
        np.mean((predicted_effect - actual_effect) ** 2)
    )

    bias = np.mean(predicted_effect - actual_effect)

    result = {
        "discount_amount": dose,
        "test_customers": len(X_test),
        "mean_predicted_effect": np.mean(predicted_effect),
        "mean_true_effect": np.mean(actual_effect),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "min_predicted_effect": np.min(predicted_effect),
        "max_predicted_effect": np.max(predicted_effect),
    }

    results.append(result)

    print(f"\nDiscount: Rs. {dose}")
    print(f"Mean predicted effect: {result['mean_predicted_effect']:.6f}")
    print(f"Mean synthetic true effect: {result['mean_true_effect']:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Bias: {bias:.6f}")

    if warnings:
        print("Warnings:", warnings)


# --------------------------------------------------
# 8. Save results
# --------------------------------------------------

results_df = pd.DataFrame(results)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print("\n========== FINAL SUMMARY ==========")
print(results_df.to_string(index=False))

print(f"\nResults saved to: {OUTPUT_PATH}")