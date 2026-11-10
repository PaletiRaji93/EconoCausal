import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from backend.app.causal_engine.budget_optimizer import (
    optimize_discount_allocation,
)


# --------------------------------------------------
# 1. Settings
# --------------------------------------------------

INPUT_PATH = Path(
    "reports/run_retail_budget_validation/recommendations.csv"
)

BUDGET = 1000.0
MARGIN_RATE = 0.30
DISCOUNT_LEVELS = (0.0, 5.0, 10.0, 20.0)

# --------------------------------------------------
# 2. Load recommendations
# --------------------------------------------------

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Recommendations not found: {INPUT_PATH}"
    )

df = pd.read_csv(INPUT_PATH)

required = [
    "avg_basket",
    "baseline_conversion_probability",
    "estimated_effect_per_discount_dollar",
    "true_effect_per_dollar",
    "recommended_discount",
    "expected_discount_spend",
    "estimated_incremental_contribution",
]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

print("\nLoaded recommendations:", len(df))


# --------------------------------------------------
# 3. Extract values
# --------------------------------------------------

basket = df["avg_basket"].to_numpy(dtype=float)
p0 = df["baseline_conversion_probability"].to_numpy(dtype=float)

predicted_slope = df[
    "estimated_effect_per_discount_dollar"
].to_numpy(dtype=float)

true_slope = df[
    "true_effect_per_dollar"
].to_numpy(dtype=float)

recommended_dose = df[
    "recommended_discount"
].to_numpy(dtype=float)

margin = basket * MARGIN_RATE


# --------------------------------------------------
# 4. Evaluate the recommended allocation
#    using synthetic ground truth
# --------------------------------------------------

true_probability = np.clip(
    p0 + true_slope * recommended_dose,
    0.0,
    1.0,
)

true_expected_spend = (
    true_probability * recommended_dose
)

true_incremental_contribution = (
    true_probability * (margin - recommended_dose)
    - p0 * margin
)

model_expected_spend = df[
    "expected_discount_spend"
].sum()

model_contribution = df[
    "estimated_incremental_contribution"
].sum()

ground_truth_spend = true_expected_spend.sum()
ground_truth_contribution = (
    true_incremental_contribution.sum()
)


# --------------------------------------------------
# 5. Check allocation counts and budget
# --------------------------------------------------

print("\n========== RECOMMENDED ALLOCATION ==========")

print(
    df["recommended_discount"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nModel-estimated expected spend:",
      round(model_expected_spend, 4))

print("Synthetic-ground-truth expected spend:",
      round(ground_truth_spend, 4))

print("Budget:",
      round(BUDGET, 4))

print("Ground-truth budget remaining:",
      round(BUDGET - ground_truth_spend, 4))

print("Ground-truth budget compliant:",
      ground_truth_spend <= BUDGET + 1e-7)

print("\nModel-estimated incremental contribution:",
      round(model_contribution, 4))

print("Synthetic-ground-truth incremental contribution:",
      round(ground_truth_contribution, 4))


# --------------------------------------------------
# 6. Compare discounted and non-discounted customers
# --------------------------------------------------

treated = recommended_dose > 0
untreated = ~treated

print("\n========== CUSTOMER TARGETING ==========")

print("Discounted customers:", int(treated.sum()))
print("Non-discounted customers:", int(untreated.sum()))

print(
    "Mean true effect slope - discounted:",
    round(float(true_slope[treated].mean()), 6)
)

print(
    "Mean true effect slope - non-discounted:",
    round(float(true_slope[untreated].mean()), 6)
)


# --------------------------------------------------
# 7. Build an oracle allocation
# --------------------------------------------------
# This uses known synthetic treatment-effect slopes.
# It is an upper benchmark, not a deployable model.

oracle_recommendations, oracle_summary = (
    optimize_discount_allocation(
        customers=df,
        baseline_conversion=p0,
        effect_per_dollar=true_slope,
        budget=BUDGET,
        discount_levels=DISCOUNT_LEVELS,
        margin_rate=MARGIN_RATE,
    )
)

oracle_dose = oracle_recommendations[
    "recommended_discount"
].to_numpy(dtype=float)

oracle_probability = np.clip(
    p0 + true_slope * oracle_dose,
    0.0,
    1.0,
)

oracle_spend = (
    oracle_probability * oracle_dose
).sum()

oracle_contribution = (
    oracle_probability * (margin - oracle_dose)
    - p0 * margin
).sum()


# --------------------------------------------------
# 8. Compare actual model allocation with oracle
# --------------------------------------------------

print("\n========== ORACLE BENCHMARK ==========")

print(
    "Oracle customers discounted:",
    int((oracle_dose > 0).sum())
)

print(
    "Oracle expected spend:",
    round(float(oracle_spend), 4)
)

print(
    "Oracle synthetic incremental contribution:",
    round(float(oracle_contribution), 4)
)

print(
    "\nModel allocation synthetic contribution:",
    round(float(ground_truth_contribution), 4)
)

if oracle_contribution > 0:
    ratio = (
        ground_truth_contribution / oracle_contribution
    )

    print(
        "Model allocation / oracle contribution:",
        f"{ratio:.2%}"
    )


# --------------------------------------------------
# 9. Save customer-level validation
# --------------------------------------------------

df["synthetic_true_conversion_probability"] = true_probability
df["synthetic_true_expected_discount_spend"] = true_expected_spend
df["synthetic_true_incremental_contribution"] = (
    true_incremental_contribution
)

output_path = Path(
    "reports/retail_budget_validation_customers.csv"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(output_path, index=False)

print("\nCustomer validation saved to:", output_path)
print("\nBudget validation completed.")