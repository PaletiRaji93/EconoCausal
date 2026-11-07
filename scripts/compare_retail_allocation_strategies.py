
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from backend.app.causal_engine.budget_optimizer import (
    optimize_discount_allocation,
)


# ==================================================
# 1. SETTINGS
# ==================================================

INPUT_PATH = Path(
    "reports/run_retail_budget_validation/recommendations.csv"
)

OUTPUT_DIR = Path("reports/retail_strategy_comparison")

BUDGET = 1000.0
MARGIN_RATE = 0.30
DISCOUNT_LEVELS = (0.0, 5.0, 10.0, 20.0)
POSITIVE_DISCOUNT_LEVELS = (5.0, 10.0, 20.0)
RANDOM_SEED = 42


# ==================================================
# 2. LOAD EXISTING DML RECOMMENDATIONS
# ==================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Recommendations not found: {INPUT_PATH}"
    )

df = pd.read_csv(INPUT_PATH)

required_columns = [
    "avg_basket",
    "baseline_conversion_probability",
    "estimated_effect_per_discount_dollar",
    "true_effect_per_dollar",
    "recommended_discount",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

print("Customers loaded:", len(df))


# ==================================================
# 3. EXTRACT CUSTOMER INFORMATION
# ==================================================

basket = df["avg_basket"].to_numpy(dtype=float)

p0 = df[
    "baseline_conversion_probability"
].to_numpy(dtype=float)

predicted_slope = df[
    "estimated_effect_per_discount_dollar"
].to_numpy(dtype=float)

true_slope = df[
    "true_effect_per_dollar"
].to_numpy(dtype=float)

dml_dose = df[
    "recommended_discount"
].to_numpy(dtype=float)

margin = basket * MARGIN_RATE

rng = np.random.default_rng(RANDOM_SEED)


# ==================================================
# 4. COMMON EVALUATION FUNCTION
# ==================================================

def evaluate_strategy(name, dose, estimated_spend):
    """
    Evaluate a discount allocation using synthetic
    ground-truth treatment effects.

    estimated_spend is calculated using the model's
    predicted conversion probabilities.

    true_spend and contribution use synthetic ground truth.
    """

    dose = np.asarray(dose, dtype=float)

    estimated_spend = float(estimated_spend)

    true_probability = np.clip(
        p0 + true_slope * dose,
        0.0,
        1.0,
    )

    true_spend = float(
        (true_probability * dose).sum()
    )

    incremental_contribution = float(
        (
            true_probability * (margin - dose)
            - p0 * margin
        ).sum()
    )

    return {
        "strategy": name,
        "customers_considered": len(df),
        "customers_discounted": int((dose > 0).sum()),
        "customers_without_discount": int((dose == 0).sum()),
        "discount_5": int((dose == 5).sum()),
        "discount_10": int((dose == 10).sum()),
        "discount_20": int((dose == 20).sum()),
        "model_estimated_spend": estimated_spend,
        "synthetic_true_expected_spend": true_spend,
        "budget": BUDGET,
        "model_budget_compliant": bool(
            estimated_spend <= BUDGET + 1e-7
        ),
        "synthetic_true_budget_compliant": bool(
            true_spend <= BUDGET + 1e-7
        ),
        "synthetic_true_budget_remaining": (
            BUDGET - true_spend
        ),
        "budget_utilization_pct": (
            true_spend / BUDGET * 100
        ),
        "synthetic_true_incremental_contribution": (
            incremental_contribution
        ),
    }


# ==================================================
# 5. STRATEGY 1 — DML ALLOCATION
# ==================================================

print("\nEvaluating DML strategy...")

dml_estimated_spend = float(
    df["expected_discount_spend"].sum()
)

dml_result = evaluate_strategy(
    "DML",
    dml_dose,
    dml_estimated_spend,
)


# ==================================================
# 6. STRATEGY 2 — RANDOM ALLOCATION
# ==================================================

print("Creating random allocation...")

random_dose = np.zeros(len(df), dtype=float)

customer_order = rng.permutation(len(df))

remaining_budget = BUDGET
random_estimated_spend = 0.0

for customer_index in customer_order:

    # Randomly choose a positive discount.
    proposed_dose = float(
        rng.choice(POSITIVE_DISCOUNT_LEVELS)
    )

    # Estimate conversion probability using the
    # DML-predicted treatment effect.
    estimated_probability = np.clip(
        p0[customer_index]
        + predicted_slope[customer_index] * proposed_dose,
        0.0,
        1.0,
    )

    # Estimate expected discount spending.
    estimated_spend = (
        estimated_probability * proposed_dose
    )

    # Assign only if the expected spend fits the budget.
    if estimated_spend <= remaining_budget:

        random_dose[customer_index] = proposed_dose

        random_estimated_spend += estimated_spend

        remaining_budget -= estimated_spend


random_result = evaluate_strategy(
    "Random",
    random_dose,
    random_estimated_spend,
)


# ==================================================
# 7. STRATEGY 3 — ORACLE ALLOCATION
# ==================================================

print("Creating oracle allocation...")

# Oracle uses synthetic ground-truth treatment effects.
# This is an ideal benchmark, not a deployable model.

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

oracle_estimated_probability = np.clip(
    p0 + true_slope * oracle_dose,
    0.0,
    1.0,
)

oracle_estimated_spend = float(
    (oracle_estimated_probability * oracle_dose).sum()
)

oracle_result = evaluate_strategy(
    "Oracle",
    oracle_dose,
    oracle_estimated_spend,
)


# ==================================================
# 8. CREATE COMPARISON TABLE
# ==================================================

comparison = pd.DataFrame(
    [
        dml_result,
        random_result,
        oracle_result,
    ]
)

oracle_contribution = oracle_result[
    "synthetic_true_incremental_contribution"
]

if oracle_contribution > 0:

    comparison["contribution_vs_oracle_pct"] = (
        comparison[
            "synthetic_true_incremental_contribution"
        ]
        / oracle_contribution
        * 100
    )

else:
    comparison["contribution_vs_oracle_pct"] = np.nan


# ==================================================
# 9. DISPLAY RESULTS
# ==================================================

print("\n========== STRATEGY COMPARISON ==========")

display_columns = [
    "strategy",
    "customers_discounted",
    "discount_5",
    "discount_10",
    "discount_20",
    "model_estimated_spend",
    "synthetic_true_expected_spend",
    "synthetic_true_budget_compliant",
    "budget_utilization_pct",
    "synthetic_true_incremental_contribution",
    "contribution_vs_oracle_pct",
]

print(
    comparison[display_columns].round(4).to_string(
        index=False
    )
)


# ==================================================
# 10. SAVE RESULTS
# ==================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

comparison_path = (
    OUTPUT_DIR / "strategy_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False,
)

customer_results = df.copy()

customer_results["dml_discount"] = dml_dose
customer_results["random_discount"] = random_dose
customer_results["oracle_discount"] = oracle_dose

customer_results.to_csv(
    OUTPUT_DIR / "customer_strategy_allocations.csv",
    index=False,
)

print("\nComparison saved to:", comparison_path)

print(
    "Customer allocations saved to:",
    OUTPUT_DIR / "customer_strategy_allocations.csv",
)

print("\nStrategy comparison completed successfully.")
