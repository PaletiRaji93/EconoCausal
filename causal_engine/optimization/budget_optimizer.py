
from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def optimize_discount_allocation(
    customers: pd.DataFrame,
    baseline_conversion,
    effect_per_dollar,
    budget: float,
    discount_levels=(0, 5, 10, 20),
    margin_rate: float = 0.30,
):
    """
    Allocate discounts using Mixed-Integer Linear Programming (MILP).

    Objective:
        Maximize estimated incremental contribution.

    Constraints:
        1. Each customer receives at most one discount.
        2. Total expected discount spend cannot exceed the budget.

    Important:
        - Discount spend is expected spend, not guaranteed realized spend.
        - The causal effect is assumed linear in discount amount.
        - Results depend on the quality of the baseline and causal models.
        - This is a synthetic-data demonstration, not a production guarantee.
    """

    # ---------------------------------------------------------
    # 1. Validate inputs
    # ---------------------------------------------------------
    if not isinstance(customers, pd.DataFrame):
        raise TypeError("customers must be a pandas DataFrame.")

    if "avg_basket" not in customers.columns:
        raise ValueError("customers must contain an 'avg_basket' column.")

    if customers.index.has_duplicates:
        raise ValueError("Customer index must contain unique values.")

    if not np.isfinite(budget) or budget < 0:
        raise ValueError("budget must be a finite, non-negative number.")

    if not np.isfinite(margin_rate) or not 0 < margin_rate <= 1:
        raise ValueError("margin_rate must be greater than 0 and at most 1.")

    n_customers = len(customers)

    p0 = np.asarray(baseline_conversion, dtype=float).reshape(-1)
    slopes = np.asarray(effect_per_dollar, dtype=float).reshape(-1)
    baskets = customers["avg_basket"].to_numpy(dtype=float)

    if len(p0) != n_customers or len(slopes) != n_customers:
        raise ValueError(
            "baseline_conversion and effect_per_dollar must match "
            "the number of customers."
        )

    if not (
        np.isfinite(p0).all()
        and np.isfinite(slopes).all()
        and np.isfinite(baskets).all()
    ):
        raise ValueError("Input arrays must contain only finite values.")

    if ((p0 < 0) | (p0 > 1)).any():
        raise ValueError("Baseline conversion probabilities must be between 0 and 1.")

    if (baskets < 0).any():
        raise ValueError("Average basket values cannot be negative.")

    levels = np.asarray(discount_levels, dtype=float)

    if levels.ndim != 1 or len(levels) == 0:
        raise ValueError("discount_levels must be a non-empty one-dimensional list.")

    if not np.isfinite(levels).all() or (levels < 0).any():
        raise ValueError("Discount levels must be finite and non-negative.")

    if len(np.unique(levels)) != len(levels):
        raise ValueError("Discount levels must be unique.")

    if not np.any(levels == 0):
        raise ValueError("discount_levels must include zero.")

    positive_levels = np.sort(levels[levels > 0])

    # ---------------------------------------------------------
    # 2. Prepare the recommendation output
    # ---------------------------------------------------------
    result = customers.copy()

    result["baseline_conversion_probability"] = p0
    result["estimated_effect_per_discount_dollar"] = slopes
    result["recommended_discount"] = 0.0
    result["estimated_conversion_probability"] = p0
    result["expected_discount_spend"] = 0.0
    result["estimated_incremental_contribution"] = 0.0

    # No positive discount options or no customers.
    if n_customers == 0 or len(positive_levels) == 0 or budget == 0:
        summary = {
            "customers_considered": int(n_customers),
            "customers_offered_discount": 0,
            "customers_without_discount": int(n_customers),
            "budget": float(budget),
            "expected_discount_spend": 0.0,
            "budget_remaining": float(budget),
            "budget_utilization": 0.0 if budget == 0 else 0.0,
            "estimated_incremental_contribution": 0.0,
            "discount_levels": levels.tolist(),
            "margin_rate_assumption": float(margin_rate),
            "synthetic_data": True,
            "optimizer": "SciPy MILP",
        }
        return result, summary

    # ---------------------------------------------------------
    # 3. Create one decision variable for each customer/discount
    # ---------------------------------------------------------
    # A decision variable is 1 if that customer receives that dose,
    # otherwise 0.
    #
    # Each variable represents one customer and one positive dose.

    customer_ids = []
    dose_values = []
    contribution_values = []
    spend_values = []
    probability_values = []

    contribution_margin = baskets * margin_rate

    for customer_id in range(n_customers):
        for dose in positive_levels:

            estimated_lift = slopes[customer_id] * dose

            predicted_probability = np.clip(
                p0[customer_id] + estimated_lift,
                0.0,
                1.0,
            )

            # Contribution after discount minus baseline contribution.
            incremental_contribution = (
                predicted_probability
                * (contribution_margin[customer_id] - dose)
                - p0[customer_id] * contribution_margin[customer_id]
            )

            expected_spend = predicted_probability * dose

            if expected_spend <= 0:
                continue

            customer_ids.append(customer_id)
            dose_values.append(dose)
            contribution_values.append(incremental_contribution)
            spend_values.append(expected_spend)
            probability_values.append(predicted_probability)

    n_variables = len(customer_ids)

    if n_variables == 0:
        summary = {
            "customers_considered": int(n_customers),
            "customers_offered_discount": 0,
            "customers_without_discount": int(n_customers),
            "budget": float(budget),
            "expected_discount_spend": 0.0,
            "budget_remaining": float(budget),
            "budget_utilization": 0.0,
            "estimated_incremental_contribution": 0.0,
            "discount_levels": levels.tolist(),
            "margin_rate_assumption": float(margin_rate),
            "synthetic_data": True,
            "optimizer": "SciPy MILP",
        }
        return result, summary

    customer_ids = np.asarray(customer_ids, dtype=int)
    dose_values = np.asarray(dose_values, dtype=float)
    contribution_values = np.asarray(contribution_values, dtype=float)
    spend_values = np.asarray(spend_values, dtype=float)
    probability_values = np.asarray(probability_values, dtype=float)

    # ---------------------------------------------------------
    # 4. Build MILP constraints
    # ---------------------------------------------------------
    # Rows 0 to n_customers-1:
    #   Each customer can receive at most one discount.
    #
    # Last row:
    #   Total expected discount spend <= budget.

    variable_ids = np.arange(n_variables)

    row_indices = np.concatenate([
        customer_ids,
        np.full(n_variables, n_customers, dtype=int),
    ])

    column_indices = np.concatenate([
        variable_ids,
        variable_ids,
    ])

    constraint_values = np.concatenate([
        np.ones(n_variables),
        spend_values,
    ])

    constraint_matrix = coo_matrix(
        (
            constraint_values,
            (row_indices, column_indices),
        ),
        shape=(n_customers + 1, n_variables),
    ).tocsr()

    lower_bounds = np.full(n_customers + 1, -np.inf)
    upper_bounds = np.concatenate([
        np.ones(n_customers),
        [float(budget)],
    ])

    constraints = LinearConstraint(
        constraint_matrix,
        lower_bounds,
        upper_bounds,
    )

    # ---------------------------------------------------------
    # 5. Solve the MILP optimization problem
    # ---------------------------------------------------------
    # scipy.optimize.milp minimizes, so negate contribution
    # to maximize it.
    objective = -contribution_values

    optimization_result = milp(
        c=objective,
        integrality=np.ones(n_variables),
        bounds=Bounds(
            np.zeros(n_variables),
            np.ones(n_variables),
        ),
        constraints=constraints,
    )

    if not optimization_result.success:
        raise RuntimeError(
            "MILP optimization failed: "
            f"{optimization_result.message}"
        )

    selected = optimization_result.x > 0.5

    # ---------------------------------------------------------
    # 6. Apply selected discounts to customers
    # ---------------------------------------------------------
    selected_positions = np.flatnonzero(selected)

    for position in selected_positions:
        customer_id = customer_ids[position]

        result.iloc[
            customer_id,
            result.columns.get_loc("recommended_discount"),
        ] = dose_values[position]

        result.iloc[
            customer_id,
            result.columns.get_loc("estimated_conversion_probability"),
        ] = probability_values[position]

        result.iloc[
            customer_id,
            result.columns.get_loc("expected_discount_spend"),
        ] = spend_values[position]

        result.iloc[
            customer_id,
            result.columns.get_loc("estimated_incremental_contribution"),
        ] = contribution_values[position]

    # ---------------------------------------------------------
    # 7. Calculate summary
    # ---------------------------------------------------------
    total_spend = float(result["expected_discount_spend"].sum())
    total_contribution = float(
        result["estimated_incremental_contribution"].sum()
    )

    customers_offered = int(
        (result["recommended_discount"] > 0).sum()
    )

    summary = {
        "customers_considered": int(n_customers),
        "customers_offered_discount": customers_offered,
        "customers_without_discount": int(n_customers - customers_offered),
        "budget": float(budget),
        "expected_discount_spend": total_spend,
        "budget_remaining": float(max(0.0, budget - total_spend)),
        "budget_utilization": (
            float(total_spend / budget) if budget > 0 else 0.0
        ),
        "estimated_incremental_contribution": total_contribution,
        "discount_levels": levels.tolist(),
        "margin_rate_assumption": float(margin_rate),
        "synthetic_data": True,
        "optimizer": "SciPy MILP",
        "solver_status": str(optimization_result.message),
    }

    return result, summary
cls