"""Budget-constrained discount allocation for EconoCausal.

Uses SciPy Mixed-Integer Linear Programming (MILP) to maximize
estimated incremental contribution under an expected discount budget.

Results are based on synthetic data and business assumptions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def optimize_discount_allocation(
    customers: pd.DataFrame,
    baseline_conversion: np.ndarray,
    effect_per_dollar: np.ndarray,
    budget: float,
    discount_levels: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0),
    margin_rate: float = 0.30,
) -> tuple[pd.DataFrame, dict]:
    """Optimize customer discount allocation using SciPy MILP.

    Each customer can receive at most one positive discount.

    The objective maximizes estimated incremental contribution,
    subject to the expected discount spend not exceeding the budget.

    Assumptions:
    - baseline_conversion estimates conversion without a discount.
    - effect_per_dollar estimates conversion lift per discount dollar.
    - avg_basket is the expected gross basket value.
    - Discount cost is incurred only when a customer converts.
    - avg_basket * margin_rate represents baseline contribution margin.
    - The estimated causal effect is linear in discount amount.
    """

    # ---------------------------------------------------------
    # 1. Validate inputs
    # ---------------------------------------------------------
    if not isinstance(customers, pd.DataFrame):
        raise TypeError("customers must be a pandas DataFrame")

    if "avg_basket" not in customers.columns:
        raise ValueError("customers must contain avg_basket")

    if not customers.index.is_unique:
        raise ValueError("customers must have a unique index")

    if not np.isfinite(budget) or budget < 0:
        raise ValueError("budget must be finite and non-negative")

    if not np.isfinite(margin_rate) or not 0 < margin_rate <= 1:
        raise ValueError("margin_rate must be in (0, 1]")

    n = len(customers)

    p0 = np.asarray(baseline_conversion, dtype=float).reshape(-1)
    slope = np.asarray(effect_per_dollar, dtype=float).reshape(-1)
    basket = customers["avg_basket"].to_numpy(dtype=float)

    if len(p0) != n or len(slope) != n:
        raise ValueError("Predictions must have one value per customer")

    if not (
        np.isfinite(p0).all()
        and np.isfinite(slope).all()
        and np.isfinite(basket).all()
    ):
        raise ValueError("Inputs must contain only finite values")

    if ((p0 < 0) | (p0 > 1)).any():
        raise ValueError("Baseline probabilities must be in [0, 1]")

    if (basket < 0).any():
        raise ValueError("avg_basket must be non-negative")

    levels = np.asarray(discount_levels, dtype=float)

    if levels.ndim != 1 or len(levels) == 0:
        raise ValueError("discount_levels must be a non-empty list")

    if not np.isfinite(levels).all() or (levels < 0).any():
        raise ValueError("Discount levels must be finite and non-negative")

    if len(np.unique(levels)) != len(levels):
        raise ValueError("Discount levels must be unique")

    if not np.any(levels == 0):
        raise ValueError("Discount levels must include zero")

    levels = np.sort(levels)
    positive_levels = levels[levels > 0]

    # ---------------------------------------------------------
    # 2. Initialize recommendation results
    # ---------------------------------------------------------
    result = customers.copy()

    result["baseline_conversion_probability"] = p0
    result["estimated_effect_per_discount_dollar"] = slope
    result["recommended_discount"] = 0.0
    result["estimated_conversion_probability"] = p0
    result["expected_discount_spend"] = 0.0
    result["estimated_incremental_contribution"] = 0.0

    def make_summary(spend: float, contribution: float, offered: int):
        return {
            "customers_considered": int(n),
            "customers_offered_discount": int(offered),
            "customers_without_discount": int(n - offered),
            "budget": float(budget),
            "expected_discount_spend": float(spend),
            "budget_remaining": float(max(0.0, budget - spend)),
            "budget_utilization": float(spend / budget) if budget > 0 else 0.0,
            "estimated_incremental_contribution": float(contribution),
            "discount_levels": levels.tolist(),
            "margin_rate_assumption": float(margin_rate),
            "synthetic_data": True,
            "optimizer": "SciPy MILP",
        }

    if n == 0 or len(positive_levels) == 0 or budget == 0:
        return result, make_summary(0.0, 0.0, 0)

    # ---------------------------------------------------------
    # 3. Create possible customer-discount offers
    # ---------------------------------------------------------
    customer_ids = []
    doses = []
    probabilities = []
    spends = []
    contributions = []

    baseline_margin = basket * margin_rate

    for i in range(n):
        for dose in positive_levels:

            # Estimated conversion probability after discount.
            probability = float(
                np.clip(p0[i] + slope[i] * dose, 0.0, 1.0)
            )

            # Estimated incremental contribution versus no discount.
            contribution = (
                probability * (baseline_margin[i] - dose)
                - p0[i] * baseline_margin[i]
            )

            # Expected discount expense.
            spend = probability * dose

            # Offers with negative contribution cannot improve
            # a maximization objective, so they can be excluded.
            if contribution > 0 and spend > 0:
                customer_ids.append(i)
                doses.append(dose)
                probabilities.append(probability)
                spends.append(spend)
                contributions.append(contribution)

    if not customer_ids:
        return result, make_summary(0.0, 0.0, 0)

    customer_ids = np.asarray(customer_ids, dtype=int)
    doses = np.asarray(doses, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    spends = np.asarray(spends, dtype=float)
    contributions = np.asarray(contributions, dtype=float)

    n_variables = len(customer_ids)

    # ---------------------------------------------------------
    # 4. Build optimization constraints
    # ---------------------------------------------------------
    # Each decision variable represents one customer-discount offer.
    #
    # Constraint A:
    # Each customer can receive at most one positive discount.
    #
    # Constraint B:
    # Total expected discount spend cannot exceed the budget.

    variable_ids = np.arange(n_variables)

    row_indices = np.concatenate([
        customer_ids,
        np.full(n_variables, n, dtype=int),
    ])

    column_indices = np.concatenate([
        variable_ids,
        variable_ids,
    ])

    matrix_values = np.concatenate([
        np.ones(n_variables),
        spends,
    ])

    constraint_matrix = coo_matrix(
        (matrix_values, (row_indices, column_indices)),
        shape=(n + 1, n_variables),
    ).tocsr()

    lower_bounds = np.full(n + 1, -np.inf)

    upper_bounds = np.concatenate([
        np.ones(n),
        [float(budget)],
    ])

    constraints = LinearConstraint(
        constraint_matrix,
        lb=lower_bounds,
        ub=upper_bounds,
    )

    # ---------------------------------------------------------
    # 5. Solve the MILP optimization problem
    # ---------------------------------------------------------
    # SciPy milp minimizes the objective.
    # Negating contribution converts maximization into minimization.

    optimization_result = milp(
        c=-contributions,
        integrality=np.ones(n_variables),
        bounds=Bounds(
            lb=np.zeros(n_variables),
            ub=np.ones(n_variables),
        ),
        constraints=constraints,
        options={"disp": False},
    )

    if not optimization_result.success:
        raise RuntimeError(
            "MILP optimization failed: "
            + str(optimization_result.message)
        )

    # ---------------------------------------------------------
    # 6. Apply selected customer-discount offers
    # ---------------------------------------------------------
    # # Identify which discount offers the MILP solver selected.
    selected = optimization_result.x > 0.5

    # Get positions of selected offers.
    selected_positions = np.flatnonzero(selected)

    # Get corresponding customer positions.
    selected_customers = customer_ids[selected_positions]

    # Initialize customer recommendations.
    result["recommended_discount"] = 0.0
    result["estimated_conversion_probability"] = p0
    result["expected_discount_spend"] = 0.0
    result["estimated_incremental_contribution"] = 0.0

    # Assign the selected recommendations using positional indexing.
    result.iloc[
        selected_customers,
        result.columns.get_loc("recommended_discount"),
    ] = doses[selected_positions]

    result.iloc[
        selected_customers,
        result.columns.get_loc("estimated_conversion_probability"),
    ] = probabilities[selected_positions]

    result.iloc[
        selected_customers,
        result.columns.get_loc("expected_discount_spend"),
    ] = spends[selected_positions]

    result.iloc[
        selected_customers,
        result.columns.get_loc("estimated_incremental_contribution"),
    ] = contributions[selected_positions]

    # ---------------------------------------------------------
    # 7. Calculate final budget summary
    # ---------------------------------------------------------
    total_spend = float(result["expected_discount_spend"].sum())

    total_contribution = float(
        result["estimated_incremental_contribution"].sum()
    )

    customers_offered = int(
        (result["recommended_discount"] > 0).sum()
    )

    if total_spend > budget + 1e-7:
        raise RuntimeError("MILP solution violated the budget constraint")

    summary = make_summary(
        spend=total_spend,
        contribution=total_contribution,
        offered=customers_offered,
    )

    summary["solver_status"] = str(optimization_result.message)

    return result, summary
