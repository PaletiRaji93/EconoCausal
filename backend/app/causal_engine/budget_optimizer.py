
"""Budget-constrained discount allocation for EconoCausal.

Uses estimated causal conversion lift and synthetic business assumptions.
The greedy allocator is a heuristic, not a globally optimal solver.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def optimize_discount_allocation(
    customers: pd.DataFrame,
    baseline_conversion: np.ndarray,
    effect_per_dollar: np.ndarray,
    budget: float,
    discount_levels: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0),
    margin_rate: float = 0.30,
) -> tuple[pd.DataFrame, dict]:
    """Recommend discounts without exceeding the expected discount budget.

    Assumptions:
    - `baseline_conversion` estimates conversion probability without a discount.
    - `effect_per_dollar` estimates conversion-probability lift per discount dollar.
    - `avg_basket` is the expected gross basket value.
    - Discount cost is incurred only when a customer converts.
    - Contribution margin is represented by `margin_rate`.

    The allocator greedily chooses positive-value offers by contribution per
    expected discount cost, with at most one offer per customer.
    """
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if not 0 < margin_rate <= 1:
        raise ValueError("margin_rate must be in (0, 1]")
    if not customers.index.is_unique:
        raise ValueError("customers must have a unique index")

    required = {"avg_basket"}
    missing = required - set(customers.columns)
    if missing:
        raise ValueError(f"Missing required customer columns: {sorted(missing)}")

    levels = sorted(set(float(d) for d in discount_levels))
    if not levels or levels[0] != 0.0 or any(d < 0 for d in levels):
        raise ValueError("Discount levels must include 0 and be non-negative")

    p0 = np.asarray(baseline_conversion, dtype=float).reshape(-1)
    slope = np.asarray(effect_per_dollar, dtype=float).reshape(-1)
    basket = customers["avg_basket"].to_numpy(dtype=float)

    n = len(customers)
    if len(p0) != n or len(slope) != n:
        raise ValueError("Predictions must have one value per customer")
    if not np.isfinite(p0).all() or not np.isfinite(slope).all():
        raise ValueError("Predictions must contain finite values")
    if not np.isfinite(basket).all() or (basket < 0).any():
        raise ValueError("avg_basket must contain finite non-negative values")
    if ((p0 < 0) | (p0 > 1)).any():
        raise ValueError("Baseline conversion probabilities must be in [0, 1]")

    # Every customer starts with no discount and zero incremental contribution.
    chosen_dose = np.zeros(n, dtype=float)
    chosen_probability = p0.copy()
    chosen_spend = np.zeros(n, dtype=float)
    chosen_value = np.zeros(n, dtype=float)

    offers = []
    for i in range(n):
        for dose in levels:
            if dose == 0:
                continue

            lift = slope[i] * dose
            probability = float(np.clip(p0[i] + lift, 0.0, 1.0))

            # Incremental contribution versus no discount.
            contribution = (
                probability * (basket[i] - dose) * margin_rate
                - p0[i] * basket[i] * margin_rate
            )

            # Expected discount expense, assuming it is paid on conversion.
            spend = probability * dose

            if contribution > 0 and spend > 0:
                offers.append({
                    "customer_position": i,
                    "discount_amount": dose,
                    "conversion_probability": probability,
                    "expected_discount_spend": spend,
                    "incremental_contribution": contribution,
                    "value_per_spend": contribution / spend,
                })

    # Greedy heuristic: highest estimated contribution per expected dollar first.
    offers.sort(
        key=lambda offer: (
            -offer["value_per_spend"],
            -offer["incremental_contribution"],
            offer["customer_position"],
        )
    )

    assigned = np.zeros(n, dtype=bool)
    expected_spend = 0.0
    total_value = 0.0

    for offer in offers:
        i = offer["customer_position"]
        if assigned[i]:
            continue

        cost = offer["expected_discount_spend"]
        if expected_spend + cost > budget + 1e-9:
            continue

        assigned[i] = True
        chosen_dose[i] = offer["discount_amount"]
        chosen_probability[i] = offer["conversion_probability"]
        chosen_spend[i] = cost
        chosen_value[i] = offer["incremental_contribution"]
        expected_spend += cost
        total_value += offer["incremental_contribution"]

    result = customers.copy()
    result["baseline_conversion_probability"] = p0
    result["estimated_effect_per_discount_dollar"] = slope
    result["recommended_discount"] = chosen_dose
    result["estimated_conversion_probability"] = chosen_probability
    result["expected_discount_spend"] = chosen_spend
    result["estimated_incremental_contribution"] = chosen_value

    summary = {
        "customers_considered": int(n),
        "customers_offered_discount": int(assigned.sum()),
        "customers_without_discount": int((~assigned).sum()),
        "budget": float(budget),
        "expected_discount_spend": float(expected_spend),
        "budget_remaining": float(max(0.0, budget - expected_spend)),
        "budget_utilization": float(expected_spend / budget) if budget else 0.0,
        "estimated_incremental_contribution": float(total_value),
        "discount_levels": levels,
        "margin_rate_assumption": float(margin_rate),
        "synthetic_data": True,
        "optimizer": "greedy heuristic",
    }

    if expected_spend > budget + 1e-7:
        raise RuntimeError("Budget constraint violated")

    return result, summary
