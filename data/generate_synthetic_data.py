
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N_CUSTOMERS = 3000

OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "sample"
    / "retail_campaign.csv"
)


def generate_data(
    n_customers: int = N_CUSTOMERS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Generate reproducible synthetic retail campaign data."""
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 71, size=n_customers)
    past_purchases = rng.poisson(4, size=n_customers)
    previous_spend = np.clip(
        rng.normal(120, 45, size=n_customers), 10, None
    )
    loyalty_score = rng.uniform(0, 1, size=n_customers)

    # Campaign assignment depends on observed customer features.
    # This creates potential confounding in observational analysis.
    assignment_logit = (
        -0.8
        + 0.12 * past_purchases
        + 0.9 * loyalty_score
        - 0.008 * (age - 40)
    )
    assignment_probability = 1 / (
        1 + np.exp(-assignment_logit)
    )
    assigned = rng.binomial(1, assignment_probability)

    discount_options = np.array([5.0, 10.0, 20.0])
    selected_discount = rng.choice(
        discount_options, size=n_customers
    )
    discount_amount = assigned * selected_discount

    # True treatment response varies with customer characteristics.
    true_effect = (
        0.04
        + 0.12 * loyalty_score
        + 0.015 * np.minimum(past_purchases, 5)
    )

    # Illustrative conversion response to the assigned discount.
    baseline_logit = (
        -1.6
        + 0.10 * past_purchases
        + 0.7 * loyalty_score
        + 0.002 * (previous_spend - 120)
    )
    treatment_effect = (
        true_effect * discount_amount / 10.0
    )
    purchase_probability = 1 / (
        1 + np.exp(
            -(baseline_logit + treatment_effect)
        )
    )
    purchase_probability = np.clip(
        purchase_probability, 0.01, 0.99
    )
    converted = rng.binomial(1, purchase_probability)

    order_value = np.where(
        converted == 1,
        np.clip(
            rng.normal(80 + 0.3 * previous_spend, 20),
            10,
            None,
        ),
        0.0,
    )

    return pd.DataFrame({
        "customer_id": np.arange(1, n_customers + 1),
        "age": age,
        "past_purchases": past_purchases,
        "previous_spend": previous_spend.round(2),
        "loyalty_score": loyalty_score.round(4),
        "assigned": assigned,
        "discount_amount": discount_amount,
        "converted": converted,
        "order_value": order_value.round(2),
    })


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    df = generate_data()
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Dataset saved to: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nTreatment distribution:")
    print(df["assigned"].value_counts())
    print("\nConversion rate by assignment:")
    print(df.groupby("assigned")["converted"].mean())
    print("\nFirst five rows:")
    print(df.head())
