
"""Prepare retail campaign data for causal inference.

Treatment: assigned
Outcome: converted
Features: customer characteristics measured before treatment

Post-treatment variables such as discount_amount and order_value
are intentionally excluded from the feature matrix.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "sample" / "retail_campaign.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

FEATURE_COLUMNS = [
    "age",
    "past_purchases",
    "previous_spend",
    "loyalty_score",
]

TREATMENT_COLUMN = "assigned"
OUTCOME_COLUMN = "converted"


def load_campaign_data(path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load campaign records and verify required columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Generate the synthetic dataset first."
        )

    df = pd.read_csv(path)

    required = {
        "customer_id",
        *FEATURE_COLUMNS,
        TREATMENT_COLUMN,
        OUTCOME_COLUMN,
        "discount_amount",
        "order_value",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("The dataset is empty.")

    if df["customer_id"].isna().any():
        raise ValueError("Customer IDs contain missing values.")

    if df["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer IDs were found.")

    if df[list(required - {"customer_id"})].isna().any().any():
        raise ValueError("Required data contains missing values.")

    if not df[TREATMENT_COLUMN].isin([0, 1]).all():
        raise ValueError("Treatment must contain only 0 or 1.")

    if not df[OUTCOME_COLUMN].isin([0, 1]).all():
        raise ValueError("Outcome must contain only 0 or 1.")

    if df[TREATMENT_COLUMN].nunique() != 2:
        raise ValueError(
            "Both treated and untreated customers are required."
        )

    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """Separate pretreatment features, treatment and outcome."""
    X = df[FEATURE_COLUMNS].copy()
    treatment = df[TREATMENT_COLUMN].astype(int).copy()
    outcome = df[OUTCOME_COLUMN].astype(int).copy()

    if not all(
        pd.api.types.is_numeric_dtype(X[column])
        for column in X.columns
    ):
        raise ValueError("All model features must be numeric.")

    if not X.map(lambda column: column.notna().all()).all():
        raise ValueError("Features contain missing values.")

    return X, treatment, outcome


def save_prepared_data(
    df: pd.DataFrame,
    X: pd.DataFrame,
    treatment: pd.Series,
    outcome: pd.Series,
) -> Path:
    """Save a clean modelling dataset for reproducible analysis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prepared = X.copy()
    prepared["assigned"] = treatment.to_numpy()
    prepared["converted"] = outcome.to_numpy()

    # Keep the customer identifier for traceability, not modelling.
    prepared.insert(0, "customer_id", df["customer_id"].to_numpy())

    output_path = OUTPUT_DIR / "causal_dataset.csv"
    prepared.to_csv(output_path, index=False)

    return output_path


def main() -> None:
    """Run preprocessing and print a concise summary."""
    df = load_campaign_data()
    X, treatment, outcome = prepare_features(df)

    output_path = save_prepared_data(df, X, treatment, outcome)

    print("Preprocessing completed successfully.")
    print(f"Input rows: {len(df)}")
    print(f"Feature columns: {list(X.columns)}")
    print(f"Treatment column: {TREATMENT_COLUMN}")
    print(f"Outcome column: {OUTCOME_COLUMN}")
    print(f"Treated customers: {int(treatment.sum())}")
    print(f"Untreated customers: {int((treatment == 0).sum())}")
    print(f"Overall conversion rate: {outcome.mean():.4f}")
    print(f"Saved prepared dataset: {output_path}")

    print(
        "\nNote: discount_amount and order_value are excluded "
        "from the feature matrix to avoid post-treatment leakage."
    )


if __name__ == "__main__":
    main()
