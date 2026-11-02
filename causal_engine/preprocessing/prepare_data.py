
"""Prepare retail campaign data for causal inference.

Treatment: assigned
Outcome: converted
Features: customer characteristics measured before treatment

Post-treatment variables such as discount_amount and order_value
are intentionally excluded from the feature matrix.
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "sample"
    / "retail_campaign.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

FEATURE_COLUMNS = [
    "age",
    "past_purchases",
    "previous_spend",
    "loyalty_score",
]

TREATMENT_COLUMN = "assigned"
OUTCOME_COLUMN = "converted"


# ---------------------------------------------------------
# 2. LOAD AND VALIDATE RAW DATA
# ---------------------------------------------------------

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

    # Customer ID validation
    if df["customer_id"].isna().any():
        raise ValueError("Customer IDs contain missing values.")

    if df["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer IDs were found.")

    # Missing-value validation
    validation_columns = list(required - {"customer_id"})

    missing_counts = df[validation_columns].isnull().sum()

    if missing_counts.any():
        raise ValueError(
            "Missing values found:\n"
            f"{missing_counts[missing_counts > 0]}"
        )

    # Treatment and outcome must be binary
    if not df[TREATMENT_COLUMN].isin([0, 1]).all():
        raise ValueError("Treatment must contain only 0 or 1.")

    if not df[OUTCOME_COLUMN].isin([0, 1]).all():
        raise ValueError("Outcome must contain only 0 or 1.")

    if df[TREATMENT_COLUMN].nunique() != 2:
        raise ValueError(
            "Both treated and untreated customers are required."
        )

    # Numeric feature validation
    for column in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"Feature '{column}' must be numeric."
            )

    # Non-negative economic values
    for column in ["discount_amount", "order_value"]:
        if (df[column] < 0).any():
            raise ValueError(
                f"Column '{column}' cannot contain negative values."
            )

    # Untreated customers should not receive discounts
    untreated = df[TREATMENT_COLUMN] == 0

    if (df.loc[untreated, "discount_amount"] != 0).any():
        raise ValueError(
            "Untreated customers must have zero discount amount."
        )

    return df


# ---------------------------------------------------------
# 3. PREPARE FEATURES, TREATMENT AND OUTCOME
# ---------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separate pretreatment features, treatment and outcome."""

    X = df[FEATURE_COLUMNS].copy()

    treatment = df[TREATMENT_COLUMN].astype(int).copy()

    outcome = df[OUTCOME_COLUMN].astype(int).copy()

    # Check that all features are numeric
    for column in X.columns:
        if not pd.api.types.is_numeric_dtype(X[column]):
            raise ValueError(
                f"Feature '{column}' must be numeric."
            )

    # Correct missing-value validation
    if X.isnull().any().any():
        raise ValueError(
            "Features contain missing values."
        )

    if treatment.isnull().any():
        raise ValueError(
            "Treatment contains missing values."
        )

    if outcome.isnull().any():
        raise ValueError(
            "Outcome contains missing values."
        )

    return X, treatment, outcome


# ---------------------------------------------------------
# 4. SAVE PREPARED DATASET
# ---------------------------------------------------------

def save_prepared_data(
    df: pd.DataFrame,
    X: pd.DataFrame,
    treatment: pd.Series,
    outcome: pd.Series,
) -> Path:
    """Save a clean modelling dataset."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepared = X.copy()

    prepared["assigned"] = treatment.to_numpy()

    prepared["converted"] = outcome.to_numpy()

    # Preserve customer ID for traceability
    # It will not be used as a model feature.
    prepared.insert(
        0,
        "customer_id",
        df["customer_id"].to_numpy(),
    )

    output_path = OUTPUT_DIR / "causal_dataset.csv"

    prepared.to_csv(
        output_path,
        index=False,
    )

    return output_path


# ---------------------------------------------------------
# 5. MAIN EXECUTION
# ---------------------------------------------------------

def main() -> None:
    """Run preprocessing and print a concise summary."""

    print("\nStarting causal data preprocessing...\n")

    df = load_campaign_data()

    X, treatment, outcome = prepare_features(df)

    output_path = save_prepared_data(
        df,
        X,
        treatment,
        outcome,
    )

    print("Preprocessing completed successfully!")

    print("\nDATASET SUMMARY")
    print("------------------------------")

    print(f"Total customers: {len(df)}")

    print(f"Feature columns: {list(X.columns)}")

    print(f"Treatment column: {TREATMENT_COLUMN}")

    print(f"Outcome column: {OUTCOME_COLUMN}")

    print(f"Treated customers: {int(treatment.sum())}")

    print(
        "Untreated customers: "
        f"{int((treatment == 0).sum())}"
    )

    print(
        f"Overall conversion rate: {outcome.mean():.4f}"
    )

    print(
        f"Treated conversion rate: "
        f"{outcome[treatment == 1].mean():.4f}"
    )

    print(
        f"Untreated conversion rate: "
        f"{outcome[treatment == 0].mean():.4f}"
    )

    print("\nOUTPUT")
    print("------------------------------")

    print(f"Saved prepared dataset:\n{output_path}")

    print("\nExcluded post-treatment columns:")
    print("- discount_amount")
    print("- order_value")

    print("\nCausal preprocessing finished successfully!")


if __name__ == "__main__":
    main()
