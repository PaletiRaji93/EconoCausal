
from pathlib import Path

import pandas as pd


DATA_PATH = (
    Path(__file__).resolve().parent
    / "sample"
    / "retail_campaign.csv"
)

REQUIRED_COLUMNS = {
    "customer_id",
    "age",
    "past_purchases",
    "previous_spend",
    "loyalty_score",
    "assigned",
    "discount_amount",
    "converted",
    "order_value",
}


def validate_data(df: pd.DataFrame) -> None:
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer IDs found.")

    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Missing values found.")

    if not df["assigned"].isin([0, 1]).all():
        raise ValueError("assigned must contain only 0 or 1.")

    if not df["converted"].isin([0, 1]).all():
        raise ValueError("converted must contain only 0 or 1.")

    if (df["discount_amount"] < 0).any():
        raise ValueError("Discount cannot be negative.")

    if (df["order_value"] < 0).any():
        raise ValueError("Order value cannot be negative.")

    untreated = df["assigned"] == 0
    if not (df.loc[untreated, "discount_amount"] == 0).all():
        raise ValueError(
            "Unassigned customers must have zero discount."
        )

    print("Dataset validation passed.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Duplicate IDs: {df['customer_id'].duplicated().sum()}")


if __name__ == "__main__":
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Generate the dataset first by running "
            "'python data/generate_synthetic_data.py'."
        )

    dataset = pd.read_csv(DATA_PATH)
    validate_data(dataset)
