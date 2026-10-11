
"""Exploratory data analysis for EconoCausal retail campaign data."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "sample" / "retail_campaign.csv"
REPORT_DIR = PROJECT_ROOT / "reports"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the existing synthetic campaign dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run python data/generate_synthetic_data.py first."
        )

    return pd.read_csv(path)


def generate_eda_report(df: pd.DataFrame) -> dict:
    """Calculate descriptive statistics and save EDA outputs."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    required_columns = {
        "customer_id",
        "assigned",
        "discount_amount",
        "converted",
        "order_value",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Cannot analyze an empty dataset.")

    # 1. Dataset overview
    overview = pd.DataFrame(
        {
            "metric": [
                "rows",
                "columns",
                "duplicate_rows",
                "missing_cells",
            ],
            "value": [
                len(df),
                len(df.columns),
                int(df.duplicated().sum()),
                int(df.isna().sum().sum()),
            ],
        }
    )
    overview.to_csv(REPORT_DIR / "dataset_overview.csv", index=False)

    # 2. Missing values by column
    missing_report = (
        df.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .reset_index(names="column")
    )
    missing_report["missing_percent"] = (
        missing_report["missing_count"] / len(df) * 100
    )
    missing_report.to_csv(
        REPORT_DIR / "missing_values.csv", index=False
    )

    # 3. Treatment-group comparison
    group_summary = (
        df.groupby("assigned", observed=True)
        .agg(
            customers=("customer_id", "count"),
            conversion_rate=("converted", "mean"),
            average_order_value=("order_value", "mean"),
            total_order_value=("order_value", "sum"),
            average_discount=("discount_amount", "mean"),
        )
        .reset_index()
    )
    group_summary.to_csv(
        REPORT_DIR / "treatment_group_summary.csv", index=False
    )

    # 4. Customer segment analysis
    if "loyalty_score" in df.columns:
        segment_data = df.copy()
        segment_data["customer_segment"] = pd.cut(
            segment_data["loyalty_score"],
            bins=[-0.001, 0.35, 0.70, 1.001],
            labels=["occasional", "regular", "loyal"],
        )

        segment_summary = (
            segment_data.groupby("customer_segment", observed=True)
            .agg(
                customers=("customer_id", "count"),
                conversion_rate=("converted", "mean"),
                average_order_value=("order_value", "mean"),
                average_discount=("discount_amount", "mean"),
            )
            .reset_index()
        )

        segment_summary.to_csv(
            REPORT_DIR / "customer_segment_summary.csv", index=False
        )

    # 5. Conversion rate by treatment group
    conversion_rates = (
        df.groupby("assigned")["converted"].mean()
        .reindex([0, 1], fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        ["No discount assigned", "Discount assigned"],
        conversion_rates.values,
    )
    ax.set_ylabel("Observed conversion rate")
    ax.set_title("Conversion Rate by Treatment Assignment")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(
        REPORT_DIR / "conversion_by_treatment.png", dpi=150
    )
    plt.close(fig)

    # 6. Discount distribution among treated customers
    treated = df.loc[df["assigned"] == 1, "discount_amount"]

    fig, ax = plt.subplots(figsize=(7, 5))
    if not treated.empty:
        treated.value_counts().sort_index().plot(
            kind="bar", ax=ax
        )
    ax.set_xlabel("Discount amount")
    ax.set_ylabel("Number of customers")
    ax.set_title("Discount Distribution Among Treated Customers")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "discount_distribution.png", dpi=150)
    plt.close(fig)

    # 7. Numerical feature correlation
    numeric_data = df.select_dtypes(include="number").drop(
        columns=["customer_id"], errors="ignore"
    )
    correlation = numeric_data.corr()
    correlation.to_csv(REPORT_DIR / "correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(
        correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto",
    )
    ax.set_xticks(range(len(correlation.columns)))
    ax.set_xticklabels(
        correlation.columns, rotation=45, ha="right"
    )
    ax.set_yticks(range(len(correlation.index)))
    ax.set_yticklabels(correlation.index)
    ax.set_title("Numerical Feature Correlations")
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "correlation_matrix.png", dpi=150)
    plt.close(fig)

    # 8. Potential data-quality warnings
    warnings = []

    if df.duplicated().any():
        warnings.append("Exact duplicate rows exist.")

    if df.isna().any().any():
        warnings.append("Missing values exist.")

    if df["assigned"].nunique() < 2:
        warnings.append(
            "Only one treatment group is present; "
            "treatment comparisons are not possible."
        )

    if (df["discount_amount"] < 0).any():
        warnings.append("Negative discount amounts exist.")

    report = {
        "rows": len(df),
        "columns": len(df.columns),
        "overall_conversion_rate": float(df["converted"].mean()),
        "treatment_summary": group_summary.to_dict(orient="records"),
        "warnings": warnings,
        "report_directory": str(REPORT_DIR),
    }

    return report


def main() -> None:
    """Execute EDA and print a concise summary."""
    df = load_data()
    report = generate_eda_report(df)

    print("\nEconoCausal EDA completed.")
    print(f"Rows: {report['rows']}")
    print(f"Columns: {report['columns']}")
    print(
        "Overall observed conversion rate: "
        f"{report['overall_conversion_rate']:.3f}"
    )
    print("\nTreatment group summary:")
    print(pd.DataFrame(report["treatment_summary"]).to_string(index=False))

    print("\nData-quality warnings:")
    if report["warnings"]:
        for warning in report["warnings"]:
            print(f"- {warning}")
    else:
        print("No basic data-quality warnings detected.")

    print(f"\nReports saved to: {report['report_directory']}")


if __name__ == "__main__":
    main()
