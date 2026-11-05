
"""Propensity score estimation and diagnostics for EconoCausal.

Treatment: assigned
Features: pretreatment customer characteristics

This script:
1. Estimates propensity scores using Logistic Regression.
2. Checks treatment-group overlap.
3. Calculates covariate balance before weighting.
4. Calculates covariate balance after ATE inverse-propensity weighting.
5. Saves diagnostic plots and CSV reports.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# 1. PROJECT CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "causal_dataset.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "propensity"

FEATURE_COLUMNS = [
    "age",
    "past_purchases",
    "previous_spend",
    "loyalty_score",
]

TREATMENT_COLUMN = "assigned"

# Avoid unstable weights caused by scores extremely close to 0 or 1.
WEIGHT_SCORE_CLIP = 0.01


# ---------------------------------------------------------
# 2. LOAD PREPARED DATA
# ---------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load the prepared causal dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {DATA_PATH}. "
            "Run prepare_data.py first."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURE_COLUMNS + [TREATMENT_COLUMN]

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in prepared dataset: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Prepared dataset is empty.")

    if df[required_columns].isnull().any().any():
        raise ValueError("Missing values found in modelling columns.")

    if not df[TREATMENT_COLUMN].isin([0, 1]).all():
        raise ValueError("Treatment must contain only 0 and 1.")

    if df[TREATMENT_COLUMN].nunique() != 2:
        raise ValueError("Both treatment groups are required.")

    return df


# ---------------------------------------------------------
# 3. ESTIMATE PROPENSITY SCORES
# ---------------------------------------------------------

def estimate_propensity_scores(
    df: pd.DataFrame,
) -> tuple[Pipeline, np.ndarray]:
    """Estimate P(T=1 | X) using Logistic Regression."""

    X = df[FEATURE_COLUMNS]
    treatment = df[TREATMENT_COLUMN]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X, treatment)

    propensity_scores = model.predict_proba(X)[:, 1]

    return model, propensity_scores


# ---------------------------------------------------------
# 4. CALCULATE STANDARDIZED MEAN DIFFERENCES
# ---------------------------------------------------------

def calculate_smd(
    df: pd.DataFrame,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Calculate standardized mean differences between groups."""

    treatment = df[TREATMENT_COLUMN].to_numpy()

    treated_mask = treatment == 1
    control_mask = treatment == 0

    results = []

    for column in FEATURE_COLUMNS:

        values = df[column].to_numpy()

        treated_values = values[treated_mask]
        control_values = values[control_mask]

        if weights is None:
            treated_weights = np.ones(len(treated_values))
            control_weights = np.ones(len(control_values))
        else:
            treated_weights = weights[treated_mask]
            control_weights = weights[control_mask]

        treated_mean = np.average(
            treated_values,
            weights=treated_weights,
        )

        control_mean = np.average(
            control_values,
            weights=control_weights,
        )

        # Use the unweighted pooled standard deviation as reference.
        treated_variance = np.var(
            treated_values,
            ddof=1,
        )

        control_variance = np.var(
            control_values,
            ddof=1,
        )

        pooled_sd = np.sqrt(
            (treated_variance + control_variance) / 2
        )

        if pooled_sd == 0:
            smd = 0.0 if treated_mean == control_mean else np.inf
        else:
            smd = (treated_mean - control_mean) / pooled_sd

        results.append(
            {
                "feature": column,
                "treated_mean": treated_mean,
                "control_mean": control_mean,
                "smd": smd,
                "absolute_smd": abs(smd),
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------
# 5. CALCULATE ATE INVERSE-PROPENSITY WEIGHTS
# ---------------------------------------------------------

def calculate_ate_weights(
    treatment: np.ndarray,
    propensity_scores: np.ndarray,
) -> np.ndarray:
    """Calculate inverse-propensity weights for ATE diagnostics."""

    clipped_scores = np.clip(
        propensity_scores,
        WEIGHT_SCORE_CLIP,
        1 - WEIGHT_SCORE_CLIP,
    )

    weights = (
        treatment / clipped_scores
        + (1 - treatment) / (1 - clipped_scores)
    )

    return weights


# ---------------------------------------------------------
# 6. SAVE PROPENSITY SCORE DISTRIBUTION
# ---------------------------------------------------------

def save_propensity_plot(
    treatment: np.ndarray,
    propensity_scores: np.ndarray,
) -> None:
    """Plot propensity scores for treated and control customers."""

    plt.figure(figsize=(9, 5))

    plt.hist(
        propensity_scores[treatment == 1],
        bins=30,
        alpha=0.6,
        density=True,
        label="Treated customers",
    )

    plt.hist(
        propensity_scores[treatment == 0],
        bins=30,
        alpha=0.6,
        density=True,
        label="Untreated customers",
    )

    plt.xlabel("Estimated propensity score")
    plt.ylabel("Density")
    plt.title("Propensity Score Distribution")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "propensity_distribution.png",
        dpi=300,
    )

    plt.close()


# ---------------------------------------------------------
# 7. SAVE COVARIATE BALANCE PLOT
# ---------------------------------------------------------

def save_balance_plot(
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> None:
    """Compare absolute standardized mean differences."""

    comparison = pd.DataFrame(
        {
            "feature": before["feature"],
            "Before weighting": before["absolute_smd"],
            "After weighting": after["absolute_smd"],
        }
    )

    x = np.arange(len(comparison))
    width = 0.35

    plt.figure(figsize=(9, 5))

    plt.bar(
        x - width / 2,
        comparison["Before weighting"],
        width,
        label="Before weighting",
    )

    plt.bar(
        x + width / 2,
        comparison["After weighting"],
        width,
        label="After weighting",
    )

    plt.axhline(
        0.1,
        linestyle="--",
        label="Common balance reference (0.1)",
    )

    plt.xticks(
        x,
        comparison["feature"],
        rotation=20,
    )

    plt.ylabel("Absolute standardized mean difference")
    plt.title("Covariate Balance Before and After Weighting")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "covariate_balance.png",
        dpi=300,
    )

    plt.close()


# ---------------------------------------------------------
# 8. MAIN EXECUTION
# ---------------------------------------------------------

def main() -> None:
    """Run propensity estimation and diagnostic reporting."""

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nStarting propensity score analysis...\n")

    df = load_data()

    treatment = df[TREATMENT_COLUMN].to_numpy()

    model, propensity_scores = estimate_propensity_scores(df)

    # Save propensity scores alongside customer IDs.
    score_report = pd.DataFrame(
        {
            "customer_id": df["customer_id"],
            "assigned": treatment,
            "propensity_score": propensity_scores,
        }
    )

    score_report.to_csv(
        REPORT_DIR / "propensity_scores.csv",
        index=False,
    )

    # Check score overlap.
    treated_scores = propensity_scores[treatment == 1]
    control_scores = propensity_scores[treatment == 0]

    treated_min = treated_scores.min()
    treated_max = treated_scores.max()

    control_min = control_scores.min()
    control_max = control_scores.max()

    overlap_lower = max(treated_min, control_min)
    overlap_upper = min(treated_max, control_max)

    overlap_exists = overlap_lower < overlap_upper

    # Calculate ATE inverse-propensity weights.
    weights = calculate_ate_weights(
        treatment,
        propensity_scores,
    )

    # Covariate balance before and after weighting.
    balance_before = calculate_smd(df)

    balance_after = calculate_smd(
        df,
        weights=weights,
    )

    balance_report = balance_before[
        ["feature", "absolute_smd"]
    ].rename(
        columns={"absolute_smd": "before_weighting"}
    ).merge(
        balance_after[
            ["feature", "absolute_smd"]
        ].rename(
            columns={"absolute_smd": "after_weighting"}
        ),
        on="feature",
    )

    balance_report.to_csv(
        REPORT_DIR / "covariate_balance.csv",
        index=False,
    )

    # Save diagnostic plots.
    save_propensity_plot(
        treatment,
        propensity_scores,
    )

    save_balance_plot(
        balance_before,
        balance_after,
    )

    # Save a concise summary.
    summary = {
        "total_customers": len(df),
        "treated_customers": int(treatment.sum()),
        "untreated_customers": int((treatment == 0).sum()),
        "propensity_min": float(propensity_scores.min()),
        "propensity_max": float(propensity_scores.max()),
        "treated_score_min": float(treated_min),
        "treated_score_max": float(treated_max),
        "control_score_min": float(control_min),
        "control_score_max": float(control_max),
        "overlap_exists": bool(overlap_exists),
        "overlap_lower": float(overlap_lower),
        "overlap_upper": float(overlap_upper),
        "weights_min": float(weights.min()),
        "weights_max": float(weights.max()),
    }

    pd.Series(summary).to_csv(
        REPORT_DIR / "propensity_summary.csv",
        header=["value"],
    )

    # Display results.
    print("PROPENSITY SCORE ANALYSIS")
    print("----------------------------------")

    print(f"Total customers: {len(df)}")

    print(
        f"Propensity score range: "
        f"{propensity_scores.min():.4f} - "
        f"{propensity_scores.max():.4f}"
    )

    print(
        f"Treated score range: "
        f"{treated_min:.4f} - {treated_max:.4f}"
    )

    print(
        f"Untreated score range: "
        f"{control_min:.4f} - {control_max:.4f}"
    )

    print(f"Common support overlap exists: {overlap_exists}")

    if overlap_exists:
        print(
            f"Common support interval: "
            f"{overlap_lower:.4f} - {overlap_upper:.4f}"
        )
    else:
        print("WARNING: No common support interval detected.")

    print("\nCOVARIATE BALANCE")
    print("----------------------------------")

    print(balance_report.to_string(index=False))

    print("\nOUTPUT FILES")
    print("----------------------------------")

    print(f"Reports saved in: {REPORT_DIR}")

    print("\nGenerated files:")
    print("- propensity_scores.csv")
    print("- propensity_summary.csv")
    print("- covariate_balance.csv")
    print("- propensity_distribution.png")
    print("- covariate_balance.png")

    print("\nPropensity score analysis completed.")


if __name__ == "__main__":
    main()
