
import json
from datetime import date
from pathlib import Path

import pandas as pd


# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]

REPORTS = ROOT / "reports"

GBM_MODEL = REPORTS / "run_retail_continuous" / "model.json"
LINEAR_MODEL = REPORTS / "run_retail_linear" / "model.json"

BUDGET_SUMMARY = (
    REPORTS / "run_retail_budget_validation" / "summary.json"
)

DOSE_VALIDATION = (
    REPORTS / "retail_dose_effect_validation.csv"
)

RANKING_VALIDATION = (
    REPORTS / "retail_effect_ranking_validation.csv"
)

STRATEGY_COMPARISON = (
    REPORTS / "retail_strategy_comparison" / "strategy_comparison.csv"
)

OUTPUT_PATH = (
    REPORTS / "EconoCausal_Model_Evaluation_Report.md"
)


# ==========================================================
# 2. HELPER FUNCTIONS
# ==========================================================

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required JSON file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required CSV file not found: {path}"
        )

    return pd.read_csv(path)


def fmt(value, digits=4):
    if pd.isna(value):
        return "N/A"

    return f"{float(value):.{digits}f}"


def money(value):
    if pd.isna(value):
        return "N/A"

    return f"₹{float(value):,.2f}"


def markdown_table(df):
    if df.empty:
        return "_No data available._"

    columns = list(df.columns)

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(
        ["---"] * len(columns)
    ) + " |"

    rows = []

    for _, row in df.iterrows():
        values = [
            str(value).replace("|", "\\|")
            for value in row.tolist()
        ]

        rows.append(
            "| " + " | ".join(values) + " |"
        )

    return "\n".join([header, separator] + rows)


# ==========================================================
# 3. LOAD MODEL METADATA
# ==========================================================

print("Loading model metadata...")

gbm = load_json(GBM_MODEL)
linear = load_json(LINEAR_MODEL)

gbm_metrics = gbm["metrics"]
linear_metrics = linear["metrics"]

features = gbm["feature_cols"]

train_rows = gbm["train_rows"]
validation_rows = gbm["validation_rows"]
test_rows = gbm["test_rows"]


# ==========================================================
# 4. LOAD BUDGET RESULTS
# ==========================================================

print("Loading budget optimization results...")

budget = load_json(BUDGET_SUMMARY)

customers_considered = budget["customers_considered"]
customers_offered = budget["customers_offered_discount"]

estimated_spend = budget["expected_discount_spend"]
estimated_contribution = budget[
    "estimated_incremental_contribution"
]


# ==========================================================
# 5. LOAD DOSE VALIDATION
# ==========================================================

print("Loading dose-effect validation...")

dose_df = load_csv(DOSE_VALIDATION)

dose_columns = [
    "discount_amount",
    "test_customers",
    "mean_predicted_effect",
    "mean_true_effect",
    "mae",
]

dose_columns = [
    col for col in dose_columns
    if col in dose_df.columns
]

dose_table = dose_df[dose_columns].copy()

for col in [
    "mean_predicted_effect",
    "mean_true_effect",
    "mae",
]:
    if col in dose_table.columns:
        dose_table[col] = dose_table[col].map(
            lambda x: fmt(x, 6)
        )


# ==========================================================
# 6. LOAD CUSTOMER RANKING VALIDATION
# ==========================================================

print("Loading customer ranking validation...")

ranking_df = load_csv(RANKING_VALIDATION)

ranking_columns = [
    "predicted_decile",
    "customers",
    "mean_predicted_effect",
    "mean_true_effect",
]

ranking_columns = [
    col for col in ranking_columns
    if col in ranking_df.columns
]

ranking_table = ranking_df[ranking_columns].copy()

for col in [
    "mean_predicted_effect",
    "mean_true_effect",
]:
    if col in ranking_table.columns:
        ranking_table[col] = ranking_table[col].map(
            lambda x: fmt(x, 6)
        )


# These metrics were calculated by the separate
# customer-ranking validation script.
# Update these values if that validation is rerun.

pearson_correlation = 0.7888
spearman_correlation = 0.7761
top_10_overlap = 0.5528


# ==========================================================
# 7. LOAD STRATEGY COMPARISON
# ==========================================================

print("Loading strategy comparison...")

strategy_df = load_csv(STRATEGY_COMPARISON)

required_strategy_columns = [
    "strategy",
    "customers_discounted",
    "model_estimated_spend",
    "synthetic_true_expected_spend",
    "synthetic_true_budget_compliant",
    "synthetic_true_incremental_contribution",
    "contribution_vs_oracle_pct",
]

missing = [
    col for col in required_strategy_columns
    if col not in strategy_df.columns
]

if missing:
    raise ValueError(
        f"Strategy comparison is missing columns: {missing}"
    )

strategy_table = strategy_df[
    required_strategy_columns
].copy()

strategy_table["model_estimated_spend"] = (
    strategy_table["model_estimated_spend"]
    .map(money)
)

strategy_table["synthetic_true_expected_spend"] = (
    strategy_table["synthetic_true_expected_spend"]
    .map(money)
)

strategy_table["synthetic_true_incremental_contribution"] = (
    strategy_table["synthetic_true_incremental_contribution"]
    .map(money)
)

strategy_table["contribution_vs_oracle_pct"] = (
    strategy_table["contribution_vs_oracle_pct"]
    .map(lambda x: f"{float(x):.2f}%")
)

strategy_table["synthetic_true_budget_compliant"] = (
    strategy_table["synthetic_true_budget_compliant"]
    .map(lambda x: "Yes" if x else "No")
)


# ==========================================================
# 8. PREPARE MODEL COMPARISON
# ==========================================================

model_comparison = pd.DataFrame([
    {
        "Model": "GBM nuisance",
        "MAE": fmt(
            gbm_metrics["mae_vs_synthetic_truth"],
            6,
        ),
        "RMSE": fmt(
            gbm_metrics["rmse_vs_synthetic_truth"],
            6,
        ),
    },
    {
        "Model": "Linear nuisance",
        "MAE": fmt(
            linear_metrics["mae_vs_synthetic_truth"],
            6,
        ),
        "RMSE": fmt(
            linear_metrics["rmse_vs_synthetic_truth"],
            6,
        ),
    },
])


# ==========================================================
# 9. BUILD THE REPORT
# ==========================================================

print("Generating Markdown report...")

report = f"""
# EconoCausal — Model Evaluation Report

**Project:** Dynamic Pricing via Double Machine Learning

**Report date:** {date.today().isoformat()}

**Project type:** Synthetic retail discount optimization

---

## 1. Executive Summary

EconoCausal is a causal machine learning project designed to
estimate how different discount amounts affect customer
conversion probabilities.

Traditional predictive models estimate whether a customer
may purchase a product. However, prediction alone does not
identify whether offering a discount caused an increase in
purchases.

EconoCausal uses Double Machine Learning (DML) to estimate
customer-level heterogeneous treatment effects and combines
these estimates with a budget-constrained optimization
algorithm.

The objective is to recommend discount allocations that
maximize estimated incremental contribution while respecting
a predefined discount budget.

The project uses synthetic retail data. Therefore, the
results demonstrate the technical workflow and do not prove
real-world business effectiveness.

### Main results

- Customers evaluated: **{customers_considered:,}**
- Primary model: **LinearDML with GBM nuisance models**
- Test customers: **{test_rows:,}**
- Customers receiving discounts: **{customers_offered:,}**
- Model-estimated discount spending: **{money(estimated_spend)}**
- Model-estimated incremental contribution: **{money(estimated_contribution)}**
- DML contribution relative to synthetic oracle: **87.49%**

---

## 2. Business Problem

Retail businesses frequently provide discounts to customers
to encourage purchases.

However, offering the same discount to every customer may
waste marketing budgets.

Some customers may purchase without a discount, while
others may respond positively to a discount.

The main business problem is:

**How can a business identify customers who are more likely
to respond to discounts and allocate its limited discount
budget effectively?**

EconoCausal addresses this problem through causal effect
estimation and budget optimization.

---

## 3. Proposed Solution

The project consists of the following stages:

1. Retail dataset preparation and validation.
2. Exploratory data analysis.
3. Causal relationship analysis.
4. Double Machine Learning model training.
5. Individual treatment-effect estimation.
6. Customer ranking and validation.
7. Budget-constrained discount optimization.
8. Backend API and frontend visualization.
9. Model and allocation evaluation.

The system estimates the incremental effect of different
discount amounts and uses those estimates to recommend
discount allocations.

---

## 4. Dataset Description

**Dataset:** Synthetic retail dataset

**Number of original generated records:** 20,000

**Customers considered in budget optimization:** {customers_considered:,}

**Treatment variable:** `discount_amount`

**Outcome variable:** `conversion`

**Available discount levels:** 0, 5, 10 and 20

### Customer features

{markdown_table(pd.DataFrame({"Feature": features}))}

### Dataset notes

- The dataset is synthetic.
- The discount amount is the treatment variable.
- Conversion is the outcome variable.
- Synthetic ground-truth treatment effects are used only
  for evaluation.
- Ground-truth effect columns are not model input features.

---

## 5. Double Machine Learning Model

The primary causal estimator is **LinearDML from EconML**.

The model estimates heterogeneous treatment effects using
machine learning nuisance models and cross-fitting.

### Primary model configuration

| Parameter | Value |
|---|---|
| Estimator | LinearDML |
| Nuisance models | GBM |
| Cross-validation folds | {gbm["cv"]} |
| Random seed | {gbm["seed"]} |
| Treatment | discount_amount |
| Outcome | conversion |
| Evaluation discount | {gbm_metrics["evaluation_dose"]} |
| Synthetic data | Yes |

### Data split

| Dataset portion | Records |
|---|---:|
| Training | {train_rows:,} |
| Validation | {validation_rows:,} |
| Testing | {test_rows:,} |
| Total | {train_rows + validation_rows + test_rows:,} |

### Treatment support

| Discount amount | Training customers |
|---|---:|
""" + "\n".join(
    f"| {dose:g} | {count:,} |"
    for dose, count in zip(
        gbm["support"]["levels"],
        gbm["support"]["counts"],
    )
) + f"""

The model assumes that the treatment effect is linear in
the discount amount within the observed treatment support.

This assumption should be considered when interpreting
predictions for different discount levels.

---

## 6. Model Performance Evaluation

The primary GBM nuisance model is compared against a
linear nuisance model.

Both models use the same dataset split and evaluation dose.

### Performance metrics

{markdown_table(model_comparison)}

### Primary GBM model results

| Metric | Value |
|---|---:|
| Mean predicted effect | {fmt(gbm_metrics["mean_predicted_effect"], 6)} |
| Mean synthetic true effect | {fmt(gbm_metrics["mean_synthetic_true_effect"], 6)} |
| MAE | {fmt(gbm_metrics["mae_vs_synthetic_truth"], 6)} |
| RMSE | {fmt(gbm_metrics["rmse_vs_synthetic_truth"], 6)} |

The GBM nuisance model performs better than the linear
nuisance model on both MAE and RMSE in this synthetic
evaluation.

The predicted mean effect is higher than the synthetic
ground-truth mean effect, indicating positive estimation
bias.

The model is retained as the primary model because its
observed test-set errors are lower.

---

## 7. Dose-Effect Validation

The model was evaluated at three discount amounts:
5, 10 and 20.

### Results

{markdown_table(dose_table)}

### Interpretation

The predicted and synthetic true effects increase with
the discount amount.

However, the predicted effects are consistently higher
than the synthetic ground-truth effects.

This indicates that the model captures the direction and
dose relationship in this synthetic dataset but has
calibration error.

The validation does not establish that the same relationship
will occur in real retail customers.

---

## 8. Customer Ranking Validation

Customers were ranked according to their predicted
treatment effects at a discount of 10.

The ranked customers were divided into ten groups.

### Ranking metrics

| Metric | Result |
|---|---:|
| Pearson correlation | {pearson_correlation:.4f} |
| Spearman correlation | {spearman_correlation:.4f} |
| Top 10% overlap | {top_10_overlap:.4f} |

### Customer effect by predicted decile

{markdown_table(ranking_table)}

### Interpretation

The mean predicted effect and mean synthetic true effect
increase across the customer deciles.

This indicates that the model has a useful customer-ranking
signal in the synthetic test data.

However, the top 10% overlap is approximately
{top_10_overlap:.2%}, so predicted and synthetic true
top-ranked customer groups are not identical.

---

## 9. Budget-Constrained Discount Optimization

The project uses SciPy Mixed-Integer Linear Programming
(MILP) to allocate discounts under a fixed budget.

### Business assumptions

| Parameter | Value |
|---|---:|
| Budget | {money(budget["budget"])} |
| Contribution margin rate | {budget["margin_rate_assumption"]:.0%} |
| Available discounts | 0, 5, 10, 20 |
| Optimizer | {budget["optimizer"]} |
| Solver status | {budget["solver_status"]} |

The optimization objective is to maximize estimated
incremental contribution while limiting expected
discount spending.

The calculation uses the assumed contribution margin:

`avg_basket × margin_rate`

Expected discount spending is based on the predicted
conversion probability and recommended discount.

### Optimizer results

| Metric | Result |
|---|---:|
| Customers considered | {customers_considered:,} |
| Customers offered discounts | {customers_offered:,} |
| Customers without discounts | {budget["customers_without_discount"]:,} |
| Model-estimated expected spend | {money(estimated_spend)} |
| Budget remaining | {money(budget["budget_remaining"])} |
| Budget utilization | {budget["budget_utilization"]:.2%} |
| Model-estimated incremental contribution | {money(estimated_contribution)} |

The optimization solver reported successful termination.

**Important:** The model-estimated contribution is an
optimization estimate, not an observed business profit.

---

## 10. Strategy Comparison

Three allocation strategies were evaluated:

1. DML: uses estimated heterogeneous treatment effects.
2. Random: randomly selects customers and discount amounts.
3. Oracle: uses synthetic ground-truth treatment effects.

The Oracle strategy is an ideal synthetic benchmark and
cannot be used as a real-world deployment strategy.

### Results

{markdown_table(strategy_table)}

### Interpretation

The DML strategy generated a synthetic-ground-truth
incremental contribution of approximately ₹205.78.

The Random strategy generated a negative synthetic-ground-
truth incremental contribution of approximately ₹575.35.

The Oracle benchmark generated approximately ₹235.21.

Therefore, the DML strategy achieved approximately 87.49%
of the Oracle benchmark's synthetic incremental contribution.

All three strategies satisfied the model-estimated
₹1,000 budget constraint. Their synthetic-ground-truth
spending was also within the budget in this particular run.

This result supports the usefulness of causal-effect-based
targeting compared with the tested random allocation.

---

## 11. Limitations

The following limitations must be considered:

1. The main retail dataset is synthetic rather than real
   customer transaction data.

2. Model performance against synthetic ground truth does
   not guarantee real-world causal effectiveness.

3. The DML model assumes a linear treatment effect in
   discount amount within the observed support.

4. The estimated treatment effects show positive bias
   compared with synthetic ground truth.

5. The baseline conversion model used by the optimization
   workflow is a demonstration model and needs further
   independent validation.

6. The business objective relies on an assumed contribution
   margin rate of 30%.

7. Expected discount spending is not the same as actual
   realized spending.

8. The random comparison is one seeded random allocation,
   not an average over repeated random trials.

9. Unobserved confounding, changing customer behavior and
   real-world implementation constraints require further
   investigation.

---

## 12. Future Improvements

Possible future improvements include:

- Evaluate the model using real, appropriately licensed
  retail discount and transaction data.
- Conduct randomized discount experiments.
- Improve treatment-effect calibration.
- Investigate generalized propensity and continuous-dose
  overlap diagnostics.
- Test multiple random seeds and repeated random baselines.
- Evaluate alternative causal estimators.
- Validate contribution margin assumptions with actual
  business data.
- Monitor model performance and allocation outcomes.

---

## 13. Conclusion

EconoCausal demonstrates an end-to-end causal machine
learning workflow for retail discount optimization.

It combines data preparation, exploratory analysis,
Double Machine Learning, heterogeneous treatment-effect
estimation and budget-constrained optimization.

The synthetic evaluation shows that the GBM nuisance
configuration performs better than the tested linear
configuration.

The DML allocation also outperforms the tested random
allocation and achieves approximately 87.49% of the
synthetic Oracle benchmark's incremental contribution.

The project demonstrates how causal machine learning
can support customer-level discount allocation decisions.

However, validation with real customer data and controlled
experiments is necessary before making real business
effectiveness claims.

---

## 14. Reproducibility and Result Files

Important project artifacts:

| Artifact | Location |
|---|---|
| Primary DML model metadata | `reports/run_retail_continuous/model.json` |
| Primary DML model artifact | `reports/run_retail_continuous/model.joblib` |
| Linear model metadata | `reports/run_retail_linear/model.json` |
| Budget summary | `reports/run_retail_budget_validation/summary.json` |
| Budget recommendations | `reports/run_retail_budget_validation/recommendations.csv` |
| Dose validation | `reports/retail_dose_effect_validation.csv` |
| Ranking validation | `reports/retail_effect_ranking_validation.csv` |
| Strategy comparison | `reports/retail_strategy_comparison/strategy_comparison.csv` |
| Customer allocations | `reports/retail_strategy_comparison/customer_strategy_allocations.csv` |

**Final note:** This report documents synthetic educational
evaluation results. It should not be interpreted as evidence
of validated real-world financial returns.
"""


# ==========================================================
# 10. SAVE REPORT
# ==========================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as file:
    file.write(report.strip() + "\n")


# ==========================================================
# 11. COMPLETION MESSAGE
# ==========================================================

print("\n========================================")
print("ECONOCAUSAL EVALUATION REPORT GENERATED")
print("========================================")

print("\nReport location:")
print(OUTPUT_PATH)

print("\nReport size:")
print(f"{OUTPUT_PATH.stat().st_size:,} bytes")

print("\nReport generation completed successfully.")
