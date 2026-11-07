# EconoCausal — Model Evaluation Report

**Project:** Dynamic Pricing via Double Machine Learning

**Report date:** 2026-11-05

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

- Customers evaluated: **17,990**
- Primary model: **LinearDML with GBM nuisance models**
- Test customers: **3,598**
- Customers receiving discounts: **1,232**
- Model-estimated discount spending: **₹999.99**
- Model-estimated incremental contribution: **₹420.18**
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

**Customers considered in budget optimization:** 17,990

**Treatment variable:** `discount_amount`

**Outcome variable:** `conversion`

**Available discount levels:** 0, 5, 10 and 20

### Customer features

| Feature |
| --- |
| age |
| income_k |
| tenure_months |
| past_purchases |
| avg_basket |
| region |
| is_member |

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
| Cross-validation folds | 3 |
| Random seed | 42 |
| Treatment | discount_amount |
| Outcome | conversion |
| Evaluation discount | 10.0 |
| Synthetic data | Yes |

### Data split

| Dataset portion | Records |
|---|---:|
| Training | 10,794 |
| Validation | 3,598 |
| Testing | 3,598 |
| Total | 17,990 |

### Treatment support

| Discount amount | Training customers |
|---|---:|
| 0 | 5,866 |
| 5 | 1,705 |
| 10 | 1,568 |
| 20 | 1,655 |

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

| Model | MAE | RMSE |
| --- | --- | --- |
| GBM nuisance | 0.017063 | 0.021066 |
| Linear nuisance | 0.018676 | 0.023645 |

### Primary GBM model results

| Metric | Value |
|---|---:|
| Mean predicted effect | 0.078728 |
| Mean synthetic true effect | 0.062204 |
| MAE | 0.017063 |
| RMSE | 0.021066 |

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

| discount_amount | test_customers | mean_predicted_effect | mean_true_effect | mae |
| --- | --- | --- | --- | --- |
| 5 | 3598 | 0.039364 | 0.031102 | 0.008532 |
| 10 | 3598 | 0.078728 | 0.062204 | 0.017063 |
| 20 | 3598 | 0.157456 | 0.124407 | 0.034126 |

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
| Pearson correlation | 0.7888 |
| Spearman correlation | 0.7761 |
| Top 10% overlap | 0.5528 |

### Customer effect by predicted decile

| predicted_decile | customers | mean_predicted_effect | mean_true_effect |
| --- | --- | --- | --- |
| 1 | 360 | 0.046861 | 0.029061 |
| 2 | 360 | 0.061759 | 0.044706 |
| 3 | 360 | 0.068970 | 0.051167 |
| 4 | 359 | 0.073938 | 0.057773 |
| 5 | 360 | 0.077829 | 0.061880 |
| 6 | 360 | 0.081453 | 0.066673 |
| 7 | 359 | 0.085456 | 0.070159 |
| 8 | 360 | 0.090047 | 0.073003 |
| 9 | 360 | 0.095650 | 0.079479 |
| 10 | 360 | 0.105323 | 0.088145 |

### Interpretation

The mean predicted effect and mean synthetic true effect
increase across the customer deciles.

This indicates that the model has a useful customer-ranking
signal in the synthetic test data.

However, the top 10% overlap is approximately
55.28%, so predicted and synthetic true
top-ranked customer groups are not identical.

---

## 9. Budget-Constrained Discount Optimization

The project uses SciPy Mixed-Integer Linear Programming
(MILP) to allocate discounts under a fixed budget.

### Business assumptions

| Parameter | Value |
|---|---:|
| Budget | ₹1,000.00 |
| Contribution margin rate | 30% |
| Available discounts | 0, 5, 10, 20 |
| Optimizer | SciPy MILP |
| Solver status | Optimization terminated successfully. (HiGHS Status 7: Optimal) |

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
| Customers considered | 17,990 |
| Customers offered discounts | 1,232 |
| Customers without discounts | 16,758 |
| Model-estimated expected spend | ₹999.99 |
| Budget remaining | ₹0.01 |
| Budget utilization | 100.00% |
| Model-estimated incremental contribution | ₹420.18 |

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

| strategy | customers_discounted | model_estimated_spend | synthetic_true_expected_spend | synthetic_true_budget_compliant | synthetic_true_incremental_contribution | contribution_vs_oracle_pct |
| --- | --- | --- | --- | --- | --- | --- |
| DML | 1232 | ₹999.99 | ₹934.95 | Yes | ₹205.78 | 87.49% |
| Random | 348 | ₹999.95 | ₹906.02 | Yes | ₹-575.35 | -244.61% |
| Oracle | 1336 | ₹999.57 | ₹999.57 | Yes | ₹235.21 | 100.00% |

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
