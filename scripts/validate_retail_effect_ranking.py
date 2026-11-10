import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr

from backend.app.preprocessing.retail_adapter import load_retail
from backend.app.preprocessing.dataset import make_splits


# 1. Paths and settings
DATA_PATH = Path("data/processed/retail_synthetic.csv")
MODEL_PATH = Path("reports/run_retail_continuous/model.joblib")
OUTPUT_PATH = Path("reports/retail_effect_ranking_validation.csv")

SEED = 42
DOSE = 10


# 2. Load dataset
print("\nLoading dataset...")

ds = load_retail(
    path=str(DATA_PATH),
    outcome="conversion",
    treatment_kind="continuous",
    eligible_only=True,
    validate=True,
)


# 3. Recreate test split
splits = make_splits(ds.T, ds.Y, seed=SEED)
test_idx = splits["test"]

X_test = ds.X.iloc[test_idx]


# 4. Load synthetic ground truth
raw_df = pd.read_csv(DATA_PATH)

eligible_df = raw_df.loc[
    raw_df["eligible"].astype(bool)
].reset_index(drop=True)

true_slope = eligible_df[
    "true_effect_per_dollar"
].to_numpy()[test_idx]

true_effect = true_slope * DOSE


# 5. Load GBM model
print("\nLoading GBM model...")

model = joblib.load(MODEL_PATH)


# 6. Predict customer-level effects
predicted_effect, warnings = model.effect_at_dose(
    X_test,
    DOSE,
)

predicted_effect = np.asarray(
    predicted_effect
).reshape(-1)


# 7. Calculate correlations
pearson = pearsonr(
    predicted_effect,
    true_effect,
).statistic

spearman = spearmanr(
    predicted_effect,
    true_effect,
).statistic


# 8. Calculate top 10% agreement
n_top = max(
    1,
    int(np.ceil(len(predicted_effect) * 0.10))
)

predicted_top = set(
    np.argsort(predicted_effect)[-n_top:]
)

true_top = set(
    np.argsort(true_effect)[-n_top:]
)

top_overlap = len(
    predicted_top.intersection(true_top)
) / n_top


# 9. Create customer-level comparison
comparison = pd.DataFrame({
    "predicted_effect": predicted_effect,
    "true_effect": true_effect,
})

comparison["predicted_decile"] = (
    pd.qcut(
        comparison["predicted_effect"],
        q=10,
        labels=False,
        duplicates="drop",
    ) + 1
)


# 10. Decile analysis
decile_results = (
    comparison.groupby("predicted_decile")
    .agg(
        customers=("predicted_effect", "size"),
        mean_predicted_effect=("predicted_effect", "mean"),
        mean_true_effect=("true_effect", "mean"),
    )
    .reset_index()
)

decile_results["absolute_difference"] = np.abs(
    decile_results["mean_predicted_effect"]
    - decile_results["mean_true_effect"]
)


# 11. Print results
print("\n========== CUSTOMER RANKING VALIDATION ==========")

print(f"Discount amount: Rs. {DOSE}")
print(f"Test customers: {len(X_test)}")

print(f"\nPearson correlation: {pearson:.4f}")
print(f"Spearman correlation: {spearman:.4f}")
print(f"Top 10% overlap: {top_overlap:.4f}")

print("\n========== EFFECT BY PREDICTED DECILE ==========")
print(decile_results.to_string(index=False))


# 12. Save results
OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

decile_results.to_csv(
    OUTPUT_PATH,
    index=False,
)

print(f"\nResults saved to: {OUTPUT_PATH}")

if warnings:
    print("\nModel warnings:", warnings)