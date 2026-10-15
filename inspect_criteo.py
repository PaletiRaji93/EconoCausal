
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/criteo-research-uplift-v2.1.csv")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH.resolve()}")

print("Dataset:", DATA_PATH.resolve())

total_rows = 0
missing_values = None
column_names = None
treatment_counts = {}
conversion_counts = {}
visit_counts = {}

for chunk in pd.read_csv(DATA_PATH, chunksize=100_000):
    total_rows += len(chunk)

    if column_names is None:
        column_names = chunk.columns.tolist()
        missing_values = pd.Series(0, index=column_names)

    missing_values += chunk.isna().sum()

    for column, counts in [
        ("treatment", treatment_counts),
        ("conversion", conversion_counts),
        ("visit", visit_counts),
    ]:
        if column in chunk.columns:
            values = chunk[column].value_counts(dropna=False)
            for key, value in values.items():
                counts[str(key)] = counts.get(str(key), 0) + int(value)

print("\nTotal rows:", total_rows)
print("Total columns:", len(column_names))
print("Columns:", column_names)
print("\nMissing values per column:")
print(missing_values[missing_values > 0])

print("\nTreatment counts:", treatment_counts)
print("Conversion counts:", conversion_counts)
print("Visit counts:", visit_counts)

print("\nInitial data inspection completed.")
