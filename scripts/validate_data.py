"""Validate a dataset file and print a data-quality report.

  python scripts/validate_data.py --dataset retail --path data/processed/retail_synthetic.csv
  python scripts/validate_data.py --dataset criteo --path data/raw/criteo-uplift-v2.1.csv.gz --rows 100000
Exit code 1 if validation fails.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

import argparse
import json

from backend.app.core.config import settings
from backend.app.core.errors import EconoCausalError
from backend.app.core.logging_config import setup_logging


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["retail", "criteo"], required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--outcome", default=None)
    ap.add_argument("--rows", type=int, default=None, help="Criteo sample size")
    ap.add_argument("--mapping", default=None, help="optional ColumnMapping JSON")
    a = ap.parse_args()
    setup_logging(settings.log_level)
    from backend.app.preprocessing.column_mapping import ColumnMapping
    mapping = ColumnMapping.from_json(a.mapping) if a.mapping else None
    try:
        if a.dataset == "retail":
            from backend.app.preprocessing.retail_adapter import load_retail
            ds = load_retail(a.path, outcome=a.outcome or "conversion", mapping=mapping)
        else:
            from backend.app.preprocessing.criteo_adapter import load_criteo
            ds = load_criteo(a.path, outcome=a.outcome or "visit", n_rows=a.rows, mapping=mapping)
    except EconoCausalError as exc:
        print(f"VALIDATION FAILED: {exc}")
        return 1
    print("VALIDATION OK")
    print(json.dumps({"features": ds.feature_cols, "excluded": ds.excluded_cols,
                      "metadata": ds.metadata}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
