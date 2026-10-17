"""Create a reproducible Criteo development sample in data/processed (raw file stays untouched).

  python scripts/prepare_criteo.py --path data/raw/criteo-uplift-v2.1.csv.gz --rows 200000 --outcome visit
First run with --header-only to confirm the real column names.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

import argparse
import json

from backend.app.core.config import settings
from backend.app.core.logging_config import setup_logging
from backend.app.preprocessing.criteo_adapter import (DATASET_INFO, EXPECTED_COLUMNS, inspect_header,
                                                      load_criteo)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", required=True)
    ap.add_argument("--rows", type=int, default=settings.criteo_sample_rows)
    ap.add_argument("--seed", type=int, default=settings.seed)
    ap.add_argument("--outcome", choices=["visit", "conversion"], default="visit")
    ap.add_argument("--header-only", action="store_true")
    a = ap.parse_args()
    setup_logging(settings.log_level)
    header = inspect_header(a.path)
    print("Header:", header)
    print("Expected:", EXPECTED_COLUMNS)
    print("Missing vs expected:", [c for c in EXPECTED_COLUMNS if c not in header])
    print("Extra vs expected:", [c for c in header if c not in EXPECTED_COLUMNS])
    print("LICENSE:", DATASET_INFO["license_note"])
    if a.header_only:
        return
    ds = load_criteo(a.path, outcome=a.outcome, n_rows=a.rows, seed=a.seed)
    out = settings.processed_dir / f"criteo_sample_{a.rows}_seed{a.seed}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.df.to_csv(out, index=False)
    out.with_suffix(".meta.json").write_text(json.dumps(ds.metadata, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {len(ds.df)} rows -> {out}")
    print(json.dumps(ds.metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
