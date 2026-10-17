"""Run EDA and save figures/JSON under reports/<name>/eda.

  python scripts/run_eda.py --dataset retail --path data/processed/retail_synthetic.csv
  python scripts/run_eda.py --dataset criteo --path data/processed/criteo_sample_200000_seed42.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

import argparse

from backend.app.core.config import settings
from backend.app.core.logging_config import setup_logging
from backend.app.eda.eda_report import run_eda


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["retail", "criteo"], required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--outcome", default=None)
    a = ap.parse_args()
    setup_logging(settings.log_level)
    if a.dataset == "retail":
        from backend.app.preprocessing.retail_adapter import load_retail
        ds = load_retail(a.path, outcome=a.outcome or "conversion")
    else:
        from backend.app.preprocessing.criteo_adapter import load_criteo
        # a processed sample is smaller than n_rows, so it is read in full
        ds = load_criteo(a.path, outcome=a.outcome or "visit", n_rows=5_000_000)
    out = settings.reports_dir / f"eda_{ds.name}"
    summary = run_eda(ds, out)
    print("EDA saved to", out)
    for w in summary["warnings"]:
        print("WARNING:", w)


if __name__ == "__main__":
    main()
