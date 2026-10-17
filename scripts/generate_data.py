"""Generate deterministic SYNTHETIC retail campaign data.

Run from repo root:
  python scripts/generate_data.py --n 20000 --seed 42 --levels 5 10 20
NOTE: if you already have your own generate_data.py, keep it and pass a ColumnMapping JSON to the
adapter instead (see PART2_README).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

import argparse

from backend.app.core.config import settings
from backend.app.core.logging_config import setup_logging
from backend.app.preprocessing.retail_adapter import generate_retail_data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=settings.seed)
    ap.add_argument("--levels", type=float, nargs="+", default=[10.0], help="discount amounts for treated rows")
    ap.add_argument("--randomized", action="store_true", help="assign treatment at random (no confounding)")
    ap.add_argument("--out", default=str(settings.processed_dir / "retail_synthetic.csv"))
    a = ap.parse_args()
    setup_logging(settings.log_level)
    df = generate_retail_data(a.n, a.seed, a.levels, confounded=not a.randomized)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} SYNTHETIC rows to {out}")
    print(df.groupby("treatment")["conversion"].agg(["mean", "size"]))


if __name__ == "__main__":
    main()
