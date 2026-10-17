"""Train binary-treatment DML + diagnostics + uplift evaluation + refutation.

  python scripts/train_dml.py --dataset retail --path data/processed/retail_synthetic.csv --nuisance gbm
  python scripts/train_dml.py --dataset criteo --path data/processed/criteo_sample_200000_seed42.csv --outcome visit
Outputs go to reports/run_<dataset>/ (figures, results.json, model.joblib).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

import argparse
import json

from backend.app.causal_engine.pipeline import PipelineConfig, run_binary_pipeline
from backend.app.core.config import settings
from backend.app.core.errors import EconoCausalError
from backend.app.core.logging_config import setup_logging


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["retail", "criteo"], required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--outcome", default=None)
    ap.add_argument("--estimator", choices=["linear", "forest"], default="linear")
    ap.add_argument("--nuisance", choices=["linear", "gbm"], default="gbm")
    ap.add_argument("--seed", type=int, default=settings.seed)
    ap.add_argument("--rows", type=int, default=None, help="Criteo rows (only if --path is the raw file)")
    ap.add_argument("--no-refute", action="store_true")
    ap.add_argument("--dowhy", action="store_true", help="also run DoWhy refuters (slow)")
    a = ap.parse_args()
    setup_logging(settings.log_level)
    try:
        if a.dataset == "retail":
            from backend.app.preprocessing.retail_adapter import load_retail
            ds = load_retail(a.path, outcome=a.outcome or "conversion")
            randomized = False
        else:
            from backend.app.preprocessing.criteo_adapter import load_criteo
            ds = load_criteo(a.path, outcome=a.outcome or "visit", n_rows=a.rows or 5_000_000, seed=a.seed)
            randomized = True
        cfg = PipelineConfig(estimator=a.estimator, nuisance=a.nuisance, seed=a.seed,
                             randomized=randomized, refute=not a.no_refute, run_dowhy=a.dowhy)
        res = run_binary_pipeline(ds, cfg)
    except EconoCausalError as exc:
        print(f"ERROR: {exc}")
        return 1
    print("Model version:", res["model_version_id"])
    print("Estimated ATE (test):", json.dumps(res["dml"]["ate_test"]))
    for name, m in res["uplift"]["metrics"].items():
        print(f"  {name:48s} qini={m['qini_coefficient']:.4f}  uplift@10%={m['uplift_at_k']['10%']:.4f}")
    for w in res["warnings"]:
        print("WARNING:", w)
    print("Reports:", settings.reports_dir / f"run_{ds.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
