"""End-to-end Part 2 pipeline on small synthetic retail data (skipped if EconML is missing)."""
import json

import numpy as np
import pytest

pytest.importorskip("econml", reason="EconML not installed: pip install econml")

from backend.app.causal_engine.pipeline import PipelineConfig, run_binary_pipeline  # noqa: E402
from backend.app.dml.dml_estimator import DMLModel  # noqa: E402
from backend.app.preprocessing.retail_adapter import generate_retail_data, load_retail  # noqa: E402


def test_pipeline_runs_and_writes_real_outputs(tmp_path):
    ds = load_retail(df=generate_retail_data(3000, seed=21))
    cfg = PipelineConfig(estimator="linear", nuisance="linear", cv=2, seed=21, n_refute_sims=2)
    res = run_binary_pipeline(ds, cfg, out_dir=tmp_path)

    for f in ("results.json", "model.joblib", "qini_curves.png", "propensity_overlap.png",
              "covariate_balance.png", "cate_distribution.png"):
        assert (tmp_path / f).exists(), f
    assert (tmp_path / "dag" / "dag.png").exists() and (tmp_path / "eda" / "eda_summary.json").exists()

    assert sum(res["splits"].values()) == len(ds.df)
    assert set(res["uplift"]["metrics"]) >= {"DML (causal)", "Random", "T-learner (causal baseline)"}
    assert np.isfinite(res["dml"]["ate_test"]["ate"])
    assert {r["test"] for r in res["refutation"]} == {"placebo_treatment", "random_common_cause", "data_subset"}
    assert "ground_truth_eval" in res                               # synthetic data only
    assert any("inverse-propensity" in w for w in res["warnings"])  # observational caveat surfaced
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["model_version_id"] == res["model_version_id"]

    model = DMLModel.load(tmp_path / "model.joblib")
    assert np.isfinite(model.predict_cate(ds.X.iloc[:20])).all()


def test_pipeline_is_reproducible(tmp_path):
    ds = load_retail(df=generate_retail_data(2000, seed=4))
    cfg = PipelineConfig(estimator="linear", nuisance="linear", cv=2, seed=4, refute=False, save_model=False)
    a = run_binary_pipeline(ds, cfg, out_dir=tmp_path / "a")
    b = run_binary_pipeline(ds, cfg, out_dir=tmp_path / "b")
    assert a["dml"]["ate_test"]["ate"] == b["dml"]["ate_test"]["ate"]
