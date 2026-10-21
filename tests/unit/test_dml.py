"""DML fit/predict on small deterministic data. Skipped (not faked) if EconML is missing."""
import numpy as np
import pandas as pd
import pytest

from backend.app.core.errors import InsufficientSupportError
from backend.app.preprocessing.dataset import make_splits
from backend.app.preprocessing.retail_adapter import generate_retail_data, load_retail

pytest.importorskip("econml", reason="EconML not installed: pip install econml")

from backend.app.dml.continuous_dml import ContinuousDML, check_dose_support  # noqa: E402
from backend.app.dml.dml_estimator import DMLModel, heterogeneity_summary  # noqa: E402


def _small():
    ds = load_retail(df=generate_retail_data(2500, seed=11))
    sp = make_splits(ds.T, ds.Y, seed=11)
    return ds, sp


def test_binary_dml_shapes_finite_and_sign():
    ds, sp = _small()
    m = DMLModel("linear", "linear", cv=2, seed=11).fit(ds, sp["train"])
    cate = m.predict_cate(ds.X.iloc[sp["test"]])
    assert cate.shape == (len(sp["test"]),) and np.isfinite(cate).all()
    ate = m.ate(ds.X.iloc[sp["test"]])["ate"]
    truth = ds.df["true_effect"].iloc[sp["test"]].mean()      # synthetic truth, evaluation only
    assert ate > 0 and abs(ate - truth) < 0.08


def test_model_save_load_roundtrip_and_version_id(tmp_path):
    ds, sp = _small()
    m = DMLModel("linear", "linear", cv=2, seed=3).fit(ds, sp["train"])
    path = m.save(tmp_path / "m.joblib")
    m2 = DMLModel.load(path)
    X = ds.X.iloc[sp["test"]]
    assert np.allclose(m.predict_cate(X), m2.predict_cate(X))
    parts = m.summary()["version_id"].split("-")
    assert parts[0] == "synthetic_retail" or "synthetic_retail" in m.summary()["version_id"]
    assert m.summary()["version_id"].endswith("-3")
    assert path.with_suffix(".json").exists()


def test_unfitted_and_bad_input_errors():
    m = DMLModel("linear", "linear")
    with pytest.raises(RuntimeError):
        m.predict_cate(pd.DataFrame({"a": [1]}))
    ds, sp = _small()
    m.fit(ds, sp["train"])
    with pytest.raises(ValueError):
        m.predict_cate(ds.X.iloc[:5].drop(columns=["age"]))


def test_single_group_training_rejected():
    ds, sp = _small()
    only_control = sp["train"][ds.T[sp["train"]] == 0]
    with pytest.raises(InsufficientSupportError):
        DMLModel("linear", "linear").fit(ds, only_control)


def test_heterogeneity_summary_structure():
    ds, sp = _small()
    cate = np.random.default_rng(0).normal(size=len(sp["test"]))
    h = heterogeneity_summary(cate, ds.X.iloc[sp["test"]])
    assert set(h["quantiles"]) == {"q05", "q25", "q50", "q75", "q95"}
    assert "region" in h["segments"] and "age" in h["segments"]


def test_dose_support_rules():
    with pytest.raises(InsufficientSupportError):
        check_dose_support(np.r_[np.zeros(100), np.full(100, 10.0)])      # only 2 levels
    with pytest.raises(InsufficientSupportError):
        check_dose_support(np.full(100, 5.0))                              # no zero dose
    info = check_dose_support(np.r_[np.zeros(100), np.full(50, 5.0), np.full(50, 10.0), np.full(50, 20.0)])
    assert info["max_dose"] == 20.0 and info["min_positive_dose"] == 5.0


def test_continuous_dml_effect_and_no_extrapolation():
    ds = load_retail(df=generate_retail_data(3000, seed=5, discount_levels=(5, 10, 20)),
                     treatment_kind="continuous")
    sp = make_splits(ds.T > 0, ds.Y, seed=5)
    m = ContinuousDML("linear", cv=2, seed=5).fit(ds, sp["train"])
    X = ds.X.iloc[sp["test"]]
    eff, _ = m.effect_at_dose(X, 10.0)
    assert eff.shape == (len(X),) and np.isfinite(eff).all()
    per_unit = m.effect_per_unit(X)
    assert np.allclose(eff, per_unit * 10.0)
    with pytest.raises(InsufficientSupportError):
        m.effect_at_dose(X, 50.0)                                          # beyond observed support
    with pytest.raises(ValueError):
        m.effect_at_dose(X, -1.0)


def test_continuous_dml_rejects_single_dose_level():
    ds = load_retail(df=generate_retail_data(2000, seed=5, discount_levels=(10,)), treatment_kind="continuous")
    sp = make_splits(ds.T > 0, ds.Y, seed=5)
    with pytest.raises(InsufficientSupportError):
        ContinuousDML("linear", cv=2).fit(ds, sp["train"])
