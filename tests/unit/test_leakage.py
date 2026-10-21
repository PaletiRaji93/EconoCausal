"""Leakage prevention and treatment/outcome separation."""
import numpy as np
import pandas as pd
import pytest

from backend.app.core.errors import LeakageError
from backend.app.preprocessing.column_mapping import ColumnMapping, criteo_mapping, retail_mapping
from backend.app.preprocessing.feature_selection import select_features
from backend.app.preprocessing.retail_adapter import generate_retail_data, load_retail


def test_retail_features_are_pretreatment_only():
    ds = load_retail(df=generate_retail_data(500, seed=1))
    banned = {"customer_id", "eligible", "treatment", "discount_amount", "conversion",
              "purchase_value", "margin_rate", "true_effect", "true_effect_per_dollar"}
    assert not banned & set(ds.feature_cols)
    assert ds.id_col == "customer_id"
    assert ds.treatment_col not in ds.feature_cols and ds.outcome_col not in ds.feature_cols


def test_continuous_mode_excludes_binary_treatment():
    ds = load_retail(df=generate_retail_data(500, seed=1, discount_levels=(5, 10, 20)),
                     treatment_kind="continuous")
    assert ds.treatment_col == "discount_amount"
    assert "treatment" not in ds.feature_cols and "discount_amount" not in ds.feature_cols


def test_explicit_post_treatment_feature_raises():
    df = pd.DataFrame({"f0": [1.0, 2, 3, 4], "exposure": [0, 1, 0, 1], "treatment": [0, 1, 0, 1],
                       "visit": [0, 1, 1, 0]})
    m = ColumnMapping(treatment="treatment", outcome="visit", features=["f0", "exposure"])
    with pytest.raises(LeakageError):
        select_features(df, m)


def test_auto_mode_drops_suspicious_names():
    df = pd.DataFrame({"f0": np.arange(6.0), "customer_id": range(6), "post_spend": np.arange(6.0),
                       "treatment": [0, 1] * 3, "y": [0, 1, 1, 0, 0, 1]})
    feats, excl = select_features(df, ColumnMapping(treatment="treatment", outcome="y", features=None))
    assert feats == ["f0"] and "customer_id" in excl and "post_spend" in excl


def test_outcome_copy_detected():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, 500)
    df = pd.DataFrame({"f0": rng.normal(size=500), "leaky": y + 0.0, "treatment": rng.binomial(1, 0.5, 500),
                       "y": y})
    with pytest.raises(LeakageError):
        select_features(df, ColumnMapping(treatment="treatment", outcome="y", features=["f0", "leaky"]))


def test_requested_but_absent_feature_raises():
    df = pd.DataFrame({"a": [1, 2], "treatment": [0, 1], "y": [0, 1]})
    with pytest.raises(LeakageError):
        select_features(df, ColumnMapping(treatment="treatment", outcome="y", features=["a", "zzz"]))


def test_criteo_mapping_defaults():
    m = criteo_mapping("conversion")
    assert m.outcome == "conversion" and "exposure" in m.excluded and "visit" in m.excluded
    assert "exposure" not in m.features and len(m.features) == 12
    with pytest.raises(ValueError):
        criteo_mapping("revenue")


def test_mapping_json_roundtrip(tmp_path):
    m = retail_mapping()
    m.save_json(tmp_path / "m.json")
    assert ColumnMapping.from_json(tmp_path / "m.json") == m
