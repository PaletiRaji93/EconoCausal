"""Schema, missing/invalid values, preprocessing, splits, Criteo adapter (tiny fake files)."""
import numpy as np
import pandas as pd
import pytest

from backend.app.core.errors import DataValidationError
from backend.app.preprocessing.criteo_adapter import (check_schema, inspect_header, load_criteo,
                                                      sample_criteo)
from backend.app.preprocessing.column_mapping import criteo_mapping
from backend.app.preprocessing.dataset import make_splits
from backend.app.preprocessing.feature_selection import build_preprocessor
from backend.app.preprocessing.retail_adapter import generate_retail_data, load_retail
from backend.app.validation.data_validation import read_csv_checked, validate_dataset


def _fake_criteo(n=3000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(12)})
    df["treatment"] = rng.binomial(1, 0.85, n)
    df["visit"] = rng.binomial(1, 0.05 + 0.02 * df["treatment"])
    df["conversion"] = rng.binomial(1, 0.005, n)
    df["exposure"] = df["treatment"] * rng.binomial(1, 0.03, n)
    return df


def _base():
    df = generate_retail_data(600, seed=1)
    feats = ["age", "income_k", "past_purchases", "region"]
    return df, feats


def test_valid_data_passes():
    df, feats = _base()
    rep = validate_dataset(df, "treatment", "conversion", feats, "customer_id")
    assert rep.ok, rep.errors
    assert rep.summary["n_treated"] + rep.summary["n_control"] == len(df)


def test_non_binary_treatment_rejected():
    df, feats = _base()
    df.loc[0, "treatment"] = 2
    rep = validate_dataset(df, "treatment", "conversion", feats)
    assert not rep.ok and any("0/1" in e for e in rep.errors)


def test_missing_column_reported():
    df, feats = _base()
    rep = validate_dataset(df.drop(columns=["age"]), "treatment", "conversion", feats)
    assert not rep.ok and "age" in rep.errors[0]


def test_missing_values_and_duplicates_flagged():
    df, feats = _base()
    df.loc[:30, "income_k"] = np.nan
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    rep = validate_dataset(df, "treatment", "conversion", feats, "customer_id")
    assert any("income_k" in w for w in rep.warnings)
    assert any("Identifier" in e for e in rep.errors)  # duplicated customer ids


def test_constant_outcome_and_single_group_rejected():
    df, feats = _base()
    df["conversion"] = 0
    assert not validate_dataset(df, "treatment", "conversion", feats).ok
    df2, feats = _base()
    df2["treatment"] = 1
    assert not validate_dataset(df2, "treatment", "conversion", feats).ok


def test_infinite_and_text_outcome_rejected():
    df, feats = _base()
    df.loc[3, "age"] = np.inf
    assert not validate_dataset(df, "treatment", "conversion", feats).ok
    df, feats = _base()
    df["conversion"] = df["conversion"].astype(object)
    df.loc[2, "conversion"] = "yes"
    assert not validate_dataset(df, "treatment", "conversion", feats).ok


def test_read_csv_checked_errors(tmp_path):
    with pytest.raises(DataValidationError):
        read_csv_checked(tmp_path / "nope.csv")
    empty = tmp_path / "e.csv"
    empty.write_text("")
    with pytest.raises(DataValidationError):
        read_csv_checked(empty)
    header_only = tmp_path / "h.csv"
    header_only.write_text("a,b\n")
    with pytest.raises(DataValidationError):
        read_csv_checked(header_only)


def test_retail_generator_deterministic_and_labelled():
    a, b = generate_retail_data(400, seed=5), generate_retail_data(400, seed=5)
    assert a.equals(b)
    assert not a.equals(generate_retail_data(400, seed=6))
    ds = load_retail(df=a)
    assert ds.metadata["synthetic"] is True


def test_eligibility_filter_applied():
    df = generate_retail_data(800, seed=2)
    ds = load_retail(df=df)
    assert (ds.df["eligible"] == 1).all()
    assert len(ds.df) == int(df["eligible"].sum())
    assert ds.df.loc[ds.df["eligible"] == 1, "treatment"].sum() > 0


def test_criteo_header_schema_and_sampling(tmp_path):
    p = tmp_path / "criteo.csv"
    _fake_criteo().to_csv(p, index=False)
    assert "f0" in inspect_header(p)
    s1, info = sample_criteo(p, 500, seed=7, chunksize=700)
    s2, _ = sample_criteo(p, 500, seed=7, chunksize=700)
    assert len(s1) == 500 and info["rows_scanned"] == 3000
    assert s1.equals(s2)                       # reproducible
    s3, _ = sample_criteo(p, 500, seed=8, chunksize=700)
    assert not s1.equals(s3)


def test_criteo_missing_column_and_missing_file(tmp_path):
    p = tmp_path / "bad.csv"
    _fake_criteo().drop(columns=["f3"]).to_csv(p, index=False)
    with pytest.raises(DataValidationError):
        load_criteo(p, n_rows=1000)
    with pytest.raises(DataValidationError):
        inspect_header(tmp_path / "missing.csv")
    with pytest.raises(DataValidationError):
        check_schema(["a", "b"], criteo_mapping())


def test_criteo_load_excludes_exposure_and_other_outcome(tmp_path):
    p = tmp_path / "criteo.csv"
    _fake_criteo().to_csv(p, index=False)
    ds = load_criteo(p, outcome="visit", n_rows=2000, seed=1)
    assert ds.feature_cols == [f"f{i}" for i in range(12)]
    assert "exposure" in ds.excluded_cols and "conversion" in ds.excluded_cols
    assert ds.metadata["seed"] == 1 and ds.metadata["rows_sampled"] == 2000
    assert "not a known discount" in ds.metadata["note"]


def test_preprocessor_handles_numeric_and_categorical_and_missing():
    df = generate_retail_data(300, seed=3)[["age", "region", "is_member"]].copy()
    df.loc[:10, "age"] = np.nan
    Xp = build_preprocessor(df).fit_transform(df)
    assert np.isfinite(Xp).all()
    assert Xp.shape[1] == 1 + 1 + 2  # age, is_member, 2 region dummies


def test_splits_disjoint_cover_and_stratified():
    df = generate_retail_data(1000, seed=4)
    sp = make_splits(df["treatment"].to_numpy(), df["conversion"].to_numpy(), seed=1)
    allidx = np.concatenate(list(sp.values()))
    assert len(allidx) == len(set(allidx)) == len(df)
    r = [df["treatment"].to_numpy()[v].mean() for v in sp.values()]
    assert max(r) - min(r) < 0.03
