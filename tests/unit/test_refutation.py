"""Refutation logic (native engine, needs no heavy deps) + DoWhy path (skipped if unavailable)."""
import numpy as np
import pandas as pd
import pytest

from backend.app.validation.refutation import native_refutations


def _rct(n=4000, effect=0.2, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    t = rng.binomial(1, 0.5, n)
    y = (rng.random(n) < 0.3 + effect * t).astype(float)
    return X, t, y


def diff_in_means(X, t, y, seed):          # cheap stand-in estimator, valid for an RCT
    return float(y[t == 1].mean() - y[t == 0].mean())


def test_real_effect_passes_all_native_tests():
    X, t, y = _rct()
    res = {r.test: r for r in native_refutations(diff_in_means, X, t, y, seed=1, n_sims=6)}
    assert set(res) == {"placebo_treatment", "random_common_cause", "data_subset"}
    assert all(r.engine == "native" for r in res.values())
    assert all(r.status == "pass" for r in res.values()), {k: v.status for k, v in res.items()}
    assert abs(res["placebo_treatment"].mean_new_effect) < 0.05
    assert all("prove" in r.limitations for r in res.values())


def test_null_effect_is_inconclusive_not_pass():
    """With no true effect the estimate is noise; stability tests must not report a confident 'pass'.
    (Statistical rule: ~3% of null datasets can still look like an effect - a known limitation.)"""
    X, t, y = _rct(20000, effect=0.0, seed=0)
    res = native_refutations(diff_in_means, X, t, y, seed=1, n_sims=6)
    assert all(r.status in ("inconclusive", "warn") for r in res)


def test_broken_estimator_triggers_warning():
    X, t, y = _rct()

    def always_same_but_placebo_sensitive(Xd, td, yd, seed):
        # an estimator that ignores treatment structure: returns a constant -> placebo not near 0
        return 0.2
    res = {r.test: r for r in native_refutations(always_same_but_placebo_sensitive, X, t, y, n_sims=3)}
    assert res["placebo_treatment"].status == "warn"      # visible failure, not hidden


def test_results_serialisable():
    X, t, y = _rct(1500)
    for r in native_refutations(diff_in_means, X, t, y, n_sims=2):
        d = r.to_dict()
        assert d["status"] in {"pass", "warn", "inconclusive", "error"}
        assert isinstance(d["new_effects"], list)


def test_dowhy_engine_runs_or_records_errors():
    pytest.importorskip("dowhy", reason="DoWhy not installed: pip install dowhy")
    pytest.importorskip("econml", reason="EconML not installed")
    from backend.app.preprocessing.retail_adapter import generate_retail_data
    from backend.app.validation.refutation import run_dowhy_refutations
    df = generate_retail_data(1500, seed=2)
    df = df[df.eligible == 1]
    res = run_dowhy_refutations(df, "treatment", "conversion", ["age", "income_k", "past_purchases"],
                                n_sims=2, max_rows=1200)
    assert {r.test for r in res} == {"placebo_treatment", "random_common_cause", "data_subset"}
    assert all(r.engine == "dowhy" and r.status in {"pass", "warn", "error"} for r in res)
