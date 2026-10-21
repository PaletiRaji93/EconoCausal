"""Qini / AUUC / uplift@k on data with a known answer."""
import numpy as np
import pytest

from backend.app.evaluation.uplift_metrics import (auuc, bootstrap_qini_ci, evaluate_scores, qini_coefficient,
                                                   qini_curve, uplift_at_k, uplift_by_decile)


def _data(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = rng.binomial(1, 0.5, n)
    y = rng.binomial(1, 0.2 + 0.2 * (x > 0) * t)       # true effect 0.2 only when x>0
    return x, t, y


def test_informative_score_beats_random_and_reverse_is_negative():
    x, t, y = _data()
    good, rev = qini_coefficient(y, t, x), qini_coefficient(y, t, -x)
    noise = qini_coefficient(y, t, np.random.default_rng(1).normal(size=len(y)))
    assert good > 100 and rev < -100 and abs(noise) < 0.2 * good
    assert abs(good + rev) < 0.1 * good                  # symmetric around random


def test_curve_endpoints_and_total_incremental():
    x, t, y = _data()
    xs, q = qini_curve(y, t, x)
    assert xs[0] == 0 and xs[-1] == 1 and q[0] == 0
    expected = 0.1 * (t == 1).sum()                      # avg effect 0.1 x treated count
    assert abs(q[-1] - expected) < 0.15 * expected


def test_uplift_at_k_recovers_known_effect():
    x, t, y = _data()
    assert abs(uplift_at_k(y, t, x, 0.3) - 0.2) < 0.03
    assert abs(uplift_at_k(y, t, x, 1.0) - 0.1) < 0.03


def test_uplift_at_k_exact_small_example():
    y = np.array([1, 0, 1, 0]); t = np.array([1, 0, 1, 0]); s = np.array([4, 3, 2, 1.0])
    assert uplift_at_k(y, t, s, 0.5) == 1.0               # top2: treated 1, control 0
    assert uplift_at_k(y, t, s, 1.0) == 1.0


def test_auuc_is_area_under_curve_and_weights_are_used():
    x, t, y = _data(4000)
    a = auuc(y, t, x)
    xs, q = qini_curve(y, t, x)
    manual = float(np.sum((xs[1:] - xs[:-1]) * (q[1:] + q[:-1]) / 2))
    assert abs(a - manual) < 1e-6 * max(1.0, abs(manual))
    w = np.where(t == 1, 2.0, 1.0)
    assert qini_curve(y, t, x, w)[1][-1] != qini_curve(y, t, x)[1][-1]


def test_deciles_monotone_signal():
    x, t, y = _data()
    tab = uplift_by_decile(y, t, x)
    assert len(tab) == 10 and tab["n"].sum() == len(y)
    assert tab["uplift"].iloc[:4].mean() > tab["uplift"].iloc[-4:].mean() + 0.1


def test_input_validation():
    x, t, y = _data(100)
    with pytest.raises(ValueError):
        qini_curve(y, np.ones_like(t), x)                 # no controls
    with pytest.raises(ValueError):
        qini_curve(y, t, x[:-1])                          # length mismatch
    with pytest.raises(ValueError):
        qini_curve(y, t, np.where(np.arange(100) == 0, np.nan, x))
    with pytest.raises(ValueError):
        uplift_at_k(y, t, x, 0)
    with pytest.raises(ValueError):
        qini_curve(y, t + 1, x)


def test_evaluate_scores_and_bootstrap_ci():
    x, t, y = _data(6000)
    m = evaluate_scores(y, t, x, n_random=5)
    assert m["qini_coefficient"] > m["random_ranking_qini_mean"] + 3 * m["random_ranking_qini_sd"]
    lo, hi = bootstrap_qini_ci(y, t, x, n_boot=30)
    assert lo < m["qini_coefficient"] < hi
