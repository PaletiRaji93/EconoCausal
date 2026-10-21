"""Propensity, overlap and balance diagnostics, DAG and EDA outputs."""
import numpy as np
import pandas as pd
import pytest

from backend.app.eda.eda_report import run_eda
from backend.app.graph.dag_builder import (DagSpec, assumptions, build_graph, save_dag, spec_from_dataset,
                                           to_gml)
from backend.app.preprocessing.retail_adapter import generate_retail_data, load_retail
from backend.app.propensity.balance import balance_summary, balance_table, standardized_mean_diff
from backend.app.propensity.propensity_model import fit_propensity, ipw_weights, overlap_diagnostics


def test_confounded_data_gets_informative_propensity_and_weighting_improves_balance():
    ds = load_retail(df=generate_retail_data(6000, seed=1))
    t = (ds.T > 0).astype(int)
    pipe, e = fit_propensity(ds.X, t, seed=1)
    diag = overlap_diagnostics(e, t)
    assert diag["auc_oof"] > 0.58 and 0 < diag["common_support"][0] < diag["common_support"][1] < 1
    Xp = pipe.named_steps["prep"].transform(ds.X)
    tab = balance_table(Xp, t, list(pipe.named_steps["prep"].get_feature_names_out()), ipw_weights(e, t))
    s = balance_summary(tab)
    assert s["max_abs_smd_after"] < s["max_abs_smd_before"]
    assert s["warnings"]


def test_randomised_data_has_uninformative_propensity():
    ds = load_retail(df=generate_retail_data(6000, seed=2, confounded=False))
    t = (ds.T > 0).astype(int)
    _, e = fit_propensity(ds.X, t, seed=2)
    assert abs(overlap_diagnostics(e, t)["auc_oof"] - 0.5) < 0.04


def test_weak_overlap_triggers_warnings():
    rng = np.random.default_rng(0)
    t = rng.binomial(1, 0.5, 2000)
    e = np.where(t == 1, rng.uniform(0.7, 0.99, 2000), rng.uniform(0.01, 0.3, 2000))
    d = overlap_diagnostics(e, t)
    assert not d["overlap_ok"] and any("common support" in w.lower() for w in d["warnings"])


def test_smd_known_value_and_zero_variance():
    x = np.r_[np.zeros(50), np.ones(50)]
    t = np.r_[np.zeros(50), np.ones(50)]
    assert standardized_mean_diff(np.r_[np.random.default_rng(0).normal(size=50), 1 + np.random.default_rng(1).normal(size=50)], t) > 0.8
    assert standardized_mean_diff(np.ones(100), t) == 0.0


def test_propensity_rejects_non_binary():
    ds = load_retail(df=generate_retail_data(500, seed=1))
    with pytest.raises(ValueError):
        fit_propensity(ds.X, ds.df["discount_amount"].to_numpy(), seed=1)


def test_dag_is_acyclic_encodes_design_and_rejects_cycles():
    obs = DagSpec("T", "Y", ["a", "b"], randomized=False)
    g = build_graph(obs)
    assert ("a", "T") in g.edges and ("T", "Y") in g.edges
    rct = DagSpec("T", "Y", ["a", "b"], randomized=True)
    assert ("a", "T") not in build_graph(rct).edges
    with pytest.raises(ValueError):
        build_graph(DagSpec("T", "Y", ["a"], extra_edges=[("Y", "a", "bad")]))
    assert "graph [" in to_gml(g)
    names = [x["name"] for x in assumptions(DagSpec("T", "Y", ["a"], unobserved=["U"]))]
    assert any("Overlap" in n for n in names) and any("unobserved" in n.lower() for n in names)


def test_save_dag_and_eda_write_files(tmp_path):
    ds = load_retail(df=generate_retail_data(2000, seed=3))
    payload = save_dag(spec_from_dataset(ds, randomized=False), tmp_path / "dag")
    assert (tmp_path / "dag" / "dag.png").exists() and "not a discovered fact" in payload["disclaimer"]
    s = run_eda(ds, tmp_path / "eda")
    assert (tmp_path / "eda" / "eda_summary.json").exists()
    assert s["outcome_by_group"]["treated"]["n"] > 0 and s["imbalanced_features"]
    assert "spending" in s                     # retail has purchase_value
