"""Orchestrates: validate -> split -> propensity/balance -> DML -> baselines -> uplift ->
refutation -> save. Binary treatment only (continuous: see dml/continuous_dml.py).

Everything reported comes from actually fitted models; nothing is hardcoded.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from backend.app.core.config import settings
from backend.app.core.seed import set_global_seed
from backend.app.dml.dml_estimator import (DMLModel, build_binary_estimator, heterogeneity_summary,
                                           response_model_scores, t_learner_scores)
from backend.app.eda.eda_report import run_eda
from backend.app.evaluation.uplift_metrics import (bootstrap_qini_ci, evaluate_scores, plot_qini,
                                                   qini_curve, uplift_by_decile)
from backend.app.graph.dag_builder import save_dag, spec_from_dataset
from backend.app.preprocessing.dataset import CausalDataset, make_splits
from backend.app.propensity.balance import balance_summary, balance_table, plot_balance
from backend.app.propensity.propensity_model import (fit_propensity, ipw_weights, overlap_diagnostics,
                                                     plot_propensity)
from backend.app.validation.data_validation import validate_dataset
from backend.app.validation.refutation import native_refutations, run_dowhy_refutations

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    estimator: str = "linear"        # "linear" | "forest"
    nuisance: str = "gbm"            # "linear" | "gbm"
    cv: int = 3
    seed: int = 42
    fractions: tuple = (0.6, 0.2, 0.2)
    randomized: bool = False         # True for Criteo (random ad assignment)
    refute: bool = True
    n_refute_sims: int = 5
    refute_max_rows: int = 20_000
    run_dowhy: bool = False          # needs dowhy+econml; slower
    propensity_model: str = "logistic"
    uplift_ks: tuple = (0.1, 0.2, 0.3)
    save_model: bool = True


def _fit_ate_factory(cfg: PipelineConfig):
    """Return fit_ate_fn(X_df, t, y, seed) refitting the same DML configuration."""
    from backend.app.preprocessing.feature_selection import build_preprocessor

    def fit_ate(X: pd.DataFrame, t: np.ndarray, y: np.ndarray, seed: int) -> float:
        prep = build_preprocessor(X)
        Xp = prep.fit_transform(X)
        est = build_binary_estimator(cfg.estimator, cfg.nuisance, cfg.cv, seed)
        est.fit(np.asarray(y, float), (np.asarray(t) > 0).astype(int), X=Xp)
        return float(np.mean(np.asarray(est.effect(Xp))))
    return fit_ate


def run_binary_pipeline(ds: CausalDataset, cfg: Optional[PipelineConfig] = None,
                        out_dir: Optional[str | Path] = None) -> dict:
    cfg = cfg or PipelineConfig()
    set_global_seed(cfg.seed)
    base = Path(out_dir) if out_dir else settings.reports_dir / f"run_{ds.name}"
    base.mkdir(parents=True, exist_ok=True)
    result: dict = {"dataset": ds.name, "config": asdict(cfg), "warnings": []}

    # 1) validation
    vr = validate_dataset(ds.df, ds.treatment_col, ds.outcome_col, ds.feature_cols, ds.id_col)
    vr.raise_if_failed()
    result["validation"] = vr.to_dict()

    # 2) EDA + DAG (descriptive / assumptions)
    result["eda"] = run_eda(ds, base / "eda")
    dag = save_dag(spec_from_dataset(ds, cfg.randomized), base / "dag")
    result["dag"] = {"assumptions": dag["assumptions"], "files": ["dag/dag.png", "dag/dag.json"]}

    # 3) splits (test set touched only for final evaluation)
    sp = make_splits(ds.T, ds.Y, cfg.seed, cfg.fractions)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    result["splits"] = {k: int(len(v)) for k, v in sp.items()}
    t_all = (ds.T > 0).astype(int)

    # 4) propensity + overlap + balance (train data)
    Xtr = ds.X.iloc[tr]
    pipe, e_oof = fit_propensity(Xtr, t_all[tr], cfg.seed, model=cfg.propensity_model)
    diag = overlap_diagnostics(e_oof, t_all[tr])
    prep = pipe.named_steps["prep"]
    Xp = prep.transform(Xtr)
    w = ipw_weights(e_oof, t_all[tr])
    bt = balance_table(Xp, t_all[tr], list(prep.get_feature_names_out()), w)
    plot_propensity(e_oof, t_all[tr], base / "propensity_overlap.png")
    plot_balance(bt, base / "covariate_balance.png")
    result["propensity"] = {**diag, "balance": balance_summary(bt),
                            "balance_table": bt.round(4).to_dict(orient="records"),
                            "method_note": "stabilised, trimmed IPW used for balance checks only; DML does not "
                                           "use these weights."}
    result["warnings"] += diag["warnings"] + result["propensity"]["balance"]["warnings"]

    # 5) DML
    model = DMLModel(cfg.estimator, cfg.nuisance, cfg.cv, cfg.seed).fit(ds, tr)
    cate_va, cate_te = model.predict_cate(ds.X.iloc[va]), model.predict_cate(ds.X.iloc[te])
    ate_te = model.ate(ds.X.iloc[te])
    result["dml"] = {"summary": model.summary(), "ate_test": ate_te,
                     "heterogeneity_test": heterogeneity_summary(cate_te, ds.X.iloc[te]),
                     "cate_validation_mean": float(cate_va.mean()),
                     "interpretation": "Estimated conditional effects (model-based), not observed individual facts."}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.hist(cate_te, bins=40)
    ax.axvline(ate_te["ate"], c="k", ls="--", label="mean (ATE est.)")
    ax.set_xlabel("estimated treatment effect (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(base / "cate_distribution.png", dpi=120)
    plt.close(fig)

    # 6) uplift evaluation on TEST only (+ baselines)
    y_te, t_te = ds.Y[te].astype(float), t_all[te]
    weights = None
    if not cfg.randomized:
        e_te = np.clip(pipe.predict_proba(ds.X.iloc[te])[:, 1], 0.01, 0.99)
        weights = ipw_weights(e_te, t_te)
        result["warnings"].append("Observational data: uplift metrics use inverse-propensity weights; "
                                  "results depend on the propensity model being adequate.")
    rng = np.random.default_rng(cfg.seed)
    scores = {"DML (causal)": cate_te,
              "T-learner (causal baseline)": t_learner_scores(ds, tr, te, cfg.seed),
              "Response model (predictive, ignores treatment)": response_model_scores(ds, tr, te, cfg.seed),
              "Random": rng.random(len(te))}
    metrics, curves = {}, {}
    for name, s in scores.items():
        m = evaluate_scores(y_te, t_te, s, weights, cfg.uplift_ks, cfg.seed)
        if name == "DML (causal)":
            m["qini_ci95"] = list(bootstrap_qini_ci(y_te, t_te, s, weights, n_boot=100, seed=cfg.seed))
            m["decile_table"] = uplift_by_decile(y_te, t_te, s, weights).round(5).to_dict(orient="records")
        metrics[name] = m
        curves[name] = qini_curve(y_te, t_te, s, weights)
    plot_qini(curves, base / "qini_curves.png")
    result["uplift"] = {"metrics": metrics, "evaluated_on": "held-out test split",
                        "definitions": "See evaluation/uplift_metrics.py docstring."}

    # 7) ground truth (synthetic only; educational)
    gt = [c for c in ("true_effect",) if c in ds.df.columns]
    if gt:
        truth = ds.df[gt[0]].to_numpy(float)[te]
        from scipy.stats import spearmanr
        result["ground_truth_eval"] = {
            "note": "Synthetic ground truth - educational evaluation only; never available in real data.",
            "rmse": float(np.sqrt(np.mean((cate_te - truth) ** 2))),
            "spearman": float(spearmanr(cate_te, truth).statistic),
            "mean_true_effect": float(truth.mean()), "mean_estimated_effect": float(cate_te.mean())}

    # 8) refutation (train subsample)
    if cfg.refute:
        sub = tr if len(tr) <= cfg.refute_max_rows else rng.choice(tr, cfg.refute_max_rows, replace=False)
        refs = native_refutations(_fit_ate_factory(cfg), ds.X.iloc[sub], t_all[sub], ds.Y[sub],
                                  cfg.seed, cfg.n_refute_sims)
        if cfg.run_dowhy:
            refs += run_dowhy_refutations(ds.df.iloc[sub], ds.treatment_col, ds.outcome_col,
                                          ds.feature_cols, cfg.seed, cfg.n_refute_sims)
        result["refutation"] = [r.to_dict() for r in refs]
        bad = [r.test for r in refs if r.status in ("warn", "error")]
        if bad:
            result["warnings"].append(f"Refutation warnings/errors: {bad}")

    # 9) persist
    if cfg.save_model:
        mp = model.save(base / "model.joblib")
        result["model_path"] = str(mp)
    result["model_version_id"] = model.meta["version_id"]
    (base / "results.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    logger.info("Pipeline complete -> %s", base)
    return result
