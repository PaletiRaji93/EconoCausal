"""Causal refutation tests.

Two engines (always labelled in the output):
- "native": re-fits THIS project's DML estimator with a perturbation (always available).
- "dowhy":  runs DoWhy's refuters against a DoWhy CausalModel using the EconML DML estimator
            (needs `dowhy` + `econml`; slower; DoWhy/EconML version quirks are possible).

Tests
- placebo_treatment: permute treatment -> estimated effect should collapse towards 0.
- random_common_cause: add a pure-noise covariate -> estimate should barely move.
- data_subset: refit on random subsets -> estimate should be stable.

PASSING THESE TESTS DOES NOT PROVE CAUSALITY. They cannot detect unobserved confounding.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from backend.app.core.errors import require

logger = logging.getLogger(__name__)

# fit_ate_fn(X_df, t, y, seed) -> float  (average effect estimated on the supplied data)
FitAteFn = Callable[[pd.DataFrame, np.ndarray, np.ndarray, int], float]


@dataclass
class RefutationResult:
    test: str
    engine: str
    status: str                       # "pass" | "warn" | "inconclusive" | "error"
    original_effect: float
    new_effects: list[float] = field(default_factory=list)
    mean_new_effect: Optional[float] = None
    method: str = ""
    interpretation: str = ""
    limitations: str = ("Cannot detect unobserved confounding; passing does not prove causality.")

    def to_dict(self) -> dict:
        return asdict(self)


def native_refutations(fit_ate_fn: FitAteFn, X: pd.DataFrame, t: np.ndarray, y: np.ndarray,
                       seed: int = 42, n_sims: int = 5, subset_frac: float = 0.8,
                       tol_placebo: float = 0.25, tol_stability: float = 0.25) -> list[RefutationResult]:
    """Run the three native refutations. Tolerances are relative to |original ATE|."""
    t, y = np.asarray(t), np.asarray(y)
    orig = float(fit_ate_fn(X, t, y, seed))
    rng = np.random.default_rng(seed)
    results: list[RefutationResult] = []

    # 1) placebo treatment (permutation)
    eff = [float(fit_ate_fn(X, rng.permutation(t), y, seed + 1 + i)) for i in range(n_sims)]
    m = float(np.mean(eff))
    # An original estimate that is not distinguishable from placebo noise (|ATE| < 3 sd of the placebo
    # effects) is effectively "no detectable effect": relative-stability checks would be meaningless.
    noise = float(np.std(eff)) if len(eff) >= 3 else 0.0
    small = abs(orig) < max(1e-8, 3.0 * noise)
    if small:
        status, text = "inconclusive", "Original effect is ~0, so a placebo comparison is uninformative."
    elif np.mean(np.abs(eff)) < tol_placebo * abs(orig):
        status, text = "pass", (f"Placebo effects average {m:.4g} vs original {orig:.4g}: the estimator does "
                                "not invent effects from a random treatment.")
    else:
        status, text = "warn", (f"Placebo effects (mean |effect| {np.mean(np.abs(eff)):.4g}) are not small "
                                f"relative to the original {orig:.4g}: estimates may be noisy/spurious.")
    results.append(RefutationResult("placebo_treatment", "native", status, orig, eff, m,
                                    "permute treatment labels; refit", text))

    # 2) random common cause
    eff = []
    for i in range(n_sims):
        Xn = X.copy()
        Xn["__random_cause__"] = rng.normal(size=len(Xn))
        eff.append(float(fit_ate_fn(Xn, t, y, seed + 100 + i)))
    m = float(np.mean(eff))
    rel = abs(m - orig) / abs(orig) if not small else np.inf
    status = "inconclusive" if small else ("pass" if rel <= tol_stability else "warn")
    text = ("Original effect within placebo noise; relative change undefined." if small else
            f"Adding a random covariate moved the estimate by {rel:.1%} (tolerance {tol_stability:.0%}).")
    results.append(RefutationResult("random_common_cause", "native", status, orig, eff, m,
                                    "append N(0,1) noise covariate; refit", text))

    # 3) data subset
    eff = []
    n = len(t)
    for i in range(n_sims):
        idx = rng.choice(n, size=int(subset_frac * n), replace=False)
        eff.append(float(fit_ate_fn(X.iloc[idx], t[idx], y[idx], seed + 200 + i)))
    m, sd = float(np.mean(eff)), float(np.std(eff))
    rel = abs(m - orig) / abs(orig) if not small else np.inf
    status = "inconclusive" if small else ("pass" if rel <= tol_stability else "warn")
    text = ("Original effect within placebo noise; relative change undefined." if small else
            f"Subset estimates average {m:.4g} (sd {sd:.4g}) vs original {orig:.4g}; change {rel:.1%}.")
    results.append(RefutationResult("data_subset", "native", status, orig, eff, m,
                                    f"refit on random {int(subset_frac*100)}% subsets", text))
    return results


def run_dowhy_refutations(df: pd.DataFrame, treatment: str, outcome: str, features: list[str],
                          seed: int = 42, n_sims: int = 5, nuisance: str = "linear", cv: int = 2,
                          max_rows: int = 5000) -> list[RefutationResult]:
    """Run DoWhy refuters on a DoWhy CausalModel using the EconML LinearDML estimator.

    Uses a subsample (`max_rows`) because DoWhy refits the estimator for every simulation.
    Each refuter's failure is recorded as status='error' with the message (never hidden).
    """
    dowhy = require("dowhy")
    require("econml", "econml")
    from backend.app.dml.dml_estimator import make_nuisance

    data = df[[treatment, outcome, *features]].dropna()
    if len(data) > max_rows:
        data = data.sample(max_rows, random_state=seed)
    data = pd.get_dummies(data, columns=[c for c in features if not pd.api.types.is_numeric_dtype(data[c])],
                          dtype=float)
    feats = [c for c in data.columns if c not in (treatment, outcome)]
    my, mt = make_nuisance(nuisance, seed, discrete_treatment=True)
    model = dowhy.CausalModel(data=data, treatment=treatment, outcome=outcome,
                              common_causes=feats, effect_modifiers=feats)
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified, method_name="backdoor.econml.dml.LinearDML", control_value=0, treatment_value=1,
        target_units="ate", confidence_intervals=False,
        method_params={"init_params": {"model_y": my, "model_t": mt, "discrete_treatment": True,
                                       "cv": cv, "random_state": seed}, "fit_params": {}})
    orig = float(np.ravel(estimate.value)[0])
    specs = [("placebo_treatment", "placebo_treatment_refuter", {"placebo_type": "permute"}),
             ("random_common_cause", "random_common_cause", {}),
             ("data_subset", "data_subset_refuter", {"subset_fraction": 0.8})]
    results = []
    for name, method, kw in specs:
        try:
            ref = model.refute_estimate(identified, estimate, method_name=method,
                                        num_simulations=n_sims, random_seed=seed, **kw)
            new = float(np.ravel(getattr(ref, "new_effect"))[0])
            if name == "placebo_treatment":
                ok = abs(new) <= 0.25 * abs(orig)       # placebo effect should be near 0
            else:
                ok = abs(new - orig) <= 0.25 * abs(orig)  # estimate should stay stable
            results.append(RefutationResult(
                name, "dowhy", "pass" if ok else "warn", orig, [new], new, f"DoWhy {method}",
                "DoWhy output (truncated): " + str(ref).replace("\n", " ")[:400]))
        except Exception as exc:  # recorded visibly, not swallowed
            logger.exception("DoWhy refuter %s failed", method)
            results.append(RefutationResult(name, "dowhy", "error", orig, [], None, f"DoWhy {method}",
                                            f"Refuter failed: {type(exc).__name__}: {exc}"))
    return results
