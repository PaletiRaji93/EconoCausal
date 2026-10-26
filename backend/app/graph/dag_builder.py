"""Configurable causal DAG + written assumptions.

The DAG encodes ASSUMPTIONS made by the analyst. It is not discovered from data and is not proof of
causal structure. DoWhy is used (when installed) to check that the effect is identifiable from
the graph via the backdoor criterion.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx

from backend.app.preprocessing.dataset import CausalDataset

logger = logging.getLogger(__name__)


@dataclass
class DagSpec:
    treatment: str
    outcome: str
    confounders: list[str]                       # observed pretreatment covariates
    randomized: bool = False                     # True -> no confounder->treatment edges
    unobserved: list[str] = field(default_factory=list)   # named unmeasured confounders (assumed)
    extra_edges: list[tuple[str, str, str]] = field(default_factory=list)  # (src, dst, reason)
    edge_reasons: dict[str, str] = field(default_factory=dict)


def spec_from_dataset(ds: CausalDataset, randomized: bool) -> DagSpec:
    return DagSpec(treatment=ds.treatment_col, outcome=ds.outcome_col,
                   confounders=list(ds.feature_cols), randomized=randomized,
                   unobserved=["U_unmeasured"])


def build_graph(spec: DagSpec, include_unobserved: bool = True) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edge(spec.treatment, spec.outcome)
    for c in spec.confounders:
        g.add_edge(c, spec.outcome)
        if not spec.randomized:
            g.add_edge(c, spec.treatment)
    if include_unobserved:
        for u in spec.unobserved:
            g.add_edge(u, spec.outcome)
            if not spec.randomized:
                g.add_edge(u, spec.treatment)
    for s, d, _ in spec.extra_edges:
        g.add_edge(s, d)
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("The specified graph contains a cycle; a causal DAG must be acyclic.")
    return g


def to_gml(g: nx.DiGraph) -> str:
    """GML string accepted by dowhy.CausalModel(graph=...)."""
    lines = ["graph [", "  directed 1"]
    for n in g.nodes:
        lines.append(f'  node [ id "{n}" label "{n}" ]')
    for s, d in g.edges:
        lines.append(f'  edge [ source "{s}" target "{d}" ]')
    lines.append("]")
    return "\n".join(lines)


def assumptions(spec: DagSpec) -> list[dict]:
    """Human-readable assumptions to display in the report/UI. Each is an ASSUMPTION."""
    a = []
    if spec.randomized:
        a.append({"name": "Randomised treatment", "kind": "design assumption",
                  "text": f"'{spec.treatment}' was assigned at random, so no covariate affects assignment "
                          "(no covariate->treatment edges). Verify with balance checks."})
    else:
        a.append({"name": "Unconfoundedness (conditional ignorability)", "kind": "identification assumption",
                  "text": "Given the listed pretreatment covariates, treatment assignment is independent of "
                          "potential outcomes. Untestable; violated if an unmeasured common cause exists."})
    a += [
        {"name": "Overlap / positivity", "kind": "identification assumption",
         "text": "Every covariate profile has a non-zero chance of being treated and untreated. "
                 "Checked empirically by propensity-score overlap diagnostics."},
        {"name": "Consistency / SUTVA", "kind": "identification assumption",
         "text": "A unit's observed outcome equals its potential outcome under the treatment received, "
                 "and one customer's treatment does not affect another's outcome (no interference)."},
        {"name": "Pretreatment covariates only", "kind": "design choice",
         "text": "Covariates are measured before treatment. Post-treatment variables (e.g. exposure, "
                 "revenue) are excluded because adjusting for them can introduce bias."},
        {"name": "Correlation vs causation", "kind": "caveat",
         "text": "The DAG is an assumed structure. Association in data does not prove these arrows."},
    ]
    for u in spec.unobserved:
        a.append({"name": f"Possible unobserved confounder ({u})", "kind": "threat to validity",
                  "text": "Drawn as a dashed node to remind readers that unmeasured causes of both "
                          "treatment and outcome would bias estimates; refutation tests cannot rule this out."})
    return a


def edge_table(spec: DagSpec) -> list[dict]:
    rows = [{"from": spec.treatment, "to": spec.outcome,
             "reason": spec.edge_reasons.get("treatment->outcome", "Treatment may change the outcome (effect of interest).")}]
    rows.append({"from": f"{len(spec.confounders)} pretreatment covariates", "to": spec.outcome,
                 "reason": "Customer characteristics influence baseline outcome."})
    if not spec.randomized:
        rows.append({"from": f"{len(spec.confounders)} pretreatment covariates", "to": spec.treatment,
                     "reason": "Campaign targeting depends on observed customer characteristics (confounding)."})
    for s, d, why in spec.extra_edges:
        rows.append({"from": s, "to": d, "reason": why})
    return rows


def check_identifiability(spec: DagSpec, data=None) -> dict:
    """Use DoWhy to check backdoor identifiability on the OBSERVED graph. Raises if DoWhy missing."""
    from backend.app.core.errors import require
    dowhy = require("dowhy")
    import pandas as pd

    g = build_graph(spec, include_unobserved=False)
    cols = [spec.treatment, spec.outcome, *spec.confounders]
    df = data[cols] if data is not None else pd.DataFrame({c: [0.0, 1.0] for c in cols})
    model = dowhy.CausalModel(data=df, treatment=spec.treatment, outcome=spec.outcome, graph=to_gml(g))
    est = model.identify_effect(proceed_when_unidentifiable=True)
    return {"estimand": str(est), "backdoor_adjustment_set": list(spec.confounders)}


def save_dag(spec: DagSpec, out_dir: str | Path, collapse_above: int = 8) -> dict:
    """Save dag.png and dag.json (edges, assumptions). Large covariate sets are drawn as one node."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    g_full = build_graph(spec)
    collapse = len(spec.confounders) > collapse_above
    if collapse:
        label = f"Pretreatment covariates ({len(spec.confounders)})"
        cs = DagSpec(spec.treatment, spec.outcome, [label], spec.randomized, spec.unobserved,
                     spec.extra_edges)
        g = build_graph(cs)
        conf_nodes = [label]
    else:
        g, conf_nodes = g_full, list(spec.confounders)
    pos = {spec.treatment: (0, 0), spec.outcome: (3, 0)}
    for i, c in enumerate(conf_nodes):
        pos[c] = (1.5, 1.2 + 0.6 * (i - (len(conf_nodes) - 1) / 2))
    for i, u in enumerate(spec.unobserved):
        pos[u] = (1.5, -1.5 - 0.6 * i)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    obs = [n for n in g.nodes if n not in spec.unobserved]
    nx.draw_networkx_nodes(g, pos, nodelist=obs, node_color="#cfe3f5", node_size=1800, ax=ax)
    if spec.unobserved:
        nx.draw_networkx_nodes(g, pos, nodelist=spec.unobserved, node_color="#f5d5cf",
                               node_shape="s", node_size=1800, ax=ax)
    solid = [(s, d) for s, d in g.edges if s not in spec.unobserved]
    dashed = [(s, d) for s, d in g.edges if s in spec.unobserved]
    nx.draw_networkx_edges(g, pos, edgelist=solid, arrows=True, node_size=1800, ax=ax)
    nx.draw_networkx_edges(g, pos, edgelist=dashed, style="dashed", arrows=True, node_size=1800, ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=7, ax=ax)
    ax.set_title("ASSUMED causal DAG (dashed = unobserved; not discovered from data)")
    ax.axis("off")
    fig.tight_layout()
    png = out / "dag.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)
    payload = {"treatment": spec.treatment, "outcome": spec.outcome, "randomized": spec.randomized,
               "confounders": spec.confounders, "unobserved_assumed": spec.unobserved,
               "edges": [list(e) for e in g_full.edges] if not collapse else edge_table(spec),
               "edge_reasons": edge_table(spec), "assumptions": assumptions(spec),
               "gml": to_gml(build_graph(spec, include_unobserved=False)),
               "disclaimer": "This graph is an assumption, not a discovered fact."}
    (out / "dag.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
