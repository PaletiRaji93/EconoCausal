"""Configurable column mapping so no dataset-specific names are hardcoded in the pipeline."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ColumnMapping:
    treatment: str
    outcome: str
    features: Optional[list[str]] = None      # None -> auto: every column not excluded
    id_col: Optional[str] = None
    eligible_col: Optional[str] = None
    other_outcomes: list[str] = field(default_factory=list)  # never features
    excluded: dict[str, str] = field(default_factory=dict)   # col -> reason
    rename: dict[str, str] = field(default_factory=dict)     # source col -> canonical col

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ColumnMapping":
        return cls(**d)

    @classmethod
    def from_json(cls, path: str | Path) -> "ColumnMapping":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


CRITEO_FEATURES = [f"f{i}" for i in range(12)]


def criteo_mapping(outcome: str = "visit") -> ColumnMapping:
    """Default Criteo v2.1 mapping. VERIFY against the real file header first.

    outcome='visit'      -> site visit (higher base rate, less noisy)
    outcome='conversion' -> purchase conversion (very rare event)
    `exposure` is post-treatment (did the user actually see the ad) -> excluded.
    """
    if outcome not in ("visit", "conversion"):
        raise ValueError("Criteo outcome must be 'visit' or 'conversion'")
    other = "conversion" if outcome == "visit" else "visit"
    return ColumnMapping(
        treatment="treatment",
        outcome=outcome,
        features=list(CRITEO_FEATURES),
        other_outcomes=[other],
        excluded={
            "exposure": "post-treatment: whether the ad was actually seen (affected by treatment)",
            other: "alternative outcome (post-treatment)",
        },
    )


def retail_mapping(outcome: str = "conversion", treatment: str = "treatment") -> ColumnMapping:
    """Default mapping for the synthetic retail data in retail_adapter.generate_retail_data.

    `treatment` is 'treatment' (binary) or 'discount_amount' (continuous dose).
    """
    excluded = {
        "purchase_value": "post-treatment revenue (only observed after conversion)",
        "margin_rate": "business assumption (constant), not a customer covariate",
        "true_effect": "synthetic ground truth - evaluation only",
        "true_effect_per_dollar": "synthetic ground truth - evaluation only",
    }
    if treatment == "treatment":
        excluded["discount_amount"] = "treatment dose (determined by treatment assignment)"
    else:
        excluded["treatment"] = "binary version of the treatment dose"
    return ColumnMapping(
        treatment=treatment,
        outcome=outcome,
        features=None,
        id_col="customer_id",
        eligible_col="eligible",
        other_outcomes=["purchase_value"] if outcome != "purchase_value" else ["conversion"],
        excluded=excluded,
    )
