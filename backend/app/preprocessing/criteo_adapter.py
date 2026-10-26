"""Criteo Uplift v2.1 adapter (public advertising-uplift data).

IMPORTANT scientific notes
- Treatment = advertising treatment (randomised assignment), NOT a monetary discount.
- `exposure` (ad actually seen) is post-treatment and is excluded from features.
- Verify the real file header and the license on the official page before use.
- The raw file is only READ; samples are written elsewhere.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from backend.app.core.config import settings
from backend.app.core.errors import DataValidationError
from backend.app.preprocessing.column_mapping import ColumnMapping, criteo_mapping
from backend.app.preprocessing.dataset import CausalDataset, schema_hash
from backend.app.preprocessing.feature_selection import select_features
from backend.app.validation.data_validation import validate_dataset

logger = logging.getLogger(__name__)

DATASET_INFO = {
    "name": "Criteo Uplift Prediction Dataset",
    "version": "v2.1",
    "official_page": "https://ailab.criteo.com/criteo-uplift-prediction-dataset/",
    "files_page": "https://huggingface.co/datasets/criteo/criteo-uplift/tree/main",
    "license_note": ("Expected: non-commercial CC BY-NC-SA 4.0. YOU MUST CONFIRM the license on the "
                     "official page and record it in docs/dataset_card.md before use."),
}
EXPECTED_COLUMNS = [f"f{i}" for i in range(12)] + ["treatment", "conversion", "visit", "exposure"]


def inspect_header(path: str | Path) -> list[str]:
    """Read only the header row (works for .csv and .csv.gz)."""
    p = Path(path)
    if not p.exists():
        raise DataValidationError(
            f"Criteo file not found: {p}. Download it from {DATASET_INFO['files_page']} "
            f"and place it in {settings.raw_dir}")
    try:
        return list(pd.read_csv(p, nrows=0).columns)
    except Exception as exc:  # parsing/compression errors -> actionable message
        raise DataValidationError(f"Cannot read header of {p}: {exc}") from exc


def check_schema(columns: list[str], mapping: ColumnMapping) -> list[str]:
    """Raise if mapped columns are absent. Return list of unexpected extra columns."""
    need = [mapping.treatment, mapping.outcome, *(mapping.features or [])]
    missing = [c for c in need if c not in columns]
    if missing:
        raise DataValidationError(
            f"Criteo file is missing mapped columns {missing}. Header found: {columns}. "
            "Adjust the ColumnMapping (JSON) to match your file.")
    return [c for c in columns if c not in EXPECTED_COLUMNS]


def sample_criteo(path: str | Path, n_rows: int, seed: int, chunksize: int = 500_000
                  ) -> tuple[pd.DataFrame, dict]:
    """Seeded uniform random sample of exactly min(n_rows, total) rows, streaming the file.

    Method: every row gets a random key; the n_rows smallest keys are kept across chunks.
    Equivalent to a uniform sample without replacement, using bounded memory.
    """
    rng = np.random.default_rng(seed)
    pool: Optional[pd.DataFrame] = None
    scanned = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        scanned += len(chunk)
        chunk = chunk.copy()
        chunk["_key"] = rng.random(len(chunk))
        pool = chunk if pool is None else pd.concat([pool, chunk], ignore_index=True)
        if len(pool) > n_rows:
            pool = pool.nsmallest(n_rows, "_key")
    if pool is None or pool.empty:
        raise DataValidationError(f"No rows could be read from {path}.")
    df = pool.sort_values("_key").drop(columns="_key").reset_index(drop=True)
    return df, {"rows_scanned": int(scanned), "rows_sampled": int(len(df))}


def load_criteo(path: str | Path, outcome: str = "visit", n_rows: Optional[int] = None,
                seed: Optional[int] = None, mapping: Optional[ColumnMapping] = None,
                validate: bool = True) -> CausalDataset:
    """Load + validate a Criteo sample and return a CausalDataset."""
    seed = settings.seed if seed is None else seed
    n_rows = settings.criteo_sample_rows if n_rows is None else n_rows
    if not (1_000 <= n_rows <= 5_000_000):
        logger.warning("n_rows=%d is outside the usual 100k-500k development range", n_rows)
    mapping = mapping or criteo_mapping(outcome)
    header = inspect_header(path)
    extras = check_schema(header, mapping)
    df, info = sample_criteo(path, n_rows, seed)
    features, excluded = select_features(df, mapping)
    if validate:
        validate_dataset(df, mapping.treatment, mapping.outcome, features).raise_if_failed()
    meta = {
        **DATASET_INFO,
        "source_file": Path(path).name,
        "sampling_method": "seeded uniform random sample via per-row random keys (no replacement)",
        "seed": seed,
        **info,
        "treatment_col": mapping.treatment, "outcome_col": mapping.outcome,
        "treated_share": round(float(df[mapping.treatment].mean()), 4),
        "outcome_rate_by_group": {
            "treated": round(float(df.loc[df[mapping.treatment] == 1, mapping.outcome].mean()), 6),
            "control": round(float(df.loc[df[mapping.treatment] == 0, mapping.outcome].mean()), 6)},
        "extra_columns_in_file": extras,
        "schema_hash": schema_hash(df),
        "outcome_interpretation": {
            "visit": "user visited the advertiser site (post-treatment behaviour)",
            "conversion": "user purchased (rare event; high variance)"}[mapping.outcome]
            if mapping.outcome in ("visit", "conversion") else "custom",
        "design": "randomised ad treatment (assignment is random, ~85% treated in v2.1)",
        "note": "Treatment is advertising exposure, not a known discount amount.",
    }
    return CausalDataset(df=df, name="criteo", treatment_col=mapping.treatment,
                         outcome_col=mapping.outcome, feature_cols=features,
                         id_col=None, excluded_cols=excluded, metadata=meta)
