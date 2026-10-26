"""Central settings. Values come from environment variables (see .env.example).

Paths are resolved against the repository root so scripts work from any folder.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

try:  # python-dotenv is optional
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


def _path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p


@dataclass(frozen=True)
class Settings:
    env: str = "local"
    seed: int = 42
    data_dir: Path = REPO_ROOT / "data"
    reports_dir: Path = REPO_ROOT / "reports"
    database_url: str = "sqlite:///./econocausal.db"
    mlflow_tracking_uri: str = "file:./mlruns"
    log_level: str = "INFO"
    criteo_sample_rows: int = 200_000
    cors_origins: str = "http://localhost:5173"

    @classmethod
    def from_env(cls) -> "Settings":
        g = os.environ.get
        return cls(
            env=g("ECONO_ENV", "local"),
            seed=int(g("ECONO_SEED", "42")),
            data_dir=_path(g("ECONO_DATA_DIR", "data")),
            reports_dir=_path(g("ECONO_REPORTS_DIR", "reports")),
            database_url=g("ECONO_DATABASE_URL", "sqlite:///./econocausal.db"),
            mlflow_tracking_uri=g("ECONO_MLFLOW_TRACKING_URI", "file:./mlruns"),
            log_level=g("ECONO_LOG_LEVEL", "INFO"),
            criteo_sample_rows=int(g("ECONO_CRITEO_SAMPLE_ROWS", "200000")),
            cors_origins=g("ECONO_CORS_ORIGINS", "http://localhost:5173"),
        )

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def sample_dir(self) -> Path:
        return self.data_dir / "sample"


settings = Settings.from_env()
