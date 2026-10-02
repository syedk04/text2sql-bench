"""Where things live on disk.

Everything downloaded or generated at runtime sits under one data directory,
``./data`` by default, which is gitignored. Set ``T2S_DATA_DIR`` to put it
somewhere else.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR_ENV = "T2S_DATA_DIR"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return PROJECT_ROOT / "data"


def bird_dir() -> Path:
    """Root of the extracted BIRD Mini-Dev files."""
    return data_dir() / "bird"


def databases_dir() -> Path:
    return bird_dir() / "dev_databases"


def questions_path() -> Path:
    """The pinned Hugging Face question file (500 records)."""
    return bird_dir() / "mini_dev_sqlite.json"


def db_path(db_id: str) -> Path:
    if not db_id or any(sep in db_id for sep in ("/", "\\", "..")):
        raise ValueError(f"invalid db_id: {db_id!r}")
    return databases_dir() / db_id / f"{db_id}.sqlite"


def cache_dir() -> Path:
    return data_dir() / "cache"


def log_dir() -> Path:
    return data_dir() / "logs"
