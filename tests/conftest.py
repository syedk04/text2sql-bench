from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from synthetic import make_databases  # noqa: E402
from text2sql import config as t2s_config  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def bird_data_present() -> bool:
    return t2s_config.databases_dir().is_dir() and t2s_config.questions_path().is_file()


def pytest_collection_modifyitems(config, items):
    if bird_data_present():
        return
    skip = pytest.mark.skip(reason="real BIRD data not found under data/bird (run download-dbs)")
    for item in items:
        if "bird" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def synthetic_data_dir(tmp_path, monkeypatch) -> Path:
    """A throwaway data dir (with a space in its path) holding synthetic BIRD-like data."""
    root = tmp_path / "data dir"
    monkeypatch.setenv(t2s_config.DATA_DIR_ENV, str(root))
    make_databases(t2s_config.databases_dir())
    t2s_config.questions_path().parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / "questions.json", t2s_config.questions_path())
    return root
