from pathlib import Path

import pytest

from text2sql import config


def test_default_data_dir_is_under_project(monkeypatch):
    monkeypatch.delenv(config.DATA_DIR_ENV, raising=False)
    assert config.data_dir() == config.PROJECT_ROOT / "data"


def test_env_override_moves_everything(monkeypatch, tmp_path):
    root = tmp_path / "some dir"
    monkeypatch.setenv(config.DATA_DIR_ENV, str(root))
    assert config.data_dir() == root.resolve()
    assert config.bird_dir() == root.resolve() / "bird"
    assert config.cache_dir() == root.resolve() / "cache"
    assert config.log_dir() == root.resolve() / "logs"
    expected = root.resolve() / "bird" / "dev_databases" / "financial" / "financial.sqlite"
    assert config.db_path("financial") == expected


@pytest.mark.parametrize("bad", ["", "../etc", "a/b", r"a\b"])
def test_db_path_rejects_path_tricks(bad):
    with pytest.raises(ValueError):
        config.db_path(bad)


def test_paths_are_paths():
    assert isinstance(config.questions_path(), Path)
