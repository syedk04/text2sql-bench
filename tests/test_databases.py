import sqlite3
import zipfile
from pathlib import Path

import pytest

from synthetic import SCHEMAS, make_db
from text2sql import config
from text2sql.data.bird import EXPECTED_DB_COUNT, load_questions
from text2sql.data.databases import extract_sqlite, list_databases, table_names


def _build_zip(tmp_path: Path, hostile: bool = False) -> Path:
    src = tmp_path / "src"
    zpath = tmp_path / "minidev.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for db_id, stmts in SCHEMAS.items():
            db = make_db(src / f"{db_id}.sqlite", stmts)
            base = f"minidev/MINIDEV/dev_databases/{db_id}"
            zf.writestr(base + "/", "")
            zf.write(db, f"{base}/{db_id}.sqlite")
            zf.writestr(f"{base}/database_description/t.csv", "col,desc\n")
        zf.writestr("minidev/MINIDEV/mini_dev_sqlite.json", "[]")
        zf.writestr("minidev/MINIDEV/mini_dev_sqlite_gold.sql", "SELECT 1\tshop\n")
        zf.writestr("minidev/MINIDEV/mini_dev_mysql.json", "[]")
        zf.writestr("minidev/MINIDEV_mysql/BIRD_dev.sql", "-- mysql dump")
        zf.writestr("minidev/MINIDEV_postgresql/dev_databases/x/x.sqlite", "not used")
        zf.writestr("minidev/MINIDEV/dev_databases/shop/other.sqlite", "wrong name")
        if hostile:
            zf.writestr("../../evil/dev_databases/evil/evil.sqlite", "x")
            zf.writestr("minidev/dev_databases/a/database_description/../../../pwn.csv", "x")
    return zpath


def _files_under(path: Path) -> set[Path]:
    return {p for p in path.rglob("*") if p.is_file()}


def test_extracts_only_sqlite_material(tmp_path):
    zpath = _build_zip(tmp_path)
    root = tmp_path / "bird out"
    report = extract_sqlite(zpath, root)
    assert report.databases == sorted(SCHEMAS)
    assert report.descriptions == 3
    assert sorted(report.legacy) == ["mini_dev_sqlite.json", "mini_dev_sqlite_gold.sql"]
    assert list_databases(root / "dev_databases") == sorted(SCHEMAS)
    for db_id in SCHEMAS:
        assert table_names(root / "dev_databases" / db_id / f"{db_id}.sqlite")
    assert (root / "legacy" / "mini_dev_sqlite_gold.sql").read_text() == "SELECT 1\tshop\n"
    assert not (root / "dev_databases" / "x").exists()
    assert not (root / "dev_databases" / "shop" / "other.sqlite").exists()
    assert not list(root.rglob("*.part"))


def test_hostile_member_names_stay_inside_output(tmp_path):
    zpath = _build_zip(tmp_path, hostile=True)
    root = tmp_path / "out"
    before = _files_under(tmp_path)
    extract_sqlite(zpath, root)
    new_files = _files_under(tmp_path) - before
    assert new_files
    for p in new_files:
        assert root in p.parents, p
    assert not (tmp_path / "pwn.csv").exists()


def test_extract_is_idempotent(tmp_path):
    zpath = _build_zip(tmp_path)
    root = tmp_path / "out"
    first = extract_sqlite(zpath, root)
    second = extract_sqlite(zpath, root)
    assert first.skipped_existing == 0
    assert second.skipped_existing == len(first.databases) + first.descriptions + 2


def test_cli_download_dbs_with_local_zip(tmp_path, monkeypatch, capsys):
    from text2sql import cli

    zpath = _build_zip(tmp_path)
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path / "data"))
    assert cli.main(["download-dbs", "--zip", str(zpath)]) == 0
    assert list_databases(config.databases_dir()) == sorted(SCHEMAS)
    assert "description CSVs" in capsys.readouterr().out


def test_cli_download_dbs_missing_zip(tmp_path, monkeypatch):
    from text2sql import cli

    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path / "data"))
    assert cli.main(["download-dbs", "--zip", str(tmp_path / "nope.zip")]) == 1


def test_table_names_does_not_modify_file(tmp_path):
    db = make_db(tmp_path / "a b" / "s.sqlite", SCHEMAS["shop"])
    before = db.read_bytes()
    assert table_names(db) == ["item"]
    assert db.read_bytes() == before


@pytest.mark.bird
def test_real_databases_all_open():
    qs = load_questions(config.questions_path(), expect_total=500)
    needed = {q.db_id for q in qs}
    present = set(list_databases(config.databases_dir()))
    assert len(needed) == EXPECTED_DB_COUNT
    assert needed <= present
    for db_id in sorted(needed):
        assert table_names(config.db_path(db_id)), db_id
        uri = config.db_path(db_id).resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok", db_id
        finally:
            conn.close()
