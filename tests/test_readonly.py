import sqlite3

import pytest

from synthetic import SCHEMAS, make_db
from text2sql.sql.executor import ReadOnlyError, open_readonly, readonly_uri

WRITES = [
    "INSERT INTO item (name, price, category) VALUES ('x', 1, 'y')",
    "UPDATE item SET price = 0",
    "DELETE FROM item",
    "DROP TABLE item",
    "CREATE TABLE t (a)",
    "CREATE TEMP TABLE t (a)",
    "CREATE INDEX ix ON item (name)",
    "CREATE VIEW v AS SELECT 1",
    "CREATE TRIGGER tr AFTER INSERT ON item BEGIN SELECT 1; END",
    "ALTER TABLE item ADD COLUMN z",
    "ALTER TABLE item RENAME TO item2",
    "REPLACE INTO item (id, name) VALUES (1, 'z')",
    "INSERT OR REPLACE INTO item (id, name) VALUES (1, 'z')",
    "PRAGMA query_only = OFF",
    "PRAGMA writable_schema = ON",
    "PRAGMA journal_mode = WAL",
    "PRAGMA user_version = 7",
    "ATTACH DATABASE ':memory:' AS other",
    "DETACH DATABASE main",
    "VACUUM",
    "REINDEX",
    "ANALYZE",
    "BEGIN IMMEDIATE",
    "SAVEPOINT s",
    "WITH x AS (SELECT 1) DELETE FROM item",
    "SELECT load_extension('nope')",
]


@pytest.fixture
def db(tmp_path):
    return make_db(tmp_path / "folder with spaces" / "shop db.sqlite", SCHEMAS["shop"])


def test_uri_handles_spaces(db):
    uri = readonly_uri(db)
    assert uri.startswith("file:") and uri.endswith("?mode=ro&immutable=1")
    assert " " not in uri


def test_reads_work(db):
    conn = open_readonly(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM item").fetchone() == (5,)
        assert conn.execute(
            "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 3) "
            "SELECT SUM(i) FROM n"
        ).fetchone() == (6,)
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        assert tables == [("item",)]
        assert conn.execute("SELECT IIF(1, 'a', 'b'), CAST('2.5' AS REAL)").fetchone() == (
            "a",
            2.5,
        )
    finally:
        conn.close()


@pytest.mark.parametrize("sql", WRITES)
def test_every_write_fails_and_file_is_untouched(db, sql):
    before = db.read_bytes()
    conn = open_readonly(db)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute(sql)
        conn.commit()
        # the guard pragma must still be on afterwards
        assert conn.execute("SELECT COUNT(*) FROM item").fetchone() == (5,)
    finally:
        conn.close()
    assert db.read_bytes() == before
    leftovers = sorted(p.name for p in db.parent.iterdir())
    assert leftovers == [db.name]


def test_executescript_cannot_write(db):
    before = db.read_bytes()
    conn = open_readonly(db)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            conn.executescript("SELECT 1; DELETE FROM item; DROP TABLE item;")
    finally:
        conn.close()
    assert db.read_bytes() == before


def test_missing_file_is_not_created(tmp_path):
    target = tmp_path / "nope.sqlite"
    with pytest.raises(ReadOnlyError):
        open_readonly(target)
    assert not target.exists()


def test_wal_mode_database_gets_no_side_files(tmp_path):
    import sqlite3 as sq

    path = tmp_path / "wal db" / "w.sqlite"
    path.parent.mkdir()
    conn = sq.connect(path)
    assert conn.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
    conn.execute("CREATE TABLE t (a)")
    conn.execute("INSERT INTO t VALUES (1), (2)")
    conn.commit()
    conn.close()
    assert sorted(p.name for p in path.parent.iterdir()) == ["w.sqlite"]
    before = path.read_bytes()

    from text2sql.sql.executor import execute

    res = execute(path, "SELECT SUM(a) FROM t")
    assert res.ok and res.rows == [(3,)]
    assert execute(path, "DELETE FROM t", guard=False).status == "error"
    assert sorted(p.name for p in path.parent.iterdir()) == ["w.sqlite"]
    assert path.read_bytes() == before
