import pytest

from text2sql.sql.guard import UnsafeSQL, check_select_only, is_select_only

ACCEPT = [
    "SELECT 1",
    "select name from item where price > 5",
    "SELECT 1;",
    "SELECT 1;  ",
    "  -- leading comment\nSELECT 1",
    "SELECT `item name`, \"other col\" FROM `my table`",
    "SELECT IIF(price > 5, 'big', 'small') FROM item",
    "SELECT CAST(SUM(price) AS REAL) / COUNT(*) FROM item",
    "SELECT CAST(COUNT(*) AS FLOAT) * 100 / 7",
    "WITH t AS (SELECT price FROM item) SELECT MAX(price) FROM t",
    "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 5) "
    "SELECT * FROM n",
    "SELECT a FROM x UNION SELECT a FROM y",
    "SELECT a FROM x INTERSECT SELECT a FROM y",
    "SELECT a FROM x EXCEPT SELECT a FROM y",
    "(SELECT 1)",
    "SELECT name FROM item ORDER BY price DESC LIMIT 1 OFFSET 2",
    "SELECT STRFTIME('%Y', d), SUBSTR(s, 1, 4), INSTR(s, 'x') FROM t",
    "SELECT RANK() OVER (PARTITION BY a ORDER BY b) FROM t",
    "SELECT * FROM a INNER JOIN b ON a.id = b.id WHERE a.x LIKE '%DELETE%'",
    "SELECT 'DROP TABLE x; --' AS s",
    "SELECT COUNT(*) FROM t WHERE x IN (SELECT y FROM u)",
]

REJECT = [
    "",
    "   ",
    ";",
    "SELECT 1; SELECT 2",
    "SELECT 1; DROP TABLE item",
    "SELECT 1; PRAGMA query_only = OFF",
    "INSERT INTO item VALUES (1)",
    "UPDATE item SET a = 1",
    "DELETE FROM item",
    "DROP TABLE item",
    "CREATE TABLE t (a)",
    "CREATE TEMP VIEW v AS SELECT 1",
    "ALTER TABLE item ADD COLUMN b",
    "PRAGMA table_info(item)",
    "PRAGMA writable_schema = ON",
    "ATTACH DATABASE 'x.db' AS x",
    "ATTACH 'x.db' AS x",
    "DETACH x",
    "/* SELECT */ ATTACH DATABASE 'x.db' AS x",
    "SELECT 1 /* ; */; ATTACH DATABASE 'x.db' AS x",
    "WITH x AS (SELECT 1) DELETE FROM item",
    "WITH x AS (SELECT 1) INSERT INTO item SELECT * FROM x",
    "WITH x AS (SELECT 1) UPDATE item SET a = 1",
    "SELECT * INTO copy FROM item",
    "BEGIN",
    "COMMIT",
    "VACUUM",
    "REINDEX",
    "ANALYZE",
    "VALUES (1)",
    "this is not sql at all",
    "SELECT FROM WHERE",
    "REPLACE INTO item VALUES (1)",
]


@pytest.mark.parametrize("sql", ACCEPT)
def test_accepts_read_only_queries(sql):
    check_select_only(sql)
    assert is_select_only(sql)


@pytest.mark.parametrize("sql", REJECT)
def test_rejects_everything_else(sql):
    with pytest.raises(UnsafeSQL):
        check_select_only(sql)
    assert not is_select_only(sql)


def test_rejection_reasons_are_readable():
    with pytest.raises(UnsafeSQL, match="exactly one statement, found 2"):
        check_select_only("SELECT 1; SELECT 2")
    with pytest.raises(UnsafeSQL, match="empty"):
        check_select_only("")
    with pytest.raises(UnsafeSQL, match="DELETE"):
        check_select_only("WITH x AS (SELECT 1) DELETE FROM item")


def test_non_string_is_rejected():
    with pytest.raises(UnsafeSQL):
        check_select_only(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT " + "abs(" * 300 + "1" + ")" * 300,
        "SELECT " + "(" * 500 + "1" + ")" * 500,
        "SELECT " + "CASE WHEN 1 THEN " * 300 + "1" + " END" * 300,
        "SELECT * FROM " + "(SELECT * FROM " * 300 + "t" + ")" * 300,
    ],
)
def test_deep_nesting_is_rejected_not_crashed(sql):
    with pytest.raises(UnsafeSQL):
        check_select_only(sql)
    assert not is_select_only(sql)


def test_unexpected_parser_errors_become_rejections(monkeypatch):
    import sqlglot

    def boom(*a, **k):
        raise KeyError("parser bug")

    monkeypatch.setattr(sqlglot, "parse", boom)
    with pytest.raises(UnsafeSQL, match="KeyError"):
        check_select_only("SELECT 1")
