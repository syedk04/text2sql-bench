"""Static SELECT-only check, run before any SQL reaches SQLite.

This is the outer layer of the executor's defences: it parses the statement with
sqlglot (SQLite dialect) and refuses anything that is not exactly one read-only
query. The read-only connection is still the real guarantee; this layer gives a
clear rejection reason and catches multi-statement input early.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError


class UnsafeSQL(ValueError):
    """The SQL is not a single read-only query."""


def _node_types(*names: str) -> tuple[type[exp.Expression], ...]:
    # Look names up defensively so a sqlglot release that renames one class
    # does not break the import; unknown names are simply skipped.
    found = (getattr(exp, n, None) for n in names)
    return tuple(t for t in found if isinstance(t, type))


_FORBIDDEN = _node_types(
    "Insert",
    "Update",
    "Delete",
    "Create",
    "Drop",
    "Alter",
    "TruncateTable",
    "Command",
    "Pragma",
    "Attach",
    "Detach",
    "Transaction",
    "Commit",
    "Rollback",
    "Merge",
    "Into",
    "Set",
    "Use",
    "Copy",
    "LoadData",
    "Analyze",
)


def check_select_only(sql: str) -> exp.Expression:
    """Return the parsed query, or raise :class:`UnsafeSQL` explaining why not."""
    if not isinstance(sql, str) or not sql.strip():
        raise UnsafeSQL("empty SQL")
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except SqlglotError as exc:
        raise UnsafeSQL(f"could not parse SQL: {exc}") from exc
    if not statements:
        raise UnsafeSQL("empty SQL")
    if len(statements) != 1:
        raise UnsafeSQL(f"expected exactly one statement, found {len(statements)}")
    root = statements[0]
    if not isinstance(root, exp.Query):
        raise UnsafeSQL(f"only SELECT queries are allowed, got {root.key.upper()}")
    for node in root.walk():
        if isinstance(node, _FORBIDDEN):
            raise UnsafeSQL(f"{node.key.upper()} is not allowed inside a query")
    return root


def is_select_only(sql: str) -> bool:
    try:
        check_select_only(sql)
    except UnsafeSQL:
        return False
    return True
