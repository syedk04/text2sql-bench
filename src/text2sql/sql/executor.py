"""Read-only SQLite access.

Generated SQL must never be able to change a database. Each layer below would
be enough on its own for ordinary writes; they are stacked so a gap in one does
not matter:

1. the file is opened through a ``mode=ro`` URI, so SQLite refuses to write it;
2. ``PRAGMA query_only = ON`` rejects any statement that would write;
3. an authorizer callback allows only reads, function calls and recursive CTEs,
   and is installed after the pragma so nothing can switch the pragma back off.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# Authorizer actions that a pure query needs. Everything else (writes, DDL,
# PRAGMA, ATTACH, transactions, ANALYZE, ...) is denied.
_ALLOWED_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
)
# Functions that are callable through SQLITE_FUNCTION but have side effects.
_DENIED_FUNCTIONS = frozenset({"load_extension"})


class ReadOnlyError(RuntimeError):
    """The database could not be opened read-only."""


def _authorizer(
    action: int, arg1: str | None, arg2: str | None, db_name: str | None, source: str | None
) -> int:
    if action not in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in _DENIED_FUNCTIONS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def readonly_uri(db_path: str | Path) -> str:
    # as_uri() percent-encodes spaces and other characters, which SQLite's URI
    # parser decodes again; plain string paths with spaces would not survive.
    return Path(db_path).resolve().as_uri() + "?mode=ro"


def open_readonly(db_path: str | Path) -> sqlite3.Connection:
    """Open ``db_path`` so that no statement on the connection can modify it."""
    path = Path(db_path)
    if not path.is_file():
        raise ReadOnlyError(f"database not found: {path}")
    conn = sqlite3.connect(readonly_uri(path), uri=True, check_same_thread=False)
    try:
        if hasattr(conn, "enable_load_extension"):
            conn.enable_load_extension(False)
        conn.execute("PRAGMA query_only = ON")
        if conn.execute("PRAGMA query_only").fetchone()[0] != 1:
            raise ReadOnlyError("PRAGMA query_only did not take effect")
        conn.set_authorizer(_authorizer)
    except Exception:
        conn.close()
        raise
    return conn
