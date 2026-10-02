"""Read-only SQLite access.

Generated SQL must never be able to change a database. Each layer below would
be enough on its own for ordinary writes; they are stacked so a gap in one does
not matter:

1. the file is opened through a ``mode=ro`` URI, so SQLite refuses to write it;
2. ``PRAGMA query_only = ON`` rejects any statement that would write;
3. an authorizer callback allows only reads, function calls and recursive CTEs,
   and is installed after the pragma so nothing can switch the pragma back off.

On top of that, :func:`execute` runs the sqlglot SELECT-only check first, stops
queries that run past a deadline (a SQLite progress handler, which works on
Windows too, unlike signal-based timeouts) and can cap the number of rows fetched.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from text2sql.sql.guard import UnsafeSQL, check_select_only

Status = Literal["ok", "error", "timeout", "rejected"]
# How many SQLite VM instructions run between deadline checks.
PROGRESS_STEPS = 10_000
# Largest string or blob a query may build (SQLite's default is 1 GB). Stops
# printf/zeroblob/randomblob tricks from eating memory; BIRD values are tiny.
MAX_VALUE_BYTES = 100_000_000

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
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_VALUE_BYTES)
        conn.execute("PRAGMA query_only = ON")
        if conn.execute("PRAGMA query_only").fetchone()[0] != 1:
            raise ReadOnlyError("PRAGMA query_only did not take effect")
        conn.set_authorizer(_authorizer)
    except Exception:
        conn.close()
        raise
    return conn


@dataclass
class ExecResult:
    status: Status
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    truncated: bool = False
    error: str | None = None
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def execute(
    db_path: str | Path,
    sql: str,
    *,
    timeout_s: float = 30.0,
    row_limit: int | None = None,
    guard: bool = True,
    deadline: float | None = None,
) -> ExecResult:
    """Run one query read-only and report what happened instead of raising.

    ``timeout_s`` covers execution and fetching. ``deadline`` (a
    ``time.monotonic()`` value) overrides it, which lets a caller share one time
    budget across several queries. ``row_limit=None`` fetches everything; with a
    limit, one extra row is requested to tell whether the result was cut off.
    """
    start = time.monotonic()
    if deadline is None:
        deadline = start + timeout_s
    if row_limit is not None and row_limit < 0:
        raise ValueError("row_limit must be >= 0 or None")

    def done(status: Status, **kw: Any) -> ExecResult:
        return ExecResult(status=status, elapsed_s=time.monotonic() - start, **kw)

    if guard:
        try:
            check_select_only(sql)
        except UnsafeSQL as exc:
            return done("rejected", error=str(exc))
        except Exception as exc:  # belt and braces: execute() never raises
            return done("rejected", error=f"could not check SQL ({type(exc).__name__}: {exc})")
    if time.monotonic() >= deadline:
        return done("timeout", error="time budget used up before the query started")

    try:
        conn = open_readonly(db_path)
    except (ReadOnlyError, sqlite3.Error) as exc:
        return done("error", error=f"could not open database: {exc}")

    timed_out = False

    def check_deadline() -> int:
        nonlocal timed_out
        if time.monotonic() > deadline:
            timed_out = True
            return 1  # non-zero aborts the running statement
        return 0

    conn.set_progress_handler(check_deadline, PROGRESS_STEPS)
    try:
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description or []]
        if row_limit is None:
            rows = cursor.fetchall()
            truncated = False
        else:
            rows = cursor.fetchmany(row_limit + 1)
            truncated = len(rows) > row_limit
            rows = rows[:row_limit]
        if time.monotonic() > deadline:
            # A single expensive step (say, upper() on a huge string) can run
            # long between progress-handler checks. The official scorer counts
            # wall time, so a late finish is still a timeout.
            return done("timeout", error=f"query finished after the {deadline - start:.1f}s limit")
        return done("ok", rows=rows, columns=columns, truncated=truncated)
    except Exception as exc:  # sqlite3 errors, decode errors, overflow, ...
        if timed_out:
            return done("timeout", error=f"query exceeded {deadline - start:.1f}s")
        return done("error", error=f"{type(exc).__name__}: {exc}")
    finally:
        conn.close()
