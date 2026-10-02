"""BIRD Mini-Dev SQLite databases.

The databases are only distributed inside ``minidev.zip`` (about 760 MiB), which
also carries MySQL and PostgreSQL dumps we don't need. We extract just the
SQLite files and their column-description CSVs into ``data/bird/dev_databases``,
plus the zip's own question/gold files into ``data/bird/legacy`` (those are the
files the published baseline numbers were scored against, and they differ from
the Hugging Face copy in a handful of records; see bird-bench/mini_dev issue #40).
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

ZIP_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip"
LEGACY_FILES = ("mini_dev_sqlite.json", "mini_dev_sqlite_gold.sql")

# Matched against the member path after the top-level folder(s), so the layout
# can shift (minidev/MINIDEV/... today) without breaking extraction.
_SQLITE_RE = re.compile(r"(?:^|/)dev_databases/([A-Za-z0-9_]+)/([A-Za-z0-9_]+)\.sqlite$")
_DESC_RE = re.compile(
    r"(?:^|/)dev_databases/([A-Za-z0-9_]+)/database_description/([^/]+\.csv)$"
)


@dataclass
class ExtractReport:
    databases: list[str] = field(default_factory=list)
    descriptions: int = 0
    legacy: list[str] = field(default_factory=list)
    skipped_existing: int = 0


def _is_sqlite_dir(member: str) -> bool:
    # The MySQL/PostgreSQL material lives in sibling folders (MINIDEV_mysql, ...).
    parts = member.split("/")
    return not any(p.lower().endswith(("_mysql", "_postgresql")) for p in parts)


def _copy_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path) -> bool:
    """Copy one member to ``target``; returns False if an identical-size file exists."""
    if target.exists() and target.stat().st_size == info.file_size:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    with zf.open(info) as src, open(tmp, "wb") as out:
        shutil.copyfileobj(src, out, length=1 << 20)
    tmp.replace(target)
    return True


def extract_sqlite(zip_path: Path, bird_root: Path) -> ExtractReport:
    """Extract SQLite databases, description CSVs and legacy files.

    Target paths are built from the regex captures, never from the raw member
    name, so a crafted archive cannot write outside ``bird_root``.
    """
    report = ExtractReport()
    db_root = bird_root / "dev_databases"
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if info.is_dir() or not _is_sqlite_dir(name):
                continue
            target: Path | None = None
            if m := _SQLITE_RE.search(name):
                db_id, stem = m.groups()
                if db_id != stem:
                    continue
                target = db_root / db_id / f"{db_id}.sqlite"
                report.databases.append(db_id)
            elif m := _DESC_RE.search(name):
                db_id, csv_name = m.groups()
                if csv_name in (".", "..") or "\\" in csv_name:
                    continue
                target = db_root / db_id / "database_description" / csv_name
                report.descriptions += 1
            else:
                base = name.rsplit("/", 1)[-1]
                if base in LEGACY_FILES and "dev_databases" not in name:
                    target = bird_root / "legacy" / base
                    report.legacy.append(base)
            if target is not None and not _copy_member(zf, info, target):
                report.skipped_existing += 1
    report.databases.sort()
    return report


def download_zip(dest: Path) -> Path:
    from text2sql.net import download_file

    return download_file(ZIP_URL, dest, timeout=300.0)


def list_databases(db_root: Path) -> list[str]:
    if not db_root.is_dir():
        return []
    return sorted(
        p.name for p in db_root.iterdir() if (p / f"{p.name}.sqlite").is_file()
    )


def table_names(db_file: Path) -> list[str]:
    """Open a database read-only and list its tables (a cheap integrity check)."""
    from text2sql.sql.executor import readonly_uri

    uri = readonly_uri(db_file)
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]
