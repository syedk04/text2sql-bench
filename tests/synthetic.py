"""Small made-up SQLite databases matching tests/fixtures/questions.json.

CI never sees the real BIRD data, so tests run against these instead.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMAS: dict[str, list[str]] = {
    "shop": [
        "CREATE TABLE item (id INTEGER PRIMARY KEY, name TEXT, price REAL, category TEXT)",
        "INSERT INTO item (name, price, category) VALUES "
        "('apple', 1.5, 'fruit'), ('melon', 6.0, 'fruit'), ('soap', 7.25, 'home'), "
        "('broom', 12.0, 'home'), ('pen', 2.0, 'office')",
        "CREATE INDEX item_category ON item (category)",
    ],
    "school": [
        "CREATE TABLE student (id INTEGER PRIMARY KEY, name TEXT, grade INTEGER, score REAL)",
        "INSERT INTO student (name, grade, score) VALUES "
        "('Ana', 10, 88.5), ('Ben', 11, 92.0), ('Cy', 10, 75.0), ('Dee', 12, 99.5)",
    ],
    "zoo": [
        "CREATE TABLE animal (id INTEGER PRIMARY KEY, name TEXT, species TEXT, legs INTEGER)",
        "INSERT INTO animal (name, species, legs) VALUES "
        "('Leo', 'lion', 4), ('Nala', 'lion', 4), ('Polly', 'parrot', 2), "
        "('Dumbo', 'elephant', 4), ('Kiwi', 'parrot', 2)",
    ],
}


def make_db(path: Path, statements: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        for stmt in statements:
            conn.execute(stmt)
        conn.commit()
    finally:
        conn.close()
    return path


def make_databases(db_root: Path) -> dict[str, Path]:
    return {
        db_id: make_db(db_root / db_id / f"{db_id}.sqlite", stmts)
        for db_id, stmts in SCHEMAS.items()
    }
