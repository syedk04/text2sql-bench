"""Cross-check our evaluator against the official BIRD Mini-Dev scorer.

The official ``evaluation_ex.py`` / ``evaluation_utils.py`` from
github.com/bird-bench/mini_dev carry no licence, so they are not copied into this
repository. Instead they are downloaded at a pinned commit into the gitignored
data folder and loaded with importlib, and their own functions do the scoring:
``package_sqls`` to read the inputs, ``run_sqls_parallel`` (one worker process,
30 s ``func_timeout``) to execute, ``sort_results`` to order the results.

Three small shims are needed to call them as a library, none of which touches
the scoring logic:

* ``evaluation_utils`` imports ``psycopg2`` and ``pymysql`` at module level for
  the MySQL/PostgreSQL paths; empty stand-in modules are written next to the
  scripts so the SQLite path can be imported without those drivers;
* ``exec_result`` is a global that only the script's ``__main__`` block creates,
  so it is initialised before running;
* the scripts open files with the platform default encoding; ``open`` inside
  ``evaluation_utils`` is pointed at UTF-8 so Windows reads the same text that
  Linux (where the published numbers were produced) does.

Gate A (hard): per-question agreement between this scorer and ours on published
baseline predictions, against the pinned Hugging Face gold.
Gate B (advisory): the official scorer on the zip's legacy gold files should
reproduce the published total within a point.
"""

from __future__ import annotations

import builtins
import functools
import importlib.util
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

from text2sql import config
from text2sql.data.bird import Question

OFFICIAL_REPO = "bird-bench/mini_dev"
OFFICIAL_SHA = "abd11b6db92a1c9f809b32f7564c7c71b34d67f0"
RAW_BASE = f"https://raw.githubusercontent.com/{OFFICIAL_REPO}/{OFFICIAL_SHA}"

SCRIPTS: dict[str, str] = {
    "evaluation/evaluation_ex.py": (
        "da1bbcd4530be83692d7c650c814ea9704bb710d0c953eb75d02ccb38233cf89"
    ),
    "evaluation/evaluation_utils.py": (
        "f6943d249caac5aeaef9bce21d43dbf29dcef85a0c965a76df032a9542f308bf"
    ),
}


@dataclass(frozen=True)
class Baseline:
    name: str
    path: str  # inside the official repo
    sha256: str
    published_ex: float  # SQLite EX in the mini_dev README


BASELINES: dict[str, Baseline] = {
    "gpt-4": Baseline(
        "gpt-4",
        "llm/exp_result/sql_output_kg/predict_mini_dev_gpt-4_sqlite.json",
        "afb13727cd9ba01aa4a4761769567b80746581e13895b63e54e6a5cc38d0c1a6",
        47.80,
    ),
    "llama-3-70b": Baseline(
        "llama-3-70b",
        "llm/exp_result/sql_output_kg/predict_mini_dev_meta-llama-3-70b-instruct-2_sqlite.json",
        "d4e0efbafaa26435c2110559b7a24c91efa272ce82b6ec6d2a7176019582e1ae",
        40.80,
    ),
}

_STUB_MODULES = ("psycopg2", "pymysql")


class OfficialError(RuntimeError):
    pass


def official_dir() -> Path:
    return config.data_dir() / "official" / OFFICIAL_SHA


def _fetch_pinned(rel_path: str, sha256: str, dest: Path) -> Path:
    from text2sql.net import fetch_bytes, sha256_bytes, sha256_file, write_atomic

    if dest.is_file() and sha256_file(dest) == sha256:
        return dest
    data = fetch_bytes(f"{RAW_BASE}/{rel_path}")
    digest = sha256_bytes(data)
    if digest != sha256:
        raise OfficialError(f"sha256 mismatch for {rel_path}: got {digest}, expected {sha256}")
    write_atomic(dest, data)
    return dest


def write_stub_modules(root: Path) -> Path:
    stubs = root / "stubs"
    stubs.mkdir(parents=True, exist_ok=True)
    for name in _STUB_MODULES:
        stub = stubs / f"{name}.py"
        if not stub.exists():
            stub.write_text(
                f'"""Stand-in for {name}: only the SQLite path of the BIRD scorer is used."""\n',
                encoding="utf-8",
            )
    return stubs


def fetch_official_scripts(root: Path | None = None) -> Path:
    root = root or official_dir()
    for rel, digest in SCRIPTS.items():
        _fetch_pinned(rel, digest, root / Path(rel).name)
    write_stub_modules(root)
    return root


def fetch_baseline(name: str, root: Path | None = None) -> Path:
    if name not in BASELINES:
        raise OfficialError(f"unknown baseline {name!r}; choose from {sorted(BASELINES)}")
    b = BASELINES[name]
    root = root or official_dir()
    return _fetch_pinned(b.path, b.sha256, root / "baselines" / Path(b.path).name)


def write_gold_inputs(questions: Sequence[Question], workdir: Path) -> tuple[Path, Path]:
    """Write questions in the official input formats: ``SQL<TAB>db_id`` lines and a
    difficulty JSONL, both in question order."""
    workdir.mkdir(parents=True, exist_ok=True)
    gold = workdir / "gold.sql"
    diff = workdir / "diff.jsonl"
    with open(gold, "w", encoding="utf-8", newline="\n") as g, open(
        diff, "w", encoding="utf-8", newline="\n"
    ) as d:
        for q in questions:
            if any(c in q.gold_sql for c in "\t\r\n"):
                raise OfficialError(
                    f"gold SQL for {q.question_id} contains a tab or newline and cannot be "
                    "written in the official one-line format"
                )
            g.write(f"{q.gold_sql}\t{q.db_id}\n")
            d.write(json.dumps({"question_id": q.question_id, "difficulty": q.difficulty}) + "\n")
    return gold, diff


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise OfficialError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_official(
    root: Path | None = None, *, fetch: bool = True
) -> tuple[ModuleType, ModuleType]:
    """Import the pinned official scripts as modules (fetching them if needed).

    ``fetch=False`` loads whatever scripts are already in ``root`` without the
    download and checksum step; the tests use it with stand-in scripts.
    """
    root = root or official_dir()
    if fetch:
        # The real evaluation_ex.py imports func_timeout at the top; say how to
        # get it rather than failing inside their import.
        if importlib.util.find_spec("func_timeout") is None:
            raise OfficialError(
                "the official scorer needs func-timeout: run `uv sync --group official`"
            )
        fetch_official_scripts(root)
    else:
        write_stub_modules(root)
    # Worker processes are spawned (Windows) and inherit sys.path, so they can
    # import the scripts and stubs by name too.
    for p in (str(root / "stubs"), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    utils = _load_module("evaluation_utils", root / "evaluation_utils.py")
    utils.open = functools.partial(builtins.open, encoding="utf-8")  # type: ignore[attr-defined]
    ex = _load_module("evaluation_ex", root / "evaluation_ex.py")
    return utils, ex


def run_official(
    pred_path: Path,
    gold_path: Path,
    db_root: Path | None = None,
    *,
    timeout_s: float = 30.0,
    root: Path | None = None,
    fetch: bool = True,
) -> list[int]:
    """Score with the official functions; returns the 0/1 vector in question order."""
    utils, ex = load_official(root, fetch=fetch)
    db_root_str = str(db_root or config.databases_dir()).replace("\\", "/").rstrip("/") + "/"
    pred_queries, _ = utils.package_sqls(str(pred_path), db_root_str, mode="pred")
    gt_queries, db_paths = utils.package_sqls(str(gold_path), db_root_str, mode="gt")
    pairs = list(zip(pred_queries, gt_queries))  # noqa: B905 - official truncation semantics
    ex.exec_result = []  # type: ignore[attr-defined]  # set by the script's __main__
    ex.run_sqls_parallel(
        pairs, db_places=db_paths, num_cpus=1, meta_time_out=timeout_s, sql_dialect="SQLite"
    )
    results = utils.sort_results(ex.exec_result)
    if [r["sql_idx"] for r in results] != list(range(len(pairs))):
        raise OfficialError(
            f"official run returned {len(results)} results for {len(pairs)} pairs"
        )
    return [int(r["res"]) for r in results]


@dataclass
class Agreement:
    baseline: str
    n: int
    official: list[int]
    ours: list[int]
    disagreements: list[int] = field(default_factory=list)  # positions

    @property
    def official_ex(self) -> float:
        return 100.0 * sum(self.official) / self.n if self.n else 0.0

    @property
    def ours_ex(self) -> float:
        return 100.0 * sum(self.ours) / self.n if self.n else 0.0

    @property
    def ok(self) -> bool:
        return self.n > 0 and not self.disagreements


def compare_vectors(baseline: str, official: Sequence[int], ours: Sequence[int]) -> Agreement:
    if len(official) != len(ours):
        raise OfficialError(f"length mismatch: official {len(official)}, ours {len(ours)}")
    diffs = [i for i, (a, b) in enumerate(zip(official, ours, strict=True)) if a != b]
    return Agreement(baseline, len(ours), list(official), list(ours), diffs)
