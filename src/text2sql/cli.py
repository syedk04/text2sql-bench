"""Command line entry point: ``text2sql <command> ...``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from text2sql import __version__, config

DEFAULT_MANIFEST = config.PROJECT_ROOT / "manifests" / "dev50.json"


def _cmd_download_questions(args: argparse.Namespace) -> int:
    from text2sql.data.bird import HF_REVISION, download_questions, load_questions

    dest = Path(args.dest) if args.dest else config.questions_path()
    from_file = Path(args.from_file) if args.from_file else None
    download_questions(dest, from_file=from_file)
    qs = load_questions(dest)
    source = from_file or f"Hugging Face revision {HF_REVISION[:12]}"
    print(f"wrote {len(qs)} questions to {dest} (from {source})")
    return 0


def _cmd_download_dbs(args: argparse.Namespace) -> int:
    from text2sql.data.databases import download_zip, extract_sqlite, list_databases

    if args.zip:
        zip_path = Path(args.zip)
        if not zip_path.is_file():
            print(f"zip not found: {zip_path}")
            return 1
    else:
        zip_path = config.data_dir() / "raw" / "minidev.zip"
        if not zip_path.is_file():
            print(f"downloading minidev.zip (about 760 MiB) to {zip_path} ...")
            download_zip(zip_path)
    report = extract_sqlite(zip_path, config.bird_dir())
    present = list_databases(config.databases_dir())
    print(
        f"databases: {len(report.databases)} in zip, {len(present)} on disk; "
        f"{report.descriptions} description CSVs; legacy files: {', '.join(report.legacy)}; "
        f"{report.skipped_existing} already up to date"
    )
    return 0


def _cmd_build_manifest(args: argparse.Namespace) -> int:
    from text2sql.data.bird import HF_FILE, HF_REPO, HF_REVISION, load_questions
    from text2sql.data.manifest import build_manifest, write_manifest
    from text2sql.net import sha256_file

    qpath = Path(args.questions) if args.questions else config.questions_path()
    questions = load_questions(qpath)
    source = {
        "hf_repo": HF_REPO,
        "hf_revision": HF_REVISION,
        "file": HF_FILE,
        "sha256": sha256_file(qpath),
    }
    manifest = build_manifest(questions, seed=args.seed, source=source)
    out = Path(args.out)
    write_manifest(manifest, out)
    print(f"wrote {manifest['n']} question ids to {out}")
    return 0


def _cmd_eval_preds(args: argparse.Namespace) -> int:
    import json

    from text2sql.data.bird import load_questions
    from text2sql.eval.ex import (
        breakdown,
        format_breakdown,
        load_official_predictions,
        score_many,
    )

    qpath = Path(args.questions) if args.questions else config.questions_path()
    questions = load_questions(qpath)
    preds = load_official_predictions(args.preds)
    if len(preds) != len(questions):
        print(f"error: {len(preds)} predictions but {len(questions)} questions")
        return 1
    malformed = [i for i, p in enumerate(preds) if not p.well_formed]
    db_mismatch = [
        i for i, (p, q) in enumerate(zip(preds, questions, strict=True))
        if p.well_formed and p.db_id != q.db_id
    ]

    def progress(done: int, total: int, _r: object) -> None:
        if done % 50 == 0 or done == total:
            print(f"  scored {done}/{total}", flush=True)

    results = score_many(
        questions, [p.sql for p in preds], timeout_s=args.timeout, progress=progress
    )
    print(format_breakdown(breakdown(results)))
    statuses: dict[str, int] = {}
    for r in results:
        key = f"pred:{r.pred_status}/gold:{r.gold_status}"
        statuses[key] = statuses.get(key, 0) + 1
    print("outcomes: " + ", ".join(f"{k}={v}" for k, v in sorted(statuses.items())))
    if malformed:
        print(f"warning: {len(malformed)} malformed prediction entries at positions {malformed}")
    if db_mismatch:
        print(f"warning: prediction db_id differs from gold at positions {db_mismatch}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            for r in results:
                fh.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
        print(f"per-question results written to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="text2sql", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser(
        "download-questions",
        help="fetch the pinned BIRD Mini-Dev SQLite question file from Hugging Face",
    )
    p.add_argument(
        "--from-file",
        help="use a file you downloaded yourself (still checked against the pinned sha256)",
    )
    p.add_argument("--dest", help="output path (default: data/bird/mini_dev_sqlite.json)")
    p.set_defaults(handler=_cmd_download_questions)

    p = sub.add_parser(
        "download-dbs",
        help="extract the BIRD Mini-Dev SQLite databases from minidev.zip (downloads it if needed)",
    )
    p.add_argument("--zip", help="path to an already downloaded minidev.zip")
    p.set_defaults(handler=_cmd_download_dbs)

    from text2sql.data.manifest import DEFAULT_SEED

    p = sub.add_parser("build-manifest", help="draw the fixed 50-question dev slice")
    p.add_argument("--questions", help="question file (default: data/bird/mini_dev_sqlite.json)")
    p.add_argument("--out", default=str(DEFAULT_MANIFEST), help="output manifest path")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.set_defaults(handler=_cmd_build_manifest)

    p = sub.add_parser(
        "eval-preds",
        help="score a prediction file in the official BIRD format (JSON object of SQL strings)",
    )
    p.add_argument("--preds", required=True, help="prediction JSON")
    p.add_argument("--questions", help="question file (default: data/bird/mini_dev_sqlite.json)")
    p.add_argument("--timeout", type=float, default=30.0, help="seconds per question")
    p.add_argument("--out", help="optional JSONL file for per-question results")
    p.set_defaults(handler=_cmd_eval_preds)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
