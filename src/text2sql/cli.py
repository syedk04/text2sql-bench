"""Command line entry point: ``text2sql <command> ...``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from text2sql import __version__, config


def _cmd_download_questions(args: argparse.Namespace) -> int:
    from text2sql.data.bird import HF_REVISION, download_questions, load_questions

    dest = Path(args.dest) if args.dest else config.questions_path()
    from_file = Path(args.from_file) if args.from_file else None
    download_questions(dest, from_file=from_file)
    qs = load_questions(dest)
    source = from_file or f"Hugging Face revision {HF_REVISION[:12]}"
    print(f"wrote {len(qs)} questions to {dest} (from {source})")
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
