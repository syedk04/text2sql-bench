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
        i
        for i, (p, q) in enumerate(zip(preds, questions, strict=True))
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


def _cmd_gate_gold(args: argparse.Namespace) -> int:
    from text2sql.data.bird import load_questions
    from text2sql.data.manifest import load_manifest
    from text2sql.eval.gates import gate_gold

    qpath = Path(args.questions) if args.questions else config.questions_path()
    questions = load_questions(qpath)
    if args.manifest:
        wanted = set(load_manifest(args.manifest))
        questions = [q for q in questions if q.question_id in wanted]

    def progress(done: int, total: int, _r: object) -> None:
        if done % 50 == 0 or done == total:
            print(f"  checked {done}/{total}", flush=True)

    import sqlite3

    report = gate_gold(questions, timeout_s=args.timeout, progress=progress)
    print(
        f"gold vs gold: {report.passed}/{report.n} correct "
        f"(timeout {args.timeout:g}s, SQLite {sqlite3.sqlite_version})"
    )
    for r in report.failures:
        print(f"  FAIL question {r.question_id} ({r.db_id}): {r.error}")
    print("slowest: " + ", ".join(f"{r.question_id}={r.elapsed_s:.1f}s" for r in report.slowest))
    print("GATE PASSED" if report.ok else "GATE FAILED")
    return 0 if report.ok else 1


def _cmd_gate_official(args: argparse.Namespace) -> int:
    import json

    from text2sql.data.bird import load_questions
    from text2sql.eval.ex import load_official_predictions, score_many
    from text2sql.eval.official import (
        BASELINES,
        compare_vectors,
        fetch_baseline,
        official_dir,
        run_official,
        write_gold_inputs,
    )

    baseline = BASELINES[args.baseline]
    label = "legacy" if args.legacy else "hf"
    workdir = official_dir() / "runs" / f"{baseline.name}-{label}"
    if args.legacy:
        legacy = config.bird_dir() / "legacy"
        questions = load_questions(
            legacy / "mini_dev_sqlite.json", expect_total=500, allow_duplicate_ids=True
        )
        gold_path = legacy / "mini_dev_sqlite_gold.sql"
        gold_source = "zip legacy files"
    else:
        questions = load_questions(config.questions_path(), expect_total=500)
        gold_path, _ = write_gold_inputs(questions, workdir)
        gold_source = "pinned Hugging Face file"
    pred_path = fetch_baseline(baseline.name)
    preds = load_official_predictions(pred_path)
    if len(preds) != len(questions):
        print(f"error: baseline has {len(preds)} entries, expected {len(questions)}")
        return 1

    print(f"official scorer on {baseline.name} ({gold_source}), one worker, {args.timeout:g}s ...")
    official = run_official(pred_path, gold_path, timeout_s=args.timeout)
    print("our scorer ...", flush=True)
    ours_results = score_many(questions, [p.sql for p in preds], timeout_s=args.timeout)
    agreement = compare_vectors(baseline.name, official, [r.correct for r in ours_results])

    print(
        f"official EX {agreement.official_ex:.2f}  ours EX {agreement.ours_ex:.2f}  "
        f"published {baseline.published_ex:.2f}"
    )
    print(f"agreement: {agreement.n - len(agreement.disagreements)}/{agreement.n}")
    for i in agreement.disagreements:
        r = ours_results[i]
        print(
            f"  position {i} question {r.question_id}: official={official[i]} "
            f"ours={r.correct} ({r.pred_status}/{r.gold_status}) {r.error or ''}"
        )
    workdir.mkdir(parents=True, exist_ok=True)
    report_path = workdir / "agreement.json"
    report_path.write_text(
        json.dumps(
            {
                "baseline": baseline.name,
                "gold": gold_source,
                "timeout_s": args.timeout,
                "official_ex": agreement.official_ex,
                "ours_ex": agreement.ours_ex,
                "published_ex": baseline.published_ex,
                "disagreements": agreement.disagreements,
                "official": official,
                "ours": [r.__dict__ for r in ours_results],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"details: {report_path}")
    if args.legacy:
        delta = agreement.official_ex - baseline.published_ex
        verdict = "within" if abs(delta) <= 1.0 else "OUTSIDE"
        print(f"Gate B (advisory): official reproduction {delta:+.2f} points, {verdict} +-1.0")
    print("GATE A PASSED" if agreement.ok else "GATE A FAILED")
    return 0 if agreement.ok else 1


def _cmd_update_readme(args: argparse.Namespace) -> int:
    from text2sql.results.report import render_markdown, replace_between_markers, update_readme
    from text2sql.results.store import list_runs, read_run, runs_dir

    root = Path(args.runs_dir) if args.runs_dir else runs_dir()
    readme = Path(args.readme) if args.readme else config.PROJECT_ROOT / "README.md"
    runs = [read_run(p) for p in list_runs(root)]
    if args.check:
        text = readme.read_text(encoding="utf-8")
        if replace_between_markers(text, render_markdown(runs)) != text:
            print(f"{readme} results table is out of date; run `text2sql update-readme`")
            return 1
        print(f"{readme} is up to date ({len(runs)} run(s))")
        return 0
    changed = update_readme(readme, runs)
    print(f"{'updated' if changed else 'unchanged'}: {readme} ({len(runs)} run(s))")
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

    p = sub.add_parser("gate-gold", help="score every gold query against itself (must be 100%%)")
    p.add_argument("--questions", help="question file (default: data/bird/mini_dev_sqlite.json)")
    p.add_argument("--manifest", help="only check the questions listed in this manifest")
    p.add_argument("--timeout", type=float, default=30.0, help="seconds per question")
    p.set_defaults(handler=_cmd_gate_gold)

    from text2sql.eval.official import BASELINES

    p = sub.add_parser(
        "gate-official",
        help="check our scorer against the official BIRD script on a published baseline",
    )
    p.add_argument("--baseline", choices=sorted(BASELINES), default="gpt-4")
    p.add_argument(
        "--legacy",
        action="store_true",
        help="score against the zip's original gold files (Gate B, reproduces published EX)",
    )
    p.add_argument("--timeout", type=float, default=30.0, help="seconds per question")
    p.set_defaults(handler=_cmd_gate_official)

    p = sub.add_parser("update-readme", help="rewrite the README results table from runs/")
    p.add_argument("--runs-dir", help="folder of runs (default: runs/)")
    p.add_argument("--readme", help="README to update (default: README.md)")
    p.add_argument("--check", action="store_true", help="only check the table is current")
    p.set_defaults(handler=_cmd_update_readme)

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
