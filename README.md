# text2sql-bench

A text-to-SQL agent measured on [BIRD Mini-Dev](https://github.com/bird-bench/mini_dev)
(500 SQLite questions over 11 databases), with a read-only SQL executor and an
execution-accuracy evaluator checked question by question against the official
BIRD script.

Status: the groundwork is done and checked. Data loading, the safe executor, the
evaluator, the results table and the model client (rate limiting, disk cache,
call log) are built and tested. No model has been run yet, so the results table
below is empty.

## Results

<!-- results:start -->
No runs yet.
<!-- results:end -->

EX is execution accuracy as defined by BIRD. The interval is a Wilson 95%
interval; at n=500 it is roughly plus or minus 4.4 points around 50%.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run text2sql download-questions      # 500 questions, pinned Hugging Face revision
uv run text2sql download-dbs            # 11 SQLite databases from minidev.zip (about 760 MiB)
uv run text2sql build-manifest          # rebuilds manifests/dev50.json (already committed)
```

- `download-questions` fetches `birdsql/bird_mini_dev` at revision
  `f65faf4ae3b638c1fa6df1d3370c8d92c8366301` and checks its sha256. If Python
  cannot reach Hugging Face (for example behind TLS-inspecting antivirus),
  download the file by hand and pass `--from-file PATH`; the same checks apply.
- `download-dbs` downloads `minidev.zip` from the BIRD site, or takes
  `--zip PATH` for a copy you already have. Only the SQLite files and their
  column description CSVs are extracted, into `data/bird/dev_databases/`.
- Everything downloaded lives under `data/` (gitignored). Set `T2S_DATA_DIR`
  to put it elsewhere.

`manifests/dev50.json` is the fixed 50-question practice slice (15 simple,
25 moderate, 10 challenging, spread round-robin over all 11 databases, seed
20261002). Day-to-day work uses it; only full 500-question runs go in the table.

## Safe executor

Model-written SQL runs through four layers, any one of which blocks ordinary
writes on its own:

1. a sqlglot parse that accepts exactly one statement, which must be a query
   (SELECT, WITH, UNION, INTERSECT, EXCEPT), with no write, DDL, PRAGMA, ATTACH
   or transaction node anywhere inside it;
2. the database file opened through a `mode=ro&immutable=1` SQLite URI, which
   also means no lock or `-wal`/`-shm` files appear next to it (the benchmark
   databases never change, so treating them as immutable is safe);
3. `PRAGMA query_only = ON`;
4. a SQLite authorizer that only allows reads, function calls and recursive
   CTEs (and refuses `load_extension`).

Queries stop at a deadline through a SQLite progress handler, which also works
on Windows and covers the time spent fetching rows. A query that still finishes
after its deadline (one very expensive step can run between checks) is reported
as a timeout, matching the official wall-clock limit. Strings and blobs a query
builds are capped at 100 MB. An optional row cap reports whether the result was
cut off. SQL nested too deeply for the parser is rejected rather than allowed
to crash a batch. The tests try inserts, updates, deletes, drops,
ATTACH, VACUUM, PRAGMA changes and DML hidden in a CTE, in a folder whose path
contains spaces, and check the database file is byte for byte unchanged.

## Evaluator

`text2sql eval-preds --preds FILE` scores a prediction file in the official
BIRD format with the official rules:

- run the predicted SQL, then the gold SQL, and fetch all rows of each;
- correct if and only if the two results are equal as sets of rows (row order
  and duplicate rows are ignored, column order is not);
- one 30 second budget covers both queries;
- an error or timeout on either side scores 0, and every question stays in the
  denominator.

Two deliberate differences, both on the safe side: queries go through the
executor above, so a prediction the guard rejects scores 0; and an empty
prediction scores 0 without running anything. Neither changes any score on the
published baselines (see Gate A).

## Gates

These were run locally on Windows 11 with the real data. Commands:

```
uv sync --group official                          # adds func-timeout for the official script
uv run text2sql gate-gold                         # gold vs gold, all 500
uv run text2sql gate-official --baseline gpt-4    # Gate A
uv run text2sql gate-official --baseline gpt-4 --legacy   # Gate A plus Gate B
uv run pytest --run-slow                          # the same gates as tests
```

The official `evaluation_ex.py` and `evaluation_utils.py` have no licence, so
they are not copied here. `gate-official` downloads them at commit
`abd11b6db92a1c9f809b32f7564c7c71b34d67f0`, checks their sha256 and runs their
own `package_sqls`, `run_sqls_parallel` (one worker, 30 s timeout) and
`sort_results`. The only shims are empty stand-ins for the MySQL and PostgreSQL
drivers they import, initialising one global their `__main__` block normally
creates, and reading their input files as UTF-8 so Windows matches Linux.

### Gold vs gold

Every gold query scored against itself must be correct.

| SQLite | Python | Result at 30 s | Slowest question |
|---|---|---:|---|
| 3.43.1 | 3.12.0 | 500/500 | 4.0 s |
| 3.45.1 | 3.12.3 | 498/500 | 30 s (timeout) |

The two failures on SQLite 3.45.1 are questions 518 (`card_games`) and 701
(`codebase_community`). They are not scoring errors: the gold queries
themselves do not finish within 30 seconds. Both join against a subquery in the
FROM clause. SQLite up to 3.43.2 materialises that subquery once; from 3.44.0
the query planner runs it as a co-routine inside the join loop and re-evaluates
it for every outer row. I checked this with the official SQLite command line
builds: 3.31.1, 3.40.0, 3.41.2, 3.42.0 and 3.43.2 finish each query in under
half a second, 3.44.0, 3.44.2 and 3.45.0 do not finish within 20 seconds.
Through Python's sqlite3, question 518 takes 213 seconds on 3.45.1, and
question 701 did not finish within 5 minutes on 3.45.1 or within 25 minutes on
3.53.1 (Python 3.12.14).

So on SQLite 3.44 or newer, questions 518 and 701 score 0 for every system,
under the official script as well as this one. Scores in this repository record
the SQLite version of the run so this stays visible.

### Gate A: agreement with the official script (hard gate)

Published baseline predictions from the BIRD repository, scored against the
pinned Hugging Face gold by the official script and by this evaluator:

| Baseline | SQLite | Official EX | This evaluator | Per-question agreement |
|---|---|---:|---:|---:|
| gpt-4 | 3.45.1 | 47.80 | 47.80 | 500/500 |
| gpt-4 | 3.43.1 | 47.80 | 47.80 | 500/500 |
| llama-3-70b | 3.45.1 | 40.60 | 40.60 | 500/500 |
| llama-3-70b | 3.43.1 | 40.60 | 40.60 | 500/500 |

The gold timeouts above hit both scorers in the same way, so agreement holds on
both SQLite versions.

### Gate B: reproducing the published numbers (advisory)

The published numbers were produced against the question and gold files inside
`minidev.zip`, which differ slightly from the Hugging Face copy (see below).
The official script on those legacy files, within plus or minus 1.0 point of
the published EX:

| Baseline | SQLite | Official EX | Published | Difference |
|---|---|---:|---:|---:|
| gpt-4 | 3.45.1 | 48.40 | 47.80 | +0.60 |
| gpt-4 | 3.43.1 | 48.40 | 47.80 | +0.60 |
| llama-3-70b | 3.45.1 | 41.40 | 40.80 | +0.60 |
| llama-3-70b | 3.43.1 | 41.40 | 40.80 | +0.60 |

Both are within the band. This evaluator agrees with the official script on all
500 legacy questions as well. Oddly, the Hugging Face gold reproduces the
published gpt-4 figure exactly (47.80, Gate A), while the zip's own files give
48.40; I have not found which exact files the published table was computed on.

## Data notes

- Questions and gold SQL come from the Hugging Face copy; the zip is used only
  for the database files and the Gate B check. The two copies differ
  ([bird-bench/mini_dev issue #40](https://github.com/bird-bench/mini_dev/issues/40)):
  the zip lists questions 137 and 138 twice and lacks 119 and 120, which puts
  the last 16 positions in a different order, and the gold SQL of questions
  879 and 1322 was corrected on Hugging Face. 16 question texts and 6
  difficulty labels differ as a result.
- Gold SQL is never edited here. If a gold query fails a check, the check is
  what gets fixed.

## Tests

```
uv run ruff check .
uv run pytest -q
```

CI runs on Ubuntu and Windows with small synthetic databases only. Tests that
need the real data are marked `bird` and skip when `data/bird` is missing; the
full gates are also marked `slow` and only run with `--run-slow`. With SQLite
3.43.1, `uv run pytest --run-slow` passes every test, the slow gates taking about 9 minutes; on
SQLite 3.44 or newer the gold vs gold test fails on questions 518 and 701 for
the reason given above.

## Attribution

BIRD Mini-Dev: Li et al., "Can LLM Already Serve as A Database Interface? A BIg
Bench for Large-Scale Database Grounded Text-to-SQLs", NeurIPS 2023, and the
[mini_dev](https://github.com/bird-bench/mini_dev) release. The dataset is
licensed CC BY-SA 4.0. It is downloaded at runtime and not redistributed here;
`manifests/dev50.json` stores only question ids, database names and
difficulty labels.

## License

Code: MIT (see LICENSE). Data: CC BY-SA 4.0, from its original source.
