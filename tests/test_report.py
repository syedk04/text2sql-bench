import pytest

from text2sql.results.report import (
    EMPTY_TABLE,
    END_MARKER,
    START_MARKER,
    ReportError,
    render_markdown,
    replace_between_markers,
    summarize,
    update_readme,
)
from text2sql.results.store import QuestionResult, Run, RunMeta, write_run


def _run(track="T0", split="dev50", correct=(1, 0, 1, 1), model="m-1") -> Run:
    diffs = ("simple", "moderate", "challenging", "moderate")
    results = [
        QuestionResult(
            question_id=i,
            db_id="shop",
            difficulty=diffs[i % 4],
            pred_sql="SELECT 1",
            correct=c,
            status="ok",
            prompt_tokens=1000,
            completion_tokens=200,
            calls=1 + i % 2,
            latency_s=2.0,
            cost_usd=0.001,
        )
        for i, c in enumerate(correct)
    ]
    meta = RunMeta(
        run_id=f"{track}-{split}", track=track, provider="fake", model=model, split=split
    )
    return Run(meta, results)


def test_summarize():
    s = summarize(_run())
    assert (s.n, s.correct, s.ex) == (4, 3, 75.0)
    assert s.ci_low < 75.0 < s.ci_high
    assert s.by_difficulty == {"simple": (1, 1), "moderate": (1, 2), "challenging": (1, 1)}
    assert s.tokens_per_q == 1200
    assert s.calls_per_q == 1.5
    assert s.latency_per_q == 2.0
    assert s.cost_per_q == pytest.approx(0.001)


def test_render_markdown():
    table = render_markdown([_run("T1"), _run("T0")])
    lines = table.splitlines()
    assert lines[0].startswith("| Track | Model | Split | n | EX % [95% CI] |")
    assert lines[1].count("---") == 12
    assert lines[2].startswith("| T0 |") and lines[3].startswith("| T1 |")
    assert "75.0 [" in lines[2]
    assert "| 100.0 | 50.0 | 100.0 |" in lines[2]
    assert "1,200" in lines[2] and "$0.0010" in lines[2]


def test_render_empty():
    assert render_markdown([]) == EMPTY_TABLE


def test_missing_difficulty_shows_dash():
    run = _run(correct=(1,))
    assert "| 100.0 | - | - |" in render_markdown([run])


def test_replace_between_markers():
    text = f"# T\n\n{START_MARKER}\nold\n{END_MARKER}\n\nrest\n"
    out = replace_between_markers(text, "NEW")
    assert out == f"# T\n\n{START_MARKER}\nNEW\n{END_MARKER}\n\nrest\n"
    assert replace_between_markers(out, "NEW") == out


@pytest.mark.parametrize(
    "text",
    ["no markers", f"{END_MARKER}\n{START_MARKER}", f"{START_MARKER}{END_MARKER}{START_MARKER}"],
)
def test_bad_markers(text):
    with pytest.raises(ReportError):
        replace_between_markers(text, "x")


def test_update_readme_keeps_line_endings(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_bytes(f"# T\r\n{START_MARKER}\r\nold\r\n{END_MARKER}\r\n".encode())
    assert update_readme(readme, [_run()]) is True
    raw = readme.read_bytes().decode()
    assert "| T0 |" in raw and "old" not in raw
    assert "\n" not in raw.replace("\r\n", "")
    assert update_readme(readme, [_run()]) is False


def test_cli_update_readme(tmp_path, capsys):
    from text2sql import cli

    runs = tmp_path / "runs"
    write_run(_run(), runs / "T0-dev50")
    readme = tmp_path / "README.md"
    readme.write_text(f"x\n{START_MARKER}\n{END_MARKER}\n", encoding="utf-8")
    args = ["update-readme", "--runs-dir", str(runs), "--readme", str(readme)]
    assert cli.main(args) == 0
    assert "| T0 |" in readme.read_text(encoding="utf-8")
    assert "1 run" in capsys.readouterr().out
    assert cli.main([*args, "--check"]) == 0
    readme.write_text(f"x\n{START_MARKER}\nstale\n{END_MARKER}\n", encoding="utf-8")
    assert cli.main([*args, "--check"]) == 1
