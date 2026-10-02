import hashlib
import json
from pathlib import Path

import pytest

from text2sql.data import bird
from text2sql.data.bird import DatasetError, install_questions, load_questions, sidecar_path

FIXTURE = Path(__file__).parent / "fixtures" / "questions.json"


def test_install_writes_file_and_sidecar(tmp_path):
    data = FIXTURE.read_bytes()
    dest = tmp_path / "with space" / "q.json"
    install_questions(data, dest, expected_sha256=None, expect_total=12)
    assert dest.read_bytes() == data
    digest = hashlib.sha256(data).hexdigest()
    assert sidecar_path(dest).read_text().split()[0] == digest
    assert len(load_questions(dest)) == 12


def test_sha_mismatch_is_refused_and_nothing_written(tmp_path):
    dest = tmp_path / "q.json"
    with pytest.raises(DatasetError, match="sha256 mismatch"):
        install_questions(FIXTURE.read_bytes(), dest, expected_sha256="0" * 64)
    assert not dest.exists()


def test_wrong_count_is_refused(tmp_path):
    data = FIXTURE.read_bytes()
    with pytest.raises(DatasetError, match="expected 500"):
        install_questions(data, tmp_path / "q.json", expected_sha256=None)


def test_garbage_is_refused(tmp_path):
    with pytest.raises(DatasetError, match="not valid UTF-8 JSON"):
        install_questions(b"\xff\xfe nope", tmp_path / "q.json", expected_sha256=None)


def test_url_is_pinned_to_a_revision():
    assert bird.HF_REVISION in bird.HF_URL
    assert len(bird.HF_REVISION) == 40


def test_from_file_skips_network_and_still_checks_the_pinned_hash(tmp_path, monkeypatch):
    import text2sql.net as net

    def boom(*a, **k):
        raise AssertionError("network used")

    monkeypatch.setattr(net, "fetch_bytes", boom)
    src = tmp_path / "manual.json"
    src.write_bytes(FIXTURE.read_bytes())
    dest = tmp_path / "out.json"
    with pytest.raises(DatasetError, match="sha256 mismatch"):
        bird.download_questions(dest, from_file=src)
    assert not dest.exists()


def test_cli_from_file_uses_pinned_checks(tmp_path, monkeypatch, capsys):
    from text2sql import cli

    data = FIXTURE.read_bytes()
    seen = {}

    def fake_install(raw, dest, **kwargs):
        seen["raw"] = raw
        return install_questions(raw, dest, expected_sha256=None, expect_total=12)

    monkeypatch.setattr(bird, "install_questions", fake_install)
    src = tmp_path / "manual.json"
    src.write_bytes(data)
    dest = tmp_path / "out.json"
    assert cli.main(["download-questions", "--from-file", str(src), "--dest", str(dest)]) == 0
    assert seen["raw"] == data
    assert "wrote 12 questions" in capsys.readouterr().out
    assert json.loads(dest.read_text(encoding="utf-8"))[0]["question_id"] == 100
