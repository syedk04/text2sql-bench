import text2sql


def test_package_has_version():
    assert text2sql.__version__ == "0.1.0"


def test_cli_without_command_prints_help(capsys):
    from text2sql.cli import main

    assert main([]) == 0
    assert "text2sql" in capsys.readouterr().out
