import shutil
from pathlib import Path

import nbformat
import pytest
from click.testing import CliRunner

from black_nb.cli import cli

THIS_FILE = Path(__file__)
THIS_DIR = THIS_FILE.parent


def test_formatting(tmp_path):
    src_dir = THIS_DIR / "data" / "formatting_tests"
    dst_dir = tmp_path / "formatting_tests"
    shutil.copytree(src_dir, dst_dir)

    unformatted = CliRunner().invoke(cli, ["--check", str(dst_dir)])
    assert unformatted.exit_code == 1

    formatting = CliRunner().invoke(cli, [str(dst_dir)])
    assert formatting.exit_code == 0

    with open(dst_dir / "unformatted.ipynb") as file:
        formatted_str = file.read()

        # check correct semicolon in code
        assert "1 + 1;" in formatted_str
        # check correct semicolon in comment
        assert "# cell ending in comment with a semicolon;" in formatted_str

    formatted = CliRunner().invoke(cli, ["--check", str(dst_dir)])
    assert formatted.exit_code == 0


def test_clear_output(tmp_path):
    src_dir = THIS_DIR / "data" / "clear_output_tests"
    dst_dir = tmp_path / "clear_output_tests"
    shutil.copytree(src_dir, dst_dir)

    uncleared = CliRunner().invoke(
        cli, ["--check", "--clear-output", str(dst_dir)]
    )
    assert uncleared.exit_code == 1

    clearing = CliRunner().invoke(cli, ["--clear-output", str(dst_dir)])
    assert clearing.exit_code == 0

    cleared = CliRunner().invoke(
        cli, ["--check", "--clear-output", str(dst_dir)]
    )
    assert cleared.exit_code == 0


def test_invalid_input(tmp_path):
    src_dir = THIS_DIR / "data" / "invalid_input_tests"
    dst_dir = tmp_path / "invalid_input_tests"
    shutil.copytree(src_dir, dst_dir)

    assert CliRunner().invoke(cli, [str(dst_dir)]).exit_code == 123


@pytest.mark.parametrize("option", ["--include", "--exclude"])
def test_invalid_include_exclude(option):
    result = CliRunner().invoke(cli, [option, "**()(!!*)"])
    assert result.exit_code == 2


def test_encoding_preserved(tmp_path):
    src_dir = THIS_DIR / "data" / "encoding_tests"
    dst_dir = tmp_path / "encoding_tests"
    shutil.copytree(src_dir, dst_dir)

    CliRunner().invoke(cli, [str(dst_dir / "unformatted.ipynb")])
    # attempt to read formatted file
    nbformat.read(
        str(dst_dir / "unformatted.ipynb"), as_version=nbformat.NO_CONVERT
    )
