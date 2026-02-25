from __future__ import annotations

import click
from click.testing import CliRunner

from rolypoly.rolypoly import rolypoly


def test_top_level_help_smoke() -> None:
    runner = CliRunner()
    result = runner.invoke(rolypoly, ["--help"], catch_exceptions=False)
    assert result.exit_code == 0, result.output


def test_each_command_help_smoke() -> None:
    runner = CliRunner()
    ctx = click.Context(rolypoly)
    command_names = sorted(set(rolypoly.list_commands(ctx)))

    assert command_names, "No commands were registered in the CLI entry point"

    for command_name in command_names:
        result = runner.invoke(
            rolypoly,
            [command_name, "--help"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, (
            f"`rolypoly {command_name} --help` failed\n{result.output}"
        )
