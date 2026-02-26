from __future__ import annotations

import click
from click.testing import CliRunner
from pathlib import Path

from rolypoly.rolypoly import rolypoly


def test_top_level_help_smoke() -> None:
    runner = CliRunner()
    result = runner.invoke(rolypoly, ["--help"], catch_exceptions=False)
    assert result.exit_code == 0, result.output


def pick_log_file_option(command_name: str) -> str | None:
    ctx = click.Context(rolypoly)
    command = rolypoly.get_command(ctx, command_name)
    if command is None:
        return None

    for parameter in command.params:
        if not isinstance(parameter, click.Option):
            continue
        if "--log-file" in parameter.opts:
            return "--log-file"
    return None


def test_each_command_help_smoke(tmp_path: Path) -> None:
    runner = CliRunner()
    ctx = click.Context(rolypoly)
    command_names = sorted(set(rolypoly.list_commands(ctx)))

    assert command_names, "No commands were registered in the CLI entry point"

    for command_name in command_names:
        args = [command_name, "--help"]
        log_option = pick_log_file_option(command_name)
        if log_option:
            log_file_path = tmp_path / f"{command_name}_help_smoke.log"
            args = [command_name, log_option, str(log_file_path), "--help"]

        result = runner.invoke(
            rolypoly,
            args,
            catch_exceptions=False,
        )
        assert result.exit_code == 0, (
            f"`rolypoly {command_name} --help` failed\n{result.output}"
        )
