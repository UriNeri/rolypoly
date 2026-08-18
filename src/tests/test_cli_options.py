"""Tests for automatically injected RolyPoly CLI options."""

from pathlib import Path

import rich_click as click
from click.testing import CliRunner

from rolypoly.utils.cli_options import (
    add_shared_options,
    shared_command_context,
)
from rolypoly.utils.lazy_group import LazyGroup


def test_lazy_group_injects_options_from_callback_parameters() -> None:
    """Commands should receive only the shared parameters they accept."""

    @click.command()
    def worker(
        threads: int,
        memory: str,
        temp_dir: Path | None,
        keep_tmp: bool,
        log_file: Path | None,
        log_level: str,
    ) -> None:
        click.echo(
            f"{threads}|{memory}|{temp_dir}|{keep_tmp}|{log_file}|{log_level}"
        )

    @click.group(cls=LazyGroup, context_settings={"show_default": True})
    def cli() -> None:
        pass

    cli.add_command(worker)
    runner = CliRunner()

    result = runner.invoke(cli, ["worker"])
    assert result.exit_code == 0
    assert result.output.strip() == "1|8g|None|False|rolypoly.log|INFO"

    help_result = runner.invoke(cli, ["worker", "--help"])
    assert help_result.exit_code == 0
    for option_name in (
        "--threads",
        "--memory",
        "--temp-dir",
        "--keep-tmp",
        "--log-file",
    ):
        assert option_name in help_result.output
    assert "--log-level" not in help_result.output


def test_memory_values_are_normalized() -> None:
    """Memory values should use one lower-case unit spelling."""

    @click.command()
    def command(memory: str) -> None:
        click.echo(memory)

    add_shared_options(command)
    runner = CliRunner()

    assert runner.invoke(command, ["--memory", "8GB"]).output.strip() == "8g"
    assert (
        runner.invoke(command, ["--memory", "6000MB"]).output.strip() == "6000m"
    )


def test_existing_command_option_is_not_replaced() -> None:
    """Explicit command-specific options should override the shared policy."""

    @click.command()
    @click.option("-j", "--threads", default=4)
    def command(threads: int) -> None:
        click.echo(threads)

    add_shared_options(command)
    add_shared_options(command)

    assert [parameter.name for parameter in command.params].count(
        "threads"
    ) == 1
    result = CliRunner().invoke(command, ["-j", "3"])
    assert result.exit_code == 0
    assert result.output.strip() == "3"


def test_legacy_aliases_are_kept_centrally() -> None:
    """Existing short aliases should remain accepted after centralization."""

    @click.command(name="marker-search")
    def command(memory: str, temp_dir: Path | None) -> None:
        click.echo(f"{memory}|{temp_dir}")

    add_shared_options(command)
    result = CliRunner().invoke(command, ["-m", "8GB", "-td", "work"])

    assert result.exit_code == 0
    assert result.output.strip() == "8g|work"


def test_in_process_context_supplies_shared_defaults() -> None:
    """Workflow invocations should receive defaults without LazyGroup."""

    @click.command()
    def command(log_level: str) -> str:
        return log_level

    context = shared_command_context(command)

    assert context.invoke(command) == "INFO"
