from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

import click
import polars as pl
import pytest
from click.testing import CliRunner

from rolypoly.rolypoly import rolypoly


@pytest.fixture(scope="module")
def runner() -> CliRunner:
    return CliRunner()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def render(value: str, tmp_path: Path) -> str:
    return value.replace("{tmp}", str(tmp_path))


def render_values(values: list[str], tmp_path: Path) -> list[str]:
    return [render(str(value), tmp_path) for value in values]


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


def inject_log_file_arg(
    args: list[str],
    tmp_path: Path,
    scenario_id: str,
) -> list[str]:
    if not args:
        return args

    command_name = args[0]
    if "--log-file" in args:
        return args

    log_option = pick_log_file_option(command_name)
    if not log_option:
        return args

    log_file_path = tmp_path / f"{command_name}_{scenario_id}.log"
    return args + [log_option, str(log_file_path)]


def inject_debug_log_level(args: list[str]) -> list[str]:
    if not args:
        return args

    command_name = args[0]
    option_tokens = {
        "--log-level",
        "-ll",
        "-l",
    }
    if any(token in option_tokens for token in args[1:]):
        return args

    suffix = debug_log_suffix_for_command(command_name)
    if not suffix:
        return args
    return args + suffix


def pick_debug_value(option: click.Option) -> str:
    option_type = getattr(option, "type", None)
    choices = getattr(option_type, "choices", None)
    if not choices:
        return "DEBUG"

    for candidate in choices:
        if str(candidate).lower() == "debug":
            return str(candidate)
    return "DEBUG"


def debug_log_suffix_for_command(command_name: str) -> list[str]:
    ctx = click.Context(rolypoly)
    command = rolypoly.get_command(ctx, command_name)
    if command is None:
        return []

    for parameter in command.params:
        if not isinstance(parameter, click.Option):
            continue

        if "--log-level" in parameter.opts:
            return ["--log-level", pick_debug_value(parameter)]
        if "-ll" in parameter.opts:
            return ["-ll", pick_debug_value(parameter)]
        if "-l" in parameter.opts:
            return ["-l", pick_debug_value(parameter)]

    return []


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_table(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return pl.read_csv(path, separator="\t")
    if suffix == ".csv":
        return pl.read_csv(path)
    if suffix == ".parquet":
        return pl.read_parquet(path)
    raise ValueError(f"Unsupported table format for schema checks: {path}")


def apply_preconditions(scenario: dict, tmp_path: Path) -> None:
    for command_name in scenario.get("required_commands", []):
        if shutil.which(command_name) is None:
            pytest.skip(
                f"Scenario '{scenario['id']}' skipped: required command '{command_name}' not found"
            )

    for group in scenario.get("required_any_commands", []):
        if not any(shutil.which(command_name) for command_name in group):
            pretty = ", ".join(group)
            pytest.skip(
                f"Scenario '{scenario['id']}' skipped: none of [{pretty}] are available"
            )

    for required_path in scenario.get("required_paths", []):
        path_obj = Path(render(required_path, tmp_path))
        if not path_obj.exists():
            pytest.skip(
                f"Scenario '{scenario['id']}' skipped: required path missing ({path_obj})"
            )


def load_cli_scenarios() -> list[dict]:
    scenario_path = Path(__file__).with_name("cli_scenarios.json")
    return json.loads(scenario_path.read_text())


def parse_csv_values(raw_value: str | None) -> set[str]:
    if not raw_value:
        return set()
    return {item.strip() for item in raw_value.split(",") if item.strip()}


def should_skip_scenario(scenario: dict, request: pytest.FixtureRequest) -> str | None:
    scenario_ids = parse_csv_values(
        request.config.getoption("--cli-scenarios")
        or os.environ.get("RP_CLI_SCENARIOS")
    )
    command_names = parse_csv_values(
        request.config.getoption("--cli-commands")
        or os.environ.get("RP_CLI_COMMANDS")
    )
    match_tokens = parse_csv_values(
        request.config.getoption("--cli-match")
        or os.environ.get("RP_CLI_MATCH")
    )

    scenario_id = str(scenario.get("id", ""))
    command_name = (
        str(scenario.get("args", [""])[0]) if scenario.get("args") else ""
    )
    searchable = " ".join(
        [
            scenario_id,
            str(scenario.get("description", "")),
            command_name,
        ]
    ).lower()

    if scenario_ids and scenario_id not in scenario_ids:
        return f"scenario id '{scenario_id}' not selected"

    if command_names and command_name not in command_names:
        return f"command '{command_name}' not selected"

    if match_tokens and not any(token.lower() in searchable for token in match_tokens):
        return "no cli-match token matched"

    return None


def test_top_level_help(runner: CliRunner) -> None:
    result = runner.invoke(rolypoly, ["--help"], catch_exceptions=False)
    assert result.exit_code == 0, result.output


def test_all_registered_commands_have_help(runner: CliRunner) -> None:
    ctx = click.Context(rolypoly)
    commands = sorted(set(rolypoly.list_commands(ctx)))

    assert commands, "No commands were registered in the CLI entry point"

    for command_name in commands:
        result = runner.invoke(
            rolypoly,
            [command_name, "--help"],
            catch_exceptions=False,
        )
        assert (
            result.exit_code == 0
        ), f"`rolypoly {command_name} --help` failed\n{result.output}"


@pytest.mark.parametrize("scenario", load_cli_scenarios(), ids=lambda row: row["id"])
def test_cli_scenarios(
    runner: CliRunner,
    tmp_path: Path,
    scenario: dict,
    request: pytest.FixtureRequest,
) -> None:
    skip_reason = should_skip_scenario(scenario, request)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    apply_preconditions(scenario, tmp_path)

    args = render_values(scenario["args"], tmp_path)
    args = inject_log_file_arg(args, tmp_path, str(scenario.get("id", "scenario")))
    args = inject_debug_log_level(args)

    result = runner.invoke(rolypoly, args, catch_exceptions=False)

    assert result.exit_code == 0, (
        f"Scenario '{scenario['id']}' failed with args: {args}\n{result.output}"
    )

    for expected in scenario.get("expected_files", []):
        expected_path = Path(render(expected, tmp_path))
        assert expected_path.exists(), f"Expected output file missing: {expected_path}"
        assert expected_path.stat().st_size > 0, f"Output file is empty: {expected_path}"

    for expected_dir in scenario.get("expected_dirs", []):
        expected_dir_path = Path(render(expected_dir, tmp_path))
        assert expected_dir_path.exists(), (
            f"Expected output directory missing: {expected_dir_path}"
        )
        assert expected_dir_path.is_dir(), (
            f"Expected directory path is not a directory: {expected_dir_path}"
        )

    for file_path, required_tokens in scenario.get("expected_contains", {}).items():
        rendered_path = Path(render(file_path, tmp_path))
        content = rendered_path.read_text()
        for token in required_tokens:
            assert token in content, (
                f"Expected token '{token}' not found in {rendered_path}"
            )

    for file_path, expected_columns in scenario.get(
        "expected_table_columns", {}
    ).items():
        rendered_path = Path(render(file_path, tmp_path))
        frame = read_table(rendered_path)
        for column_name in expected_columns:
            assert column_name in frame.columns, (
                f"Expected column '{column_name}' not found in {rendered_path}. "
                f"Observed columns: {frame.columns}"
            )

    for file_path, min_rows in scenario.get("expected_table_min_rows", {}).items():
        rendered_path = Path(render(file_path, tmp_path))
        frame = read_table(rendered_path)
        assert frame.height >= int(min_rows), (
            f"Expected at least {min_rows} rows in {rendered_path}, found {frame.height}"
        )

    for file_path, expected_prefix in scenario.get(
        "expected_checksum_prefix", {}
    ).items():
        rendered_path = Path(render(file_path, tmp_path))
        digest = sha256(rendered_path)
        assert digest.startswith(expected_prefix), (
            f"Checksum mismatch for {rendered_path}: expected prefix {expected_prefix}, got {digest}"
        )
