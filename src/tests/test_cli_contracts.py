from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tomllib

import click
import polars as pl
import pytest
from click.testing import CliRunner

from rolypoly.rolypoly import rolypoly


@pytest.fixture(scope="module")
def runner() -> CliRunner:
    return CliRunner()


@pytest.mark.parametrize("format", ["json", "toml"])
def test_detect_seq_layout_stdout_and_file_contract(
    runner: CliRunner, tmp_path: Path, format: str
) -> None:
    source = repo_root() / "testing_folder/inputs/reads/layout"
    inputs = tmp_path / 'inputs with "quotes" \u03b1 \U0001f41b \x7f'
    inputs.mkdir()
    for name in ("sample_R1.fq", "sample_R2.fq", "single.fq"):
        shutil.copyfile(source / name, inputs / name)
    with gzip.open(inputs / "interleaved.fq.gz", "wt") as handle:
        handle.write((source / "interleaved.fq").read_text())
    shutil.copyfile(source / "single.fq", inputs / "dedupe_final_merged_filtered.fq")
    shutil.copyfile(source / "single.fq", inputs / "dedupe_final_merged_orphan.fq")
    shutil.copyfile(
        source / "interleaved.fq", inputs / "dedupe_final_interleaved_filtered.fq"
    )
    fasta = inputs / "contigs.fa"
    fasta.write_text(">contig\nACGT\n")
    # Unknown sampled layout must remain visible despite single-end fallback.
    (inputs / "ambiguous.fq").write_text(
        "@a/1\nACGT\n+\nIIII\n@b/2\nACGT\n+\nIIII\n"
    )
    args = [
        "detect-seq-layout", "-i", str(inputs), "--format", format,
        "-g", str(tmp_path / "layout.log"), "-ll", "DEBUG",
    ]
    result = runner.invoke(rolypoly, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    parse = json.loads if format == "json" else tomllib.loads
    report = parse(result.stdout)
    assert "Detected Casava" in result.stderr
    assert report["schema_version"] == 1
    assert report["sample_size"] == 1000
    assert report["header_sample_size"] == 100
    assert report["fasta_files"] == [str(fasta)]
    info = report["fastq"]
    assert info["R1_R2_pairs"] == [
        [str(inputs / "sample_R1.fq"), str(inputs / "sample_R2.fq")]
    ]
    assert info["interleaved_files"] == [str(inputs / "interleaved.fq.gz")]
    assert info["single_end_files"] == [
        str(inputs / "ambiguous.fq"), str(inputs / "single.fq")
    ]
    assert info["rolypoly_data"]["filtered"] == {
        "interleaved": str(inputs / "dedupe_final_interleaved_filtered.fq"),
        "merged": str(inputs / "dedupe_final_merged_filtered.fq"),
    }
    assert info["rolypoly_data"]["orphan"] == {
        "merged": str(inputs / "dedupe_final_merged_orphan.fq")
    }
    details = info["file_details"][str(inputs / "interleaved.fq.gz")]
    assert details["file_type"] == "interleaved"
    assert details["is_gzipped"] is True
    assert details["pair_1_count"] == details["pair_2_count"] == 1
    assert details["average_read_length"] == 8
    assert details["average_read_quality"] == 40
    assert details["header_analysis"]["format"] == "casava"
    assert details["header_analysis"]["barcodes"] == ["AGGCTTCT"]
    assert info["file_details"][str(inputs / "ambiguous.fq")]["file_type"] == "unknown"

    output = tmp_path / "nested" / f"layout.{format}"
    saved = runner.invoke(rolypoly, args + ["-o", str(output)], catch_exceptions=False)
    assert saved.exit_code == 0, saved.output
    assert saved.stdout == ""
    assert parse(output.read_text()) == report
    original = output.read_bytes()
    repeated = runner.invoke(rolypoly, args + ["-o", str(output)], catch_exceptions=False)
    assert repeated.exit_code != 0
    assert output.read_bytes() == original


@pytest.mark.parametrize(
    "names,expected_types",
    [
        (["single.fq"], ["single"]),
        (["interleaved.fq"], ["interleaved"]),
        (["sample_R1.fq", "sample_R2.fq"], ["paired_R1", "paired_R2"]),
        (
            ["single.fq", "sample_R1.fq", "sample_R2.fq"],
            ["single", "paired_R1", "paired_R2"],
        ),
        (
            ["sample_R1.fq", "contigs.fa", "sample_R2.fq"],
            ["paired_R1", "paired_R2"],
        ),
    ],
)
def test_detect_seq_layout_explicit_inputs(
    runner: CliRunner, tmp_path: Path, names: list[str], expected_types: list[str]
) -> None:
    source = repo_root() / "testing_folder/inputs/reads/layout"
    fasta = tmp_path / "contigs.fa"
    fasta.write_text(">contig\nACGT\n")
    paths = [fasta if name == "contigs.fa" else source / name for name in names]
    fastq_paths = [path for path in paths if path != fasta]
    result = runner.invoke(
        rolypoly,
        ["detect-layout", "-i", ",".join(map(str, paths)), "-o", "-",
         "-g", str(tmp_path / "layout.log")],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["fasta_files"] == ([str(fasta)] if fasta in paths else [])
    info = report["fastq"]
    assert [
        info["file_details"][str(path)]["file_type"] for path in fastq_paths
    ] == expected_types
    if len(fastq_paths) == 2:
        assert info["R1_R2_pairs"] == [list(map(str, fastq_paths))]
    else:
        key = "interleaved_files" if expected_types == ["interleaved"] else "single_end_files"
        assert info[key] == list(map(str, fastq_paths))
        assert info["R1_R2_pairs"] == []


@pytest.mark.parametrize(
    "case", [
        "missing", "empty", "malformed", "non_fastq", "unsupported", "empty_dir",
        "mixed_directories", "invalid_format", "empty_input", "bad_directory",
        "bad_pair", "bad_filtered", "bad_gzip",
    ]
)
def test_detect_seq_layout_invalid_inputs(
    runner: CliRunner, tmp_path: Path, case: str
) -> None:
    path = tmp_path / "reads.fq"
    input_value = str(path)
    extra = []
    if case == "empty":
        path.touch()
    elif case == "malformed":
        path.write_text("@read\nACGT\n+\n")
    elif case == "non_fastq":
        path.write_text(">contig\nACGT\n")
    elif case == "unsupported":
        path = tmp_path / "notes.txt"
        path.write_text("not sequence data\n")
        input_value = str(path)
    elif case == "empty_dir":
        input_value = str(tmp_path)
    elif case == "mixed_directories":
        input_value = f"{tmp_path},{tmp_path}"
    elif case == "invalid_format":
        extra = ["--format", "yaml"]
    elif case == "empty_input":
        input_value = ""
    elif case == "bad_directory":
        path.write_text("@read\nACGT\n+\n")
        input_value = str(tmp_path)
    elif case == "bad_pair":
        path.write_text("@read\nACGT\n+\n")
        mate = repo_root() / "testing_folder/inputs/reads/layout/sample_R1.fq"
        input_value = f"{mate},{path}"
    elif case == "bad_filtered":
        path = tmp_path / "dedupe_final_merged_sample.fq"
        path.write_text("@read\nACGT\n+\n")
        input_value = str(tmp_path)
    elif case == "bad_gzip":
        path = tmp_path / "reads.fq.gz"
        path.write_bytes(b"\x1f\x8bnot a gzip stream")
        input_value = str(path)
    result = runner.invoke(
        rolypoly,
        ["detect-seq-layout", "-i", input_value, "-g", str(tmp_path / "layout.log"),
         "-o", str(tmp_path / "output.json"), *extra],
        catch_exceptions=False,
    )
    assert result.exit_code != 0, result.output
    assert result.stdout == ""
    assert not (tmp_path / "output.json").exists()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def render(value: str, tmp_path: Path) -> str:
    return (
        value.replace("{tmp}", str(tmp_path))
        .replace("{data}", os.environ["ROLYPOLY_DATA"])
    )


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
    args: list[str], tmp_path: Path, scenario_id: str
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
    option_tokens = {"--log-level", "-ll", "-l"}
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

    for fixture_name in scenario.get("fixtures", []):
        source = repo_root() / "testing_folder" / "inputs" / "mmtax"
        fixture_dir = tmp_path / "mmtax_fixture"
        fixture_dir.mkdir(parents=True, exist_ok=True)
        if fixture_name == "mmtax_mmseqs_db":
            database = fixture_dir / "ncbi_virus"
            subprocess.run(
                [
                    "mmseqs",
                    "createdb",
                    str(source / "reference.faa"),
                    str(database),
                    "--dbtype",
                    "1",
                ],
                check=True,
            )
            subprocess.run(
                [
                    "mmseqs",
                    "createtaxdb",
                    str(database),
                    str(fixture_dir / "taxonomy_tmp"),
                    "--ncbi-tax-dump",
                    str(source / "taxonomy"),
                    "--tax-mapping-file",
                    str(source / "accession2taxid.tsv"),
                ],
                check=True,
            )
        elif fixture_name == "mmtax_diamond_db":
            subprocess.run(
                [
                    "diamond",
                    "makedb",
                    "--in",
                    str(source / "reference.faa"),
                    "--db",
                    str(fixture_dir / "ncbi_virus"),
                    "--taxonmap",
                    str(source / "accession2taxid_diamond.tsv"),
                    "--taxonnodes",
                    str(source / "taxonomy" / "nodes.dmp"),
                    "--taxonnames",
                    str(source / "taxonomy" / "names.dmp"),
                    "--threads",
                    "1",
                ],
                check=True,
            )
        elif fixture_name == "nucleic_search_inputs":
            contig_source = repo_root() / "testing_folder/inputs/contigs"
            shutil.copyfile(
                contig_source / "test_contigs.fasta",
                tmp_path / "nucleic_queries_1.fasta",
            )
            shutil.copyfile(
                contig_source / "test_fasta_for_masking.fasta",
                tmp_path / "nucleic_queries_2.fasta",
            )
        else:
            raise ValueError(f"Unknown CLI fixture: {fixture_name}")


def load_cli_scenarios() -> list[dict]:
    scenario_path = Path(__file__).with_name("cli_scenarios.json")
    return json.loads(scenario_path.read_text())


def parse_csv_values(raw_value: str | None) -> set[str]:
    if not raw_value:
        return set()
    return {item.strip() for item in raw_value.split(",") if item.strip()}


def should_skip_scenario(
    scenario: dict, request: pytest.FixtureRequest
) -> str | None:
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
        [scenario_id, str(scenario.get("description", "")), command_name]
    ).lower()

    if scenario_ids and scenario_id not in scenario_ids:
        return f"scenario id '{scenario_id}' not selected"

    if command_names and command_name not in command_names:
        return f"command '{command_name}' not selected"

    if match_tokens and not any(
        token.lower() in searchable for token in match_tokens
    ):
        return "no cli-match token matched"

    return None


@pytest.mark.parametrize(
    "scenario", load_cli_scenarios(), ids=lambda row: row["id"]
)
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
    args = inject_log_file_arg(
        args, tmp_path, str(scenario.get("id", "scenario"))
    )
    args = inject_debug_log_level(args)

    result = runner.invoke(rolypoly, args, catch_exceptions=False)

    assert result.exit_code == 0, (
        f"Scenario '{scenario['id']}' failed with args: {args}\n{result.output}"
    )

    for expected in scenario.get("expected_files", []):
        expected_path = Path(render(expected, tmp_path))
        assert expected_path.exists(), (
            f"Expected output file missing: {expected_path}"
        )
        assert expected_path.stat().st_size > 0, (
            f"Output file is empty: {expected_path}"
        )

    for expected_dir in scenario.get("expected_dirs", []):
        expected_dir_path = Path(render(expected_dir, tmp_path))
        assert expected_dir_path.exists(), (
            f"Expected output directory missing: {expected_dir_path}"
        )
        assert expected_dir_path.is_dir(), (
            f"Expected directory path is not a directory: {expected_dir_path}"
        )

    for file_path, required_tokens in scenario.get(
        "expected_contains", {}
    ).items():
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

    for file_path, min_rows in scenario.get(
        "expected_table_min_rows", {}
    ).items():
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
