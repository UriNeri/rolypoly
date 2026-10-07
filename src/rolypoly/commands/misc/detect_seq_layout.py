"""Expose shared sequence-file discovery and FASTQ library detection."""

import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import rich_click as click

from rolypoly.utils.bio.library_detection import (
    handle_input_fastq,
    identify_fastq_files,
    is_fasta_file,
    resolve_sequence_inputs,
)
from rolypoly.utils.logging.loggit import setup_logging


def layout_serializable(value: Any) -> Any:
    """Normalize detection paths and omit absent optional fields for TOML."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            key: layout_serializable(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, (list, tuple)):
        return [layout_serializable(item) for item in value]
    return value


def layout_toml_value(value: Any) -> str:
    """Encode the report's JSON-compatible values as TOML inline values."""
    if isinstance(value, dict):
        return (
            "{ "
            + ", ".join(
                f"{layout_toml_value(key)} = {layout_toml_value(item)}"
                for key, item in value.items()
            )
            + " }"
        )
    if isinstance(value, list):
        return "[" + ", ".join(layout_toml_value(item) for item in value) + "]"
    # Unlike JSON, TOML disallows a literal DEL character in strings.
    return json.dumps(value, ensure_ascii=False, allow_nan=False).replace(
        "\x7f", "\\u007f"
    )


@click.command(name="detect-seq-layout")
@click.option(
    "-i",
    "--input",
    required=True,
    help="FASTA/FASTQ file, comma-separated files, or one directory (non-recursive).",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="New output file; omit or use '-' to print to stdout.",
)
@click.option(
    "-f",
    "--fmt",
    "--format",
    "format",
    type=click.Choice(["json", "toml"], case_sensitive=False),
    default="json",
    show_default=True,
    help="Structured output format (independent of the output suffix).",
)
def detect_seq_layout(
    input: str, output: Path | None, format: str, log_file: Path, log_level: str
) -> None:
    """Discover FASTA files and infer FASTQ library layouts from sampled headers.

    Uses the same detection helpers as filter-reads, assemble, and roll.
    Two explicitly listed FASTQs are treated as an ordered R1/R2 pair.
    """
    try:
        # Keep both RolyPoly logs and reader diagnostics out of the payload.
        with redirect_stdout(sys.stderr):
            logger = setup_logging(log_file, log_level)
            requested = [
                part.strip() for part in input.split(",") if part.strip()
            ]
            if len(requested) > 1 and any(
                Path(part).expanduser().is_dir() for part in requested
            ):
                raise ValueError(
                    "Supply one directory or a list of files, not both."
                )
            paths = resolve_sequence_inputs(input)
            if any(not path.is_file() for path in paths):
                raise ValueError("All resolved inputs must be sequence files.")
            fasta_files = [path for path in paths if is_fasta_file(path)]
            fastq_files = [path for path in paths if not is_fasta_file(path)]
            fastq_info: dict[str, Any] = {
                "rolypoly_data": {},
                "R1_R2_pairs": [],
                "interleaved_files": [],
                "single_end_files": [],
                "file_details": {},
            }
            if fastq_files:
                directory = Path(requested[0]).expanduser().resolve()
                if directory.is_dir():
                    detected = identify_fastq_files(
                        directory, logger=logger, strict=True
                    )
                    detected["single_end_files"] = detected.pop("single_end")
                else:
                    detected = handle_input_fastq(
                        ",".join(str(path) for path in fastq_files),
                        logger=logger,
                        strict=True,
                    )
                fastq_info.update(
                    (key, detected[key])
                    for key in fastq_info
                    if key in detected
                )
            report = layout_serializable(
                {
                    "schema_version": 1,
                    "sample_size": 1000,
                    "header_sample_size": 100,
                    "fasta_files": fasta_files,
                    "fastq": fastq_info,
                }
            )
        if format.lower() == "json":
            payload = (
                json.dumps(
                    report, indent=2, ensure_ascii=False, allow_nan=False
                )
                + "\n"
            )
        else:
            payload = (
                "\n".join(
                    f"{key} = {layout_toml_value(value)}"
                    for key, value in report.items()
                )
                + "\n"
            )
        if output is None or output == Path("-"):
            click.echo(payload, nl=False)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                handle.write(payload)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
