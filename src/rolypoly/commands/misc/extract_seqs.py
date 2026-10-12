"""Expose the existing streaming FASTA header-selection helper."""

import gzip
import sys
from contextlib import redirect_stdout
from pathlib import Path

import rich_click as click

from rolypoly.utils.bio.sequences import (
    filter_fasta_by_headers,
    resolve_sequence_inputs,
)
from rolypoly.utils.logging.loggit import setup_logging


@click.command(name="extract-seqs")
@click.option(
    "-i",
    "--input",
    required=True,
    help="Sequence file(s), comma-separated, or directory.",
)
@click.option(
    "-H",
    "--headers",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Plain text file with one full header or substring per line.",
)
@click.option(
    "-s",
    "--select",
    multiple=True,
    help="Full header or substring to select; repeat as needed.",
)
@click.option(
    "-m",
    "--match",
    type=click.Choice(["header", "substring"]),
    default="header",
    show_default=True,
    help="Exact complete-header or literal substring matching.",
)
@click.option(
    "-v", "--invert", is_flag=True, help="Write records that do not match."
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="New FASTA output; omit or use '-' for stdout. .gz compresses.",
)
def extract_seqs(
    input: str,
    headers: Path | None,
    select: tuple[str, ...],
    match: str,
    invert: bool,
    output: Path | None,
    log_file: Path,
    log_level: str,
) -> None:
    """Stream FASTA records selected using the existing filter_fasta_by_headers.

    Accepts FASTA/FASTQ inputs; outputs FASTA without qualities. Supply --headers
    and/or repeat --select. Matching is case-sensitive; exact matching uses the
    complete header, including its description.
    """
    created = False
    complete = False
    stdout = sys.stdout
    try:
        values = list(select)
        if headers is not None:
            values.extend(headers.read_text(encoding="utf-8").splitlines())
        values = list(
            dict.fromkeys(value.strip() for value in values if value.strip())
        )
        if not values:
            raise click.UsageError(
                "Supply a nonempty --headers file or --select value."
            )
        with redirect_stdout(sys.stderr):
            inputs = resolve_sequence_inputs(input)
        if any(not path.is_file() for path in inputs):
            raise click.UsageError("Inputs must be regular sequence files.")
        protected = [
            *inputs,
            *([headers.resolve()] if headers is not None else []),
        ]
        if log_file.resolve() in protected:
            raise click.UsageError(
                "The log file must not be an input or selection file."
            )
        if output is not None and output != Path("-"):
            if output.resolve() in [*protected, log_file.resolve()]:
                raise click.UsageError(
                    "Output must not overwrite an input, selection, or log file."
                )
        with redirect_stdout(sys.stderr):
            logger = setup_logging(log_file, log_level)
            handle = stdout
            if output is not None and output != Path("-"):
                output.parent.mkdir(parents=True, exist_ok=True)
                handle = (
                    gzip.open(output, "xt", encoding="utf-8")
                    if output.suffix == ".gz"
                    else output.open("x", encoding="utf-8")
                )
                created = True
            try:
                logger.info("Extracting match=%s invert=%s", match, invert)
                counts = filter_fasta_by_headers(
                    inputs,
                    values,
                    handle,  # type: ignore[arg-type]
                    wrap=match == "substring",
                    invert=invert,
                    return_counts=True,
                )
                assert counts is not None
                logger.info(
                    "Processed %d records; wrote %d",
                    counts["records_processed"],
                    counts["records_written"],
                )
                if not counts["records_written"]:
                    logger.warning(
                        "No records were selected; the sequence output is empty."
                    )
            finally:
                if created:
                    handle.close()
        complete = True
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        if created and not complete and output is not None:
            output.unlink(missing_ok=True)
