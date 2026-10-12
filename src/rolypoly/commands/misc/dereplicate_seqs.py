"""Expose the existing streaming sequence deduplication helper."""

import gzip
import sys
from contextlib import ExitStack, redirect_stdout
from pathlib import Path

import rich_click as click
from needletail import NeedletailError

from rolypoly.utils.bio.sequences import (
    remove_duplicates,
    resolve_sequence_inputs,
)
from rolypoly.utils.logging.loggit import setup_logging


@click.command(name="dereplicate-seqs")
@click.option(
    "-i",
    "--input",
    required=True,
    help="Sequence file(s), comma-separated, or directory.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="New FASTA output; omit or use '-' for stdout. .gz compresses.",
)
@click.option(
    "-b",
    "--by",
    type=click.Choice(["seq", "id", "name"]),
    default="seq",
    show_default=True,
    help="Compare sequence content, first-token IDs, or full headers.",
)
@click.option(
    "--revcomp-as-different/--revcomp-as-same",
    "revcomp_as_distinct",
    default=True,
    show_default=True,
    help="Keep reverse complements distinct or collapse them.",
)
@click.option(
    "--ignore-case/--case-sensitive",
    default=True,
    show_default=True,
    help="Case policy for comparisons; emitted records are unchanged.",
)
@click.option(
    "-d",
    "--save-duplicates",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Optional discarded-record FASTA output; .gz compresses.",
)
@click.option(
    "-m",
    "--save-dup-list",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Optional legacy headerless duplicate count/member list; .gz compresses.",
)
def dereplicate_seqs(
    input: str,
    output: Path | None,
    by: str,
    revcomp_as_distinct: bool,
    ignore_case: bool,
    save_duplicates: Path | None,
    save_dup_list: Path | None,
    log_file: Path,
    log_level: str,
) -> None:
    """Stream unique FASTA records using the existing remove_duplicates helper.

    Accepts FASTA/FASTQ inputs; outputs FASTA without qualities. Keeps the first
    representative across inputs. Reverse-complement grouping requires
    --by seq and --ignore-case.
    """
    created: list[Path] = []
    complete = False
    stdout = sys.stdout
    try:
        if not revcomp_as_distinct and (by != "seq" or not ignore_case):
            raise click.UsageError(
                "--revcomp-as-same requires --by seq and --ignore-case "
                "(the existing helper does not correctly complement lowercase "
                "bases in case-sensitive mode)."
            )
        with redirect_stdout(sys.stderr):
            inputs = resolve_sequence_inputs(input)
        if any(not path.is_file() for path in inputs):
            raise click.UsageError("Inputs must be regular sequence files.")
        if log_file.resolve() in inputs:
            raise click.UsageError("The log file must not be a sequence input.")
        paths = [
            path
            for path in (output, save_duplicates, save_dup_list)
            if path is not None and path != Path("-")
        ]
        resolved = [path.resolve() for path in paths]
        if len(set(resolved)) != len(resolved):
            raise click.UsageError("Output paths must be distinct.")
        if any(path in [*inputs, log_file.resolve()] for path in resolved):
            raise click.UsageError(
                "Outputs must not overwrite inputs or the log file."
            )
        if Path("-") in (save_duplicates, save_dup_list):
            raise click.UsageError(
                "Only the main output may use '-' for stdout."
            )
        with ExitStack() as stack, redirect_stdout(sys.stderr):
            logger = setup_logging(log_file, log_level)
            handles = []
            for path in (output, save_duplicates, save_dup_list):
                if path is None or path == Path("-"):
                    handles.append(None)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    handle = stack.enter_context(
                        gzip.open(path, "xt", encoding="utf-8")
                        if path.suffix == ".gz"
                        else path.open("x", encoding="utf-8")
                    )
                    created.append(path)
                    handles.append(handle)
            logger.info(
                "Dereplicating by=%s ignore_case=%s revcomp_as_different=%s",
                by,
                ignore_case,
                revcomp_as_distinct,
            )
            # The helper accepts text handles via open_text_writer at runtime;
            # its existing public annotations only declare paths.
            remove_duplicates(
                inputs,
                output_file=handles[0] or stdout,  # type: ignore[arg-type]
                by=by,
                revcomp_as_distinct=revcomp_as_distinct,
                ignore_case=ignore_case,
                save_duplicates=handles[1],  # type: ignore[arg-type]
                save_dup_list=handles[2],  # type: ignore[arg-type]
                logger=logger,
            )
        complete = True
    except (OSError, ValueError, NeedletailError) as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        if not complete:
            for path in created:
                path.unlink(missing_ok=True)
