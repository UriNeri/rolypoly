from pathlib import Path

import polars as pl
import rich_click as click
from rich.console import Console

console = Console()


@click.command(name="fastx_stats")
@click.option(
    "-i",
    "--input",
    required=True,
    type=click.Path(exists=True),
    help="Input file (fasta, fa, fna, faa)",
)
@click.option(
    "-o",
    "--output",
    default="stdout",
    type=click.Path(exists=False),
    help="Output path (use 'stdout' to print to console)",
)
@click.option(
    "--log-file",
    # default="command.log",
    type=click.Path(exists=False),
    help="Path to log file",
    hidden=True,
)
@click.option(
    "-ll", "--log-level", hidden=True, default="INFO", help="Log level"
)
@click.option(
    "--min_length",
    hidden=True,
    default=None,
    type=int,
    help="minimum sequence length to consider",
)
@click.option(
    "--max_length",
    hidden=True,
    default=None,
    type=int,
    help="maximum sequence length to consider",
)
@click.option(
    "--format",
    default="tsv",
    type=click.Choice(case_sensitive=False, choices=["csv", "tsv", "md"]),
    help="Output format for aggregate statistics",
)
@click.option(
    "-f",
    "--fields",
    type=click.Choice(
        case_sensitive=False, choices=["length", "gc_content", "n_count"]
    ),
    multiple=True,
    default=["length", "gc_content", "n_count"],
    help="Fields to calculate statistics for",
)
@click.option(
    "-c",
    "--circular",
    is_flag=True,
    help="Treat sequences as circular (rotate to minimal lexicographical form before analysis)",
)
def fastx_stats(
    input,
    output,
    log_file,
    log_level,
    min_length,
    max_length,
    format,
    fields,
    circular,
):
    """
    Calculate aggregate statistics for sequences (min, max, mean, median, etc.).

    This command computes summary statistics across all sequences in a FASTA/FASTQ file.
    For per-sequence calculations, use the 'fastx-calc' command instead.
    """
    from rolypoly.utils.bio.polars_fastx import (
        compute_aggregate_stats,
        fasta_stats,
        write_fastx_output,
        write_markdown_summary,
    )
    from rolypoly.utils.logging.loggit import log_start_info, setup_logging

    logger = setup_logging(log_file, log_level)
    log_start_info(logger, locals())

    write_to_stdout = output == "stdout"
    if write_to_stdout and format.lower() == "md":
        logger.error(
            "Markdown format cannot be written to stdout. Please specify an output file."
        )
        return

    # Compute per-sequence stats using fasta_stats
    fields_list = list(fields)
    if "header" not in fields_list:
        fields_list.insert(0, "header")
    fields_str = ",".join(fields_list)

    df = fasta_stats(
        input_file=input,
        output_file=None,
        min_length=min_length,
        max_length=max_length,
        fields=fields_str,
        circular=circular,
    )

    total_seqs = df.height
    logger.info(f"Processing {total_seqs} sequences from {input}")

    # Compute aggregate statistics
    agg_stats = compute_aggregate_stats(df, list(fields))
    df_output = pl.DataFrame([agg_stats])

    # Output results
    if format.lower() == "md":
        output_path = Path(output)
        write_markdown_summary(output_path, input, agg_stats, logger)
    else:
        write_fastx_output(
            df=df_output,
            output=output,
            format=format,
            logger=logger,
            write_to_stdout=write_to_stdout,
        )

    logger.info(f"✓ Aggregate statistics computed for {total_seqs} sequences")
    if write_to_stdout and format.lower() != "md":
        pass  # Already printed to stdout
    elif format.lower() != "md":
        logger.info(f"✓ Output written to {output}")

    logger.info("Sequence statistics calculation completed successfully")
