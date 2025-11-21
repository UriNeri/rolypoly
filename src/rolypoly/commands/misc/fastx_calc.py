import rich_click as click
from rich.console import Console
from pathlib import Path
import polars as pl

console = Console()

@click.command(
        name="fastx_calc"
)
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
    default="rp_sequence_calc.tsv",
    type=click.Path(exists=False),
    help="Output path (use 'stdout' to print to console)",
)
@click.option(
    "--log-file",
    type=click.Path(exists=False),
    help="Path to log file",
    hidden=True,
)
@click.option(
    "-ll",
    "--log-level",
    hidden=True,
    default="INFO",
    help="Log level",
)
@click.option(
    "--min_length",
    hidden=True,
    default=None,
    type=int,
    help="Minimum sequence length to consider",
)
@click.option(
    "--max_length",
    hidden=True,
    default=None,
    type=int,
    help="Maximum sequence length to consider",
)
@click.option(
    "--format",
    default="tsv",
    type=click.Choice(case_sensitive=False, choices=["csv", "tsv", "parquet"]),
    help="Output format for per-sequence annotations",
)
@click.option(
    "-f",
    "--fields",
    type=click.Choice(case_sensitive=False, choices=["length", "gc_content", "n_count", "hash"]),
    multiple=True,
    default=["length", "gc_content", "n_count"],
    help="""Fields to annotate for each sequence.
              Available:
              length - sequence length
              gc_content - percentage of GC nucleotides
              n_count - total number of Ns 
              hash - md5 hash of the sequence
              """,
)
@click.option(
    "-c",
    "--circular",
    is_flag=True,
    help="Treat sequences as circular (rotate to minimal lexicographical form before hashing)",
)
def fastx_calc(
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
    Calculate per-sequence metrics (length, GC content, hash, etc.).
    
    This command computes metrics for each sequence in a FASTA/FASTQ file.
    For aggregate statistics across all sequences, use the 'fastx-stats' command instead.
    
    Note:
        - No support yet for reverse complement (not in circular or hash).
    """
    from rolypoly.utils.logging.loggit import log_start_info, setup_logging
    from rolypoly.utils.bio.polars_fastx import fasta_stats, write_fastx_output

    logger = setup_logging(log_file, log_level)
    log_start_info(logger, locals())

    write_to_stdout = (output == "stdout")
    if write_to_stdout:
        format = "tsv"  # Force TSV for stdout
        logger.info("Writing to stdout (TSV format)")

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
        circular=circular
    )
    
    total_seqs = df.height
    logger.info(f"Calculated metrics for {total_seqs} sequences from {input}")
    
    # Output results
    write_fastx_output(
        df=df,
        output=output,
        format=format,
        logger=logger,
        write_to_stdout=write_to_stdout
    )
    
    if not write_to_stdout:
        logger.info(f"✓ Results written to {output}")
        logger.info(f"✓ Total sequences processed: {total_seqs}")
        logger.info("First 5 rows:")
        console.print(df.head(5))
    
    logger.info("Sequence calculations completed successfully")
