import polars as pl
import rich_click as click
from rich.console import Console

from rolypoly.utils.bio.polars_fastx import from_fastx_eager as read_fasta_df
from rolypoly.utils.bio.sequences import process_sequences, rename_sequences

console = Console()


@click.command(name="rename-seqs")
@click.option("-i", "--input", required=True, help="Input FASTA file")
@click.option("-o", "--output", required=True, help="Output FASTA file")
@click.option(
    "-m", "--mapping", required=True, help="Output mapping file (TSV)"
)
@click.option(
    "-p", "--prefix", default="CID", help="Prefix for new sequence IDs"
)
@click.option(
    "--hash/--no-hash",
    default=False,
    help="Use hash instead of a padded running number for IDs",
)
@click.option(
    "--stats/--no-stats",
    default=True,
    help="Include sequence statistics in mapping file (length, GC content)",
)
@click.option(
    "-ll", "--log-level", hidden=True, default="INFO", help="Log level"
)
def rename_seqs(
    input: str,
    output: str,
    mapping: str,
    prefix: str,
    hash: bool,
    stats: bool,
    log_level: str,
):
    """Rename sequences in a FASTA file with consistent IDs (supports numbering or hashing, appending attributes like GC and length).

    This tool renames sequences in a FASTA file using either sequential numbers
    or hashes, and generates a lookup table mapping old IDs to new IDs.
    Optionally includes sequence statistics (length, GC content).
    """
    from rolypoly.utils.logging.loggit import get_logger, setup_logging

    setup_logging(None, log_level)
    logger = get_logger()

    # Read input FASTA
    logger.info(f"Reading sequences from {input}")
    df = read_fasta_df(input)

    # Rename sequences
    logger.info(f"Renaming sequences with prefix '{prefix}'")
    df_renamed, id_map = rename_sequences(df, prefix, hash)

    # Calculate stats if requested
    if stats:
        logger.info("Calculating sequence statistics")
        df_renamed = process_sequences(df_renamed)

        # Prepare mapping DataFrame with stats
        mapping_df = pl.DataFrame(
            {
                "old_id": list(id_map.keys()),
                "new_id": list(id_map.values()),
                "length": df_renamed["length"],
                "gc_content": df_renamed["gc_content"].round(2),
            }
        )
    else:
        # Mapping DataFrame without stats
        mapping_df = pl.DataFrame(
            {"old_id": list(id_map.keys()), "new_id": list(id_map.values())}
        )

    # Write output files
    logger.info(f"Writing renamed sequences to {output}")
    with open(output, "w") as f:
        for header, seq in zip(df_renamed["header"], df_renamed["sequence"]):
            f.write(f">{header}\n{seq}\n")

    logger.info(f"Writing ID mapping to {mapping}")
    mapping_df.write_csv(mapping, separator="\t")

    logger.info("[green]Done![/green]")
