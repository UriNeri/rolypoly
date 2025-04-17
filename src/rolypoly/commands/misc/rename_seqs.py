import rich_click as click
from rich.console import Console

console = Console()


@click.command()
@click.option("-i", "--input", required=True, help="Input FASTA file")
@click.option("-o", "--output", required=True, help="Output FASTA file")
@click.option("-m", "--mapping", required=True, help="Output mapping file (TSV)")
@click.option("-p", "--prefix", default="CID", help="Prefix for new sequence IDs")
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
def main(input: str, output: str, mapping: str, prefix: str, hash: bool, stats: bool):
    """Rename sequences in a FASTA file with consistent IDs.

    This tool renames sequences in a FASTA file using either sequential numbers
    or hashes, and generates a lookup table mapping old IDs to new IDs.
    Optionally includes sequence statistics (length, GC content).
    """
    import polars as pl

    from rolypoly.utils.fax import process_sequences, read_fasta_df, rename_sequences

    # Read input FASTA
    console.print(f"Reading sequences from {input}")
    df = read_fasta_df(input)

    # Rename sequences
    console.print(f"Renaming sequences with prefix '{prefix}'")
    df_renamed, id_map = rename_sequences(df, prefix, hash)

    # Calculate stats if requested
    if stats:
        console.print("Calculating sequence statistics")
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
    console.print(f"Writing renamed sequences to {output}")
    with open(output, "w") as f:
        for header, seq in zip(df_renamed["header"], df_renamed["sequence"]):
            f.write(f">{header}\n{seq}\n")

    console.print(f"Writing ID mapping to {mapping}")
    mapping_df.write_csv(mapping, separator="\t")

    console.print("[green]Done![/green]")


if __name__ == "__main__":
    main()


# """Rename sequences in a FASTA file with consistent IDs.

# This script provides functionality to rename sequences in FASTA files with consistent
# IDs, either using sequential numbers or hashes. It also generates a lookup table
# mapping old IDs to new IDs and optionally includes sequence statistics.

# Note on naming: to be consiset, the default for the raw assembly outputs sets new names as "CID_####" (CID is contig ID, and the ### is a padded running number).
# Subsequent steps in the pipeline will prepend additional identifiers to the contig IDs - after binning we add Bin_####, and after marker search we add vid_####.
# ORFs/CDS, get a unique name regardless of their position in the genome or the contig ID - they can be traced back to the CIDs in the GFF or some table.
# """
