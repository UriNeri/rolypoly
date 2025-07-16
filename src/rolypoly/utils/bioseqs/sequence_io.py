"""Sequence file I/O operations for reading and writing FASTA/FASTQ files."""

from typing import List, Union

import polars as pl
from needletail import parse_fastx_file


def read_fasta_needletail(fasta_file: str) -> tuple[list[str], list[str]]:
    """Read sequences from a FASTA/FASTQ file using needletail"""

    seqs = []
    seq_ids = []
    for record in parse_fastx_file(fasta_file):
        seqs.append(getattr(record, 'seq', ''))  # type: ignore
        seq_ids.append(getattr(record, 'id', ''))  # type: ignore
    return seq_ids, seqs


def read_fasta_polars(
    fasta_file, idcol="seq_id", seqcol="seq", add_length=False, add_gc_content=False
):
    """Read FASTA file into polars DataFrame with optional statistics."""
    seq_ids, seqs = read_fasta_needletail(fasta_file)
    df = pl.DataFrame({idcol: seq_ids, seqcol: seqs})
    if add_length:
        df = df.with_columns(pl.col(seqcol).str.len_chars().alias("length")) # type: ignore
    if add_gc_content:
        df = df.with_columns(pl.col(seqcol).str.count_matches("G|C").alias("gc_content") / pl.col(seqcol).str.len_chars().alias("length")) 
    return df


def read_fasta_df(file_path: str) -> pl.DataFrame:
    """Reads a FASTA file into a Polars DataFrame.

    Args:
        file_path (str): Path to the input FASTA file

    Returns:
        polars.DataFrame: DataFrame with columns:
            - header: Sequence headers
            - sequence: Sequence strings
            - group: Internal grouping number (used for sequence concatenation)
    """
    # First read the raw sequences using the registered from_fastx namespace
    raw_df = pl.LazyFrame.from_fastx.init(file_path).collect() # type: ignore defined in polars_fastx.py

    # The from_fastx namespace already returns header and sequence columns
    df = raw_df

    # Drop quality column if it exists
    if "quality" in df.columns:
        df = df.drop("quality")

    # Create a running length column for sequence concatenation
    df = df.with_columns([pl.col("header").is_not_null().cum_sum().alias("group")])

    # Create a new dataframe concatenating sequences
    result = (
        df.filter(pl.col("header") != "")
        .select("header", "group")
        .join(
            df.group_by("group").agg(
                [
                    pl.when(pl.col("sequence").str.contains(">") == False)
                    .then(pl.col("sequence"))
                    .str.concat(delimiter="")
                    .alias("sequence")
                ]
            ),
            on="group",
        )
    )

    return result


def write_fasta_file(
    records=None, seqs=None, headers=None, output_file=None, format: str = "fasta"
) -> None:
    """Write sequences to a FASTA file or stdout if no output file is provided"""
    import sys

    if format == "fasta":
        seq_delim = "\n"
        header_delim = "\n>"
    elif format == "tab":
        seq_delim = "\t"
        header_delim = "\n>"
    else:
        raise ValueError(f"Invalid format: {format}")

    if output_file is None:
        output_file = sys.stdout
    else:
        output_file = open(output_file, "w")

    if records:
        for record in records:
            output_file.write(f"{header_delim}{record.id}{seq_delim}{str(record.seq)}")
    elif seqs is not None and headers is not None:
        for i, seq in enumerate(seqs):
            output_file.write(f"{header_delim}{headers[i]}{seq_delim}{seq}")
    else:
        raise ValueError("No records, seqs, or headers provided")


def filter_fasta_by_headers(
    fasta_file: str,
    headers: Union[str, List[str]],
    output_file: str,
    invert: bool = False,
) -> None:
    """Filter sequences in a FASTA file based on their headers.

    Extracts sequences whose headers match (or don't match if inverted) any of
    the provided header patterns.

    Args:
        fasta_file (str): Path to input FASTA file
        headers (Union[str, List[str]]): Either a file containing headers (one per line)
            or a list of header patterns to match
        output_file (str): Path to write filtered sequences
        invert (bool, optional): If True, keep sequences that don't match.
    """

    headers_list = []
    if not isinstance(headers, list):
        with open(headers, "r") as f:
            for line in f:
                headers_list.append(line.strip())
    else:
        headers_list = headers

    with open(output_file, "w") as out_f:
        for record in parse_fastx_file(fasta_file):
            matches = any(header in str(getattr(record, 'id', '')) for header in headers_list)  # type: ignore
            if (
                matches ^ invert
            ):  # XOR operation: write if (matches and not invert) or (not matches and invert)
                out_f.write(f">{getattr(record, 'id', '')}\n{getattr(record, 'seq', '')}\n")  # type: ignore


def add_fasta_to_gff(config, gff_file):
    """Add FASTA section to GFF file"""

    with open(gff_file, "a") as f:
        f.write("##FASTA\n")
        write_fasta_file(
            records=parse_fastx_file(config.input),
            output_file=f,
            format="fasta",
        )


def populate_pldf_withseqs_needletail(
    pldf,
    seqfile,
    chunk_size=20000000,
    trim_to_region=False,
    reverse_by_strand_col=False,
    idcol="contig_id",
    seqcol="contig_seq",
    start_col="start",
    end_col="end",
    strand_col="strand",
):
    """Populate a polars DataFrame with sequences from a FASTA file."""
    import subprocess
    from .sequence_analysis import revcomp

    merge_cols = [idcol]
    if reverse_by_strand_col:
        merge_cols.append(strand_col)
    if trim_to_region:
        merge_cols.extend([start_col, end_col])

    print(f"Initial pldf shape: {pldf.shape}")
    minipldf = pldf.select(merge_cols).unique()
    print(f"Unique entries in minipldf: {minipldf.shape}")

    minipldf = minipldf.filter(~pl.col(idcol).is_in([None, "", "nan"]))
    print(f"After filtering nulls: {minipldf.shape}")

    minipldf = minipldf.with_columns(pl.lit(None).alias(seqcol))

    seqs = []
    seq_ids = []

    # Get actual sequence count from file
    seq_count = int(
        subprocess.run(
            f"grep -F '>'  {seqfile} -c ", shell=True, capture_output=True, text=True
        ).stdout.strip()
    )
    print(f"Actual number of sequences in file: {seq_count}")

    # Reset file iterator
    index = 0
    for record in parse_fastx_file(seqfile):
        seqs.append(record.seq) # type: ignore
        seq_ids.append(record.id) # type: ignore
        index += 1

        # Process chunk when we hit chunk_size or end of file
        if len(seqs) >= chunk_size or index == seq_count:
            print(f"\nProcessing chunk {index}/{seq_count}")
            print(f"Number of sequences in chunk: {len(seqs)}")

            chunk_seqs = pl.DataFrame({idcol: seq_ids, seqcol: seqs})

            chunk_seqs = chunk_seqs.join(
                minipldf.select(merge_cols), on=idcol, how="inner"
            )  # this join get's the info columns (start, end, strand) if needed, only for the entires in this chunk that are in the minipldf.

            if trim_to_region:
                print("Trimming sequences")
                chunk_seqs = chunk_seqs.with_columns(
                    pl.struct(pl.col(seqcol), pl.col(start_col), pl.col(end_col))
                    .map_elements(
                        lambda x: str(x[seqcol][x[start_col] : x[end_col]])
                        if x[seqcol] is not None
                        else None,
                        return_dtype=pl.Utf8,
                    )
                    .alias(seqcol)
                )

            if reverse_by_strand_col:
                print("Reversing sequences")
                chunk_seqs = chunk_seqs.with_columns(
                    pl.when(pl.col(strand_col))
                    .then(
                        pl.col(seqcol).map_elements(
                            lambda x: revcomp(x) if x is not None else None,
                            return_dtype=pl.Utf8,
                        )
                    )
                    .otherwise(pl.col(seqcol))
                    .alias(seqcol)
                )

            print("Joining with nascent df")
            minipldf = minipldf.join(chunk_seqs, on=merge_cols, how="left")
            minipldf = minipldf.with_columns(
                pl.coalesce([pl.col(seqcol), pl.col(f"{seqcol}_right")]).alias(seqcol)
            ).drop(f"{seqcol}_right")

            print(f"Null count in seqcol after chunk: {minipldf[seqcol].null_count()}")

            seqs = []
            seq_ids = []
            # get count for remaining nulls, if zero, break - should be useful when fetching just a few sequences from a large file, at least if the needed seqs are closer to the start of the input fasta.
            if minipldf[seqcol].null_count() == 0:
                break

    print("\nFinal merge with original df")
    pldf = pldf.join(minipldf, on=merge_cols, how="left")
    print(f"Final null count in seqcol: {pldf[seqcol].null_count()}")

    return pldf 