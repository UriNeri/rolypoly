"""
Sequence file I/O operations for reading and writing FASTA/FASTQ files.
Basic sequence analysis and validation functions.
"""
import re
from typing import List, Union, Dict, Tuple, Optional

import polars as pl
from needletail import parse_fastx_file
# from rolypoly.utils.bio.polars_fastx import * # I think this brings in the i/o plugins, TODO: check if can be done so that I don't need to type: ignore all the time
from rolypoly.utils.various import  find_files_by_extension
from pathlib import Path
import logging

# show all columns
# pl.Config().set_tbl_cols(-1)

def read_fasta_needletail(fasta_file: str) -> Tuple[list[str], list[str]]:
    """Read sequences from a FASTA/FASTQ file using needletail"""

    seqs = []
    seq_ids = []
    for record in parse_fastx_file(fasta_file):
        seqs.append(getattr(record, 'seq', ''))  # type: ignore
        seq_ids.append(getattr(record, 'id', ''))  # type: ignore
    return seq_ids, seqs


def read_fasta_df(file_path: str) -> pl.DataFrame:
    """wrapper for legacy code"""
    from rolypoly.utils.bio.polars_fastx import from_fastx_eager
    return from_fastx_eager(file_path) # type: ignore defined in polars_fastx.py


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


def is_nucl_string(sequence, extended=False):
    """Check if a string is a valid nucleotide sequence."""
    valid_characters = set({"A", "T", "G", "C", "U", "N"})
    if extended:
        valid_characters.update({"M", "R", "W", "S", "Y", "K", "V", "H", "D", "B"})
    return all(char in valid_characters for char in sequence.upper())


def is_aa_string(sequence, extended=False):
    """Check if a string is a valid amino acid sequence."""
    valid_characters = set(
        {
            "A",
            "R",
            "N",
            "D",
            "C",
            "Q",
            "E",
            "G",
            "H",
            "I",
            "L",
            "K",
            "M",
            "F",
            "P",
            "S",
            "T",
            "W",
            "Y",
            "V",
            "O",
            "U",
            "B",
            "Z",
            "X",
            "J",
        }
    )
    if extended:
        valid_characters.update({"B", "J", "X", "Z", "*", "-", "."})
    return all(char in valid_characters for char in sequence.upper())


def guess_fasta_alpha(input_file) -> str:
    """Guess the alphabet type (nucleotide vs amino acid) of a FASTA file."""
    # only peek at the first sequence
    with open(input_file, "rb") as fin:
        input_string = get_sequence_between_newlines(
            fin.peek(2)[:1110].decode().replace(r"*/\n", "")
        )
    if is_nucl_string(input_string):
        return "nucl"
    elif is_aa_string(input_string):
        return "amino"
    else:
        return "nothing_good"


def get_sequence_between_newlines(input_string):
    """Extract sequence content between newlines (skipping header)."""
    newline_pattern = re.compile(r"\n")
    newline_positions = [
        match.start() for match in newline_pattern.finditer(input_string)
    ]
    if len(newline_positions) < 2:
        return input_string[newline_positions[0] + 1 :]
    return input_string[newline_positions[0] + 1 : newline_positions[1]]


def revcomp(seq: str) -> str:
    """Calculate reverse complement of a DNA/RNA sequence."""
    import mappy as mp

    return mp.revcomp(seq)


def process_sequences(df: pl.DataFrame) -> pl.DataFrame:
    """Process sequences and calculate statistics.

    Args:
        df (pl.DataFrame): DataFrame with sequence column

    Returns:
        pl.DataFrame: DataFrame with added statistics columns
    """

    # Calculate basic stats
    df = df.with_columns(
        [
            pl.col("sequence").str.len_chars().alias("length"),
            pl.col("sequence").str.count_matches("G|C").alias("gc_content") / pl.col("sequence").str.len_chars().alias("length"),
            pl.col("sequence").str.count_matches("N").alias("n_count"),
        ]
    )

    return df

def rename_sequences(
    df: pl.DataFrame, prefix: str = "CID", use_hash: bool = False
) -> Tuple[pl.DataFrame, Dict[str, str]]:
    """Rename sequences with consistent IDs.

    Args:
        df (pl.DataFrame): DataFrame with 'header' and 'sequence' columns
        prefix (str, optional): Prefix for new IDs. Defaults to "CID".
        use_hash (bool, optional): Use hash instead of numbers. Defaults to False.

    Returns:
        Tuple[pl.DataFrame, Dict[str, str]]:
            - DataFrame with renamed sequences
            - Dictionary mapping old IDs to new IDs
    """

    if use_hash:
        # Use polars expressions for hash generation directly
        import hashlib
        
        def _hash(seq: str) -> str:
            return hashlib.md5(seq.encode()).hexdigest()[:32]
        
        df_with_hashes = df.with_columns(
            pl.col("sequence").map_elements(_hash, return_dtype=pl.String).alias("seq_hash")
        )
        new_headers = [f"{prefix}_{h}" for h in df_with_hashes["seq_hash"]]
    else:
        # Calculate padding based on total number of sequences
        padding = len(str(len(df)))
        new_headers = [f"{prefix}_{str(i + 1).zfill(padding)}" for i in range(len(df))]

    # Create mapping dictionary
    id_map = dict(zip(df["header"], new_headers))

    return df.with_columns(pl.Series("header", new_headers)), id_map 



def find_fasta_files(
    input_path: Union[str, Path],
    extensions: List[str] = None,
    logger: Optional[logging.Logger] = None
) -> List[Path]:
    """Find all FASTA files in a directory or return single file.
    
    Args:
        input_path: Path to directory or file
        extensions: List of extensions to look for
        logger: Logger instance
        
    Returns:
        List of FASTA file paths
    """
    if extensions is None:
        extensions = ["*.fa", "*.fasta", "*.fna", "*.fa.gz", "*.fasta.gz", "*.fna.gz"]
    
    return find_files_by_extension(input_path, extensions, "FASTA files", logger)



def ensure_faidx(input_file: str, logger: Optional[logging.Logger] = None) -> None:
    """Ensure a FASTA file has a pyfastx index.
    
    Creates a pyfastx index for the input FASTA file if it doesn't exist.
    
    Args:
        input_file: Path to the FASTA file
        logger: Logger instance
    """
    import os as os 
    from rolypoly.utils.logging.loggit import get_logger
    logger = get_logger(logger)
    
    try:
        import pyfastx
        from rich.console import Console
        
        console = Console(width=150)
        
        if not os.path.exists(f"{input_file}.fxi"):
            logger.info(f"Indexing {input_file} with pyfastx")
            console.print(f"[yellow]Indexing {input_file} with pyfastx[/yellow]")
            pyfastx.Fasta(str(input_file))
            console.print("[green]Indexing complete.[/green]")
            logger.info("FASTA indexing completed")
        else:
            logger.debug(f"Index already exists for {input_file}")
            
    except ImportError:
        logger.error("pyfastx not available for FASTA indexing")
        raise
    except Exception as e:
        logger.error(f"Error creating FASTA index for {input_file}: {e}")
        raise
        

