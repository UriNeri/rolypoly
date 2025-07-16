"""Basic sequence analysis and validation functions."""

import re
from typing import Dict, Tuple

import polars as pl


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