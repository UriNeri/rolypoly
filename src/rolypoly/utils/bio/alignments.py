"""Alignment and mapping utility functions."""

import re
from pathlib import Path
from typing import List, Optional, Union
from rolypoly.utils.various import find_files_by_extension
import logging

def find_msa_files(
    input_path: Union[str, Path],
    extensions: List[str] = None,
    logger: Optional[logging.Logger] = None
) -> List[Path]:
    """Find all Multiple Sequence Alignment files in a directory or return single file.
    
    Args:
        input_path: Path to directory or file
        extensions: List of extensions to look for
        logger: Logger instance
        
    Returns:
        List of MSA file paths
    """
    if extensions is None:
        extensions = ["*.faa", "*.afa", "*.aln", "*.msa"]
    
    return find_files_by_extension(input_path, extensions, "MSA files", logger)

def calculate_percent_identity(cigar_string: str, num_mismatches: int) -> float:
    """Calculate sequence identity percentage from CIGAR string and edit distance.

    Computes the percentage identity between aligned sequences using the CIGAR
    string from an alignment and the number of mismatches (NM tag).

    Args:
        cigar_string (str): CIGAR string from sequence alignment
        num_mismatches (int): Number of mismatches (edit distance)

    Returns:
        float: Percentage identity between sequences (0-100)

    Note:
        The calculation considers matches (M), insertions (I), deletions (D),
        and exact matches (=) from the CIGAR string.

    Example:
         print(calculate_percent_identity("100M", 0))
         100.0
         print(calculate_percent_identity("100M", 2))
         98.0
    """

    cigar_tuples = re.findall(r"(\d+)([MIDNSHPX=])", cigar_string)
    matches = sum(int(length) for length, op in cigar_tuples if op in {"M", "=", "X"})
    total_length = sum(
        int(length) for length, op in cigar_tuples if op in {"M", "I", "D", "=", "X"}
    )
    return (matches - num_mismatches) / total_length * 100 