"""File format detection and analysis functions."""

import os
import gzip
from pathlib import Path
from typing import Union


def is_gzipped(file_path: str) -> bool:
    """Check if a file is gzip compressed."""
    with open(file_path, "rb") as test_f:
        return test_f.read(2).startswith(b"\x1f\x8b")


def guess_fastq_properties(file_path: str, mb_to_read: int = 20) -> dict:
    """Analyze a FASTQ file to determine its properties.

    Examines the first 20MB of a FASTQ file to determine if it's gzipped,
    paired-end, and calculate average read length.

    Args:
        file_path (str): Path to the FASTQ file
        mb_to_read (int): Number of MB to read for analysis

    Returns:
        dict: Dictionary containing:
            - is_gzipped (bool): Whether file is gzip compressed
            - paired_end (bool): Whether reads appear to be paired-end
            - average_read_length (float): Average length of reads
    """
    bytes_to_read = mb_to_read * 1024 * 1024
    is_gz = is_gzipped(file_path)
    file_size = os.path.getsize(file_path)
    if file_size < bytes_to_read:
        bytes_to_read = file_size
    paired_end = False
    average_read_length = 0
    total_length = 0
    read_count = 0

    # Open the file accordingly
    if is_gz:
        with gzip.open(file_path, "rb") as f:
            data = f.read(bytes_to_read)
    else:
        with open(file_path, "rb") as f:
            data = f.read(bytes_to_read)

    # Decode the data
    data = data.decode("utf-8", errors="ignore")

    # Split the data into lines
    lines = data.splitlines()

    # Process the lines to determine properties
    for i in range(0, len(lines) - len(lines) % 4, 4):
        if lines[i].startswith("@"):
            read_id = lines[i][1:].split()[0]
            if "/1" in read_id or "/2" in read_id:
                paired_end = True
            read_length = len(lines[i + 1])
            total_length += read_length
            read_count += 1

    if read_count > 0:
        average_read_length = total_length / read_count

    return {
        "is_gzipped": is_gz,
        "paired_end": paired_end,
        "average_read_length": average_read_length,
    }


def identify_fastq_files(
    input_path: Union[str, Path], return_rolypoly: bool = True
) -> dict:
    """Identify and categorize FASTQ files from input path.

    Args:
        input_path: Path to input directory or file
        return_rolypoly: Whether to look for and return rolypoly-formatted files first

    Returns:
        dict: Dictionary containing:
            - rolypoly_data: {lib_name: {'interleaved': path, 'merged': path}}
            - R1_R2_pairs: [(r1_path, r2_path), ...]
            - interleaved_files: [path, ...]
            - single_end: [path, ...]

    Note:
        When return_rolypoly is True and rolypoly files are found, other files
        are ignored to maintain consistency with rolypoly pipeline.
    """

    input_path = Path(input_path)
    file_info = {
        "rolypoly_data": {},
        "R1_R2_pairs": [],
        "interleaved_files": [],
        "single_end": [],
    }

    def is_paired_filename(filename: str) -> tuple[bool, str]:
        """Check if filename indicates paired-end data and extract pair info."""
        import re

        patterns = [
            (r".*_R?1[._].*", r".*_R?2[._].*"),  # Matches _R1/_R2, _1/_2
            (r".*_1\.f.*q.*", r".*_2\.f.*q.*"),  # Matches _1.fastq/_2.fastq
            (r".*\.1\.f.*q.*", r".*\.2\.f.*q.*"),  # Matches .1.fastq/.2.fastq
        ]

        for pat in patterns:
            if re.match(pat, filename): # type: ignore
                pair_file = (
                    filename.replace("_R1", "_R2")
                    .replace("_1.", "_2.")
                    .replace(".1.", ".2.")
                )
                return True, pair_file
        return False, ""

    if input_path.is_dir():
        # First look for rolypoly output files
        if return_rolypoly:
            rolypoly_files = list(input_path.glob("*_final_*.f*q*"))
            if rolypoly_files:
                for file in rolypoly_files:
                    lib_name = file.stem.split("_final_")[0]
                    if lib_name not in file_info["rolypoly_data"]:
                        file_info["rolypoly_data"][lib_name] = {
                            "interleaved": None,
                            "merged": None,
                        }
                    if "interleaved" in file.name:
                        file_info["rolypoly_data"][lib_name]["interleaved"] = file
                    elif "merged" in file.name:
                        file_info["rolypoly_data"][lib_name]["merged"] = file
                return file_info

        # Process all fastq files
        all_fastq = list(input_path.glob("*.f*q*"))
        processed_files = set()

        # First pass - identify paired files by name
        for file in all_fastq:
            if file in processed_files:
                continue

            is_paired, pair_file = is_paired_filename(file.name)
            if is_paired and (file.parent / pair_file).exists():
                file_info["R1_R2_pairs"].append((file, file.parent / pair_file))
                processed_files.add(file)
                processed_files.add(file.parent / pair_file)
                continue

            # Check remaining files
            props = guess_fastq_properties(str(file))
            if props["paired_end"]:  # Interleaved paired-end
                file_info["interleaved_files"].append(file)
            else:  # Single-end
                file_info["single_end"].append(file)
            processed_files.add(file)

    else:
        # Single file input
        props = guess_fastq_properties(str(input_path))
        if props["paired_end"]:  # Interleaved paired-end
            file_info["interleaved_files"].append(input_path)
        else:  # Single-end
            file_info["single_end"].append(input_path)

    return file_info


def ensure_faidx(input_file: str) -> None:
    """Ensure a FASTA file has a pyfastx index.

    Creates a pyfastx index for the input FASTA file if it doesn't exist.

    Args:
        input_file (str): Path to the FASTA file
    """
    import pyfastx
    from rich.console import Console
    
    console = Console(width=150)

    if not os.path.exists(f"{input_file}.fxi"):
        console.print(f"[yellow]Indexing {input_file} with pyfastx    [/yellow]")
        pyfastx.Fasta(str(input_file))
        console.print("[green]Indexing complete.[/green]") 