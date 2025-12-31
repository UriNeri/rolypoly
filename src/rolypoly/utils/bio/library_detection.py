"""File format detection and analysis functions.

This module provides comprehensive FASTQ file detection, analysis, and classification
functionality with support for paired-end, interleaved, and single-end files.
"""

import gzip
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from rolypoly.utils.logging.loggit import get_logger
from rolypoly.utils.various import find_files_by_extension, is_gzipped


def create_sample_file(
    file_path: Union[str, Path],
    subset_type: str = "top_reads",
    sample_size: Union[int, float] = 1000,
    output_file: str = "sample.fastq.gz",
    logger: Optional[logging.Logger] = None,
) -> str:
    """Create a temporary sample file from a FASTQ file for analysis.

    Args:
        file_path: Path to the input FASTQ file. If it is 2 paired end files (r1 r2) use , to separate them.
        subset_type: Type of subsert - "top_reads" or "random".
        sample_size: if top_reads than how many reads (from the top) to sample, if random than fracton of reads to sample randomly (0.0-1.0)
        # keep_pairs: Keep paired-end reads - if true input file is assumed to be paired end AND interleaved. all R1 reads in the output will have matching R2. (EDIT- I'm just going to sneakily take half the sample size, get that many random items, then take 2 consecutive reads at a time lol)
        output_file: path to output file - if ending in .gz then will be compressed. If input is 2 paired end files, will assume output also has 2 files in it (R1 and R2) separated by comma.
        logger: Logger instance

    Returns:
        name of output file
    Note:
        - If sample type is random, the total number of reads in the file will have to be computed and that coudl be slow.
        - Generally adivsory to use gzipped output file
        - ..,, to provice an even number for sample_size, if subset_type is top_reads. Otherwise if your input file is interleaved, the last read will lose its pair.
        -
    """
    logger = get_logger(logger)
    file_path = Path(file_path)
    is_paired_files = False if "," not in str(file_path) else True
    is_gz_output = True if Path(output_file).suffix == ".gz" else False

    # Helper to get total reads in a FASTQ file
    def _get_total_reads(fpath, gzipped):
        import subprocess as sp

        if gzipped:
            cmd = f"zgrep -c '^@' {fpath}"
        else:
            cmd = f"grep -c '^@' {fpath}"
        try:
            return int(
                sp.run(
                    cmd, shell=True, capture_output=True, text=True
                ).stdout.strip()
            )
        except Exception as e:
            logger.error(f"Failed to count reads in {fpath}: {e}")
            raise

    # Decide if we need to get total reads first
    need_total_reads = False
    if subset_type == "random":
        need_total_reads = True
    elif (
        subset_type == "top_reads"
        and isinstance(sample_size, float)
        and sample_size < 1.0
    ):
        need_total_reads = True
    logger.debug(f"need_total_reads: {need_total_reads}")

    if not is_paired_files:
        is_gz = is_gzipped(file_path)
        total_reads = None
        if need_total_reads:
            total_reads = _get_total_reads(file_path, is_gz)
            # If sample_size is float, convert to int number of reads
            if isinstance(sample_size, float):
                sample_size = int(sample_size * total_reads)
        logger.debug(f"Sampling {subset_type} of {sample_size} of {file_path}")
        try:
            if subset_type == "top_reads":
                sample_size_int = int(sample_size)
                sample_size_int = sample_size_int - (
                    sample_size_int % 2
                )  # ensure even for pairs
                n_lines = sample_size_int * 4  # 4 lines per read
                # Stream first n lines without loading entire file
                if is_gz:
                    f_in = gzip.open(
                        file_path, "rt", encoding="utf-8", errors="ignore"
                    )
                else:
                    f_in = open(
                        file_path, "r", encoding="utf-8", errors="ignore"
                    )
                if is_gz_output:
                    f_out = gzip.open(output_file, "wt", encoding="utf-8")
                else:
                    f_out = open(output_file, "w", encoding="utf-8")
                for i, line in enumerate(f_in):
                    if i >= n_lines:
                        break
                    f_out.write(line)
                f_out.close()
                f_in.close()
            elif subset_type == "random":
                import itertools
                from random import sample

                import numpy as np

                sample_size_int = int(sample_size)
                sample_size_int = sample_size_int - (sample_size_int % 2)
                if total_reads is None:
                    total_reads = _get_total_reads(file_path, is_gz)
                if sample_size_int > total_reads:
                    logger.warning(
                        f"Requested sample_size {sample_size_int} > total_reads {total_reads}, using all reads."
                    )
                    sample_size_int = total_reads - (total_reads % 2)
                # Each read = 4 lines, so sample indices of reads
                read_indices = sample(range(total_reads), sample_size_int)
                read_indices = np.sort(read_indices)
                # Convert to line numbers
                lines_2_get = np.sort(
                    list(
                        itertools.chain.from_iterable(
                            [
                                [i * 4 + j for j in range(4)]
                                for i in read_indices
                            ]
                        )
                    )
                )
                target_set = set(lines_2_get)
                target_iter = iter(lines_2_get)
                try:
                    next_target = next(target_iter)
                except StopIteration:
                    next_target = None
                f_in = (
                    gzip.open(
                        file_path, "rt", encoding="utf-8", errors="ignore"
                    )
                    if is_gz
                    else open(file_path, "r", encoding="utf-8", errors="ignore")
                )
                f_out = (
                    gzip.open(output_file, "wt", encoding="utf-8")
                    if is_gz_output
                    else open(output_file, "w", encoding="utf-8")
                )
                for i, line in enumerate(f_in):
                    if next_target is not None and i > lines_2_get[-1]:
                        break
                    if i in target_set:
                        f_out.write(line)
                        try:
                            next_target = next(target_iter)
                        except StopIteration:
                            next_target = None
                f_in.close()
                f_out.close()
        except Exception as e:
            logger.error(f"Error creating sample file from {file_path}: {e}")
            raise
    elif "," in str(file_path):
        # Assume 2 paired end files. Like above but first pass on R1 file we keep the read names selected, then second pass on R2 file we keep the read headers that have 2 instead of 1 in final header char.
        logger.debug(f"Sampling {subset_type} of {sample_size} of {file_path}")
        try:
            r1_path, r2_path = str(file_path).split(",")
            r1_path = Path(r1_path)
            r2_path = Path(r2_path)
            is_gz = is_gzipped(r1_path)
            r1_output_file, r2_output_file = output_file.split(",")
            total_reads = None
            if need_total_reads:
                total_reads = _get_total_reads(r1_path, is_gz)
                if isinstance(sample_size, float):
                    sample_size = int(sample_size * total_reads)
            if subset_type == "top_reads":
                sample_size_int = int(sample_size)
                sample_size_int = sample_size_int - (sample_size_int % 2)
                lines_2_get = [i for i in range(0, sample_size_int * 2, 4)]
            else:
                import itertools
                from random import sample

                import numpy as np

                sample_size_int = int(sample_size)
                sample_size_int = sample_size_int - (sample_size_int % 2)
                if total_reads is None:
                    total_reads = _get_total_reads(r1_path, is_gz)
                if sample_size_int > total_reads:
                    logger.warning(
                        f"Requested sample_size {sample_size_int} > total_reads {total_reads}, using all reads."
                    )
                    sample_size_int = total_reads - (total_reads % 2)
                read_indices = sample(range(total_reads), sample_size_int)
                read_indices = np.sort(read_indices)
                lines_2_get = np.sort(
                    list(
                        itertools.chain.from_iterable(
                            [
                                [i * 4 + j for j in range(4)]
                                for i in read_indices
                            ]
                        )
                    )
                )
            target_set = set(lines_2_get)
            target_iter = iter(lines_2_get)
            try:
                next_target = next(target_iter)
            except StopIteration:
                next_target = None
            # first pass - R1
            headers = []
            f_in = (
                gzip.open(r1_path, "rt", encoding="utf-8", errors="ignore")
                if is_gz
                else open(r1_path, "r", encoding="utf-8", errors="ignore")
            )
            f_out = (
                gzip.open(r1_output_file, "wt", encoding="utf-8")
                if is_gz_output
                else open(r1_output_file, "w", encoding="utf-8")
            )
            for i, line in enumerate(f_in):
                if next_target is None or i >= lines_2_get[-1] + 1:
                    break
                if i in target_set:
                    f_out.write(line)
                    if line.startswith("@"):
                        headers.append(line.strip())
                    try:
                        next_target = next(target_iter)
                    except StopIteration:
                        next_target = None
            f_in.close()
            f_out.close()
            # second pass - R2
            headers_2 = {str(h).removesuffix("1") + "2" for h in headers}
            f_in = (
                gzip.open(r2_path, "rt", encoding="utf-8", errors="ignore")
                if is_gz
                else open(r2_path, "r", encoding="utf-8", errors="ignore")
            )
            f_out = (
                gzip.open(r2_output_file, "wt", encoding="utf-8")
                if is_gz_output
                else open(r2_output_file, "w", encoding="utf-8")
            )
            while headers_2:
                try:
                    line = next(f_in)
                except StopIteration:
                    logger.warning(
                        f"Reached end of {r2_path} before finding all expected headers. "
                        f"{len(headers_2)} read pair(s) missing."
                    )
                    break
                if line.startswith("@"):
                    stripped = line.strip()
                    if stripped in headers_2:
                        f_out.write(line)
                        f_out.write(next(f_in))
                        f_out.write(next(f_in))
                        f_out.write(next(f_in))
                        headers_2.remove(stripped)
                        continue
            f_in.close()
            f_out.close()
            if headers_2.__len__() > 0:
                logger.warning(
                    "WHOW! not all headers were found in R2 - THIS IS NOT A GOOD SIGN"
                )
        except Exception as e:
            logger.error(f"Error creating sample file from {file_path}: {e}")


def determine_fastq_type(
    file_path: Union[str, Path],
    sample_size: int = 1000,  # Increased from whenever. should be consisent as long as the number is even..
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """Analyze FASTQ headers to determine file characteristics.

    Args:
        file_path: Path to FASTQ file or sample file
        sample_size: Number of reads (from the top of file) to use
        logger: Logger instance

    Returns:
        Dictionary containing header analysis results
    """
    import polars as pl

    logger = get_logger(logger)
    from needletail import (
        decode_phred,  # could be cool to add a mean quality score calculation here.
    )

    from rolypoly.utils.bio.polars_fastx import from_fastx_lazy as read_fastx

    results = {
        "file_type": "unknown",
        "is_gzipped": is_gzipped(file_path),
        "pair_1_count": 0,
        "pair_2_count": 0,
        "average_read_length": 0,
        "average_read_quality": 0,
    }
    try:
        file_path = Path(file_path)
        fastq_df = read_fastx(file_path)
        fastq_df = fastq_df.head(sample_size).collect()
        header_count = fastq_df.select(
            pl.col("header").str.tail(2).value_counts()
        ).unnest("header")
        logger.debug(
            f"example read headers: {fastq_df.select(pl.col('header')).head(5).to_series().to_list()}"
        )
        # Check suffix patterns for paired-end indicators
        # logger.debug(f"header_count: {header_count}")
        pair_1_count = header_count.filter(
            pl.col("header").is_in([" 1", "/1"])
        )["count"].sum()
        pair_2_count = header_count.filter(
            pl.col("header").is_in([" 2", "/2"])
        )["count"].sum()
        # Check if header looks like old Casava format,e.g. @A00178:83:HJ73JDSXX:1:1101:10285:2394 1:N:0:AGGCTTCT+AGAAGCCT (the "n:" part after the space is the important bit, and we will want to look for the leading string to it to exist in both the 1 and 2 forms)
        if pair_1_count == 0 and pair_2_count == 0:
            # Check for Casava format: space followed by 1: or 2:
            # Extract base header (before space) for reads with " 1:" and " 2:"
            headers_with_1 = fastq_df.filter(
                pl.col("header").str.contains(r" 1:")
            ).select(
                pl.col("header").str.split(" ").list.get(0).alias("base_header")
            )

            headers_with_2 = fastq_df.filter(
                pl.col("header").str.contains(r" 2:")
            ).select(
                pl.col("header").str.split(" ").list.get(0).alias("base_header")
            )

            # Check if there are overlapping base headers (indicating paired reads)
            if headers_with_1.height > 0 and headers_with_2.height > 0:
                set_1 = set(headers_with_1["base_header"].to_list())
                set_2 = set(headers_with_2["base_header"].to_list())
                overlap = set_1.intersection(set_2)

                if len(overlap) == len(set_1) and len(overlap) == len(set_2):
                    # Found matching pairs in Casava format
                    pair_1_count = headers_with_1.height
                    pair_2_count = headers_with_2.height
                    logger.warning(
                        f"Detected Casava paired-end format in headers - treating as interleaved paired-end reads... this could be wrong..."
                    )

        # total_length = fastq_df.select(pl.col("sequence").str.len_chars()).sum().item() # could have just
        average_read_length = (
            fastq_df.select(pl.col("sequence").str.len_chars()).mean().item()
        )
        from numpy import mean

        average_read_quality = fastq_df.select(
            pl.col("sequence")
            .map_elements(
                lambda x: mean(decode_phred(x)), return_dtype=pl.Float64
            )
            .mean()
        ).item()

        # add to results dict
        results["average_read_length"] = average_read_length
        results["average_read_quality"] = average_read_quality
        results["pair_1_count"] = pair_1_count
        results["pair_2_count"] = pair_2_count
        if pair_1_count == sample_size / 2 and pair_2_count == pair_1_count:
            results["file_type"] = "interleaved"
        elif pair_1_count == sample_size and pair_2_count == 0:
            results["file_type"] = "paired_R1"
        elif pair_1_count == 0 and pair_2_count == sample_size:
            results["file_type"] = "paired_R2"
        elif pair_1_count == 0 and pair_2_count == 0:
            results["file_type"] = "single"  # this is a guess
        logger.debug(f"Header analysis for {file_path}: {results}")

    except Exception as e:
        logger.error(f"Error analyzing  {file_path}: {e}")
    return results


def is_paired_filename(
    filename: str, logger: Optional[logging.Logger] = None
) -> Tuple[bool, str]:
    """Check if filename indicates paired-end data and extract pair info.

    Args:
        filename: Name of the file to check
        logger: Logger instance

    Returns:
        Tuple of (is_paired, pair_filename)
    """
    logger = get_logger(logger)

    patterns = [
        (r"(.*)_R1([._].*)$", r"\1_R2\2"),  # _R1/_R2
        (r"(.*)_1([._].*)$", r"\1_2\2"),  # _1/_2
        (
            r"(.*)\.1(\.f.*q.*)$",
            r"\1.2\2",
        ),  # .1.fastq/.2.fastq # not sre if the f*q* is required.
    ]

    for pattern, replacement in patterns:
        match = re.match(pattern, filename)
        if match:
            pair_file = re.sub(pattern, replacement, filename)
            logger.debug(
                f"Detected paired filename pattern: {filename} -> {pair_file}"
            )
            return True, pair_file

    return False, ""


def identify_fastq_files(
    input_path: Union[str, Path],
    return_rolypoly: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """Identify and categorize FASTQ files from input path.

    Args:
        input_path: Path to input directory or file
        return_rolypoly: Whether to look for and return rolypoly-formatted files first
        logger: Logger instance

    Returns:
        Dictionary containing categorized file information:
        - rolypoly_data: {lib_name: {'interleaved': path, 'merged': path}}
        - R1_R2_pairs: [(r1_path, r2_path), ...]
        - interleaved_files: [path, ...]
        - single_end: [path, ...]
        - file_details: {file_path: analysis_results}
    """
    logger = get_logger(logger)
    input_path = Path(input_path)

    logger.info(f"Identifying FASTQ files in: {input_path}")

    file_info = {
        "rolypoly_data": {},
        "R1_R2_pairs": [],
        "interleaved_files": [],
        "single_end": [],
        "file_details": {},
    }

    if input_path.is_dir():
        # First look for rolypoly output files if requested - these are expected to be named like "lib_name_final_interleaved.fq.gz" and "lib_name_final_merged.fq.gz"
        if return_rolypoly:
            rolypoly_files = list(input_path.glob("*_final_*.f*q*"))
            if rolypoly_files:
                logger.info(
                    f"Found {len(rolypoly_files)} rolypoly output files"
                )
                for file in rolypoly_files:
                    lib_name = file.stem.split("_final_")[0]
                    if lib_name not in file_info["rolypoly_data"]:
                        file_info["rolypoly_data"][lib_name] = {
                            "interleaved": None,
                            "merged": None,
                        }
                    if "interleaved" in file.name:
                        file_info["rolypoly_data"][lib_name]["interleaved"] = (
                            file
                        )
                        logger.debug(
                            f"Added rolypoly interleaved: {lib_name} -> {file}"
                        )
                    elif "merged" in file.name:
                        file_info["rolypoly_data"][lib_name]["merged"] = file
                        logger.debug(
                            f"Added rolypoly merged: {lib_name} -> {file}"
                        )

                # Analyze rolypoly files - is this neccessary? shouldn't some other part of my code be writting this and thus I can trust myself to expect... correct formatting?
                for lib_name, data in file_info["rolypoly_data"].items():
                    for file_type, file_path in data.items():
                        if file_path:
                            analysis = determine_fastq_type(
                                file_path, logger=logger
                            )
                            file_info["file_details"][str(file_path)] = analysis

                return file_info

        # Process all FASTQ files
        all_fastq = find_files_by_extension(
            input_path,
            ["*.fq", "*.fastq", "*.fq.gz", "*.fastq.gz"],
            "FASTQ files",
            logger,
        )
        processed_files = set()

        logger.info(f"Processing {len(all_fastq)} FASTQ files")

        # First pass - identify paired files by filename
        for file in all_fastq:
            if file in processed_files:
                continue

            is_paired, pair_file = is_paired_filename(file.name, logger)
            if is_paired:
                pair_path = file.parent / pair_file
                if pair_path.exists() and pair_path in all_fastq:
                    # Analyze both files
                    r1_analysis = determine_fastq_type(file, logger=logger)
                    r2_analysis = determine_fastq_type(pair_path, logger=logger)

                    file_info["file_details"][str(file)] = r1_analysis
                    file_info["file_details"][str(pair_path)] = r2_analysis

                    file_info["R1_R2_pairs"].append((file, pair_path))
                    processed_files.add(file)
                    processed_files.add(pair_path)

                    logger.debug(
                        f"Added R1/R2 pair: {file.name} <-> {pair_file}"
                    )
                    continue

        # Second pass - analyze remaining files
        for file in all_fastq:
            if file in processed_files:
                continue

            logger.debug(f"Analyzing remaining file: {file}")
            analysis = determine_fastq_type(file, logger=logger)
            file_info["file_details"][str(file)] = analysis

            # Categorize based on analysis
            # breakpoint()
            if analysis["file_type"] == "interleaved":
                file_info["interleaved_files"].append(file)
                logger.debug(f"Categorized as interleaved: {file}")
            elif analysis["file_type"] == "single":
                file_info["single_end"].append(file)
                logger.debug(f"Categorized as single-end: {file}")
            else:
                # Default to single-end if unclear
                file_info["single_end"].append(file)
                logger.warning(
                    f"Unclear file type, defaulting to single-end: {file}"
                )

            processed_files.add(file)

    else:
        # Single file input
        logger.info(f"Analyzing single file: {input_path}")
        analysis = determine_fastq_type(input_path, logger=logger)
        file_info["file_details"][str(input_path)] = analysis

        if analysis["file_type"] == "interleaved":
            file_info["interleaved_files"].append(input_path)
        else:
            file_info["single_end"].append(input_path)

    # Log debug, should usualy be printed in the summary
    logger.debug("File identification summary:")
    logger.debug(f"  - Rolypoly libraries: {len(file_info['rolypoly_data'])}")
    logger.debug(f"  - R1/R2 file pairs: {len(file_info['R1_R2_pairs'])}")
    logger.debug(
        f"  - Interleaved files: {len(file_info['interleaved_files'])}"
    )
    logger.debug(f"  - Single-end files: {len(file_info['single_end'])}")

    return file_info


def handle_input_fastq(
    input_path: Union[str, Path], logger: Optional[logging.Logger] = None
) -> Dict:
    """Handle input FASTQ files and prepare file information for processing.

    This function is designed to be compatible with the filter_reads workflow.
    It uses the consolidated file detection functions and returns information
    in a format expected by the read filtering pipeline.

    Args:
        input_path: Path to input directory or file(s)
        logger: Logger instance

    Returns:
        Dictionary containing:
        - R1_R2_pairs: List of (R1, R2) path tuples
        - interleaved_files: List of interleaved file paths
        - single_end_files: List of single-end file paths
        - file_name: Suggested base name for output files
    """
    logger = get_logger(logger)
    input_path = Path(input_path)

    # Handle comma-separated file inputs (common in filter_reads usage)
    if isinstance(input_path, (str, Path)) and "," in str(input_path):
        # Split comma-separated files
        file_paths = [Path(p.strip()) for p in str(input_path).split(",")]

        if len(file_paths) == 2:
            # Assume R1, R2 pair
            r1_path, r2_path = file_paths

            # Generate file name from R1
            file_name = r1_path.stem
            if file_name.endswith(".fq") or file_name.endswith(".fastq"):
                file_name = file_name.rsplit(".", 1)[0]
            # Remove R1/R2 indicators
            for pattern in ["_R1", "_1", ".1"]:
                if pattern in file_name:
                    file_name = file_name.replace(pattern, "")
                    break

            logger.info(f"Detected paired files: {r1_path} and {r2_path}")

            return {
                "R1_R2_pairs": [(r1_path, r2_path)],
                "interleaved_files": [],
                "single_end_files": [],
                "file_name": file_name,
            }
        else:
            # Multiple single files
            logger.info(f"Detected {len(file_paths)} individual files")

            # Use first file for naming
            file_name = file_paths[0].stem
            if file_name.endswith(".fq") or file_name.endswith(".fastq"):
                file_name = file_name.rsplit(".", 1)[0]

            return {
                "R1_R2_pairs": [],
                "interleaved_files": [],
                "single_end_files": file_paths,
                "file_name": file_name,
            }

    # Use consolidated file detection for directory or single file
    file_info = identify_fastq_files(
        input_path, return_rolypoly=False, logger=logger
    )

    # Generate appropriate file name
    file_name = "rolypoly_filtered_reads"

    if input_path.is_file():
        # Single file input
        file_name = input_path.stem
        if file_name.endswith(".fq") or file_name.endswith(".fastq"):
            file_name = file_name.rsplit(".", 1)[0]
        # Remove R1/R2 indicators
        for pattern in ["_R1", "_1", ".1", "_R2", "_2", ".2"]:
            if pattern in file_name:
                file_name = file_name.replace(pattern, "")
                break
    elif input_path.is_dir():
        # Use directory name as base
        file_name = input_path.name

    # Convert our file_info format to the expected format
    result = {
        "R1_R2_pairs": file_info["R1_R2_pairs"],
        "interleaved_files": file_info["interleaved_files"],
        "single_end_files": file_info["single_end"],  # Note: different key name
        "file_name": file_name,
    }

    # Add rolypoly data if present
    if file_info["rolypoly_data"]:
        result["rolypoly_data"] = file_info["rolypoly_data"]

    logger.info(f"File handling summary for path '{input_path.absolute()}':")
    logger.info(f"  - File name: {file_name}")
    logger.info(f"  - R1/R2 pairs: {len(result['R1_R2_pairs'])}")
    logger.info(f"  - Interleaved files: {len(result['interleaved_files'])}")
    logger.info(f"  - Single-end files: {len(result['single_end_files'])}")

    return result
