"""File format detection and analysis functions.

This module provides comprehensive FASTQ file detection, analysis, and classification
functionality with support for paired-end, interleaved, and single-end files.
"""

import gzip
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from rolypoly.utils.logging.loggit import get_logger
from rolypoly.utils.various import find_files_by_extension, is_gzipped

CASAVA_HEADER_RE = re.compile(
    r"^(?P<instrument>[^:\s]+):(?P<run_id>[^:\s]+):(?P<flowcell_id>[^:\s]+):"
    r"(?P<lane>\d+):(?P<tile>\d+):(?P<x_coord>\d+):(?P<y_coord>\d+)"
    r"(?:\s+(?P<read_num>[12]):(?P<is_filtered>[YN]):(?P<control>\d+):(?P<barcode>[^\s]+))?$"
)
SLASH_PAIR_HEADER_RE = re.compile(
    r"^(?:(?P<accession>[^:\s]+)\s+)?"
    r"(?P<instrument>[^:\s]+):(?P<run_id>[^:\s]+):(?P<flowcell_id>[^:\s]+):"
    r"(?P<lane>\d+):(?P<tile>\d+):(?P<x_coord>\d+):(?P<y_coord>\d+)"
    r"/(?P<read_num>[12])$"
)
LEGACY_PAIR_RE = re.compile(r"^(?P<base>.+?)(?:\s|/)(?P<mate>[12])$")
MATE_SUFFIX_RE = re.compile(r"([_\.-])(R?[12])$")
READ_EXT_SUFFIX_RE = re.compile(r"\.(?:f(?:ast)?q|fq|fa|fasta|fna)(?:_[A-Za-z0-9]+)?$", re.IGNORECASE)


def _append_unique(values: list[str], value: str, limit: int = 5) -> None:
    #TODO: is this really needed? maybe use set and get first n instead? but then we lose order, which I'm not sure is important. 
    if value and value not in values and len(values) < limit:
        values.append(value)


def normalize_fastq_sample_name(name: str) -> str:
    """Strip paired-read and FASTQ-style suffixes from a sample basename."""
    normalized = str(name)
    changed = True
    while changed:
        changed = False
        mate_match = MATE_SUFFIX_RE.search(normalized)
        if mate_match:
            normalized = normalized[: mate_match.start()]
            changed = True
            continue
        ext_match = READ_EXT_SUFFIX_RE.search(normalized)
        if ext_match:
            normalized = normalized[: ext_match.start()]
            changed = True
    return normalized


def analyze_fastq_header_metadata(
    headers: list[str],
) -> Dict[str, Any]:
    """Summarize header-derived metadata from a small read sample."""
    summary: Dict[str, Any] = {
        "sample_size": len(headers),
        "format": "unknown",
        "has_sequencer": False,
        "sequencers": [],
        "has_tile": False,
        "tiles": [],
        "has_xy_coordinates": False,
        "xy_coordinates": [],
        "has_barcode": False,
        "barcodes": [],
        "has_pair_mate": False,
        "pair_mates": {"1": 0, "2": 0},
        "casava_count": 0,
        "slash_pair_count": 0,
        "legacy_pair_count": 0,
        "example_headers": headers[:3],
    }

    for raw_header in headers:
        header = str(raw_header).lstrip("@").strip()
        if not header:
            continue

        casava_match = CASAVA_HEADER_RE.match(header)
        header_format = "casava"
        if casava_match:
            summary["casava_count"] += 1
        else:
            slash_pair_match = SLASH_PAIR_HEADER_RE.match(header)
            if slash_pair_match:
                casava_match = slash_pair_match
                summary["slash_pair_count"] += 1
                header_format = (
                    "prefixed_slash_pair"
                    if slash_pair_match.group("accession")
                    else "slash_pair"
                )
        if casava_match:
            summary["format"] = (
                header_format if summary["format"] == "unknown" else summary["format"]
            )
            summary["has_sequencer"] = True
            summary["has_tile"] = True
            summary["has_xy_coordinates"] = True
            summary["has_pair_mate"] = True
            instrument = casava_match.group("instrument")
            tile = casava_match.group("tile")
            x_coord = casava_match.group("x_coord")
            y_coord = casava_match.group("y_coord")
            barcode = (
                casava_match.group("barcode")
                if "barcode" in casava_match.re.groupindex
                else None
            )
            read_num = casava_match.group("read_num")
            _append_unique(summary["sequencers"], instrument)
            _append_unique(summary["tiles"], tile)
            _append_unique(summary["xy_coordinates"], f"{x_coord},{y_coord}")
            if barcode and barcode not in {"0", "N", "NN", "N:N"}:
                summary["has_barcode"] = True
                _append_unique(summary["barcodes"], barcode)
            if read_num in {"1", "2"}:
                summary["pair_mates"][read_num] += 1
            continue

        legacy_match = LEGACY_PAIR_RE.match(header)
        if legacy_match:
            summary["legacy_pair_count"] += 1
            summary["has_pair_mate"] = True
            mate = legacy_match.group("mate")
            if mate in {"1", "2"}:
                summary["pair_mates"][mate] += 1

    if summary["casava_count"] > 0 and summary["format"] == "unknown":
        summary["format"] = "casava"
    elif summary["slash_pair_count"] > 0 and summary["format"] == "unknown":
        summary["format"] = "prefixed_slash_pair"
    elif summary["legacy_pair_count"] > 0 and summary["format"] == "unknown":
        summary["format"] = "legacy_pair_suffix"

    return summary


def create_sample_file(
    file_path: Union[str, Path],
    subset_type: str = "top_reads",
    sample_size: Union[int, float] = 1000,
    output_file: str = "sample.fastq.gz",
    threads: int = 1,
    bbnorm_min_depth: int = 2,
    interleaved: Optional[bool] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Create a temporary sample file from a FASTQ file for analysis.

    Args:
        file_path: Path to the input FASTQ file. If it is 2 paired end files (r1 r2) use , to separate them.
        subset_type: Type of subsert - "top_reads" or "random".
        sample_size: if top_reads than how many reads (from the top) to sample, if random than fracton of reads to sample randomly (0.0-1.0)
        # keep_pairs: Keep paired-end reads - if true input file is assumed to be paired end AND interleaved. all R1 reads in the output will have matching R2. (EDIT- I'm just going to sneakily take half the sample size, get that many random items, then take 2 consecutive reads at a time lol)
        output_file: path to output file - if ending in .gz then will be compressed. If input is 2 paired end files, will assume output also has 2 files in it (R1 and R2) separated by comma.
        threads: Threads for bbnorm mode.
        bbnorm_min_depth: Minimum depth threshold for bbnorm mode.
        interleaved: Explicitly force single-file paired interleaving for bbnorm.
        logger: Logger instance

    Returns:
        name of output file
    Note:
        - If sample type is random, the total number of reads in the file will have to be computed and that coudl be slow.
        - Generally adivsory to use gzipped output file
        - ..,, to provide an even number for sample_size, if subset_type is top_reads. Otherwise if your input file is interleaved, the last read will lose its pair.
        -
    """
    logger = get_logger(logger)
    file_path = Path(file_path)
    is_paired_files = False if "," not in str(file_path) else True
    is_gz_output = True if Path(output_file).suffix == ".gz" else False

    # Helper to get total reads in a FASTQ file
    def get_total_reads(fpath, gzipped):
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

    if subset_type == "bbnorm":
        from bbmapy import bbnorm

        target = max(1, int(sample_size))
        logger.debug(f"Normalizing with bbnorm target={target} for {file_path}")
        try:
            if is_paired_files:
                r1_path, r2_path = str(file_path).split(",")
                out1, out2 = output_file.split(",")
                bb_stdout, bb_stderr = bbnorm(
                    **{
                        "in1": str(Path(r1_path)),
                        "in2": str(Path(r2_path)),
                        "out1": str(Path(out1)),
                        "out2": str(Path(out2)),
                        "target": target,
                        "min": bbnorm_min_depth,
                        "threads": threads,
                        "capture_output": True,
                    }
                )
            else:
                bbnorm_kwargs: Dict[str, Any] = {
                    "in": str(file_path),
                    "out": str(output_file),
                    "target": target,
                    "min": bbnorm_min_depth,
                    "threads": threads,
                    "capture_output": True,
                }
                if interleaved is not None:
                    bbnorm_kwargs["interleaved"] = "t" if interleaved else "f"
                bb_stdout, bb_stderr = bbnorm(**bbnorm_kwargs)

            if bb_stdout or bb_stderr:
                logger.info(
                    "%s",
                    "\n".join(
                        part
                        for part in [
                            str(bb_stderr).strip() if bb_stderr else "",
                            str(bb_stdout).strip() if bb_stdout else "",
                        ]
                        if part
                    ),
                )
            return str(output_file)
        except Exception as e:
            logger.error(f"Error creating bbnorm sample file from {file_path}: {e}")
            raise

    if not is_paired_files:
        is_gz = is_gzipped(file_path)
        total_reads = None
        if need_total_reads:
            total_reads = get_total_reads(file_path, is_gz)
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
                    total_reads = get_total_reads(file_path, is_gz)
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
                total_reads = get_total_reads(r1_path, is_gz)
                if isinstance(sample_size, float):
                    sample_size = int(sample_size * total_reads)
            if subset_type == "top_reads":
                sample_size_int = int(sample_size)
                sample_size_int = sample_size_int - (sample_size_int % 2)
                n_lines = sample_size_int * 4

                # For paired top-read sampling, copy complete FASTQ records from
                # both files directly. This keeps R1/R2 synchronized and avoids
                # header-only output when selecting line indices.
                f_in_r1 = (
                    gzip.open(r1_path, "rt", encoding="utf-8", errors="ignore")
                    if is_gz
                    else open(r1_path, "r", encoding="utf-8", errors="ignore")
                )
                f_out_r1 = (
                    gzip.open(r1_output_file, "wt", encoding="utf-8")
                    if is_gz_output
                    else open(r1_output_file, "w", encoding="utf-8")
                )
                for i, line in enumerate(f_in_r1):
                    if i >= n_lines:
                        break
                    f_out_r1.write(line)
                f_in_r1.close()
                f_out_r1.close()

                f_in_r2 = (
                    gzip.open(r2_path, "rt", encoding="utf-8", errors="ignore")
                    if is_gz
                    else open(r2_path, "r", encoding="utf-8", errors="ignore")
                )
                f_out_r2 = (
                    gzip.open(r2_output_file, "wt", encoding="utf-8")
                    if is_gz_output
                    else open(r2_output_file, "w", encoding="utf-8")
                )
                for i, line in enumerate(f_in_r2):
                    if i >= n_lines:
                        break
                    f_out_r2.write(line)
                f_in_r2.close()
                f_out_r2.close()
                return str(output_file)
            else:
                import itertools
                from random import sample

                import numpy as np

                sample_size_int = int(sample_size)
                sample_size_int = sample_size_int - (sample_size_int % 2)
                if total_reads is None:
                    total_reads = get_total_reads(r1_path, is_gz)
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


def probe_fastq_inputs(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    sample_size: int = 100000,
    subset_type: str = "top_reads",
    include_single_end: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Create sampled FASTQ subsets for lightweight downstream probing.

    This is intended for preflight analysis such as adapter discovery or
    lightweight scan commands. It reuses the standard input classification path,
    writes sampled subsets to ``output_dir``, and returns fresh file_info for the
    probe subset.
    """
    logger = get_logger(logger)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_info = handle_input_fastq(input_path, logger=logger)

    for file_path in source_info.get("interleaved_files", []):
        file_path = Path(file_path)
        output_file = output_dir / f"{file_path.stem}_probe.fq.gz"
        create_sample_file(
            file_path=file_path,
            subset_type=subset_type,
            sample_size=sample_size,
            output_file=str(output_file),
            interleaved=True,
            logger=logger,
        )

    for r1_path, r2_path in source_info.get("R1_R2_pairs", []):
        r1_path = Path(r1_path)
        r2_path = Path(r2_path)
        output_r1 = output_dir / f"{r1_path.stem}_probe_R1.fq.gz"
        output_r2 = output_dir / f"{r2_path.stem}_probe_R2.fq.gz"
        create_sample_file(
            file_path=f"{r1_path},{r2_path}",
            subset_type=subset_type,
            sample_size=sample_size,
            output_file=f"{output_r1},{output_r2}",
            logger=logger,
        )

    if include_single_end:
        for file_path in source_info.get("single_end_files", []):
            file_path = Path(file_path)
            output_file = output_dir / f"{file_path.stem}_probe.fq.gz"
            create_sample_file(
                file_path=file_path,
                subset_type=subset_type,
                sample_size=sample_size,
                output_file=str(output_file),
                interleaved=False,
                logger=logger,
            )

    probe_info = handle_input_fastq(output_dir, logger=logger)
    logger.info(
        "Prepared FASTQ probe subset in %s using %s reads per input.",
        output_dir,
        sample_size,
    )
    return {
        "source_file_info": source_info,
        "probe_dir": output_dir,
        "probe_file_info": probe_info,
    }


def determine_fastq_type(
    file_path: Union[str, Path],
    sample_size: int = 1000,  # Increased from whenever. should be consisent as long as the number is even..
    header_sample_size: int = 100,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """Analyze FASTQ headers to determine file characteristics.

    Args:
        file_path: Path to FASTQ file or sample file
        sample_size: Number of reads (from the top of file) to use for read stats
        header_sample_size: Number of reads to use for header metadata analysis
        logger: Logger instance

    Returns:
        Dictionary containing header analysis results
    """
    import polars as pl

    logger = get_logger(logger)
    from rolypoly.utils.bio.polars_fastx import from_fastx_lazy as read_fastx

    results = {
        "file_type": "unknown",
        "is_gzipped": is_gzipped(file_path),
        "pair_1_count": 0,
        "pair_2_count": 0,
        "average_read_length": 0,
        "average_read_quality": 0,
        "header_analysis": {},
    }
    try:
        file_path = Path(file_path)
        header_df = read_fastx(file_path).head(header_sample_size).collect()
        header_analysis = analyze_fastq_header_metadata(
            header_df.select(pl.col("header")).to_series().to_list()
        )
        results["header_analysis"] = header_analysis

        fastq_df = read_fastx(file_path).head(sample_size).collect()
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
                        "Detected Casava paired-end format in headers - treating as interleaved paired-end reads... this could be wrong..."
                    )

        average_read_length = fastq_df.select(
            pl.col("sequence").seq.length().mean().alias("average_read_length")
        ).item()
        average_read_quality = (
            fastq_df.select(
                pl.col("quality").seq.avg_quality().mean().alias(
                    "average_read_quality"
                )
            ).item()
            if "quality" in fastq_df.columns
            else 0.0
        )

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

        if (
            header_analysis.get("has_sequencer")
            or header_analysis.get("has_tile")
            or header_analysis.get("has_xy_coordinates")
            or header_analysis.get("has_barcode")
            or header_analysis.get("has_pair_mate")
        ):
            logger.debug(
                "Header metadata for %s (first %s reads): format=%s sequencer=%s tile=%s xy=%s barcode=%s pair_mates=%s",
                file_path,
                header_sample_size,
                header_analysis.get("format", "unknown"),
                ", ".join(header_analysis.get("sequencers", [])) or "none",
                ", ".join(header_analysis.get("tiles", [])) or "none",
                ", ".join(header_analysis.get("xy_coordinates", [])) or "none",
                ", ".join(header_analysis.get("barcodes", [])) or "none",
                header_analysis.get("pair_mates", {}),
            )
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
        (r"(.*)([_\.-])R1([._-].*)$", r"\g<1>\g<2>R2\3"),
        (r"(.*)([_\.-])1([._-].*)$", r"\g<1>\g<2>2\3"),
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

    logger.debug(f"Identifying FASTQ files in: {input_path}")

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

def identify_fasta_files(
    input_path: Union[str, Path],
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """Identify FASTA files from input path. for consistency with the FASTQ detection, in cases where a flat/no-quality containing fasta migth be used (e.g. to use as --trusted-contigs)

    Args:
        input_path: Path to input directory or file
        logger: Logger instance
    """
    logger = get_logger(logger)
    input_path = Path(input_path)

    logger.info(f"Identifying FASTA files in: {input_path}")

    fasta_files = find_files_by_extension(
        input_path,
        ["*.fa", "*.fasta", "*.fa.gz", "*.fasta.gz","*.fna", "*.fna.gz"],
    )
    return {"fasta_files": fasta_files}


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

    def aggregate_read_stats(file_details: Dict[str, Dict]) -> Tuple[float, float]:
        lengths = [
            float(details.get("average_read_length", 0))
            for details in file_details.values()
            if float(details.get("average_read_length", 0)) > 0
        ]
        qualities = [
            float(details.get("average_read_quality", 0))
            for details in file_details.values()
            if float(details.get("average_read_quality", 0)) > 0
        ]
        avg_len = sum(lengths) / len(lengths) if lengths else 0.0
        avg_qual = sum(qualities) / len(qualities) if qualities else 0.0
        return avg_len, avg_qual

    def derive_file_name(path: Path) -> str:
        stem = path.stem
        return normalize_fastq_sample_name(stem)

    # Handle comma-separated file inputs (common in filter_reads usage)
    if isinstance(input_path, (str, Path)) and "," in str(input_path):
        # Split comma-separated files
        file_paths = [Path(p.strip()) for p in str(input_path).split(",")]

        if len(file_paths) == 2:
            # Assume R1, R2 pair
            r1_path, r2_path = file_paths

            # Generate file name from R1
            file_name = derive_file_name(r1_path)

            logger.info(f"Detected paired files: {r1_path} and {r2_path}")
            file_details = {
                str(r1_path): determine_fastq_type(r1_path, logger=logger),
                str(r2_path): determine_fastq_type(r2_path, logger=logger),
            }
            average_read_length, average_read_quality = aggregate_read_stats(
                file_details
            )

            return {
                "R1_R2_pairs": [(r1_path, r2_path)],
                "interleaved_files": [],
                "single_end_files": [],
                "file_name": file_name,
                "file_details": file_details,
                "average_read_length": average_read_length,
                "average_read_quality": average_read_quality,
                "is_single_end_only": False,
            }
        else:
            # Multiple single files
            logger.info(f"Detected {len(file_paths)} individual files")

            # Use first file for naming
            file_name = derive_file_name(file_paths[0])
            file_details = {
                str(path): determine_fastq_type(path, logger=logger)
                for path in file_paths
            }
            average_read_length, average_read_quality = aggregate_read_stats(
                file_details
            )

            return {
                "R1_R2_pairs": [],
                "interleaved_files": [],
                "single_end_files": file_paths,
                "file_name": file_name,
                "file_details": file_details,
                "average_read_length": average_read_length,
                "average_read_quality": average_read_quality,
                "is_single_end_only": True,
            }

    # Use consolidated file detection for directory or single file
    file_info = identify_fastq_files(
        input_path, return_rolypoly=False, logger=logger
    )

    # Generate appropriate file name
    file_name = "rolypoly_filtered_reads"

    if input_path.is_file():
        # Single file input
        file_name = derive_file_name(input_path)
    elif input_path.is_dir():
        # Use directory name as base
        file_name = input_path.name

    # Convert our file_info format to the expected format
    result = {
        "R1_R2_pairs": file_info["R1_R2_pairs"],
        "interleaved_files": file_info["interleaved_files"],
        "single_end_files": file_info["single_end"],  # Note: different key name
        "file_name": file_name,
        "file_details": file_info["file_details"],
    }
    average_read_length, average_read_quality = aggregate_read_stats(
        file_info["file_details"]
    )
    result["average_read_length"] = average_read_length
    result["average_read_quality"] = average_read_quality
    result["is_single_end_only"] = (
        len(result["single_end_files"]) > 0
        and len(result["R1_R2_pairs"]) == 0
        and len(result["interleaved_files"]) == 0
    )

    # Add rolypoly data if present
    if file_info["rolypoly_data"]:
        result["rolypoly_data"] = file_info["rolypoly_data"]

    logger.debug(f"File handling summary for path '{input_path.absolute()}':")
    logger.debug(f"  - File name: {file_name}")
    logger.debug(f"  - R1/R2 pairs: {len(result['R1_R2_pairs'])}")
    logger.debug(f"  - Interleaved files: {len(result['interleaved_files'])}")
    logger.debug(f"  - Single-end files: {len(result['single_end_files'])}")
    logger.debug(
        "  - Avg read length / quality: %.2f / %.2f",
        result["average_read_length"],
        result["average_read_quality"],
    )

    return result
