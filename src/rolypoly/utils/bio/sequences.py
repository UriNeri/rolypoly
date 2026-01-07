"""
Sequence file I/O operations for reading and writing FASTA/FASTQ files.
Basic sequence analysis and validation functions.

Key functions: (subject to change...)
    - read_fasta_needletail: Read FASTA/FASTQ files into lists
    - read_fasta_df: Read FASTA/FASTQ files into polars DataFrame
    - write_fasta_file: Write sequences to FASTA files
    - filter_fasta_by_headers: Filter sequences by header patterns
    - rmdup: Remove duplicate sequences (similar to seqkit rmdup)
    - revcomp: Calculate reverse complement of sequences
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import polars as pl
from needletail import parse_fastx_file

from rolypoly.utils.various import find_files_by_extension


def read_fasta_needletail(fasta_file: str) -> Tuple[list[str], list[str]]:
    """Read sequences from a FASTA/FASTQ file using needletail"""

    seqs = []
    seq_ids = []
    for record in parse_fastx_file(fasta_file):
        seqs.append(getattr(record, "seq", ""))  # type: ignore
        seq_ids.append(getattr(record, "id", ""))  # type: ignore
    return seq_ids, seqs


def read_fasta_df(file_path: str) -> pl.DataFrame:
    """wrapper for legacy code"""
    from rolypoly.utils.bio.polars_fastx import from_fastx_eager

    return from_fastx_eager(file_path)  # type: ignore defined in polars_fastx.py


def write_fasta_file(
    records=None,
    seqs=None,
    headers=None,
    output_file=None,
    format: str = "fasta",
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

    should_close = False
    if output_file is None:
        output_handle = sys.stdout
    elif hasattr(output_file, "write"):
        output_handle = output_file
    else:
        output_handle = open(output_file, "w")
        should_close = True

    try:
        if records:
            for record in records:
                output_handle.write(
                    f"{header_delim}{record.id}{seq_delim}{str(record.seq)}"
                )
        elif seqs is not None and headers is not None:
            for i, seq in enumerate(seqs):
                if i == 0:
                    output_handle.write(f">{headers[i]}{seq_delim}{seq}")
                else:
                    output_handle.write(
                        f"{header_delim}{headers[i]}{seq_delim}{seq}"
                    )
        else:
            raise ValueError("No records, seqs, or headers provided")
    finally:
        if should_close:
            output_handle.close()


def clean_fasta_headers(
    fasta_file: str,
    output_file: str,
    strip_suffix="",
    strip_prefix="bla bla lba",
    drop_from_space=False,
) -> None:
    """remove annoying stuff from the headers of a FASTA file.

    Args:
        fasta_file (str): Path to input FASTA file
        output_file (str): Path to write filtered sequences
        strip_prefix (str, optional): if provided, a prefix to remove from fasta headers (if present)
        strip_suffix (str, optional): if provided, a suffix to remove from fasta headers (if present)
        drop_from_space (bool): if true, discard everything after the first space (not every white space, just space)

    Example:
        clean a NCBI protein header:
        clean_fasta_headers(fasta_file = some_fasta, output_file = cleaner_fasta, strip_prefix = "lcl|", drop_from_space = True, strip_suffix = ")

    Note:
        No gaurentee that all output header will be unique. # TODO add a hash +seen + logger.warning for this.

    """
    import gzip

    from rolypoly.utils.various import is_gzipped

    strip_prefix_len = 0 if not strip_prefix else len(strip_prefix)
    strip_suffix_len = 0 if not strip_suffix else len(strip_suffix)
    # Determine if files are gzipped
    is_gz_input = is_gzipped(
        fasta_file
    )  # not really needed, handled in the needletail parser
    is_gz_output = output_file.endswith(".gz")

    try:
        # Open output file with appropriate method
        if is_gz_output:
            out_f = gzip.open(output_file, "wt", encoding="utf-8")
        else:
            out_f = open(output_file, "w", encoding="utf-8")

        # Stream through records for memory efficiency
        records_processed = 0
        records_written = 0
        from rich.progress import track

        for record in track(parse_fastx_file(fasta_file), "modifying file"):
            records_processed += 1
            record_id = str(getattr(record, "id", ""))
            record_seq = str(getattr(record, "seq", ""))
            if record_id.startswith(strip_prefix):
                record_id = record_id[strip_prefix_len:]
            if record_id.endswith(strip_suffix):
                record_id = record_id[:strip_suffix_len]
            if drop_from_space:
                record_id = record_id.split(sep=" ")[0]
            out_f.write(f">{record_id}\n{record_seq}\n")
            records_written += 1
            if records_processed % 100000 == 0:
                print(
                    f"Processed {records_processed} records, written {records_written}"
                )
        out_f.close()
    except Exception as e:
        if "out_f" in locals():
            out_f.close()
        raise Exception(f"Error filtering FASTA file {fasta_file}: {e}") from e


def filter_fasta_by_headers(
    fasta_file: str,
    headers: Union[str, List[str]],
    output_file: str,
    wrap: bool = False,
    invert: bool = False,
    return_counts: bool = False,
) -> Union[None, Dict[str, int]]:
    """Filter sequences in a FASTA file based on their headers.

    Extracts sequences whose headers match (or don't match if inverted) any of
    the provided header patterns.

    Args:
        fasta_file (str): Path to input FASTA file
        headers (Union[str, List[str]]): Either a file containing headers (one per line)
            or a list of header patterns to match
        output_file (str): Path to write filtered sequences
        wrap (bool, optional): If True, match headers that contain the patterns as substrings (substring match). Default False (exact match).
        invert (bool, optional): If True, keep sequences that don't match.
        return_counts (bool, optional): If True, return counts of filtered, and written records.
    """
    import gzip

    from rolypoly.utils.various import is_gzipped

    # Load headers and optimize for lookup pattern
    headers_exact = set()  # For exact matches
    headers_patterns = []  # For substring patterns only if needed - # TODO: is this needed?
    if not isinstance(headers, list):
        with open(headers, "r") as f:
            for line in f:
                header = line.strip()
                if wrap:
                    headers_patterns.append(header)
                else:
                    headers_exact.add(header)
    else:
        if wrap:
            headers_patterns.extend(headers)
        else:
            headers_exact.update(headers)

    # Determine if files are gzipped
    is_gz_input = is_gzipped(
        fasta_file
    )  # not really needed, handled in the needletail parser
    is_gz_output = output_file.endswith(".gz")

    try:
        # Open output file with appropriate method
        if is_gz_output:
            out_f = gzip.open(output_file, "wt", encoding="utf-8")
        else:
            out_f = open(output_file, "w", encoding="utf-8")

        # Stream through records for memory efficiency
        records_processed = 0
        records_written = 0

        if not invert:
            # Optimized path for non-inverted case
            if not wrap:
                # Fast membership test (default, backward compatible)
                target_headers = headers_exact.copy()
                for record in parse_fastx_file(fasta_file):
                    records_processed += 1
                    record_id = str(getattr(record, "id", ""))
                    if record_id in target_headers:
                        record_seq = str(getattr(record, "seq", ""))
                        target_headers.remove(record_id)
                        out_f.write(f">{record_id}\n{record_seq}\n")
                        records_written += 1
                        if not target_headers:
                            break
                    if records_processed % 100000 == 0:
                        print(
                            f"Processed {records_processed} records, written {records_written}"
                        )
            else:
                # Substring match (wrap=True)
                for record in parse_fastx_file(fasta_file):
                    records_processed += 1
                    record_id = str(getattr(record, "id", ""))
                    if any(
                        pattern in record_id for pattern in headers_patterns
                    ):
                        record_seq = str(getattr(record, "seq", ""))
                        out_f.write(f">{record_id}\n{record_seq}\n")
                        records_written += 1
                    if records_processed % 100000 == 0:
                        print(
                            f"Processed {records_processed} records, written {records_written}"
                        )
        else:
            # Inverted case
            if not wrap:
                for record in parse_fastx_file(fasta_file):
                    records_processed += 1
                    record_id = str(getattr(record, "id", ""))
                    if record_id not in headers_exact:
                        record_seq = str(getattr(record, "seq", ""))
                        out_f.write(f">{record_id}\n{record_seq}\n")
                        records_written += 1
                    if records_processed % 100000 == 0:
                        print(
                            f"Processed {records_processed} records, written {records_written}"
                        )
            else:
                for record in parse_fastx_file(fasta_file):
                    records_processed += 1
                    record_id = str(getattr(record, "id", ""))
                    if not any(
                        pattern in record_id for pattern in headers_patterns
                    ):
                        record_seq = str(getattr(record, "seq", ""))
                        out_f.write(f">{record_id}\n{record_seq}\n")
                        records_written += 1
                    if records_processed % 100000 == 0:
                        print(
                            f"Processed {records_processed} records, written {records_written}"
                        )

        out_f.close()
        if return_counts:
            return {
                "records_processed": records_processed,
                "records_written": records_written,
            }

    except Exception as e:
        if "out_f" in locals():
            out_f.close()
        raise Exception(f"Error filtering FASTA file {fasta_file}: {e}") from e


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
    """Populate a polars DataFrame with sequences from a FASTA file - slow, but supports trimming to selected regions / revcomp. \n
    If you don't need those use another function."""
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
            f"grep -F '>'  {seqfile} -c ",
            shell=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    print(f"Actual number of sequences in file: {seq_count}")

    # Reset file iterator
    index = 0
    for record in parse_fastx_file(seqfile):
        seqs.append(record.seq)  # type: ignore
        seq_ids.append(record.id)  # type: ignore
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
                    pl.struct(
                        pl.col(seqcol), pl.col(start_col), pl.col(end_col)
                    )
                    .map_elements(
                        lambda x: (
                            str(x[seqcol][x[start_col] : x[end_col]])
                            if x[seqcol] is not None
                            else None
                        ),
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
                pl.coalesce([pl.col(seqcol), pl.col(f"{seqcol}_right")]).alias(
                    seqcol
                )
            ).drop(f"{seqcol}_right")

            print(
                f"Null count in seqcol after chunk: {minipldf[seqcol].null_count()}"
            )

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
        valid_characters.update(
            {"M", "R", "W", "S", "Y", "K", "V", "H", "D", "B"}
        )
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


def remove_duplicates(
    input_file: Union[str, List[str]],
    output_file: Optional[str] = None,
    by: str = "name",
    revcomp_as_distinct: bool = True,
    ignore_case: bool = False,
    save_duplicates: Optional[str] = None,
    save_dup_list: Optional[str] = None,
    return_stats: bool = False,
    return_sequences: bool = False,
    streaming: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict[str, Union[int, pl.DataFrame, List[Tuple[str, str]]]]]:
    """Remove duplicate sequences from FASTA/FASTQ files.

    Efficient deduplication similar to seqkit rmdup. Uses xxhash for fast hashing
    and supports both streaming (memory-efficient) and in-memory modes.
    Supports processing multiple input files without concatenation.

    Args:
        input_file: Path to input FASTA/FASTQ file, or list of paths to process multiple files
        output_file: Path to output file with unique sequences. If None and return_sequences=False,
                    prints to stdout. If return_sequences=True, output_file is optional.
        by: Deduplication criterion - "id", "name", or "seq"
            - "id": Use sequence ID (first word of header)
            - "name": Use full header/name
            - "seq": Use sequence content
        revcomp_as_distinct: If False and by="seq", treats reverse complement as duplicate.
                           Only applies when by="seq". Default True (revcomp is distinct).
        ignore_case: Ignore case when comparing sequences/names. Default False.
        save_duplicates: Optional path to save duplicate sequences
        save_dup_list: Optional path to save list of duplicate IDs with counts
        return_stats: Return statistics dictionary. Default False.
        return_sequences: If True, return sequences in memory instead of/in addition to writing.
                         When True and streaming=False, all sequences are loaded into memory first.
                         When True and streaming=True, sequences are collected as processed. Default False.
        streaming: If True (default), process records one at a time (low memory).
                  If False and return_sequences=True, load all sequences into memory first for faster access.
                  Only affects behavior when return_sequences=True. Default True.
        logger: Logger instance

    Returns:
        Optional dictionary with:
            - total_records: Total sequences processed
            - unique_records: Unique sequences kept
            - duplicates_removed: Number of duplicates removed
            - duplicate_groups: DataFrame with duplicate groups (if save_dup_list requested)
            - sequences: List of (header, sequence) tuples (if return_sequences=True)

    Examples:
        >>> # Remove duplicates by sequence ID, write to file
        >>> remove_duplicates("input.fasta", "output.fasta")

        >>> # Remove duplicates and return sequences in memory
        >>> result = remove_duplicates("input.fasta", return_sequences=True, return_stats=True)
        >>> for header, seq in result['sequences']:
        ...     print(f">{header}\\n{seq}")

        >>> # Non-streaming mode: load all into memory first (faster for multiple passes)
        >>> result = remove_duplicates("input.fasta", return_sequences=True, streaming=False)

        >>> # Remove duplicates by sequence, treating revcomp as duplicate
        >>> remove_duplicates("input.fasta", "output.fasta", by="seq", revcomp_as_distinct=False)

        >>> # Save duplicates and get statistics
        >>> stats = remove_duplicates("input.fasta", "output.fasta",
        ...               save_duplicates="dups.fasta",
        ...               save_dup_list="dup_list.txt",
        ...               return_stats=True)

        >>> # Process multiple files without concatenation
        >>> remove_duplicates(["file1.fasta", "file2.fasta", "file3.fasta"], "output.fasta")

    Note:
        - Uses xxhash (if available) or blake2b for fast hashing
        - Only first occurrence is kept for duplicates
        - Streaming mode: processes records one at a time (low memory)
        - Non-streaming mode: loads all sequences into memory first (only useful with return_sequences=True)
        - When processing multiple files, duplicates are detected across all files
    """
    import hashlib
    import sys

    from rolypoly.utils.logging.loggit import get_logger

    logger = get_logger(logger)

    # Normalize input to list
    input_files = [input_file] if isinstance(input_file, str) else input_file

    if len(input_files) > 1:
        logger.info(f"Processing {len(input_files)} input files")

    # Validate parameters
    if by not in ["id", "name", "seq"]:
        raise ValueError(
            f"Invalid 'by' parameter: {by}. Must be 'id', 'name', or 'seq'"
        )

    if not revcomp_as_distinct and by != "seq":
        logger.warning(
            "revcomp_as_distinct only applies when by='seq', ignoring"
        )

    if not streaming and not return_sequences:
        logger.warning(
            "streaming=False only affects behavior when return_sequences=True, ignoring"
        )

    # Open output files
    out_fh = None
    if output_file:
        out_fh = open(output_file, "w")
    elif not return_sequences:
        out_fh = sys.stdout

    dup_fh = None
    if save_duplicates:
        dup_fh = open(save_duplicates, "w")

    # Tracking data structures
    seen_hashes = set()  # Set of hashes we've seen
    duplicate_groups = {}  # Maps hash -> list of IDs that are duplicates
    unique_sequences = []  # Store unique sequences if return_sequences=True

    # Statistics
    total_records = 0
    unique_records = 0
    duplicates_removed = 0

    # Non-streaming mode: load all sequences into memory first
    if not streaming and return_sequences:
        logger.debug("Loading all sequences into memory (non-streaming mode)")
        all_records = []
        for input_path in input_files:
            for record in parse_fastx_file(input_path):
                record_id = str(getattr(record, "id", ""))
                record_seq = str(getattr(record, "seq", ""))
                all_records.append((record_id, record_seq))
        logger.debug(
            f"Loaded {len(all_records)} records into memory from {len(input_files)} file(s)"
        )
    else:
        all_records = None

    try:
        # Process records (either from memory or streaming)
        if all_records:
            records_iter = all_records
        else:
            # Create an iterator that chains all input files
            def multi_file_iterator():
                for input_path in input_files:
                    for record in parse_fastx_file(input_path):
                        yield record

            records_iter = multi_file_iterator()

        for record in records_iter:
            total_records += 1

            # Handle different record types (tuple from memory vs needletail object)
            if isinstance(record, tuple):
                record_id, record_seq = record
            else:
                record_id = str(getattr(record, "id", ""))
                record_seq = str(getattr(record, "seq", ""))

            # Extract relevant field based on 'by' parameter
            if by == "seq":
                field = record_seq
            elif by == "name":
                field = record_id  # Full header
            else:  # by == "id"
                # Extract just the ID (first word)
                field = record_id.split()[0] if record_id else record_id

            # Apply case sensitivity
            if ignore_case:
                field = field.lower()

            # Calculate hash using xxhash3 (faster than MD5/SHA) or fallback to hashlib
            field_bytes = field.encode("utf-8")
            try:
                # Try xxhash3 if available (much faster)
                import xxhash

                hash_val = xxhash.xxh3_64(field_bytes).intdigest()
            except ImportError:
                # Fallback to hashlib (slower but always available)
                hash_val = int.from_bytes(
                    hashlib.blake2b(field_bytes, digest_size=8).digest(),
                    byteorder="big",
                )

            # Check if this is a duplicate
            is_duplicate = hash_val in seen_hashes

            # If checking sequences and revcomp should be treated as duplicates
            if by == "seq" and not revcomp_as_distinct and not is_duplicate:
                rc_seq = revcomp(field if not ignore_case else field.upper())
                if ignore_case:
                    rc_seq = rc_seq.lower()
                rc_bytes = rc_seq.encode("utf-8")
                try:
                    import xxhash

                    rc_hash = xxhash.xxh3_64(rc_bytes).intdigest()
                except ImportError:
                    rc_hash = int.from_bytes(
                        hashlib.blake2b(rc_bytes, digest_size=8).digest(),
                        byteorder="big",
                    )

                if rc_hash in seen_hashes:
                    is_duplicate = True
                    hash_val = rc_hash  # Use the RC hash for tracking

            # Handle duplicate vs unique
            if is_duplicate:
                duplicates_removed += 1

                # Save to duplicate file if requested
                if dup_fh:
                    dup_fh.write(f">{record_id}\n{record_seq}\n")

                # Track for duplicate list
                if save_dup_list:
                    if hash_val not in duplicate_groups:
                        duplicate_groups[hash_val] = []
                    duplicate_groups[hash_val].append(record_id)
            else:
                # First occurrence - keep it
                unique_records += 1
                seen_hashes.add(hash_val)

                # Write to output file if specified
                if out_fh:
                    out_fh.write(f">{record_id}\n{record_seq}\n")

                # Store in memory if return_sequences=True
                if return_sequences:
                    unique_sequences.append((record_id, record_seq))

                # Initialize duplicate tracking
                if save_dup_list:
                    duplicate_groups[hash_val] = [record_id]

        logger.info(
            f"Processed {total_records} records: {unique_records} unique, {duplicates_removed} duplicates removed"
        )

    finally:
        # Close output files
        if out_fh and output_file:
            out_fh.close()
        if dup_fh:
            dup_fh.close()

    # Save duplicate list if requested
    if save_dup_list:
        with open(save_dup_list, "w") as dup_list_fh:
            # Sort by number of duplicates (descending)
            sorted_groups = sorted(
                [
                    (hash_val, ids)
                    for hash_val, ids in duplicate_groups.items()
                    if len(ids) > 1
                ],
                key=lambda x: len(x[1]),
                reverse=True,
            )

            for _, ids in sorted_groups:
                dup_list_fh.write(f"{len(ids)}\t{', '.join(ids)}\n")

    # Return results if requested
    if return_stats or return_sequences:
        result = {}

        if return_stats:
            result["total_records"] = total_records
            result["unique_records"] = unique_records
            result["duplicates_removed"] = duplicates_removed

            if save_dup_list and duplicate_groups:
                # Create DataFrame of duplicate groups
                dup_data = []
                for hash_val, ids in duplicate_groups.items():
                    if len(ids) > 1:
                        dup_data.append(
                            {"count": len(ids), "ids": ", ".join(ids)}
                        )

                if dup_data:
                    result["duplicate_groups"] = pl.DataFrame(dup_data).sort(
                        "count", descending=True
                    )

        if return_sequences:
            result["sequences"] = unique_sequences

        return result

    return None


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
            pl.col("sequence").str.count_matches("G|C").alias("gc_content")
            / pl.col("sequence").str.len_chars().alias("length"),
            pl.col("sequence").str.count_matches("N|n").alias("n_count"),
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
            pl.col("sequence")
            .map_elements(_hash, return_dtype=pl.String)
            .alias("seq_hash")
        )
        new_headers = [f"{prefix}_{h}" for h in df_with_hashes["seq_hash"]]
    else:
        # Calculate padding based on total number of sequences
        padding = len(str(len(df)))
        new_headers = [
            f"{prefix}_{str(i + 1).zfill(padding)}" for i in range(len(df))
        ]

    # Create mapping dictionary
    id_map = dict(zip(df["header"], new_headers))

    return df.with_columns(pl.Series("header", new_headers)), id_map


def find_fasta_files(
    input_path: Union[str, Path],
    extensions: List[str] = None,
    logger: Optional[logging.Logger] = None,
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
        extensions = [
            "*.fa",
            "*.fasta",
            "*.fna",
            "*.fa.gz",
            "*.fasta.gz",
            "*.fna.gz",
        ]

    return find_files_by_extension(
        input_path, extensions, "FASTA files", logger
    )


def ensure_faidx(
    input_file: str, logger: Optional[logging.Logger] = None
) -> None:
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
            console.print(
                f"[yellow]Indexing {input_file} with pyfastx[/yellow]"
            )
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
