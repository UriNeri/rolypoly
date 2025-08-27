"""
Sequence file I/O operations for reading and writing FASTA/FASTQ files.
Basic sequence analysis and validation functions.
"""
import re
from typing import List, Union, Dict, Tuple

import polars as pl
from needletail import parse_fastx_file
from .polars_fastx import * # I think this brings in the i/o plugins, TODO: check if can be done so that I don't need to type: ignore all the time
from rolypoly.utils.various import  find_files_by_extension
from pathlib import Path
import logging


def read_fasta_needletail(fasta_file: str) -> Tuple[list[str], list[str]]:
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
    return pl.DataFrame.from_fastx(file_path) # type: ignore defined in polars_fastx.py


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


import rich_click as click
@click.command(name="rename_seqs_cli")
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
def rename_seqs_cli(input: str, output: str, mapping: str, prefix: str, hash: bool, stats: bool):
    """Rename sequences in a FASTA file with consistent IDs.

    This tool renames sequences in a FASTA file using either sequential numbers
    or hashes, and generates a lookup table mapping old IDs to new IDs.
    Optionally includes sequence statistics (length, GC content).
    """
    from rolypoly.utils.logging.loggit import get_logger
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



from pathlib import Path
import hashlib
import json
from collections import Counter
from typing import Optional

import polars as pl
import rich_click as click
from rich.console import Console

from rolypoly.utils.logging.citation_reminder import remind_citations
from rolypoly.utils.bio.sequences import (
    read_fasta_df,
)

# show all columns
pl.Config().set_tbl_cols(-1)

# global console
console = Console()


@click.command()
@click.option(
    "-i",
    "--input",
    required=True,
    type=click.Path(exists=True),
    help="Input file (fasta, fa, fna, faa)",
)
@click.option(
    "-agg",
    "--aggregate",
    default=False,
    type=bool,
    help="aggregate statistics across all sequences",
)
@click.option(
    "-o",
    "--output",
    default="rp_sequence_stats.txt",
    type=click.Path(exists=False),
    help="Output path",
)
@click.option(
    "--log-file",
    default="command.log",
    type=click.Path(exists=False),
    help="Path to log file",
    hidden=True,
)
@click.option(
    "--log-level",
    hidden=True,
    default="INFO",
    help="Log level",
)
@click.option(
    "--min_length",
    hidden=True,
    default=None,
    type=int,
    help="minimum sequence length to consider",
)
@click.option(
    "--max_length",
    hidden=True,
    default=None,
    type=int,
    help="maximum sequence length to consider",
)
@click.option(
    "--format",
    default="tsv",
    type=click.Choice(case_sensitive=False, choices=["csv", "tsv", "md", "parquet"]),
    help="output format, either a parquet/csv/tsv file with the data or a markdown file with summary statistics",
)
@click.option(
    "-f",
    "--fields",
    type=click.Choice(case_sensitive=False, choices=["length", "gc_content", "n_count", "hash", "kmer_freq"]),
    multiple=True,
    default=["length", "gc_content", "n_count", "hash"],
    help="""
              comma-separated list of fields to include.  
              Available:
              length - mandatory
              gc_content - percentage of GC nucleotides
              n_count - total number of Ns 
              hash - md5 hash of the sequence
              kmer_freq - k-mer frequencies (k=3 by default)
              """,
)
def sequence_stats(
    input,
    aggregate,
    output,
    log_file,
    log_level,
    min_length,
    max_length,
    format,
    fields,
):
    """Calculate sequence statistics using Polars expressions"""
    from rolypoly.utils.logging.loggit import log_start_info, setup_logging

    logger = setup_logging(log_file, log_level)
    log_start_info(logger, locals())

    output_path = Path(output)

    # Read sequences into DataFrame
    df = read_fasta_df(input)
    total_seqs = len(df)
    df = df.with_columns(pl.col("sequence").str.len_chars().alias("length"))
    
    logger.info(f"Read {total_seqs} sequences from {input}")

    # Apply length filters
    if min_length:
        df = df.filter(pl.col("length") >= min_length)
        logger.info(f"Applied minimum length filter: {min_length}")
    if max_length:
        df = df.filter(pl.col("length") <= max_length)
        logger.info(f"Applied maximum length filter: {max_length}")

    filtered_seqs = len(df)
    logger.info(f"After filtering: {filtered_seqs} sequences")

    # Define available fields and their dependencies
    field_options = {
        "length": {"desc": "Sequence length"},
        "gc_content": {"desc": "GC content percentage"},
        "n_count": {"desc": "Count of Ns in sequence"},
        "hash": {"desc": "Sequence hash (MD5)"},
        "kmer_freq": {"desc": "K-mer frequencies", "complex": True},
    }

    # Parse fields - fields is already a tuple from multiple=True
    selected_fields = list(fields) if fields else ["length", "gc_content", "n_count", "hash"]
    
    # Always include length for summaries
    if "length" not in selected_fields:
        selected_fields.append("length")

    logger.info(f"Selected fields: {', '.join(selected_fields)}")

    # Helper function for kmer calculation
    def calculate_kmers(sequence, k=3):
        """Calculate k-mer frequencies for a sequence"""
        if len(sequence) < k:
            return {}
        kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
        return dict(Counter(kmers))

    # Helper function for MD5 hash
    def calculate_hash(sequence):
        """Calculate MD5 hash of sequence"""
        return hashlib.md5(sequence.encode()).hexdigest()

    # Calculate selected statistics
    if "gc_content" in selected_fields:
        df = df.with_columns(
            (pl.col("sequence").str.count_matches(r"[GCgc]") / pl.col("length") * 100.0)
            .alias("gc_content")
        )
    
    if "n_count" in selected_fields:
        df = df.with_columns(
            pl.col("sequence").str.count_matches(r"[Nn]").alias("n_count")
        )
    
    if "hash" in selected_fields:
        df = df.with_columns(
            pl.col("sequence").map_elements(calculate_hash, return_dtype=pl.String).alias("hash")
        )

    if "kmer_freq" in selected_fields:
        df = df.with_columns(
            pl.col("sequence").map_elements(
                lambda x: json.dumps(calculate_kmers(x, k=3)), 
                return_dtype=pl.String
            ).alias("kmer_frequencies")
        )

    # Select only the fields we want to output
    output_columns = ["header"] + selected_fields
    if "kmer_freq" in selected_fields:
        # Replace kmer_freq with kmer_frequencies in output columns
        output_columns = [col if col != "kmer_freq" else "kmer_frequencies" for col in output_columns]
    
    df_output = df.select([col for col in output_columns if col in df.columns])

    if aggregate:
        # Create aggregated statistics
        agg_stats = {}
        
        if "length" in selected_fields:
            length_stats = df.select([
                pl.col("length").min().alias("min_length"),
                pl.col("length").max().alias("max_length"),
                pl.col("length").mean().alias("mean_length"),
                pl.col("length").median().alias("median_length"),
                pl.col("length").std().alias("std_length"),
                pl.col("length").sum().alias("total_length")
            ]).to_dicts()[0]
            agg_stats.update(length_stats)
        
        if "gc_content" in selected_fields:
            gc_stats = df.select([
                pl.col("gc_content").min().alias("min_gc"),
                pl.col("gc_content").max().alias("max_gc"),
                pl.col("gc_content").mean().alias("mean_gc"),
                pl.col("gc_content").median().alias("median_gc"),
                pl.col("gc_content").std().alias("std_gc")
            ]).to_dicts()[0]
            agg_stats.update(gc_stats)
        
        if "n_count" in selected_fields:
            n_stats = df.select([
                pl.col("n_count").min().alias("min_n_count"),
                pl.col("n_count").max().alias("max_n_count"),
                pl.col("n_count").mean().alias("mean_n_count"),
                pl.col("n_count").sum().alias("total_n_count")
            ]).to_dicts()[0]
            agg_stats.update(n_stats)
        
        agg_stats["total_sequences"] = filtered_seqs
        agg_stats["sequences_before_filter"] = total_seqs
        
        # Convert to DataFrame for consistent output
        df_output = pl.DataFrame([agg_stats])

    # Output results
    if format.lower() == "parquet":
        df_output.write_parquet(output_path)
        logger.info(f"Results written to {output_path} (parquet format)")
    
    elif format.lower() == "csv":
        df_output.write_csv(output_path)
        logger.info(f"Results written to {output_path} (CSV format)")
    
    elif format.lower() == "tsv":
        df_output.write_csv(output_path, separator="\t")
        logger.info(f"Results written to {output_path} (TSV format)")
    
    elif format.lower() == "md":
        # Create markdown summary
        md_content = "# Sequence Statistics Report\n\n"
        md_content += f"**Input file:** {input}\n"
        md_content += f"**Total sequences:** {total_seqs}\n"
        md_content += f"**Sequences after filtering:** {filtered_seqs}\n\n"
        
        if aggregate:
            md_content += "## Aggregate Statistics\n\n"
            for key, value in agg_stats.items():
                if isinstance(value, float):
                    md_content += f"- **{key}:** {value:.2f}\n"
                else:
                    md_content += f"- **{key}:** {value}\n"
        else:
            md_content += "## Summary Statistics\n\n"
            # Add basic summary stats even when not aggregating
            if "length" in selected_fields:
                length_summary = df.select([
                    pl.col("length").min().alias("min"),
                    pl.col("length").max().alias("max"),
                    pl.col("length").mean().alias("mean"),
                    pl.col("length").median().alias("median")
                ]).to_dicts()[0]
                
                md_content += "### Length Statistics\n"
                for stat, value in length_summary.items():
                    md_content += f"- **{stat}:** {value:.2f}\n"
                md_content += "\n"
            
            md_content += "\n## First 10 sequences\n\n"
            md_content += df_output.head(10).to_pandas().to_markdown(index=False)
        
        with open(output_path, 'w') as f:
            f.write(md_content)
        logger.info(f"Markdown report written to {output_path}")

    # Display summary to console
    console.print(f"\n[bold green]✓[/bold green] Processed {filtered_seqs} sequences")
    console.print(f"[bold blue]Output:[/bold blue] {output_path}")
    
    if not aggregate and format.lower() != "md":
        console.print("\n[bold]First 5 rows:[/bold]")
        console.print(df_output.head(5))

    # Remind about citations
    tools = ["polars"]  # Tools used in this analysis
    remind_citations(tools)
    
    logger.info("Sequence statistics calculation completed successfully")


#  sequence statistics calculator with filtering 
def fasta_stats(
    input_file: str,
    output_file: Optional[str] = None, # if not provided, will print to stdout
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    fields: str = "header,length,gc_content,n_count,hash,codon_usage,kmer_freq",
    kmer_length: int = 3,
) -> None:
    """Calculate sequence statistics using Polars expressions
    
    Args:
        input: Input file or directory
        output: Output path
        min_length: Minimum sequence length to consider
        max_length: Maximum sequence length to consider
        fields: Comma-separated list of fields to include (available: header,sequence,length,gc_content,n_count,hash,codon_usage,kmer_freq)
    """
    import sys
    output_path = Path(output_file) if output_file else sys.stdout

    # Read sequences into DataFrame
    df = pl.DataFrame.from_fastx(input_file) # type: ignore
    
    # init_height = df.height
    # Apply length filters
    if min_length:
        df = df.filter(pl.  col("sequence").seq.length() >= min_length)
    if max_length:
        df = df.filter(pl.col("sequence").seq.length() <= max_length)
    # print(f"Filtered {init_height - df.height} sequences out of {init_height}")

    # Define available fields and their dependencies
    field_options = {
        "length": {"desc": "Sequence length"},
        "gc_content": {"desc": "GC content percentage"},
        "n_count": {"desc": "Count of Ns in sequence"},
        "hash": {"desc": "Sequence hash (MD5)"},
        "codon_usage": {"desc": "Codon usage frequencies"},
        "kmer_freq": {"desc": "K-mer frequencies"},
        "header": {"desc": "Sequence header"},
        "sequence": {"desc": "DNA/RNA sequence"}
    }

    # Parse fields
    selected_fields = ["header"]
    if fields:
        selected_fields = [f.strip().lower() for f in fields.split(",")]
        # Validate fields
        valid_fields = list(field_options.keys())
        invalid_fields = [f for f in selected_fields if f not in valid_fields]
        if invalid_fields:
            print(f"Unknown field(s): {', '.join(invalid_fields)}")
            print(f"Available fields are: {', '.join(valid_fields)}")
        selected_fields = [f for f in selected_fields if f in valid_fields]

    # Build the stats expressions
    stats_expr = []
    # for field in selected_fields: # this doesn't work  :(
    #     stats_expr.append(pl.col("sequence").seq.field(field).alias(field))
    
    if "length" in selected_fields:
        stats_expr.append(pl.col("sequence").seq.length().alias("length"))
    if "gc_content" in selected_fields:
        stats_expr.append(pl.col("sequence").seq.gc_content().alias("gc_content"))
    if "n_count" in selected_fields:
        stats_expr.append(pl.col("sequence").seq.n_count().alias("n_count"))
    if "hash" in selected_fields:
        stats_expr.append(pl.col("sequence").seq.generate_hash().alias("hash")) 
    if "codon_usage" in selected_fields:
        stats_expr.append(pl.col("sequence").seq.codon_usage().alias("codon_usage"))
    if "kmer_freq" in selected_fields:
        stats_expr.append(pl.col("sequence").seq.calculate_kmer_frequencies(kmer_length).alias("kmer_freq"))

    # Apply all the stats expressions
    df = df.with_columns(stats_expr)
    df = df.select(selected_fields)


    # Convert all nested columns to strings
    for col in df.columns:
        if col != "header":  # Keep header as is
            if isinstance(df[col].dtype, pl.Struct) or isinstance(df[col].dtype, pl.List):
                df = df.with_columns(
                    [pl.col(col).cast(pl.Utf8).alias(f"{col}")]
                )

    df.write_csv(output_path, separator="\t")
    print("Successfully wrote file after converting data types")



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