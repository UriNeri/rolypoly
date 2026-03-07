"""Sequence clustering utilities for ANI/AAI-based genome and contig clustering.

Provides functions for:
- Computing pairwise ANI/AAI from FASTA files using various backends
  (pyskani, blastn, mmseqs2)
- Parsing pre-computed alignment/ANI tables (BLAST outfmt 6,
  CheckV-style ANI table, MMseqs2 easy-search output)
- K-mer-based pairwise identity estimation (overlap coefficient)
- Clustering sequences by identity and coverage thresholds with
  multiple algorithms (centroid/greedy, connected components, Leiden
  community detection)

Similarity measure definitions (following Vclust terminology,
DOI: 10.1038/s41592-025-02701-7):

    For query sequence A aligned against reference sequence B,
    let M(A,B) = total matching (identical) nucleotides in the
    alignment, L(A,B) = total aligned length on A (merged, no
    double-counting overlapping HSPs), and |A| = full sequence length.

    ANI(A,B)        = M(A,B) / L(A,B)
        Average nucleotide identity over the aligned region only.
        This is the standard "identity" column in our edge table,
        computed as the alignment-length-weighted mean of per-HSP
        percent identities (following CheckV anicalc).

    AF(A->B)        = L(A,B) / |A|
        Alignment fraction: the proportion of A that aligned to B.
        Stored as "query_coverage" in the edge table.

    AF(B->A)        ≈ L(B, from A->B coords) / |B|
        Stored as "target_coverage". From a single directed search
        this uses the target coordinates of the A->B alignment, which
        approximates the true AF(B->A) from a B->A search.

    Global ANI(A->B) = M(A,B) / |A| = ANI(A,B) * AF(A->B)
        Identity over the full query length. Derived column
        "global_ani_query" (and "global_ani_target" for the reverse).

    tANI            = (M(A,B) + M(B,A)) / (|A| + |B|)
        Total ANI: the bidirectional average identity normalised by
        total sequence length. From the Vclust paper (DOI above):
        tANI = (ANI1 × AF1 × LEN1 + ANI2 × AF2 × LEN2) / (LEN1 + LEN2)
        Approximated from directed edges as
        (global_ani_query * |A| + global_ani_target * |B|) / (|A|+|B|).
        When sequence lengths are unknown, averaged as
        (global_ani_query + global_ani_target) / 2.
        Derived column "tani".

    K-mer overlap coefficient = |kmers(A) ∩ kmers(B)| / min(|kmers(A)|, |kmers(B)|)
        Preferred over the Jaccard index for unequal-length sequences
        as it measures what fraction of the smaller k-mer set is shared,
        avoiding penalisation for length asymmetry. Used as a fast ANI
        estimate or prefilter (Kmer-db 2 approach).

Base edge table schema (produced by all parsers/backends):
    query_id, target_id, identity, query_coverage, target_coverage,
    num_alignments

Enriched edge table (after enrich_edges_with_derived_metrics):
    + global_ani_query, global_ani_target, tani

Standard cluster assignment output:
    seq_id, cluster_id, representative_id, is_representative,
    identity_to_representative, coverage_to_representative

References:
    - CheckV ANI scripts: https://bitbucket.org/berkeleylab/checkv
    - ClusterGenomes: https://github.com/simroux/ClusterGenomes
    - MIUViG standards: 95 pct ANI, 85 pct AF for species-level vOTUs
    - Vclust (the kmer-prefilter/fast mode and most of the definitions are from there): https://doi.org/10.1038/s41592-025-02701-7
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Union

import igraph as ig
import leidenalg
import polars as pl
import pyfastani
import pyskani


# Standard column names for edge tables (base schema)
EDGE_COLUMNS = [
    "query_id",
    "target_id",
    "identity",
    "query_coverage",
    "target_coverage",
    "num_alignments",
]

# Derived columns added by enrich_edges_with_derived_metrics()
DERIVED_EDGE_COLUMNS = [
    "global_ani_query",
    "global_ani_target",
    "tani",
]

# All edge columns including derived ones
ENRICHED_EDGE_COLUMNS = EDGE_COLUMNS + DERIVED_EDGE_COLUMNS

# Recognised similarity column names for filtering/thresholding
SIMILARITY_COLUMNS = {
    "identity": "identity",
    "ani": "identity",
    "tani": "tani",
    "global_ani": "global_ani_query",
    "global_ani_query": "global_ani_query",
    "global_ani_target": "global_ani_target",
    "kmer_overlap": "kmer_overlap",
}

# Standard column names for cluster assignment tables
CLUSTER_COLUMNS = [
    "seq_id",
    "cluster_id",
    "representative_id",
    "is_representative",
    "identity_to_representative",
    "coverage_to_representative",
]


# Edge table helpers
def empty_edge_frame() -> pl.DataFrame:
    """Return an empty polars DataFrame with the standard edge schema."""
    return pl.DataFrame(
        schema={
            "query_id": pl.String,
            "target_id": pl.String,
            "identity": pl.Float64,
            "query_coverage": pl.Float64,
            "target_coverage": pl.Float64,
            "num_alignments": pl.Int64,
        }
    )


def empty_cluster_frame() -> pl.DataFrame:
    """Return an empty polars DataFrame with the standard cluster schema."""
    return pl.DataFrame(
        schema={
            "seq_id": pl.String,
            "cluster_id": pl.String,
            "representative_id": pl.String,
            "is_representative": pl.Boolean,
            "identity_to_representative": pl.Float64,
            "coverage_to_representative": pl.Float64,
        }
    )

# Derived metrics 
def enrich_edges_with_derived_metrics(
    edges: pl.DataFrame,
    seq_lengths: dict[str, int] | None = None,
) -> pl.DataFrame:
    """Add Global ANI and tANI columns to a base edge table.

    Derived columns (all on a 0-100 percentage scale):

        global_ani_query : ANI(A,B) * AF(A->B) = M(A,B) / |A|
            Approximated as identity * query_coverage / 100.
        global_ani_target : ANI(A,B) * AF(B->A) = M(B) / |B| (approx)
            Approximated as identity * target_coverage / 100.
            Note: this uses target coordinates from the A->B alignment
            which may differ slightly from a true B->A search.
        tani : (M(A,B) + M(B,A)) / (|A| + |B|)
            When *seq_lengths* is given, computed as a length-weighted
            average of the two global ANI values.  Otherwise averaged
            as (global_ani_query + global_ani_target) / 2.

    Args:
        edges: Base edge DataFrame with at least identity, query_coverage,
            target_coverage columns.
        seq_lengths: Optional mapping of sequence ID to length (nt). When
            provided, tANI is computed with proper length weighting.

    Returns:
        Copy of *edges* with three extra float columns appended.
    """
    if edges.is_empty():
        return edges.with_columns(
            pl.lit(0.0).alias("global_ani_query"),
            pl.lit(0.0).alias("global_ani_target"),
            pl.lit(0.0).alias("tani"),
        )

    # Global ANI = identity * coverage / 100 (all in 0-100 scale)
    enriched = edges.with_columns(
        (pl.col("identity") * pl.col("query_coverage") / 100.0)
        .round(2)
        .alias("global_ani_query"),
        (pl.col("identity") * pl.col("target_coverage") / 100.0)
        .round(2)
        .alias("global_ani_target"),
    )

    if seq_lengths:
        # Length-weighted tANI (Vclust formula):
        # tANI = (ANI1*AF1*LEN1 + ANI2*AF2*LEN2) / (LEN1+LEN2)
        # Since global_ani = ANI*AF, this simplifies to:
        # tANI = (global_ani_query*LEN_Q + global_ani_target*LEN_T) / (LEN_Q+LEN_T)
        ql = pl.col("query_id").replace_strict(
            seq_lengths, default=0, return_dtype=pl.Int64,
        ).cast(pl.Float64)
        tl = pl.col("target_id").replace_strict(
            seq_lengths, default=0, return_dtype=pl.Int64,
        ).cast(pl.Float64)
        total_len = ql + tl
        enriched = enriched.with_columns(
            pl.when(total_len > 0)
            .then(
                (
                    pl.col("global_ani_query") * ql
                    + pl.col("global_ani_target") * tl
                )
                / total_len
            )
            .otherwise(0.0)
            .round(2)
            .alias("tani"),
        )
    else:
        # Simple average when lengths are unknown
        enriched = enriched.with_columns(
            (
                (pl.col("global_ani_query") + pl.col("global_ani_target"))
                / 2.0
            )
            .round(2)
            .alias("tani"),
        )

    return enriched


# BLAST outfmt 6 ANI calculation (anicalc-style)


# Shared HSP schema used by BLAST6 and MMseqs tabular outputs
HSP_COLUMNS = [
    "query_id", "target_id", "pident", "length", "mismatch", "gapopen",
    "qstart", "qend", "sstart", "send", "evalue", "bitscore",
    "qlen", "slen",
]

HSP_DTYPES = {
    "query_id": pl.String,
    "target_id": pl.String,
    "pident": pl.Float64,
    "length": pl.Float64,
    "mismatch": pl.Int64,
    "gapopen": pl.Int64,
    "qstart": pl.Int64,
    "qend": pl.Int64,
    "sstart": pl.Int64,
    "send": pl.Int64,
    "evalue": pl.Float64,
    "bitscore": pl.Float64,
    "qlen": pl.Int64,
    "slen": pl.Int64,
}


# Column names for the MMseqs2 14-column output format
MMSEQS_COLUMNS_14 = [
    "query_id", "target_id", "fident", "alnlen", "mismatch", "gapopen",
    "qstart", "qend", "tstart", "tend", "evalue", "bitscore",
    "qlen", "tlen",
]

MMSEQS_COLUMNS_12 = [
    "query_id", "target_id", "fident", "alnlen", "mismatch", "gapopen",
    "qstart", "qend", "tstart", "tend", "evalue", "bitscore",
]


def normalise_alignment_coordinates(df: pl.DataFrame) -> pl.DataFrame:
    """Normalise alignment coordinates so starts are <= ends."""
    return df.with_columns(
        pl.min_horizontal("qstart", "qend").alias("qstart"),
        pl.max_horizontal("qstart", "qend").alias("qend"),
        pl.min_horizontal("sstart", "send").alias("sstart"),
        pl.max_horizontal("sstart", "send").alias("send"),
    )


def filter_hsp_rows(
    hsp_df: pl.DataFrame,
    min_alignment_length: int = 0,
    min_evalue: float = 1e-3,
    apply_pruning: bool = True,
) -> pl.DataFrame:
    """Filter/prune HSP rows using vectorised per-pair Polars operations."""
    if hsp_df.is_empty():
        return hsp_df

    filtered = hsp_df.with_columns(
        (pl.col("qend") - pl.col("qstart") + 1).alias("aln_length")
    ).filter(
        (pl.col("aln_length") >= min_alignment_length)
        & (pl.col("evalue") <= min_evalue)
    )
    if filtered.is_empty() or not apply_pruning:
        return filtered

    pair_keys = ["query_id", "target_id"]
    filtered = filtered.with_columns(
        pl.col("qlen").first().over(pair_keys).alias("query_length"),
        (
            pl.col("aln_length").cum_sum().over(pair_keys)
            - pl.col("aln_length")
        ).alias("cum_before"),
    ).filter(
        (pl.col("cum_before") < pl.col("query_length"))
        & (
            (pl.col("aln_length") + pl.col("cum_before"))
            < (1.10 * pl.col("query_length"))
        )
    )
    return filtered


def hsp_frame_to_edges(
    hsp_df: pl.DataFrame,
    min_alignment_length: int = 0,
    min_evalue: float = 1e-3,
    apply_pruning: bool = True,
    min_identity_prefilter: float | None = None,
) -> pl.DataFrame:
    """Convert a standardised HSP table into the base edge table schema."""
    if hsp_df.is_empty():
        return empty_edge_frame()

    required = ["query_id", "target_id", "pident", "length"]
    missing = [c for c in required if c not in hsp_df.columns]
    if missing:
        raise ValueError(
            "HSP table missing required columns: " + ", ".join(missing)
        )

    # Drop self hits / malformed rows first
    filtered = hsp_df.filter(
        pl.col("pident").is_not_null()
        & pl.col("length").is_not_null()
        & (pl.col("query_id") != pl.col("target_id"))
    )
    if filtered.is_empty():
        return empty_edge_frame()

    # Apply per-HSP filtering and optional CheckV-style cumulative pruning
    supports_hsp_pruning = {
        "qstart", "qend", "evalue", "qlen"
    }.issubset(set(filtered.columns))
    if supports_hsp_pruning:
        filtered = filter_hsp_rows(
            filtered,
            min_alignment_length=min_alignment_length,
            min_evalue=min_evalue,
            apply_pruning=apply_pruning,
        )
    if filtered.is_empty():
        return empty_edge_frame()

    # Vectorised pair-level ANI and count
    pair_stats = filtered.group_by(["query_id", "target_id"]).agg(
        (
            (pl.col("length") * pl.col("pident")).sum()
            / pl.col("length").sum()
        )
        .round(2)
        .alias("identity"),
        pl.len().alias("num_alignments"),
    )

    # Optional fast-path prefilter to avoid coverage work for pairs that
    # are guaranteed to be dropped by downstream identity filtering.
    if min_identity_prefilter is not None:
        pair_stats = pair_stats.filter(
            pl.col("identity") >= float(min_identity_prefilter)
        )
        if pair_stats.is_empty():
            return empty_edge_frame()
        filtered = filtered.join(
            pair_stats.select("query_id", "target_id"),
            on=["query_id", "target_id"],
            how="inner",
        )
        if filtered.is_empty():
            return empty_edge_frame()

    # Coverage uses grouped interval union lengths computed in Polars.
    supports_coverage = {
        "qstart", "qend", "sstart", "send", "qlen", "slen"
    }.issubset(set(filtered.columns))
    if not supports_coverage:
        return pair_stats.with_columns(
            pl.lit(0.0).alias("query_coverage"),
            pl.lit(0.0).alias("target_coverage"),
        ).select(EDGE_COLUMNS)

    coverage_df = compute_bidirectional_group_coverages(filtered)

    return pair_stats.join(
        coverage_df,
        on=["query_id", "target_id"],
        how="left",
    ).with_columns(
        pl.col("query_coverage").fill_null(0.0),
        pl.col("target_coverage").fill_null(0.0),
    ).select(EDGE_COLUMNS)


def read_blast6(path: Union[str, Path]) -> pl.DataFrame:
    """Read a BLAST outfmt 6 file (with qlen slen) into a polars DataFrame.

    Handles plain text and gzip-compressed files.  Lines starting with
    '#' are skipped (comment_prefix).  Both tab and space delimiters
    are supported — tab is tried first.

    Returns:
        DataFrame with typed columns matching HSP_COLUMNS.
    """
    path = Path(path)
    try:
        df = pl.read_csv(
            path,
            separator="\t",
            has_header=False,
            new_columns=HSP_COLUMNS,
            schema_overrides=HSP_DTYPES,
            comment_prefix="#",
            truncate_ragged_lines=True,
            ignore_errors=True,
        )
    except Exception:
        # Fall back to space-delimited
        df = pl.read_csv(
            path,
            separator=" ",
            has_header=False,
            new_columns=HSP_COLUMNS,
            schema_overrides=HSP_DTYPES,
            comment_prefix="#",
            truncate_ragged_lines=True,
            ignore_errors=True,
        )
    return normalise_alignment_coordinates(df)


def compute_bidirectional_group_coverages(df: pl.DataFrame) -> pl.DataFrame:
    """Compute query and target coverages per pair in one Polars pipeline."""
    if df.is_empty():
        return pl.DataFrame(
            schema={
                "query_id": pl.String,
                "target_id": pl.String,
                "query_coverage": pl.Float64,
                "target_coverage": pl.Float64,
            }
        )

    pair_keys = ["query_id", "target_id"]
    coverage_keys = pair_keys + ["axis"]

    query_intervals = df.select(
        pl.col("query_id"),
        pl.col("target_id"),
        pl.lit("query").alias("axis"),
        pl.col("qstart").cast(pl.Int64).alias("start"),
        pl.col("qend").cast(pl.Int64).alias("end"),
        pl.col("qlen").cast(pl.Float64).alias("seq_len"),
    )
    target_intervals = df.select(
        pl.col("query_id"),
        pl.col("target_id"),
        pl.lit("target").alias("axis"),
        pl.col("sstart").cast(pl.Int64).alias("start"),
        pl.col("send").cast(pl.Int64).alias("end"),
        pl.col("slen").cast(pl.Float64).alias("seq_len"),
    )

    intervals = pl.concat(
        [query_intervals, target_intervals],
        how="vertical_relaxed",
    ).filter(
        pl.col("start").is_not_null() & pl.col("end").is_not_null()
    )

    if intervals.is_empty():
        return pl.DataFrame(
            schema={
                "query_id": pl.String,
                "target_id": pl.String,
                "query_coverage": pl.Float64,
                "target_coverage": pl.Float64,
            }
        )

    prepared = (
        intervals.sort(coverage_keys + ["start", "end"])
        .with_columns(
            pl.col("end")
            .cum_max()
            .shift(1)
            .over(coverage_keys)
            .fill_null(pl.col("start") - 1)
            .alias("prev_max_end"),
        )
        .with_columns(
            pl.max_horizontal("start", pl.col("prev_max_end") + 1)
            .alias("new_start"),
        )
        .with_columns(
            pl.when(pl.col("end") >= pl.col("new_start"))
            .then(pl.col("end") - pl.col("new_start") + 1)
            .otherwise(0)
            .alias("covered_increment"),
        )
    )

    coverage_long = (
        prepared.group_by(coverage_keys)
        .agg(
            pl.col("covered_increment").sum().cast(pl.Float64).alias("covered_len"),
            pl.col("seq_len").first().alias("seq_len"),
        )
        .with_columns(
            pl.when((pl.col("seq_len").is_not_null()) & (pl.col("seq_len") > 0))
            .then((100.0 * pl.col("covered_len") / pl.col("seq_len")).round(2))
            .otherwise(0.0)
            .alias("coverage")
        )
        .select("query_id", "target_id", "axis", "coverage")
    )

    return coverage_long.group_by(pair_keys).agg(
        pl.when(pl.col("axis") == "query")
        .then(pl.col("coverage"))
        .otherwise(None)
        .max()
        .fill_null(0.0)
        .alias("query_coverage"),
        pl.when(pl.col("axis") == "target")
        .then(pl.col("coverage"))
        .otherwise(None)
        .max()
        .fill_null(0.0)
        .alias("target_coverage"),
    )


def parse_blast6_to_edges(
    blast_path: Union[str, Path],
    min_alignment_length: int = 0,
    min_evalue: float = 1e-3,
    min_identity_prefilter: float | None = None,
) -> pl.DataFrame:
    """Parse BLAST outfmt 6 (with qlen slen) into the standard edge table.

    Uses polars to read the tabular file, then groups by (query, target)
    pair and applies the CheckV-style ANI/coverage computation per group
    (pruning, interval merging, weighted identity).

    Expected BLAST command:
        blastn -query in.fa -subject in.fa -outfmt
        '6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen'

    Args:
        blast_path: Path to BLAST tabular output file.
        min_alignment_length: Minimum individual alignment length to keep.
        min_evalue: Maximum evalue for individual alignments.

    Returns:
        Polars DataFrame with standard edge columns.
    """
    blast_path = Path(blast_path)
    if not blast_path.exists():
        raise FileNotFoundError(f"BLAST output not found: {blast_path}")

    raw = read_blast6(blast_path)
    if raw.is_empty():
        return empty_edge_frame()

    return hsp_frame_to_edges(
        raw,
        min_alignment_length=min_alignment_length,
        min_evalue=min_evalue,
        apply_pruning=True,
        min_identity_prefilter=min_identity_prefilter,
    )


#  CheckV-style ANI table parsing 


def parse_ani_table(
    ani_path: Union[str, Path],
) -> pl.DataFrame:
    """Parse a CheckV-style ANI table into the standard edge table.

    Expected columns (tab-delimited): qname tname num_alns pid qcov tcov
    Handles both plain-text and gzip-compressed files.

    Args:
        ani_path: Path to the ANI table file.

    Returns:
        Polars DataFrame with standard edge columns.
    """
    ani_path = Path(ani_path)
    if not ani_path.exists():
        raise FileNotFoundError(f"ANI table not found: {ani_path}")

    _ani_col_names = [
        "query_id", "target_id", "num_alignments",
        "identity", "query_coverage", "target_coverage",
    ]
    _ani_dtypes = {
        "query_id": pl.String,
        "target_id": pl.String,
        "num_alignments": pl.Int64,
        "identity": pl.Float64,
        "query_coverage": pl.Float64,
        "target_coverage": pl.Float64,
    }
    try:
        df = pl.read_csv(
            ani_path,
            separator="\t",
            has_header=False,
            new_columns=_ani_col_names,
            schema_overrides=_ani_dtypes,
            comment_prefix="#",
            truncate_ragged_lines=True,
            ignore_errors=True,
        )
    except Exception:
        df = pl.read_csv(
            ani_path,
            separator=" ",
            has_header=False,
            new_columns=_ani_col_names,
            schema_overrides=_ani_dtypes,
            comment_prefix="#",
            truncate_ragged_lines=True,
            ignore_errors=True,
        )

    if df.is_empty():
        return empty_edge_frame()

    # Drop header rows (identity column will be null after failed cast)
    # and self-hits
    df = df.filter(
        pl.col("identity").is_not_null()
        & (pl.col("query_id") != pl.col("target_id"))
    )

    if df.is_empty():
        return empty_edge_frame()
    return df.select(EDGE_COLUMNS)


#  MMseqs2 easy-search output parsing 
def parse_mmseqs_table(
    mmseqs_path: Union[str, Path],
) -> pl.DataFrame:
    """Parse MMseqs2 easy-search output into the standard edge table.

    Expects columns:
        query target fident alnlen mismatch gapopen qstart qend tstart
        tend evalue bits qlen tlen

    The last two columns (qlen, tlen) are needed for coverage computation
    and can be added via ``--format-output query,target,fident,alnlen,...,qlen,tlen``.

    If qlen/tlen are absent, coverage values are set to 0.
    Handles both plain-text and gzip-compressed files.

    Args:
        mmseqs_path: Path to MMseqs2 tabular output.

    Returns:
        Polars DataFrame with standard edge columns.
    """
    mmseqs_path = Path(mmseqs_path)
    if not mmseqs_path.exists():
        raise FileNotFoundError(f"MMseqs2 output not found: {mmseqs_path}")

    # Read all columns as strings first to detect width, then cast
    raw = pl.read_csv(
        mmseqs_path,
        separator="\t",
        has_header=False,
        comment_prefix="#",
        truncate_ragged_lines=True,
        ignore_errors=True,
        infer_schema_length=0,  # read everything as String
    )
    if raw.is_empty():
        return empty_edge_frame()

    ncols = raw.width
    has_lengths = ncols >= 14

    col_names = MMSEQS_COLUMNS_14 if has_lengths else MMSEQS_COLUMNS_12
    # Rename only the columns we care about
    rename_map = {
        raw.columns[i]: col_names[i]
        for i in range(min(len(col_names), ncols))
    }
    raw = raw.rename(rename_map)

    # Convert MMseqs names to the shared HSP schema
    hsp = raw.rename(
        {
            "fident": "pident",
            "alnlen": "length",
            "tstart": "sstart",
            "tend": "send",
        }
    ).with_columns(
        pl.col("pident").cast(pl.Float64, strict=False),
        pl.col("length").cast(pl.Float64, strict=False),
        pl.col("qstart").cast(pl.Int64, strict=False),
        pl.col("qend").cast(pl.Int64, strict=False),
        pl.col("sstart").cast(pl.Int64, strict=False),
        pl.col("send").cast(pl.Int64, strict=False),
        pl.col("evalue").cast(pl.Float64, strict=False),
    )

    # fident from mmseqs is commonly 0-1; convert to 0-100 if needed.
    hsp = hsp.with_columns(
        pl.when(pl.col("pident") <= 1.0)
        .then(pl.col("pident") * 100.0)
        .otherwise(pl.col("pident"))
        .round(2)
        .alias("pident"),
    )

    # Keep qlen/slen when present, otherwise add null placeholders.
    if has_lengths:
        hsp = hsp.rename({"tlen": "slen"}).with_columns(
            pl.col("qlen").cast(pl.Float64, strict=False),
            pl.col("slen").cast(pl.Float64, strict=False),
        )
    else:
        hsp = hsp.with_columns(
            pl.lit(None).cast(pl.Float64).alias("qlen"),
            pl.lit(None).cast(pl.Float64).alias("slen"),
        )

    hsp = normalise_alignment_coordinates(hsp)
    hsp = hsp.select(
        [col for col in HSP_COLUMNS if col in hsp.columns]
    )

    # MMseqs is not pre-pruned like CheckV BLAST6 output; keep all rows.
    return hsp_frame_to_edges(
        hsp,
        min_alignment_length=0,
        min_evalue=float("inf"),
        apply_pruning=False,
    )


#  On-the-fly ANI calculation backends 


def compute_ani_pyskani(
    fasta_path: Union[str, Path],
    min_identity: float = 0.0,
    threads: int = 1,
    logger: logging.Logger | None = None,
) -> pl.DataFrame:
    """Compute pairwise ANI from a FASTA file using pyskani.

    Args:
        fasta_path: Path to input FASTA/FASTQ file.
        min_identity: Lower identity cutoff (0-1 scale) for pyskani
            prefiltering. Hits slightly below this may still appear so
            the caller should apply final filtering.
        threads: Number of threads for pyskani (not currently
            parallelised beyond the internal C implementation).
        logger: Optional logger instance.

    Returns:
        Polars DataFrame with standard edge columns.
    """
    from rolypoly.utils.bio.polars_fastx import load_sequences

    seq_df = load_sequences(str(fasta_path))
    if seq_df.is_empty() or seq_df.height < 2:
        return empty_edge_frame()

    seq_rows = seq_df.to_dicts()
    # pyskani truncates names at whitespace, so use sanitised index-based
    # names and map back to the original IDs afterwards.
    idx_to_name: dict[str, str] = {}
    database = pyskani.Database()
    for idx, row in enumerate(seq_rows):
        original_name = str(row.get("contig_id", row.get("header", "")))
        safe_name = f"seq_{idx}"
        idx_to_name[safe_name] = original_name
        seq = str(row.get("sequence", ""))
        database.sketch(safe_name, seq.encode("ascii"))

    cutoff = max(0.0, min_identity - 0.05) if min_identity > 0 else 0.0
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(seq_rows):
        safe_name = f"seq_{idx}"
        seq = str(row.get("sequence", ""))
        hits = database.query(safe_name, seq.encode("ascii"), cutoff=cutoff)
        for hit in hits:
            ref_safe = str(getattr(hit, "reference_name", ""))
            if ref_safe == safe_name:
                continue
            if ref_safe not in idx_to_name:
                continue
            identity = float(getattr(hit, "identity", 0.0))
            qfrac = float(getattr(hit, "query_fraction", 0.0))
            rfrac = float(getattr(hit, "reference_fraction", 0.0))
            rows.append(
                {
                    "query_id": idx_to_name[safe_name],
                    "target_id": idx_to_name[ref_safe],
                    "identity": round(identity * 100.0, 2),
                    "query_coverage": round(qfrac * 100.0, 2),
                    "target_coverage": round(rfrac * 100.0, 2),
                    "num_alignments": 1,
                }
            )

    if not rows:
        return empty_edge_frame()

    if logger:
        logger.info(
            "pyskani produced %s raw edges from %s sequences",
            len(rows),
            len(seq_rows),
        )
    return pl.DataFrame(rows).select(EDGE_COLUMNS)


def compute_ani_pyfastani(
    fasta_path: Union[str, Path],
    threads: int = 1,
    fragment_length: int = 1000,
    minimum_fraction: float = 0.05,
    percentage_identity: int = 70,
    k: int = 16,
    logger: logging.Logger | None = None,
) -> pl.DataFrame:
    """Compute pairwise ANI from a FASTA file using pyfastani (FastANI).

    pyfastani implements the FastANI algorithm natively in Python/Cython.
    The default parameter tweaks here lower the fragment length and
    minimum-fraction thresholds relative to FastANI's prokaryote-genome
    defaults so that shorter RNA-virus contigs from metatranscriptomic
    assemblies still produce hits.

    Note: FastANI reports identity and the fraction of query fragments
    that matched (matches/fragments) for a directed query. This
    function uses that as ``query_coverage`` and estimates
    ``target_coverage`` from the reverse-direction hit when available
    (falling back to query coverage if the reverse edge is absent).

    Args:
        fasta_path: Path to input FASTA/FASTQ file.
        threads: Number of threads for the fragment-mapping step.
        fragment_length: Maximum fragment length for query splitting.
            FastANI default is 3000 (suited for prokaryote genomes).
            For short viral contigs this function auto-reduces the
            effective value based on sequence-length quantiles.
        minimum_fraction: Minimum fraction of the smaller genome that
            must be shared for a hit.  FastANI default is 0.2; lowered
            to 0.05 for partial/short assemblies.
        percentage_identity: Lower-bound identity percentage for the
            internal window-size heuristic.  FastANI default is 80;
            lowered to 70 to catch more divergent pairs.
        k: K-mer size (max pyfastani.MAX_KMER_SIZE, typically 16).
        logger: Optional logger instance.

    Returns:
        Polars DataFrame with standard edge columns.
    """
    from rolypoly.utils.bio.polars_fastx import load_sequences

    seq_df = load_sequences(str(fasta_path))
    if seq_df.is_empty() or seq_df.height < 2:
        return empty_edge_frame()

    # Auto-tune fragment length for short-contig datasets while keeping
    # the caller-provided value as an upper bound. This avoids sparse
    # hits when most contigs are shorter than the nominal fragment size.
    effective_fragment_length = fragment_length
    if "seq_length" in seq_df.columns:
        q25_len = seq_df.select(
            pl.col("seq_length").quantile(0.25, interpolation="nearest")
        ).item()
        if q25_len is not None:
            effective_fragment_length = min(
                fragment_length,
                max(200, int(q25_len)),
            )

    seq_rows = seq_df.to_dicts()
    names: list[str] = []
    sequences: list[str] = []
    for row in seq_rows:
        names.append(str(row.get("contig_id", row.get("header", ""))))
        sequences.append(str(row.get("sequence", "")))

    # Build sketch with adjusted parameters
    sketch = pyfastani.Sketch(
        k=k,
        fragment_length=effective_fragment_length,
        minimum_fraction=minimum_fraction,
        percentage_identity=percentage_identity,
    )
    for name, seq in zip(names, sequences):
        sketch.add_genome(name, seq)

    mapper = sketch.index()

    if logger:
        logger.info(
            "pyfastani: indexed %d sequences (k=%d, frag_len=%d, min_frac=%.2f)",
            len(names), k, effective_fragment_length, minimum_fraction,
        )

    rows: list[dict[str, Any]] = []
    for query_name, query_seq in zip(names, sequences):
        hits = mapper.query_genome(query_seq, threads=threads)
        for hit in hits:
            target_name = str(hit.name)
            if target_name == query_name:
                continue
            identity = float(hit.identity)
            # Coverage approximated from matches/fragments ratio
            coverage = (
                round(100.0 * hit.matches / hit.fragments, 2)
                if hit.fragments > 0 else 0.0
            )
            rows.append(
                {
                    "query_id": query_name,
                    "target_id": target_name,
                    "identity": round(identity, 2),
                    "query_coverage": coverage,
                }
            )

    if not rows:
        return empty_edge_frame()

    edges = (
        pl.DataFrame(rows)
        .group_by(["query_id", "target_id"])
        .agg(
            pl.col("identity").mean().round(2).alias("identity"),
            pl.col("query_coverage").max().round(2).alias("query_coverage"),
            pl.len().alias("num_alignments"),
        )
    )

    reverse_cov = edges.select(
        pl.col("query_id").alias("target_id"),
        pl.col("target_id").alias("query_id"),
        pl.col("query_coverage").alias("target_coverage"),
    )

    edges = (
        edges.join(reverse_cov, on=["query_id", "target_id"], how="left")
        .with_columns(
            pl.col("target_coverage").fill_null(pl.col("query_coverage")).round(2)
        )
        .select(EDGE_COLUMNS)
    )

    if logger:
        logger.info(
            "pyfastani produced %s raw edges from %s sequences",
            edges.height,
            len(seq_rows),
        )
    return edges


def compute_ani_blastn(
    fasta_path: Union[str, Path],
    threads: int = 1,
    min_alignment_length: int = 0,
    min_evalue: float = 1e-3,
    min_identity_prefilter: float | None = None,
    logger: logging.Logger | None = None,
) -> pl.DataFrame:
    """Compute pairwise ANI by running blastn all-vs-all and parsing results.

    Requires blastn to be available on PATH or in the pixi environment.

    Args:
        fasta_path: Path to input FASTA file.
        threads: Number of threads for blastn.
        min_alignment_length: Minimum individual alignment length.
        min_evalue: Maximum evalue for individual alignments.
        logger: Optional logger instance.

    Returns:
        Polars DataFrame with standard edge columns.
    """
    from rolypoly.utils.various import run_command_comp

    fasta_path = Path(fasta_path)
    with tempfile.TemporaryDirectory(prefix="rolypoly_blast_") as tmpdir:
        outfile = Path(tmpdir) / "blast.out"
        # Create BLAST database
        run_command_comp(
            "makeblastdb",
            positional_args=[],
            params={
                "in": str(fasta_path),
                "dbtype": "nucl",
                "out": str(Path(tmpdir) / "blastdb"),
            },
            logger=logger,
            check_status=True,
            prefix_style="single",
        )
        # Run all-vs-all blastn
        run_command_comp(
            "blastn",
            positional_args=[],
            params={
                "query": str(fasta_path),
                "db": str(Path(tmpdir) / "blastdb"),
                "out": str(outfile),
                "num_threads": str(threads),
                "max_target_seqs": "25000",
                "evalue": str(min_evalue),
                "outfmt": str("'6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen'"),
            },
            logger=logger,
            check_status=True,
            prefix_style="single",
        )
        if not outfile.exists() or outfile.stat().st_size == 0:
            if logger:
                logger.warning("blastn produced no output")
            return empty_edge_frame()
        return parse_blast6_to_edges(
            outfile,
            min_alignment_length,
            min_evalue,
            min_identity_prefilter=min_identity_prefilter,
        )


def compute_ani_mmseqs(
    fasta_path: Union[str, Path],
    threads: int = 1,
    sensitivity: float = 7.5,
    logger: logging.Logger | None = None,
) -> pl.DataFrame:
    """Compute pairwise ANI by running mmseqs easy-search all-vs-all.

    Requires mmseqs to be available on PATH or in the pixi environment.

    Args:
        fasta_path: Path to input FASTA file.
        threads: Number of threads for mmseqs.
        sensitivity: mmseqs sensitivity parameter (-s).
        logger: Optional logger instance.

    Returns:
        Polars DataFrame with standard edge columns.
    """
    from rolypoly.utils.various import run_command_comp

    fasta_path = Path(fasta_path)
    with tempfile.TemporaryDirectory(prefix="rolypoly_mmseqs_") as tmpdir:
        outfile = Path(tmpdir) / "mmseqs.out"
        tmp_subdir = Path(tmpdir) / "tmp"
        tmp_subdir.mkdir()
        run_command_comp(
            "mmseqs",
            positional_args=[
                "easy-search",
                str(fasta_path),
                str(fasta_path),
                str(outfile),
                str(tmp_subdir),
            ],
            positional_args_location="start",
            params={
                "threads": str(threads),
                "s": str(sensitivity),
                "search-type": "3",  # nucleotide
                "format-output": (
                    "query,target,fident,alnlen,mismatch,gapopen,"
                    "qstart,qend,tstart,tend,evalue,bits,qlen,tlen"
                ),
            },
            logger=logger,
            check_status=True,
        )
        if not outfile.exists() or outfile.stat().st_size == 0:
            if logger:
                logger.warning("mmseqs easy-search produced no output")
            return empty_edge_frame()
        return parse_mmseqs_table(outfile)


#  K-mer-based identity estimation 


# Translation table for complement (DNA only, uppercase)
_COMPLEMENT_TABLE = str.maketrans("ACGT", "TGCA")


def canonical_kmer(kmer: str) -> str:
    """Return the canonical (lexicographically smaller) form of a k-mer.

    The canonical form is ``min(kmer, revcomp(kmer))``. Using canonical
    k-mers ensures that the same subsequence is counted identically
    regardless of which DNA strand it originates from.

    Args:
        kmer: Uppercase DNA k-mer (A/T/C/G only).

    Returns:
        The lexicographically smaller of the k-mer and its reverse
        complement.
    """
    rc = kmer.translate(_COMPLEMENT_TABLE)[::-1]
    return kmer if kmer <= rc else rc


def extract_kmer_sets(
    seq_df: pl.DataFrame,
    k: int = 15,
    skip_ambiguous: bool = True,
    canonical: bool = True,
) -> dict[str, tuple[set[str], int]]:
    """Extract distinct k-mer sets for each sequence.

    Args:
        seq_df: DataFrame with at least 'contig_id' and 'sequence' columns
            (as produced by polars_fastx.load_sequences).
        k: K-mer length (default 15, following Kmer-db 2).
        skip_ambiguous: When True, k-mers containing characters other
            than A/T/C/G are skipped.
        canonical: When True (default), each k-mer is reduced to its
            canonical form ``min(kmer, revcomp(kmer))`` so that both
            DNA strands contribute to the same k-mer set.  This is
            essential for comparing sequences that may be on opposite
            strands.

    Returns:
        Dict mapping sequence ID to (kmer_set, distinct_kmer_count).
    """
    valid = set("ATCG")
    kmer_sets: dict[str, tuple[set[str], int]] = {}
    for row in seq_df.iter_rows(named=True):
        seq_id = str(row.get("contig_id", row.get("header", "")))
        seq = str(row.get("sequence", "")).upper()
        kmers: set[str] = set()
        for i in range(len(seq) - k + 1):
            kmer = seq[i : i + k]
            if skip_ambiguous and not all(c in valid for c in kmer):
                continue
            if canonical:
                kmer = canonical_kmer(kmer)
            kmers.add(kmer)
        kmer_sets[seq_id] = (kmers, len(kmers))
    return kmer_sets


def compute_kmer_overlap_matrix(
    fasta_path: Union[str, Path],
    k: int = 15,
    min_overlap: float = 0.0,
    skip_ambiguous: bool = True,
    canonical: bool = True,
    logger: logging.Logger | None = None,
) -> pl.DataFrame:
    """Estimate pairwise sequence similarity using k-mer overlap coefficient.

    The overlap coefficient is defined as:
        OC(A, B) = |kmers(A) ∩ kmers(B)| / min(|kmers(A)|, |kmers(B)|)

    It is preferred over the Jaccard index for sequences of unequal
    length because it measures the fraction of the smaller k-mer set
    that is shared, without penalising length asymmetry (Vclust /
    Kmer-db 2 approach, DOI: 10.1038/s41592-025-02701-7).

    The function builds an inverted index over k-mers to avoid computing
    all O(n^2) pairwise comparisons.  Only pairs that share at least one
    k-mer are evaluated, making it efficient for sparse similarity
    graphs.

    Args:
        fasta_path: Path to input FASTA/FASTQ file.
        k: K-mer length (default 15).
        min_overlap: Minimum overlap coefficient (0-1) to report.
            Pairs below this are discarded.
        skip_ambiguous: Skip k-mers with non-ATCG characters.
        canonical: Use canonical (strand-agnostic) k-mers (default True).
        logger: Optional logger.

    Returns:
        Polars DataFrame with columns:
            query_id, target_id, kmer_overlap (0-1 scale),
            num_shared_kmers, query_kmer_count, target_kmer_count
    """
    from rolypoly.utils.bio.polars_fastx import load_sequences

    empty_schema = {
        "query_id": pl.String,
        "target_id": pl.String,
        "kmer_overlap": pl.Float64,
        "num_shared_kmers": pl.Int64,
        "query_kmer_count": pl.Int64,
        "target_kmer_count": pl.Int64,
    }

    seq_df = load_sequences(str(fasta_path))
    if seq_df.is_empty() or seq_df.height < 2:
        return pl.DataFrame(schema=empty_schema)

    if logger:
        logger.info(
            "Extracting %d-mer sets from %d sequences", k, seq_df.height
        )

    kmer_data = extract_kmer_sets(
        seq_df, k=k, skip_ambiguous=skip_ambiguous, canonical=canonical,
    )

    # Build inverted index: kmer -> list of seq_ids
    inverted: dict[str, list[str]] = {}
    for seq_id, (kmer_set, _) in kmer_data.items():
        for kmer in kmer_set:
            inverted.setdefault(kmer, []).append(seq_id)

    if logger:
        logger.info(
            "Built inverted index with %d distinct %d-mers",
            len(inverted),
            k,
        )

    # Count shared k-mers per pair using the inverted index
    pair_counts: dict[tuple[str, str], int] = {}
    for seq_ids in inverted.values():
        if len(seq_ids) < 2:
            continue
        # Only process pairs in canonical order to avoid double-counting
        for i in range(len(seq_ids)):
            for j in range(i + 1, len(seq_ids)):
                pair = (seq_ids[i], seq_ids[j])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

    if logger:
        logger.info(
            "Found %d sequence pairs sharing at least one %d-mer",
            len(pair_counts),
            k,
        )

    # Compute overlap coefficient for each pair
    rows: list[dict[str, Any]] = []
    for (a, b), shared_count in pair_counts.items():
        a_count = kmer_data[a][1]
        b_count = kmer_data[b][1]
        min_size = min(a_count, b_count)
        if min_size == 0:
            continue
        overlap = shared_count / min_size
        if overlap < min_overlap:
            continue
        rows.append(
            {
                "query_id": a,
                "target_id": b,
                "kmer_overlap": round(overlap, 4),
                "num_shared_kmers": shared_count,
                "query_kmer_count": a_count,
                "target_kmer_count": b_count,
            }
        )

    if logger:
        logger.info(
            "Retained %d pairs with overlap >= %.2f", len(rows), min_overlap
        )

    if not rows:
        return pl.DataFrame(schema=empty_schema)
    return pl.DataFrame(rows)


def kmer_to_edge_table(
    kmer_df: pl.DataFrame,
) -> pl.DataFrame:
    """Convert a k-mer overlap matrix into the standard edge table format.

    Maps the kmer_overlap (0-1) to percentage scale for the identity
    column.  Coverage columns are set to 100.0 since k-mer overlap
    does not directly measure alignment coverage.

    Args:
        kmer_df: DataFrame from compute_kmer_overlap_matrix().

    Returns:
        Base edge table DataFrame.
    """
    if kmer_df.is_empty():
        return empty_edge_frame()
    return kmer_df.select(
        pl.col("query_id"),
        pl.col("target_id"),
        (pl.col("kmer_overlap") * 100.0).round(2).alias("identity"),
        pl.lit(100.0).alias("query_coverage"),
        pl.lit(100.0).alias("target_coverage"),
        pl.lit(1).cast(pl.Int64).alias("num_alignments"),
    )


def kmer_prefilter_pairs(
    fasta_path: Union[str, Path],
    k: int = 15,
    min_overlap: float = 0.5,
    skip_ambiguous: bool = True,
    canonical: bool = True,
    logger: logging.Logger | None = None,
) -> set[tuple[str, str]]:
    """Return sequence-ID pairs that pass a k-mer overlap prefilter.

    Useful for reducing the number of pairwise alignments: only pairs
    with a k-mer overlap coefficient above *min_overlap* are returned,
    and these can then be fed into a more expensive alignment backend.

    Args:
        fasta_path: Path to FASTA file.
        k: K-mer length.
        min_overlap: Minimum overlap coefficient (0-1).
        skip_ambiguous: Skip k-mers with non-ATCG characters.
        canonical: Use canonical (strand-agnostic) k-mers (default True).
        logger: Optional logger.

    Returns:
        Set of (seq_id_a, seq_id_b) tuples (both directions).
    """
    kmer_df = compute_kmer_overlap_matrix(
        fasta_path,
        k=k,
        min_overlap=min_overlap,
        skip_ambiguous=skip_ambiguous,
        canonical=canonical,
        logger=logger,
    )
    if kmer_df.is_empty():
        return set()
    pairs: set[tuple[str, str]] = set()
    for row in kmer_df.select("query_id", "target_id").iter_rows():
        pairs.add((row[0], row[1]))
        pairs.add((row[1], row[0]))  # add both directions
    return pairs


#  Edge filtering 


def filter_edges(
    edges: pl.DataFrame,
    min_identity: float = 95.0,
    min_query_coverage: float = 0.0,
    min_target_coverage: float = 85.0,
    min_alignment_fraction: float = 0.0,
    similarity_column: str = "identity",
) -> pl.DataFrame:
    """Filter an edge table by identity and coverage thresholds.

    The *similarity_column* selects which measure is compared against
    *min_identity*.  Accepted values include the base column ``identity``
    (ANI) and the enriched columns ``tani``, ``global_ani_query``,
    ``global_ani_target``.  If the requested column is absent from
    *edges*, the function falls back to ``identity`` with a warning.

    Args:
        edges: Standard (or enriched) edge DataFrame.
        min_identity: Minimum similarity percentage (0-100) applied to
            the column selected by *similarity_column*.
        min_query_coverage: Minimum query coverage percentage (0-100).
        min_target_coverage: Minimum target coverage percentage (0-100).
        min_alignment_fraction: Minimum alignment fraction
            (min(qcov, tcov), 0-100). When set, overrides individual
            qcov/tcov thresholds.
        similarity_column: Edge column to threshold on (default
            ``identity``).  See ``SIMILARITY_COLUMNS`` for recognised
            aliases.

    Returns:
        Filtered edge DataFrame.
    """
    if edges.is_empty():
        return edges

    # Resolve alias
    col_name = SIMILARITY_COLUMNS.get(similarity_column, similarity_column)
    if col_name not in edges.columns:
        logging.getLogger(__name__).warning(
            "Similarity column '%s' not found in edges, falling back to 'identity'",
            col_name,
        )
        col_name = "identity"

    filtered = edges.filter(pl.col(col_name) >= min_identity)
    if min_query_coverage > 0:
        filtered = filtered.filter(
            pl.col("query_coverage") >= min_query_coverage
        )
    if min_target_coverage > 0:
        filtered = filtered.filter(
            pl.col("target_coverage") >= min_target_coverage
        )
    if min_alignment_fraction > 0:
        filtered = filtered.filter(
            pl.min_horizontal("query_coverage", "target_coverage")
            >= min_alignment_fraction
        )
    return filtered


#  Repetitive-sequence flagging 


def flag_repetitive_sequences(
    sequences: dict[str, str],
    k: int = 15,
    max_repeat_fraction: float = 0.40,
    logger: logging.Logger | None = None,
) -> set[str]:
    """Flag sequences whose self-dotplot suggests assembly artefacts.

    For each sequence, computes the longest forward and inverted repeat
    track via exact *k*-mer self-matching (using
    :func:`~rolypoly.utils.bio.dotplot.compute_self_dotplot_track_spans`).
    A sequence is flagged when either track spans at least
    *max_repeat_fraction* of its length.

    Flagged sequences are still included in clustering but should be
    excluded from representative selection to avoid propagating
    artefacts.

    Args:
        sequences: Mapping of sequence ID to nucleotide string.
        k: K-mer size for the dotplot analysis.
        max_repeat_fraction: Maximum fraction of sequence length that
            the longest repeat track may span before the sequence is
            flagged (default 0.40 = 40%).
        logger: Optional logger for debug output.

    Returns:
        Set of sequence IDs that were flagged as repetitive.
    """
    from rolypoly.utils.bio.dotplot import compute_self_dotplot_track_spans

    flagged: set[str] = set()
    for seq_id, seq in sequences.items():
        seq_len = len(seq)
        if seq_len < 2 * k:
            continue
        metrics = compute_self_dotplot_track_spans(seq, k=k)
        fwd_span = metrics.get("dotplot_forward_max_span", 0)
        inv_span = metrics.get("dotplot_inverted_max_span", 0)
        max_span = max(fwd_span, inv_span)
        if max_span >= max_repeat_fraction * seq_len:
            flagged.add(seq_id)
            if logger:
                logger.debug(
                    "Flagged repetitive: %s  len=%d  fwd_span=%d  inv_span=%d  (%.1f%%)",
                    seq_id, seq_len, fwd_span, inv_span,
                    100.0 * max_span / seq_len,
                )

    if logger:
        logger.info(
            "Repeat-flag check: %d / %d sequences flagged as repetitive (%.1f%%)",
            len(flagged), len(sequences),
            100.0 * len(flagged) / max(len(sequences), 1),
        )
    return flagged


#  Clustering algorithms 


def cluster_connected_components(
    edges: pl.DataFrame,
    seq_ids: list[str] | None = None,
    exclude_as_representatives: set[str] | None = None,
) -> pl.DataFrame:
    """Cluster sequences into connected components from an edge table.

    Uses a union-find data structure to group all sequences that share
    at least one edge (directly or transitively). The longest sequence
    in each component is chosen as representative.

    Args:
        edges: Standard edge DataFrame (already filtered by thresholds).
        seq_ids: Optional complete list of sequence IDs. Sequences not
            present in *edges* will appear as singleton clusters.
        exclude_as_representatives: Optional set of sequence IDs that
            should not be chosen as cluster representatives (e.g.
            repetitive / artefact sequences). If every member of a
            component is excluded, the first member is used anyway.

    Returns:
        Polars DataFrame with standard cluster columns.
    """
    # Collect all IDs
    all_ids: set[str] = set()
    if seq_ids:
        all_ids.update(seq_ids)
    if not edges.is_empty():
        all_ids.update(edges["query_id"].to_list())
        all_ids.update(edges["target_id"].to_list())
    if not all_ids:
        return empty_cluster_frame()

    id_list = sorted(all_ids)
    id_to_idx = {name: idx for idx, name in enumerate(id_list)}
    parent = list(range(len(id_list)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if not edges.is_empty():
        for qid, tid in edges.select(["query_id", "target_id"]).iter_rows():
            qi, ti = id_to_idx[qid], id_to_idx[tid]
            union(qi, ti)

    # Build components
    components: dict[int, list[str]] = {}
    for idx, name in enumerate(id_list):
        root = find(idx)
        components.setdefault(root, []).append(name)

    # Build edge lookup for identity/coverage to representative
    edge_lookup: dict[tuple[str, str], tuple[float, float]] = {}
    if not edges.is_empty():
        compact_edges = edges.select(
            "query_id", "target_id", "identity", "query_coverage", "target_coverage"
        )
        for qid, tid, ident, qcov, tcov in compact_edges.iter_rows():
            af = min(qcov, tcov)
            edge_lookup[(qid, tid)] = (ident, af)
            edge_lookup[(tid, qid)] = (ident, af)

    return build_cluster_assignment_rows(
        components, edge_lookup, exclude_as_representatives,
    )


def cluster_centroid_greedy(
    edges: pl.DataFrame,
    seq_lengths: dict[str, int] | None = None,
    seq_ids: list[str] | None = None,
    exclude_as_representatives: set[str] | None = None,
) -> pl.DataFrame:
    """Centroid / greedy incremental clustering (CD-HIT / CheckV style).

    Sequences are sorted by decreasing length. Each sequence is compared
    against existing centroids; the first centroid it passes thresholds
    with becomes its cluster representative. If none pass, it becomes a
    new centroid.

    The edge table must already be filtered by identity and coverage
    thresholds before calling this function.

    Args:
        edges: Standard edge DataFrame (already filtered).
        seq_lengths: Mapping of seq_id to sequence length. Used for
            sorting. If not provided, all sequences are treated as equal
            length (ordered alphabetically).
        seq_ids: Optional complete list of sequence IDs.
        exclude_as_representatives: Optional set of sequence IDs that
            should not become centroids. Excluded sequences are still
            assigned to clusters but never seed new centroids (unless
            no non-excluded sequence is available).

    Returns:
        Polars DataFrame with standard cluster columns.
    """
    # Collect all IDs
    all_ids: set[str] = set()
    if seq_ids:
        all_ids.update(seq_ids)
    if not edges.is_empty():
        all_ids.update(edges["query_id"].to_list())
        all_ids.update(edges["target_id"].to_list())
    if not all_ids:
        return empty_cluster_frame()

    # Sort by length descending, break ties by name.
    # Additionally, push excluded sequences to the end so that
    # non-excluded sequences are considered for centroid status first.
    _excl = exclude_as_representatives or set()
    if seq_lengths:
        sorted_ids = sorted(
            all_ids,
            key=lambda x: (x in _excl, -seq_lengths.get(x, 0), x),
        )
    else:
        sorted_ids = sorted(
            all_ids, key=lambda x: (x in _excl, x),
        )

    # Build adjacency: for each seq, store all its edges
    neighbours: dict[str, dict[str, tuple[float, float]]] = {
        sid: {} for sid in sorted_ids
    }
    if not edges.is_empty():
        compact_edges = edges.select(
            "query_id", "target_id", "identity", "query_coverage", "target_coverage"
        )
        for q, t, ident, qcov, tcov in compact_edges.iter_rows():
            af = min(qcov, tcov)
            # store bidirectional
            if t not in neighbours.get(q, {}):
                neighbours.setdefault(q, {})[t] = (ident, af)
            if q not in neighbours.get(t, {}):
                neighbours.setdefault(t, {})[q] = (ident, af)

    centroids: list[str] = []
    seq_to_centroid: dict[str, str] = {}
    seq_to_stats: dict[str, tuple[float, float]] = {}

    for sid in sorted_ids:
        if sid in seq_to_centroid:
            continue
        assigned = False
        for centroid in centroids:
            if centroid in neighbours.get(sid, {}):
                identity, af = neighbours[sid][centroid]
                seq_to_centroid[sid] = centroid
                seq_to_stats[sid] = (identity, af)
                assigned = True
                break
        if not assigned:
            # new centroid
            centroids.append(sid)
            seq_to_centroid[sid] = sid
            seq_to_stats[sid] = (100.0, 100.0)

    # Build output
    cluster_number: dict[str, int] = {}
    cluster_idx = 0
    rows: list[dict[str, Any]] = []
    for sid in sorted_ids:
        rep = seq_to_centroid[sid]
        if rep not in cluster_number:
            cluster_number[rep] = cluster_idx
            cluster_idx += 1
        cid = f"cluster_{cluster_number[rep]}"
        identity, af = seq_to_stats.get(sid, (100.0, 100.0))
        rows.append(
            {
                "seq_id": sid,
                "cluster_id": cid,
                "representative_id": rep,
                "is_representative": sid == rep,
                "identity_to_representative": identity,
                "coverage_to_representative": af,
            }
        )

    return pl.DataFrame(rows).select(CLUSTER_COLUMNS)


def cluster_leiden(
    edges: pl.DataFrame,
    seq_ids: list[str] | None = None,
    resolution: float = 1.0,
    weight_column: str = "identity",
    exclude_as_representatives: set[str] | None = None,
) -> pl.DataFrame:
    """Cluster sequences using the Leiden community detection algorithm.

    Requires the ``leidenalg`` and ``igraph`` packages. Edges are
    weighted by the chosen column (default: identity).

    Args:
        edges: Standard edge DataFrame (already filtered).
        seq_ids: Optional complete list of sequence IDs.
        resolution: Leiden resolution parameter. Higher values produce
            more, smaller clusters.
        weight_column: Edge column to use as weight.
        exclude_as_representatives: Optional set of sequence IDs that
            should not be chosen as cluster representatives.

    Returns:
        Polars DataFrame with standard cluster columns.
    """
    all_ids: set[str] = set()
    if seq_ids:
        all_ids.update(seq_ids)
    if not edges.is_empty():
        all_ids.update(edges["query_id"].to_list())
        all_ids.update(edges["target_id"].to_list())
    if not all_ids:
        return empty_cluster_frame()

    id_list = sorted(all_ids)
    id_to_idx = {name: idx for idx, name in enumerate(id_list)}

    graph = ig.Graph(n=len(id_list), directed=False)
    graph.vs["name"] = id_list

    edge_list: list[tuple[int, int]] = []
    weights: list[float] = []
    seen_pairs: set[tuple[int, int]] = set()
    if not edges.is_empty():
        compact_edges = edges.select("query_id", "target_id", weight_column)
        for qid, tid, weight in compact_edges.iter_rows():
            qi = id_to_idx[qid]
            ti = id_to_idx[tid]
            pair = (min(qi, ti), max(qi, ti))
            if pair in seen_pairs or qi == ti:
                continue
            seen_pairs.add(pair)
            edge_list.append(pair)
            weights.append(float(weight) if weight is not None else 1.0)

    if edge_list:
        graph.add_edges(edge_list)
        graph.es["weight"] = weights

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights if weights else None,
        resolution_parameter=resolution,
    )

    # Build edge lookup
    edge_lookup: dict[tuple[str, str], tuple[float, float]] = {}
    if not edges.is_empty():
        compact_edges = edges.select(
            "query_id", "target_id", "identity", "query_coverage", "target_coverage"
        )
        for qid, tid, ident, qcov, tcov in compact_edges.iter_rows():
            af = min(qcov, tcov)
            edge_lookup[(qid, tid)] = (ident, af)
            edge_lookup[(tid, qid)] = (ident, af)

    components: dict[int, list[str]] = {}
    for node_idx, community_id in enumerate(partition.membership):
        components.setdefault(community_id, []).append(id_list[node_idx])

    return build_cluster_assignment_rows(
        components, edge_lookup, exclude_as_representatives,
    )


#  Shared helpers 
def build_cluster_assignment_rows(
    components: dict[int, list[str]],
    edge_lookup: dict[tuple[str, str], tuple[float, float]],
    exclude_as_representatives: set[str] | None = None,
) -> pl.DataFrame:
    """Build the standard cluster assignment DataFrame from components.

    The representative of each component is chosen as the first member
    that is not in *exclude_as_representatives* (alphabetically, or by
    the ordering passed in). If all members are excluded, the first
    member is used anyway.

    Args:
        components: Mapping of component/community ID to list of member
            sequence IDs.
        edge_lookup: Mapping (seq_a, seq_b) -> (identity, alignment_fraction)
            for looking up stats to the representative.
        exclude_as_representatives: Optional set of sequence IDs that
            should not be chosen as cluster representatives.

    Returns:
        Polars DataFrame with standard cluster columns.
    """
    _excl = exclude_as_representatives or set()
    rows: list[dict[str, Any]] = []
    for cluster_idx, (_, members) in enumerate(sorted(components.items())):
        cid = f"cluster_{cluster_idx}"
        # Pick the first non-excluded member as representative;
        # fall back to the first member if all are excluded.
        rep = next((m for m in members if m not in _excl), members[0])
        for member in members:
            if member == rep:
                identity, af = 100.0, 100.0
            else:
                identity, af = edge_lookup.get((member, rep), (0.0, 0.0))
            rows.append(
                {
                    "seq_id": member,
                    "cluster_id": cid,
                    "representative_id": rep,
                    "is_representative": member == rep,
                    "identity_to_representative": identity,
                    "coverage_to_representative": af,
                }
            )
    if not rows:
        return empty_cluster_frame()
    return pl.DataFrame(rows).select(CLUSTER_COLUMNS)


def summarise_clusters(
    assignments: pl.DataFrame,
) -> pl.DataFrame:
    """Summarise cluster assignments into a per-cluster summary table.

    Returns a DataFrame with columns:
        cluster_id, representative_id, member_count, members
    """
    if assignments.is_empty():
        return pl.DataFrame(
            schema={
                "cluster_id": pl.String,
                "representative_id": pl.String,
                "member_count": pl.UInt32,
                "members": pl.String,
            }
        )
    summary = (
        assignments.group_by("cluster_id", maintain_order=True)
        .agg(
            pl.col("representative_id").first().alias("representative_id"),
            pl.col("seq_id").count().alias("member_count"),
            pl.col("seq_id")
            .sort_by("is_representative", descending=True)
            .str.concat(";")
            .alias("members"),
        )
        .sort("member_count", descending=True)
    )
    return summary
