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
from typing import Any, Sequence, Union

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


# ── Edge table helpers ──────────────────────────────────────────────


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


# ── Derived metrics ────────────────────────────────────────────────


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


# ── BLAST outfmt 6 ANI calculation (anicalc-style) ─────────────────


def merge_intervals(intervals: Sequence[Sequence[int]]) -> list[list[int]]:
    """Merge overlapping or adjacent intervals.

    Args:
        intervals: Sequence of (start, end) pairs (1-based, inclusive).
            Accepts both lists and tuples (e.g. from polars ``.rows()``).

    Returns:
        Non-overlapping merged intervals sorted by start.
    """
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    merged = [list(sorted_iv[0])]
    for start, stop in sorted_iv[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return merged


# Column names for the BLAST outfmt 6 (+ qlen slen) format
_BLAST6_COLUMNS = [
    "qname", "tname", "pident", "alen", "mismatch", "gapopen",
    "qstart", "qend", "sstart", "send", "evalue", "bitscore",
    "qlen", "tlen",
]

_BLAST6_DTYPES = {
    "qname": pl.String,
    "tname": pl.String,
    "pident": pl.Float64,
    "alen": pl.Float64,
    "mismatch": pl.Int64,
    "gapopen": pl.Int64,
    "qstart": pl.Int64,
    "qend": pl.Int64,
    "sstart": pl.Int64,
    "send": pl.Int64,
    "evalue": pl.Float64,
    "bitscore": pl.Float64,
    "qlen": pl.Int64,
    "tlen": pl.Int64,
}


def read_blast6(path: Union[str, Path]) -> pl.DataFrame:
    """Read a BLAST outfmt 6 file (with qlen slen) into a polars DataFrame.

    Handles plain text and gzip-compressed files.  Lines starting with
    '#' are skipped (comment_prefix).  Both tab and space delimiters
    are supported — tab is tried first.

    Returns:
        DataFrame with typed columns matching _BLAST6_COLUMNS.
    """
    path = Path(path)
    try:
        df = pl.read_csv(
            path,
            separator="\t",
            has_header=False,
            new_columns=_BLAST6_COLUMNS,
            schema_overrides=_BLAST6_DTYPES,
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
            new_columns=_BLAST6_COLUMNS,
            schema_overrides=_BLAST6_DTYPES,
            comment_prefix="#",
            truncate_ragged_lines=True,
            ignore_errors=True,
        )
    # Ensure coordinate columns are min/max normalised
    df = df.with_columns(
        pl.min_horizontal("qstart", "qend").alias("qstart"),
        pl.max_horizontal("qstart", "qend").alias("qend"),
        pl.min_horizontal("sstart", "send").alias("sstart"),
        pl.max_horizontal("sstart", "send").alias("send"),
    )
    return df


def prune_alignments(
    df: pl.DataFrame,
    min_length: int = 0,
    min_evalue: float = 1e-3,
) -> pl.DataFrame:
    """Remove short or high-evalue alignments and stop after full coverage.

    Following CheckV anicalc logic: discard alignments shorter than
    *min_length* or with evalue above *min_evalue*, and stop accumulating
    once the total aligned length reaches 110 pct of the query length.

    Operates on a polars DataFrame with at least: qstart, qend, evalue, qlen.
    Returns a filtered DataFrame (may include an extra ``aln_length`` column).
    """
    if df.is_empty():
        return df
    query_length = df["qlen"][0]

    # Compute alignment length and apply basic quality filters
    df = df.with_columns(
        (pl.col("qend") - pl.col("qstart") + 1).alias("aln_length")
    ).filter(
        (pl.col("aln_length") >= min_length) & (pl.col("evalue") <= min_evalue)
    )
    if df.is_empty():
        return df

    # Cumulative stop: keep only rows before the coverage cap.
    # cum_before is the running aligned length *before* the current row.
    df = df.with_columns(
        (pl.col("aln_length").cum_sum() - pl.col("aln_length")).alias("cum_before")
    ).filter(
        (pl.col("cum_before") < query_length)
        & (pl.col("aln_length") + pl.col("cum_before") < 1.10 * query_length)
    )
    return df


def compute_pair_ani(df: pl.DataFrame) -> float:
    """Compute average nucleotide identity for one query-target pair.

    Weighted by alignment length: sum(alen*pident) / sum(alen).
    Operates on a polars DataFrame with columns ``alen`` and ``pident``.
    """
    if df.is_empty():
        return 0.0
    result = df.select(
        (pl.col("alen") * pl.col("pident")).sum() / pl.col("alen").sum()
    )
    return round(result.item(), 2)


def compute_pair_coverages(
    df: pl.DataFrame,
) -> tuple[float, float]:
    """Compute query and target coverage for one query-target pair.

    Merges overlapping alignment coordinates before computing the
    fraction of each sequence covered.

    Operates on a polars DataFrame with columns ``qstart``, ``qend``,
    ``sstart``, ``send``, ``qlen``, ``tlen``.

    Returns:
        (query_coverage, target_coverage) as percentages (0-100).
    """
    if df.is_empty():
        return 0.0, 0.0

    qlen = df["qlen"][0]
    tlen = df["tlen"][0]

    query_coords = merge_intervals(df.select("qstart", "qend").rows())
    query_aligned = sum(stop - start + 1 for start, stop in query_coords)
    qcov = round(100.0 * query_aligned / qlen, 2) if qlen > 0 else 0.0

    target_coords = merge_intervals(df.select("sstart", "send").rows())
    target_aligned = sum(stop - start + 1 for start, stop in target_coords)
    tcov = round(100.0 * target_aligned / tlen, 2) if tlen > 0 else 0.0

    return qcov, tcov


def compute_blast6_pair_edges(
    group: pl.DataFrame,
    min_alignment_length: int,
    min_evalue: float,
) -> dict[str, Any] | None:
    """Compute ANI and coverage for one (query, target) group of HSPs.

    Applies pruning, ANI and coverage computation directly on the
    polars group sub-frame, then returns a single edge row dict or
    None if the pair should be skipped.
    """
    pruned = prune_alignments(group, min_alignment_length, min_evalue)
    if pruned.is_empty():
        return None
    ani = compute_pair_ani(pruned)
    qcov, tcov = compute_pair_coverages(pruned)
    return {
        "query_id": group["qname"][0],
        "target_id": group["tname"][0],
        "identity": ani,
        "query_coverage": qcov,
        "target_coverage": tcov,
        "num_alignments": pruned.height,
    }


def parse_blast6_to_edges(
    blast_path: Union[str, Path],
    min_alignment_length: int = 0,
    min_evalue: float = 1e-3,
) -> pl.DataFrame:
    """Parse BLAST outfmt 6 (with qlen slen) into the standard edge table.

    Uses polars to read the tabular file, then groups by (query, target)
    pair and applies the CheckV-style ANI/coverage computation per group
    (pruning, interval merging, weighted identity).

    Expected BLAST command:
        blastn -query in.fa -subject in.fa -outfmt '6 std qlen slen'

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

    # Drop self-hits early
    raw = raw.filter(pl.col("qname") != pl.col("tname"))
    if raw.is_empty():
        return empty_edge_frame()

    # Group by (query, target) and compute per-pair ANI/coverage
    rows: list[dict[str, Any]] = []
    for (_qname, _tname), group in raw.group_by(
        ["qname", "tname"], maintain_order=True
    ):
        result = compute_blast6_pair_edges(
            group, min_alignment_length, min_evalue
        )
        if result is not None:
            rows.append(result)

    if not rows:
        return empty_edge_frame()
    return pl.DataFrame(rows).select(EDGE_COLUMNS)


# ── CheckV-style ANI table parsing ─────────────────────────────────


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


# ── MMseqs2 easy-search output parsing ─────────────────────────────


# Column names for the MMseqs2 14-column output format
_MMSEQS_COLUMNS_14 = [
    "query_id", "target_id", "fident", "alnlen", "mismatch", "gapopen",
    "qstart", "qend", "tstart", "tend", "evalue", "bitscore",
    "qlen", "tlen",
]

_MMSEQS_COLUMNS_12 = [
    "query_id", "target_id", "fident", "alnlen", "mismatch", "gapopen",
    "qstart", "qend", "tstart", "tend", "evalue", "bitscore",
]


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

    col_names = _MMSEQS_COLUMNS_14 if has_lengths else _MMSEQS_COLUMNS_12
    # Rename only the columns we care about
    rename_map = {
        raw.columns[i]: col_names[i]
        for i in range(min(len(col_names), ncols))
    }
    raw = raw.rename(rename_map)

    # Cast numeric columns
    raw = raw.with_columns(
        pl.col("fident").cast(pl.Float64, strict=False),
        pl.col("alnlen").cast(pl.Int64, strict=False),
        pl.col("qstart").cast(pl.Int64, strict=False),
        pl.col("qend").cast(pl.Int64, strict=False),
        pl.col("tstart").cast(pl.Int64, strict=False),
        pl.col("tend").cast(pl.Int64, strict=False),
    )

    # Drop self-hits and rows with null fident (e.g. header lines)
    raw = raw.filter(
        pl.col("fident").is_not_null()
        & (pl.col("query_id") != pl.col("target_id"))
    )
    if raw.is_empty():
        return empty_edge_frame()

    # fident from mmseqs is 0-1 by default; convert to 0-100 if needed
    raw = raw.with_columns(
        pl.when(pl.col("fident") <= 1.0)
        .then(pl.col("fident") * 100.0)
        .otherwise(pl.col("fident"))
        .round(2)
        .alias("identity"),
    )

    # Compute per-row coverage from coordinates if qlen/tlen present
    if has_lengths:
        raw = raw.with_columns(
            pl.col("qlen").cast(pl.Float64, strict=False),
            pl.col("tlen").cast(pl.Float64, strict=False),
        )
        raw = raw.with_columns(
            pl.when(pl.col("qlen") > 0)
            .then(
                (100.0 * (pl.col("qend") - pl.col("qstart") + 1).abs()
                 / pl.col("qlen"))
                .round(2)
            )
            .otherwise(0.0)
            .alias("query_coverage"),
            pl.when(pl.col("tlen") > 0)
            .then(
                (100.0 * (pl.col("tend") - pl.col("tstart") + 1).abs()
                 / pl.col("tlen"))
                .round(2)
            )
            .otherwise(0.0)
            .alias("target_coverage"),
        )
    else:
        raw = raw.with_columns(
            pl.lit(0.0).alias("query_coverage"),
            pl.lit(0.0).alias("target_coverage"),
        )

    # Aggregate multiple HSPs per pair
    agg = (
        raw.group_by(["query_id", "target_id"])
        .agg(
            pl.col("identity")
            .mean()
            .round(2)
            .alias("identity"),
            pl.col("query_coverage")
            .sum()
            .clip(0, 100)
            .round(2)
            .alias("query_coverage"),
            pl.col("target_coverage")
            .sum()
            .clip(0, 100)
            .round(2)
            .alias("target_coverage"),
            pl.len().alias("num_alignments"),
        )
        .select(EDGE_COLUMNS)
    )
    return agg


# ── On-the-fly ANI calculation backends ────────────────────────────


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

    Note: FastANI reports only identity and the fraction of k-mer
    fragments that matched (matches/fragments).  It does not provide
    explicit target coverage, so ``target_coverage`` is set equal to
    ``query_coverage`` (symmetric for a one-way search) and the caller
    should keep this limitation in mind.

    Args:
        fasta_path: Path to input FASTA/FASTQ file.
        threads: Number of threads for the fragment-mapping step.
        fragment_length: Fragment length for query splitting.  FastANI
            default is 3000 (suited for prokaryote genomes). Lowered
            here to 1000 for shorter viral contigs.
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

    seq_rows = seq_df.to_dicts()
    names: list[str] = []
    sequences: list[str] = []
    for row in seq_rows:
        names.append(str(row.get("contig_id", row.get("header", ""))))
        sequences.append(str(row.get("sequence", "")))

    # Build sketch with adjusted parameters
    sketch = pyfastani.Sketch(
        k=k,
        fragment_length=fragment_length,
        minimum_fraction=minimum_fraction,
        percentage_identity=percentage_identity,
    )
    for name, seq in zip(names, sequences):
        sketch.add_genome(name, seq)

    mapper = sketch.index()

    if logger:
        logger.info(
            "pyfastani: indexed %d sequences (k=%d, frag_len=%d, min_frac=%.2f)",
            len(names), k, fragment_length, minimum_fraction,
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
                    "target_coverage": coverage,
                    "num_alignments": 1,
                }
            )

    if not rows:
        return empty_edge_frame()

    if logger:
        logger.info(
            "pyfastani produced %s raw edges from %s sequences",
            len(rows),
            len(seq_rows),
        )
    return pl.DataFrame(rows).select(EDGE_COLUMNS)


def compute_ani_blastn(
    fasta_path: Union[str, Path],
    threads: int = 1,
    min_alignment_length: int = 0,
    min_evalue: float = 1e-3,
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
                "-in": str(fasta_path),
                "-dbtype": "nucl",
                "-out": str(Path(tmpdir) / "blastdb"),
            },
            logger=logger,
            check_status=True,
        )
        # Run all-vs-all blastn
        run_command_comp(
            "blastn",
            positional_args=[],
            params={
                "-query": str(fasta_path),
                "-db": str(Path(tmpdir) / "blastdb"),
                "-outfmt": "6 std qlen slen",
                "-out": str(outfile),
                "-num_threads": str(threads),
                "-max_target_seqs": "25000",
                "-evalue": str(min_evalue),
            },
            logger=logger,
            check_status=True,
        )
        if not outfile.exists() or outfile.stat().st_size == 0:
            if logger:
                logger.warning("blastn produced no output")
            return empty_edge_frame()
        return parse_blast6_to_edges(outfile, min_alignment_length, min_evalue)


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
            positional_args_location="end",
            params={
                "--threads": str(threads),
                "-s": str(sensitivity),
                "--search-type": "3",  # nucleotide
                "--format-output": (
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


# ── K-mer-based identity estimation ───────────────────────────────


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


# ── Edge filtering ─────────────────────────────────────────────────


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


# ── Repetitive-sequence flagging ───────────────────────────────────


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


# ── Clustering algorithms ──────────────────────────────────────────


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
        for row in edges.select(["query_id", "target_id"]).iter_rows():
            qi, ti = id_to_idx[row[0]], id_to_idx[row[1]]
            union(qi, ti)

    # Build components
    components: dict[int, list[str]] = {}
    for idx, name in enumerate(id_list):
        root = find(idx)
        components.setdefault(root, []).append(name)

    # Build edge lookup for identity/coverage to representative
    edge_lookup: dict[tuple[str, str], tuple[float, float]] = {}
    if not edges.is_empty():
        for row in edges.iter_rows(named=True):
            key_fwd = (row["query_id"], row["target_id"])
            key_rev = (row["target_id"], row["query_id"])
            af = min(row["query_coverage"], row["target_coverage"])
            edge_lookup[key_fwd] = (row["identity"], af)
            edge_lookup[key_rev] = (row["identity"], af)

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
        for row in edges.iter_rows(named=True):
            q, t = row["query_id"], row["target_id"]
            af = min(row["query_coverage"], row["target_coverage"])
            # store bidirectional
            if t not in neighbours.get(q, {}):
                neighbours.setdefault(q, {})[t] = (row["identity"], af)
            if q not in neighbours.get(t, {}):
                neighbours.setdefault(t, {})[q] = (row["identity"], af)

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
        for row in edges.iter_rows(named=True):
            qi = id_to_idx[row["query_id"]]
            ti = id_to_idx[row["target_id"]]
            pair = (min(qi, ti), max(qi, ti))
            if pair in seen_pairs or qi == ti:
                continue
            seen_pairs.add(pair)
            edge_list.append(pair)
            weights.append(float(row.get(weight_column, 1.0)))

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
        for row in edges.iter_rows(named=True):
            af = min(row["query_coverage"], row["target_coverage"])
            edge_lookup[(row["query_id"], row["target_id"])] = (
                row["identity"],
                af,
            )
            edge_lookup[(row["target_id"], row["query_id"])] = (
                row["identity"],
                af,
            )

    components: dict[int, list[str]] = {}
    for node_idx, community_id in enumerate(partition.membership):
        components.setdefault(community_id, []).append(id_list[node_idx])

    return build_cluster_assignment_rows(
        components, edge_lookup, exclude_as_representatives,
    )


# ── Shared helpers ─────────────────────────────────────────────────


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
