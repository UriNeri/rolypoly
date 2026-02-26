from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any  # , Tuple

import polars as pl
import rich_click as click

from rolypoly.utils.logging.loggit import log_start_info, setup_logging
from rolypoly.utils.bio.polars_fastx import (
    count_kmers_df,
    frame_to_fastx,
    load_sequences,
)
from rolypoly.utils.bio.sequences import revcomp

# Ensure the FASTX plugins are registered
from rolypoly.utils.bio import polars_fastx as _polars_fastx  # noqa: F401


# Pileup / alignment constants

ANI_PILEUP_PREFILTER_K = 9
ANI_PILEUP_PREFILTER_WINDOW = 250
ANI_PILEUP_PREFILTER_MIN_SHARED = 2
ANI_PILEUP_PARASAIL_ALGORITHM = "ov"
ANI_PILEUP_PARASAIL_GAP_OPEN = 3
ANI_PILEUP_PARASAIL_GAP_EXTEND = 1
ANI_PILEUP_PARASAIL_MATCH = 5
ANI_PILEUP_PARASAIL_MISMATCH = 0

#  Process-pool worker globals (set once per worker via initializer)
ANI_WORKER_REPRESENTATIVE_MODE: str | None = None
ANI_WORKER_PILEUP_MIN_OVERLAP: int | None = None
ANI_WORKER_PILEUP_MIN_IDENTITY: float | None = None
ANI_WORKER_PREFIX_KMERS: dict[int, set[str]] | None = None
ANI_WORKER_SUFFIX_KMERS: dict[int, set[str]] | None = None
ANI_WORKER_ALIGNER_BACKEND: str | None = None
ANI_WORKER_THREADS: int | None = None
ANI_WORKER_PARASAIL_ALGORITHM: str | None = None
ANI_WORKER_PARASAIL_GAP_OPEN: int | None = None
ANI_WORKER_PARASAIL_GAP_EXTEND: int | None = None
ANI_WORKER_PARASAIL_MATCH: int | None = None
ANI_WORKER_PARASAIL_MISMATCH: int | None = None


def init_ani_cluster_pileup_worker(
    representative_mode: str,
    pileup_min_overlap: int,
    pileup_min_identity: float,
    prefix_kmers_by_idx: dict[int, set[str]],
    suffix_kmers_by_idx: dict[int, set[str]],
    pileup_aligner_backend: str,
    threads: int,
    parasail_algorithm: str,
    parasail_gap_open: int,
    parasail_gap_extend: int,
    parasail_match: int,
    parasail_mismatch: int,
) -> None:
    """Initialize global pileup settings for each process-pool worker.

    Args:
            representative_mode: Representative strategy used by workers.
            pileup_min_overlap: Minimum overlap length for merge acceptance.
            pileup_min_identity: Minimum overlap identity for merge acceptance.
            prefix_kmers_by_idx: Prefix terminal k-mers keyed by row order.
            suffix_kmers_by_idx: Suffix terminal k-mers keyed by row order.
            pileup_aligner_backend: Aligner backend (parasail or pyopal).
            threads: Worker thread/process count setting.
            parasail_algorithm: Parasail algorithm mode (ov or sw).
            parasail_gap_open: Gap-open penalty used for parasail.
            parasail_gap_extend: Gap-extension penalty used for parasail.
            parasail_match: Match score for parasail overlap matrix.
            parasail_mismatch: Mismatch score for parasail overlap matrix.
    """
    global ANI_WORKER_REPRESENTATIVE_MODE
    global ANI_WORKER_PILEUP_MIN_OVERLAP
    global ANI_WORKER_PILEUP_MIN_IDENTITY
    global ANI_WORKER_PREFIX_KMERS
    global ANI_WORKER_SUFFIX_KMERS
    global ANI_WORKER_ALIGNER_BACKEND
    global ANI_WORKER_THREADS
    global ANI_WORKER_PARASAIL_ALGORITHM
    global ANI_WORKER_PARASAIL_GAP_OPEN
    global ANI_WORKER_PARASAIL_GAP_EXTEND
    global ANI_WORKER_PARASAIL_MATCH
    global ANI_WORKER_PARASAIL_MISMATCH
    ANI_WORKER_REPRESENTATIVE_MODE = representative_mode
    ANI_WORKER_PILEUP_MIN_OVERLAP = pileup_min_overlap
    ANI_WORKER_PILEUP_MIN_IDENTITY = pileup_min_identity
    ANI_WORKER_PREFIX_KMERS = prefix_kmers_by_idx
    ANI_WORKER_SUFFIX_KMERS = suffix_kmers_by_idx
    ANI_WORKER_ALIGNER_BACKEND = pileup_aligner_backend
    ANI_WORKER_THREADS = threads
    ANI_WORKER_PARASAIL_ALGORITHM = parasail_algorithm
    ANI_WORKER_PARASAIL_GAP_OPEN = parasail_gap_open
    ANI_WORKER_PARASAIL_GAP_EXTEND = parasail_gap_extend
    ANI_WORKER_PARASAIL_MATCH = parasail_match
    ANI_WORKER_PARASAIL_MISMATCH = parasail_mismatch


@lru_cache(maxsize=4)
def get_parasail_matrix(match_score: int, mismatch_score: int):
    import parasail  # type: ignore[import-not-found]

    return parasail.matrix_create("ACGT", match_score, mismatch_score)


def terminal_overlap_identity(
    repr_overlap: str, ext_overlap: str, window: int = 30
) -> float:
    """Compute conservative edge identity for two overlap strings.

    The score is the minimum identity across the left and right windows,
    which penalizes weak overlap boundaries even when internal identity is high.

    Args:
            repr_overlap: Overlap bases from the representative sequence.
            ext_overlap: Overlap bases from the extender sequence.
            window: Number of bases to inspect on each overlap edge.

    Returns:
            float: Edge identity score in the 0-1 range.
    """
    overlap_span = min(len(repr_overlap), len(ext_overlap))
    if overlap_span == 0:
        return 0.0
    window_span = min(window, overlap_span)
    left_matches = sum(
        repr_overlap[idx] == ext_overlap[idx] for idx in range(window_span)
    )
    right_matches = sum(
        repr_overlap[overlap_span - window_span + idx]
        == ext_overlap[overlap_span - window_span + idx]
        for idx in range(window_span)
    )
    left_identity = left_matches / window_span
    right_identity = right_matches / window_span
    return min(left_identity, right_identity)


def _merge_pair_by_overlap(
    base: str,
    other: str,
    min_overlap: int,
    min_identity: float,
    aligner_backend: str = "parasail",
    threads: int = 1,
    parasail_algorithm: str = ANI_PILEUP_PARASAIL_ALGORITHM,
    parasail_gap_open: int = ANI_PILEUP_PARASAIL_GAP_OPEN,
    parasail_gap_extend: int = ANI_PILEUP_PARASAIL_GAP_EXTEND,
    parasail_match: int = ANI_PILEUP_PARASAIL_MATCH,
    parasail_mismatch: int = ANI_PILEUP_PARASAIL_MISMATCH,
) -> tuple[str, dict[str, Any] | None]:
    """Attempt to merge two sequences by terminal overlap in either orientation.

    The function checks containment first, then tries aligner-based overlap merges
    (using parasail or pyopal), and finally falls back to direct terminal scans.

    Args:
            base: Current representative sequence.
            other: Candidate sequence to merge into ``base``.
            min_overlap: Minimum overlap length required for a merge.
            min_identity: Minimum identity required for a merge.
            aligner_backend: Preferred aligner backend (parasail or pyopal).
            threads: Thread count for aligners that support threading.
            parasail_algorithm: Parasail mode, overlap (ov) or local (sw).
            parasail_gap_open: Parasail gap-open penalty.
            parasail_gap_extend: Parasail gap-extension penalty.
            parasail_match: Parasail match score.
            parasail_mismatch: Parasail mismatch score.

    Returns:
            Tuple of merged sequence and merge metadata. Metadata is ``None`` when no
            accepted merge is found.
    """
    if not base:
        return other, None
    if not other:
        return base, None
    if base in other:
        return (
            other,
            {
                "orientation": "containment_other_contains_repr",
                "repr_overlap": base,
                "ext_overlap": base,
                "repr_overlap_side": "all",
                "ext_overlap_side": "all",
                "overlap_len": len(base),
                "overlap_identity": 1.0,
            },
        )
    if other in base:
        return base, None

    def terminal_identity(
        repr_overlap: str, ext_overlap: str, window: int = 30
    ) -> float:
        overlap_span = min(len(repr_overlap), len(ext_overlap))
        if overlap_span == 0:
            return 0.0
        window_span = min(window, overlap_span)
        left_matches = sum(
            repr_overlap[idx] == ext_overlap[idx] for idx in range(window_span)
        )
        right_matches = sum(
            repr_overlap[overlap_span - window_span + idx]
            == ext_overlap[overlap_span - window_span + idx]
            for idx in range(window_span)
        )
        left_identity = left_matches / window_span
        right_identity = right_matches / window_span
        return min(left_identity, right_identity)

    def fallback_overlap_merge(
        candidate_other: str, ext_is_rc: bool
    ) -> tuple[str, dict[str, Any] | None]:
        max_overlap = min(len(base), len(candidate_other))
        for overlap in range(max_overlap, min_overlap - 1, -1):
            repr_overlap = base[-overlap:]
            ext_overlap = candidate_other[:overlap]
            mismatches = sum(
                repr_overlap[idx] != ext_overlap[idx] for idx in range(overlap)
            )
            identity = 1.0 - (mismatches / overlap)
            edge_identity = terminal_identity(repr_overlap, ext_overlap)
            if identity >= min_identity and edge_identity >= min_identity:
                return (
                    base + candidate_other[overlap:],
                    {
                        "orientation": "repr_suffix_vs_ext_prefix",
                        "repr_overlap": repr_overlap,
                        "ext_overlap": ext_overlap,
                        "repr_overlap_side": "suffix",
                        "ext_overlap_side": "prefix",
                        "overlap_len": overlap,
                        "overlap_identity": identity,
                        "overlap_edge_identity": edge_identity,
                        "ext_is_rc": ext_is_rc,
                        "ext_sequence_used": candidate_other,
                    },
                )
        for overlap in range(max_overlap, min_overlap - 1, -1):
            repr_overlap = base[:overlap]
            ext_overlap = candidate_other[-overlap:]
            mismatches = sum(
                repr_overlap[idx] != ext_overlap[idx] for idx in range(overlap)
            )
            identity = 1.0 - (mismatches / overlap)
            edge_identity = terminal_identity(repr_overlap, ext_overlap)
            if identity >= min_identity and edge_identity >= min_identity:
                return (
                    candidate_other + base[overlap:],
                    {
                        "orientation": "repr_prefix_vs_ext_suffix",
                        "repr_overlap": repr_overlap,
                        "ext_overlap": ext_overlap,
                        "repr_overlap_side": "prefix",
                        "ext_overlap_side": "suffix",
                        "overlap_len": overlap,
                        "overlap_identity": identity,
                        "overlap_edge_identity": edge_identity,
                        "ext_is_rc": ext_is_rc,
                        "ext_sequence_used": candidate_other,
                    },
                )
        return base, None

    def parasail_overlap_merge(
        candidate_other: str, ext_is_rc: bool
    ) -> tuple[str, dict[str, Any] | None]:
        try:
            import parasail  # type: ignore[import-not-found]
        except ImportError:
            return fallback_overlap_merge(candidate_other, ext_is_rc)

        try:
            if parasail_algorithm == "ov":
                matrix = get_parasail_matrix(parasail_match, parasail_mismatch)
                result = parasail.sg_trace_striped_16(
                    base,
                    candidate_other,
                    max(1, int(parasail_gap_open)),
                    max(0, int(parasail_gap_extend)),
                    matrix,
                )
            else:
                result = parasail.sw_trace_striped_16(
                    base,
                    candidate_other,
                    max(1, int(parasail_gap_open)),
                    max(0, int(parasail_gap_extend)),
                    parasail.nuc44,
                )
        except Exception:
            return fallback_overlap_merge(candidate_other, ext_is_rc)

        if result.score <= 0:
            return base, None

        beg_query = int(result.cigar.beg_query)
        beg_ref = int(result.cigar.beg_ref)
        end_query = int(result.end_query)
        end_ref = int(result.end_ref)
        tb_query = str(result.traceback.query)
        tb_ref = str(result.traceback.ref)
        tb_comp = str(result.traceback.comp)

        if "-" in tb_query or "-" in tb_ref:
            return fallback_overlap_merge(candidate_other, ext_is_rc)

        overlap_len = sum(
            1
            for left, right in zip(tb_query, tb_ref)
            if left != "-" and right != "-"
        )
        if overlap_len < min_overlap:
            return base, None

        matches = sum(1 for mark in tb_comp if mark == "|")
        identity = matches / overlap_len if overlap_len > 0 else 0.0
        if identity < min_identity:
            return base, None

        edge_identity = terminal_identity(
            base[beg_query : end_query + 1],
            candidate_other[beg_ref : end_ref + 1],
        )
        if edge_identity < min_identity:
            return base, None

        repr_overlap = base[beg_query : end_query + 1]
        ext_overlap = candidate_other[beg_ref : end_ref + 1]

        if end_query == len(base) - 1 and beg_ref == 0:
            merged = base + candidate_other[end_ref + 1 :]
            return (
                merged,
                {
                    "orientation": "repr_suffix_vs_ext_prefix",
                    "repr_overlap": repr_overlap,
                    "ext_overlap": ext_overlap,
                    "repr_overlap_side": "suffix",
                    "ext_overlap_side": "prefix",
                    "overlap_len": overlap_len,
                    "overlap_identity": identity,
                    "overlap_edge_identity": edge_identity,
                    "ext_is_rc": ext_is_rc,
                    "ext_sequence_used": candidate_other,
                    "aligner_backend": "parasail",
                    "parasail_algorithm": parasail_algorithm,
                },
            )

        if beg_query == 0 and end_ref == len(candidate_other) - 1:
            merged = candidate_other + base[end_query + 1 :]
            return (
                merged,
                {
                    "orientation": "repr_prefix_vs_ext_suffix",
                    "repr_overlap": repr_overlap,
                    "ext_overlap": ext_overlap,
                    "repr_overlap_side": "prefix",
                    "ext_overlap_side": "suffix",
                    "overlap_len": overlap_len,
                    "overlap_identity": identity,
                    "overlap_edge_identity": edge_identity,
                    "ext_is_rc": ext_is_rc,
                    "ext_sequence_used": candidate_other,
                    "aligner_backend": "parasail",
                    "parasail_algorithm": parasail_algorithm,
                },
            )

        return fallback_overlap_merge(candidate_other, ext_is_rc)

    def pyopal_overlap_merge(
        candidate_other: str, ext_is_rc: bool, threads: int = 1
    ) -> tuple[str, dict[str, Any] | None]:
        try:
            import pyopal  # type: ignore[import-not-found]
        except ImportError:
            return fallback_overlap_merge(candidate_other, ext_is_rc)

        try:
            result = next(
                iter(
                    pyopal.align(
                        base,
                        [candidate_other],
                        algorithm="ov",
                        mode="full",
                        threads=threads,
                        ordered=True,
                    )
                )
            )
        except Exception:
            return fallback_overlap_merge(candidate_other, ext_is_rc)

        q_start = int(result.query_start)
        q_end = int(result.query_end)
        t_start = int(result.target_start)
        t_end = int(result.target_end)
        query_len = int(result.query_length)
        target_len = int(result.target_length)

        overlap_len = min((q_end - q_start + 1), (t_end - t_start + 1))
        if overlap_len < min_overlap:
            return fallback_overlap_merge(candidate_other, ext_is_rc)

        repr_overlap = base[q_start : q_start + overlap_len]
        ext_overlap = candidate_other[t_start : t_start + overlap_len]
        identity = float(result.identity())
        edge_identity = terminal_identity(repr_overlap, ext_overlap)
        if identity < min_identity or edge_identity < min_identity:
            return fallback_overlap_merge(candidate_other, ext_is_rc)

        tolerance = 3
        if q_end >= query_len - 1 - tolerance and t_start <= tolerance:
            merged = base + candidate_other[t_end + 1 :]
            if len(merged) <= len(base):
                return fallback_overlap_merge(candidate_other, ext_is_rc)
            return (
                merged,
                {
                    "orientation": "repr_suffix_vs_ext_prefix",
                    "repr_overlap": repr_overlap,
                    "ext_overlap": ext_overlap,
                    "repr_overlap_side": "suffix",
                    "ext_overlap_side": "prefix",
                    "overlap_len": overlap_len,
                    "overlap_identity": identity,
                    "overlap_edge_identity": edge_identity,
                    "ext_is_rc": ext_is_rc,
                    "ext_sequence_used": candidate_other,
                    "aligner_backend": "pyopal",
                },
            )

        if q_start <= tolerance and t_end >= target_len - 1 - tolerance:
            merged = candidate_other[:t_start] + base
            if len(merged) <= len(base):
                return fallback_overlap_merge(candidate_other, ext_is_rc)
            return (
                merged,
                {
                    "orientation": "repr_prefix_vs_ext_suffix",
                    "repr_overlap": repr_overlap,
                    "ext_overlap": ext_overlap,
                    "repr_overlap_side": "prefix",
                    "ext_overlap_side": "suffix",
                    "overlap_len": overlap_len,
                    "overlap_identity": identity,
                    "overlap_edge_identity": edge_identity,
                    "ext_is_rc": ext_is_rc,
                    "ext_sequence_used": candidate_other,
                    "aligner_backend": "pyopal",
                },
            )

        return fallback_overlap_merge(candidate_other, ext_is_rc)

    best_merged = base
    best_meta: dict[str, Any] | None = None
    aligner_backend = aligner_backend.lower()
    if aligner_backend not in {"parasail", "pyopal"}:
        aligner_backend = "parasail"
    for candidate_other, ext_is_rc in ((other, False), (revcomp(other), True)):
        if aligner_backend == "pyopal":
            merged_candidate, meta_candidate = pyopal_overlap_merge(
                candidate_other, ext_is_rc, threads=threads
            )
        else:
            merged_candidate, meta_candidate = parasail_overlap_merge(
                candidate_other, ext_is_rc
            )
        if meta_candidate is None:
            continue
        if best_meta is None:
            best_merged = merged_candidate
            best_meta = meta_candidate
            continue
        if len(merged_candidate) > len(best_merged):
            best_merged = merged_candidate
            best_meta = meta_candidate
            continue
        if len(merged_candidate) == len(best_merged):
            if float(meta_candidate.get("overlap_identity", 0.0)) > float(
                best_meta.get("overlap_identity", 0.0)
            ):
                best_merged = merged_candidate
                best_meta = meta_candidate

    if best_meta is not None:
        return best_merged, best_meta

    return base, None


#  Alignment formatting helpers
def format_overlap_alignment(repr_overlap: str, ext_overlap: str) -> str:
    """Return positional overlap alignment glyphs ('|' match, '.' mismatch)."""
    if not repr_overlap or not ext_overlap:
        return ""
    overlap_span = min(len(repr_overlap), len(ext_overlap))
    return "".join(
        "|" if left == right else "."
        for left, right in zip(
            repr_overlap[:overlap_span], ext_overlap[:overlap_span]
        )
    )


def compact_alignment_block(
    repr_overlap: str, align_line: str, ext_overlap: str, max_bases: int = 50
) -> tuple[str, str, str]:
    if len(repr_overlap) <= max_bases:
        return repr_overlap, align_line, ext_overlap
    # Use a simple left window (no ellipsis) so the output stays compact and readable.
    return (
        repr_overlap[:max_bases],
        align_line[:max_bases],
        ext_overlap[:max_bases],
    )


def build_step_alignment_view(
    representative_before: str,
    extender_sequence: str,
    repr_overlap: str,
    align_line: str,
    ext_overlap: str,
    repr_side: str,
    ext_side: str,
    flank_size: int = 10,
) -> tuple[str, str, str, str, str]:
    repr_overlap_view, align_view, ext_overlap_view = compact_alignment_block(
        repr_overlap, align_line, ext_overlap
    )

    left_flank_len = flank_size
    right_flank_len = flank_size

    if repr_side == "suffix":
        rep_left = representative_before[
            max(
                0, len(representative_before) - len(repr_overlap) - flank_size
            ) : max(0, len(representative_before) - len(repr_overlap))
        ]
        rep_right = ""
        rep_label = f"rep(-{flank_size}+ovl)"
    else:
        rep_left = ""
        rep_right = representative_before[
            len(repr_overlap) : len(repr_overlap) + flank_size
        ]
        rep_label = f"rep(ovl+{flank_size})"

    if ext_side == "prefix":
        ext_left = ""
        ext_right = extender_sequence[
            len(ext_overlap) : len(ext_overlap) + flank_size
        ]
        ext_label = f"ext(ovl+{flank_size})"
    else:
        ext_left = extender_sequence[
            max(
                0, len(extender_sequence) - len(ext_overlap) - flank_size
            ) : max(0, len(extender_sequence) - len(ext_overlap))
        ]
        ext_right = ""
        ext_label = f"ext(-{flank_size}+ovl)"

    rep_display = (
        f"{rep_left.rjust(left_flank_len)}"
        f"{repr_overlap_view}"
        f"{rep_right.ljust(right_flank_len)}"
    )
    align_display = f"{' ' * left_flank_len}{align_view}{' ' * right_flank_len}"
    ext_display = (
        f"{ext_left.rjust(left_flank_len)}"
        f"{ext_overlap_view}"
        f"{ext_right.ljust(right_flank_len)}"
    )

    return rep_label, rep_display, align_display, ext_label, ext_display


# Terminal k-mer helpers ------


def terminal_kmer_set(
    sequence: str, side: str, k: int, window: int
) -> set[str]:
    if k <= 0 or window <= 0 or len(sequence) < k:
        return set()
    region_span = min(window, len(sequence))
    region = (
        sequence[:region_span] if side == "prefix" else sequence[-region_span:]
    )
    if len(region) < k:
        return set()
    return {region[idx : idx + k] for idx in range(0, len(region) - k + 1)}


def extract_kmer_symbols(kmer_counts: list[dict[str, Any]] | None) -> set[str]:
    if not kmer_counts:
        return set()
    kmers: set[str] = set()
    for entry in kmer_counts:
        if not isinstance(entry, dict):
            continue
        if "substrings" in entry and entry["substrings"]:
            kmers.add(str(entry["substrings"]))
            continue
        for key, value in entry.items():
            if key in {"count", "counts", "proportion"}:
                continue
            if value is not None:
                kmers.add(str(value))
                break
    return kmers


def compute_terminal_kmer_maps(
    seq_rows: list[dict[str, Any]], logger=None
) -> tuple[dict[int, set[str]], dict[int, set[str]]]:
    """Precompute prefix/suffix terminal k-mer sets for overlap prefiltering.

    Args:
            seq_rows: Sequence rows containing ``_order`` and ``sequence``.
            logger: Optional logger for stage-level timing diagnostics.

    Returns:
            Tuple of dictionaries keyed by ``_order`` index:
            (prefix_kmers_by_idx, suffix_kmers_by_idx).
    """
    if not seq_rows:
        return {}, {}
    start_time = perf_counter()

    seq_df_start = perf_counter()
    seq_df = pl.DataFrame(seq_rows).select(
        pl.col("_order").cast(pl.Int64),
        pl.col("sequence").cast(pl.String).str.to_uppercase().alias("sequence"),
    )
    seq_df_seconds = perf_counter() - seq_df_start

    window_start = perf_counter()
    windowed_df = seq_df.with_columns(
        pl.col("sequence")
        .str.slice(0, ANI_PILEUP_PREFILTER_WINDOW)
        .alias("prefix_window"),
        pl.col("sequence")
        .str.slice(-ANI_PILEUP_PREFILTER_WINDOW, ANI_PILEUP_PREFILTER_WINDOW)
        .alias("suffix_window"),
    )
    window_seconds = perf_counter() - window_start

    terminal_start = perf_counter()
    terminals_df = pl.concat(
        [
            windowed_df.select(
                pl.col("_order"),
                pl.lit("prefix").alias("side"),
                pl.col("prefix_window").alias("terminal_seq"),
            ),
            windowed_df.select(
                pl.col("_order"),
                pl.lit("suffix").alias("side"),
                pl.col("suffix_window").alias("terminal_seq"),
            ),
        ],
        how="vertical_relaxed",
    ).with_columns(
        pl.concat_str(
            [pl.col("side"), pl.lit("_"), pl.col("_order").cast(pl.String)]
        ).alias("terminal_id")
    )
    terminal_seconds = perf_counter() - terminal_start

    kmer_start = perf_counter()
    kmer_df = count_kmers_df(
        terminals_df,
        seq_col="terminal_seq",
        id_col="terminal_id",
        k=ANI_PILEUP_PREFILTER_K,
        relative=False,
    )
    kmer_seconds = perf_counter() - kmer_start

    map_start = perf_counter()
    prefix_kmers_by_idx: dict[int, set[str]] = {}
    suffix_kmers_by_idx: dict[int, set[str]] = {}
    for row in kmer_df.select("_order", "side", "kmer_counts").to_dicts():
        idx = int(row["_order"])
        side = str(row["side"])
        kmers = extract_kmer_symbols(row.get("kmer_counts"))
        if side == "prefix":
            prefix_kmers_by_idx[idx] = kmers
        else:
            suffix_kmers_by_idx[idx] = kmers

    for row in seq_rows:
        idx = int(row["_order"])
        prefix_kmers_by_idx.setdefault(idx, set())
        suffix_kmers_by_idx.setdefault(idx, set())
    map_seconds = perf_counter() - map_start
    total_seconds = perf_counter() - start_time

    if logger is not None:
        logger.debug(
            "ANI pileup kmer stages: seqdf_s=%.3f window_s=%.3f terminals_s=%.3f kmer_count_s=%.3f map_extract_s=%.3f total_s=%.3f rows=%s terminals=%s k=%s window=%s",
            seq_df_seconds,
            window_seconds,
            terminal_seconds,
            kmer_seconds,
            map_seconds,
            total_seconds,
            len(seq_rows),
            terminals_df.height,
            ANI_PILEUP_PREFILTER_K,
            ANI_PILEUP_PREFILTER_WINDOW,
        )

    return prefix_kmers_by_idx, suffix_kmers_by_idx


def build_nascent_chain_label(
    seed_contig_id: str, extension_steps: list[dict[str, Any]]
) -> str:
    """Build a directional chain label from seed and extension steps.

    Args:
            seed_contig_id: Initial seed contig identifier.
            extension_steps: Ordered extension step payloads with placement fields.

    Returns:
            str: Arrow-joined chain label reflecting left/right insertions.
    """
    chain_tokens: list[str] = [seed_contig_id]
    for step in extension_steps:
        contig_id = str(step.get("with_contig_id", ""))
        if not contig_id:
            continue
        placement = str(step.get("placement", "right"))
        if placement == "left":
            chain_tokens.insert(0, contig_id)
        else:
            chain_tokens.append(contig_id)
    return "->".join(chain_tokens)


#  Build pileup sequence


def build_pileup_sequence(
    cluster_rows: list[dict[str, Any]],
    min_overlap: int,
    min_identity: float,
    prefix_kmers_by_idx: dict[int, set[str]],
    suffix_kmers_by_idx: dict[int, set[str]],
    aligner_backend: str,
    threads: int = 1,
    parasail_algorithm: str = ANI_PILEUP_PARASAIL_ALGORITHM,
    parasail_gap_open: int = ANI_PILEUP_PARASAIL_GAP_OPEN,
    parasail_gap_extend: int = ANI_PILEUP_PARASAIL_GAP_EXTEND,
    parasail_match: int = ANI_PILEUP_PARASAIL_MATCH,
    parasail_mismatch: int = ANI_PILEUP_PARASAIL_MISMATCH,
) -> tuple[str, set[str], list[dict[str, Any]], dict[str, Any]]:
    """Construct an extended representative by graph-guided pileup within a cluster.

    Workflow:
    1) Build candidate overlap edges after terminal k-mer prefiltering.
    2) Assemble chains from possible seeds and keep the best merged sequence.
    3) Run rescue merges for remaining contributors not captured in chain traversal.

    Args:
            cluster_rows: Rows for a single ANI cluster.
            min_overlap: Minimum overlap length required for merges.
            min_identity: Minimum identity required for merges.
            prefix_kmers_by_idx: Prefix k-mer sets keyed by ``_order``.
            suffix_kmers_by_idx: Suffix k-mer sets keyed by ``_order``.
            aligner_backend: Pairwise backend for overlap checks.
            threads: Thread count for alignment backends.
            parasail_algorithm: Parasail mode, overlap (ov) or local (sw).
            parasail_gap_open: Parasail gap-open penalty.
            parasail_gap_extend: Parasail gap-extension penalty.
            parasail_match: Parasail match score.
            parasail_mismatch: Parasail mismatch score.

    Returns:
            Tuple containing merged sequence, contributor contig IDs, extension step
            records, and a metrics dictionary for logging/diagnostics.
    """
    start_time = perf_counter()
    if not cluster_rows:
        return (
            "",
            set(),
            [],
            {
                "cluster_size": 0,
                "candidate_pairs": 0,
                "prefilter_passed": 0,
                "prefilter_skipped": 0,
                "edges_built": 0,
                "pyopal_batch_calls": 0,
                "pyopal_targets_total": 0,
                "edge_build_seconds": 0.0,
                "chain_seconds": 0.0,
                "total_seconds": 0.0,
            },
        )

    seed = max(
        cluster_rows,
        key=lambda row: (len(str(row["sequence"])), -int(row["_order"])),
    )
    seed_contig_id = str(seed["contig_id"])
    seed_idx = int(seed["_order"])
    row_by_idx = {int(row["_order"]): row for row in cluster_rows}
    pyopal_module = None
    rc_cache_by_idx: dict[int, str] = {}
    if aligner_backend == "pyopal":
        try:
            import pyopal as _pyopal  # type: ignore[import-not-found]

            pyopal_module = _pyopal
        except ImportError:
            pyopal_module = None
        for row in cluster_rows:
            row_idx = int(row["_order"])
            rc_cache_by_idx[row_idx] = revcomp(str(row["sequence"]))

    edge_build_start = perf_counter()
    candidate_pairs = 0
    prefilter_passed = 0
    prefilter_skipped = 0
    pyopal_batch_calls = 0
    pyopal_targets_total = 0

    edges: list[dict[str, Any]] = []
    for source in cluster_rows:
        source_idx = int(source["_order"])
        source_seq = str(source["sequence"])
        candidate_targets: list[dict[str, Any]] = []
        for target in cluster_rows:
            target_idx = int(target["_order"])
            if source_idx == target_idx:
                continue
            candidate_pairs += 1
            if (
                len(
                    suffix_kmers_by_idx.get(source_idx, set())
                    & prefix_kmers_by_idx.get(target_idx, set())
                )
                < ANI_PILEUP_PREFILTER_MIN_SHARED
            ):
                prefilter_skipped += 1
                continue
            prefilter_passed += 1
            candidate_targets.append(
                {
                    "target_idx": target_idx,
                    "target_contig_id": str(target["contig_id"]),
                    "target_sequence": str(target["sequence"]),
                }
            )

        if (
            pyopal_module is not None
            and aligner_backend == "pyopal"
            and candidate_targets
        ):
            pyopal_batch_calls += 1
            pyopal_targets_total += len(candidate_targets)
            pyopal_entries: list[dict[str, Any]] = []
            for payload in candidate_targets:
                target_idx = int(payload["target_idx"])
                target_seq = str(payload["target_sequence"])
                pyopal_entries.append(
                    {
                        "target_idx": target_idx,
                        "target_contig_id": str(payload["target_contig_id"]),
                        "ext_sequence_used": target_seq,
                        "ext_is_rc": False,
                    }
                )
                pyopal_entries.append(
                    {
                        "target_idx": target_idx,
                        "target_contig_id": str(payload["target_contig_id"]),
                        "ext_sequence_used": rc_cache_by_idx.get(
                            target_idx, revcomp(target_seq)
                        ),
                        "ext_is_rc": True,
                    }
                )

            try:
                results = list(
                    pyopal_module.align(
                        source_seq,
                        [
                            entry["ext_sequence_used"]
                            for entry in pyopal_entries
                        ],
                        algorithm="ov",
                        mode="full",
                        threads=max(1, int(threads)),
                        ordered=True,
                    )
                )
            except Exception:
                results = []

            if len(results) == len(pyopal_entries):
                for entry, result in zip(pyopal_entries, results):
                    q_start = int(result.query_start)
                    q_end = int(result.query_end)
                    t_start = int(result.target_start)
                    t_end = int(result.target_end)
                    query_len = int(result.query_length)
                    overlap_len = min(
                        (q_end - q_start + 1), (t_end - t_start + 1)
                    )
                    if overlap_len < min_overlap:
                        continue

                    repr_overlap = source_seq[q_start : q_start + overlap_len]
                    ext_overlap = str(entry["ext_sequence_used"])[
                        t_start : t_start + overlap_len
                    ]
                    identity = float(result.identity())
                    edge_identity = terminal_overlap_identity(
                        repr_overlap, ext_overlap
                    )
                    if identity < min_identity or edge_identity < min_identity:
                        continue

                    tolerance = 3
                    if not (
                        q_end >= query_len - 1 - tolerance
                        and t_start <= tolerance
                    ):
                        continue

                    merged_candidate = (
                        source_seq
                        + str(entry["ext_sequence_used"])[t_end + 1 :]
                    )
                    added_bp = len(merged_candidate) - len(source_seq)
                    if added_bp <= 0:
                        continue

                    meta = {
                        "orientation": "repr_suffix_vs_ext_prefix",
                        "repr_overlap": repr_overlap,
                        "ext_overlap": ext_overlap,
                        "repr_overlap_side": "suffix",
                        "ext_overlap_side": "prefix",
                        "overlap_len": overlap_len,
                        "overlap_identity": identity,
                        "overlap_edge_identity": edge_identity,
                        "ext_is_rc": bool(entry["ext_is_rc"]),
                        "ext_sequence_used": str(entry["ext_sequence_used"]),
                        "aligner_backend": "pyopal",
                    }

                    edges.append(
                        {
                            "source_idx": source_idx,
                            "target_idx": int(entry["target_idx"]),
                            "target_contig_id": str(entry["target_contig_id"]),
                            "target_sequence": str(entry["ext_sequence_used"]),
                            "added_bp": added_bp,
                            "overlap_len": overlap_len,
                            "overlap_identity": identity,
                            "overlap_edge_identity": edge_identity,
                            "merge_meta": meta,
                        }
                    )
                continue

        for payload in candidate_targets:
            target_idx = int(payload["target_idx"])
            target_seq = str(payload["target_sequence"])
            merged_candidate, meta = _merge_pair_by_overlap(
                source_seq,
                target_seq,
                min_overlap,
                min_identity,
                aligner_backend,
                threads=threads,
                parasail_algorithm=parasail_algorithm,
                parasail_gap_open=parasail_gap_open,
                parasail_gap_extend=parasail_gap_extend,
                parasail_match=parasail_match,
                parasail_mismatch=parasail_mismatch,
            )
            if not meta:
                continue
            if str(meta.get("orientation", "")) != "repr_suffix_vs_ext_prefix":
                continue
            added_bp = len(merged_candidate) - len(source_seq)
            if added_bp <= 0:
                continue
            edges.append(
                {
                    "source_idx": source_idx,
                    "target_idx": target_idx,
                    "target_contig_id": str(payload["target_contig_id"]),
                    "target_sequence": str(
                        meta.get("ext_sequence_used", target_seq)
                    ),
                    "added_bp": added_bp,
                    "overlap_len": int(meta.get("overlap_len", 0)),
                    "overlap_identity": float(
                        meta.get("overlap_identity", 0.0)
                    ),
                    "overlap_edge_identity": float(
                        meta.get(
                            "overlap_edge_identity",
                            meta.get("overlap_identity", 0.0),
                        )
                    ),
                    "merge_meta": meta,
                }
            )
    edge_build_seconds = perf_counter() - edge_build_start

    outgoing: dict[int, list[dict[str, Any]]] = {}
    incoming: dict[int, list[dict[str, Any]]] = {}
    for edge in edges:
        outgoing.setdefault(int(edge["source_idx"]), []).append(edge)
        incoming.setdefault(int(edge["target_idx"]), []).append(edge)

    def edge_score(edge: dict[str, Any]) -> tuple[float, float, int, int]:
        return (
            float(edge["overlap_edge_identity"]),
            float(edge["overlap_identity"]),
            int(edge["added_bp"]),
            int(edge["overlap_len"]),
        )

    for key in outgoing:
        outgoing[key].sort(key=edge_score, reverse=True)
    for key in incoming:
        incoming[key].sort(key=edge_score, reverse=True)

    def assemble_chain_from_seed(
        local_seed_idx: int,
    ) -> tuple[str, set[str], list[dict[str, Any]], str]:
        used: set[int] = {local_seed_idx}
        right_steps: list[dict[str, Any]] = []
        left_steps: list[dict[str, Any]] = []
        right_idx = local_seed_idx
        left_idx = local_seed_idx

        while True:
            progress = False

            for edge in outgoing.get(right_idx, []):
                next_idx = int(edge["target_idx"])
                if next_idx in used:
                    continue
                right_steps.append(edge)
                used.add(next_idx)
                right_idx = next_idx
                progress = True
                break

            for edge in incoming.get(left_idx, []):
                prev_idx = int(edge["source_idx"])
                if prev_idx in used:
                    continue
                left_steps.append(edge)
                used.add(prev_idx)
                left_idx = prev_idx
                progress = True
                break

            if not progress:
                break

        seed_row = row_by_idx[local_seed_idx]
        seed_label = str(seed_row["contig_id"])
        merged_local = str(seed_row["sequence"])
        contributor_ids_local: set[str] = {seed_label}
        extension_steps_local: list[dict[str, Any]] = []

        for edge in left_steps:
            base_before = merged_local
            overlap_len = int(edge["overlap_len"])
            ext_seq = str(row_by_idx[int(edge["source_idx"])]["sequence"])
            merged_local = ext_seq + merged_local[overlap_len:]
            contributor_ids_local.add(
                str(row_by_idx[int(edge["source_idx"])]["contig_id"])
            )
            extension_steps_local.append(
                {
                    "with_contig_id": str(
                        row_by_idx[int(edge["source_idx"])]["contig_id"]
                    ),
                    "placement": "left",
                    "added_bp": len(merged_local) - len(base_before),
                    "new_length": len(merged_local),
                    "representative_before": base_before,
                    "extender_sequence": ext_seq,
                    "merged_sequence": merged_local,
                    "merge_meta": dict(edge["merge_meta"]),
                }
            )

        for edge in right_steps:
            base_before = merged_local
            overlap_len = int(edge["overlap_len"])
            ext_seq = str(edge["target_sequence"])
            merged_local = merged_local + ext_seq[overlap_len:]
            contributor_ids_local.add(str(edge["target_contig_id"]))
            extension_steps_local.append(
                {
                    "with_contig_id": str(edge["target_contig_id"]),
                    "placement": "right",
                    "added_bp": len(merged_local) - len(base_before),
                    "new_length": len(merged_local),
                    "representative_before": base_before,
                    "extender_sequence": ext_seq,
                    "merged_sequence": merged_local,
                    "merge_meta": dict(edge["merge_meta"]),
                }
            )

        nascent_chain_local = build_nascent_chain_label(
            seed_label, extension_steps_local
        )
        return (
            merged_local,
            contributor_ids_local,
            extension_steps_local,
            nascent_chain_local,
        )

    chain_start = perf_counter()
    best_merged = str(seed["sequence"])
    best_contributors: set[str] = {seed_contig_id}
    best_steps: list[dict[str, Any]] = []
    best_nascent_chain = seed_contig_id
    if edges:
        seed_candidates = {seed_idx}
        seed_candidates.update(int(edge["source_idx"]) for edge in edges)
        seed_candidates.update(int(edge["target_idx"]) for edge in edges)

        for candidate_seed_idx in sorted(seed_candidates):
            (
                merged_candidate,
                contributors_candidate,
                steps_candidate,
                nascent_chain_candidate,
            ) = assemble_chain_from_seed(candidate_seed_idx)
            if len(merged_candidate) > len(best_merged):
                best_merged = merged_candidate
                best_contributors = contributors_candidate
                best_steps = steps_candidate
                best_nascent_chain = nascent_chain_candidate
                continue
            if len(merged_candidate) == len(best_merged) and len(
                contributors_candidate
            ) > len(best_contributors):
                best_merged = merged_candidate
                best_contributors = contributors_candidate
                best_steps = steps_candidate
                best_nascent_chain = nascent_chain_candidate

    chain_seconds = perf_counter() - chain_start

    rescue_start = perf_counter()
    rescue_steps = 0
    while True:
        best_rescue: dict[str, Any] | None = None
        for row in cluster_rows:
            candidate_id = str(row["contig_id"])
            if candidate_id in best_contributors:
                continue
            candidate_sequence = str(row["sequence"])
            merged_candidate, meta = _merge_pair_by_overlap(
                best_merged,
                candidate_sequence,
                min_overlap,
                min_identity,
                aligner_backend,
                parasail_algorithm=parasail_algorithm,
                parasail_gap_open=parasail_gap_open,
                parasail_gap_extend=parasail_gap_extend,
                parasail_match=parasail_match,
                parasail_mismatch=parasail_mismatch,
            )
            if not meta:
                continue
            added_bp = len(merged_candidate) - len(best_merged)
            if added_bp <= 0:
                continue
            candidate_payload = {
                "contig_id": candidate_id,
                "merged": merged_candidate,
                "added_bp": added_bp,
                "meta": dict(meta),
            }
            if best_rescue is None:
                best_rescue = candidate_payload
                continue
            if int(candidate_payload["added_bp"]) > int(
                best_rescue["added_bp"]
            ):
                best_rescue = candidate_payload
                continue
            if int(candidate_payload["added_bp"]) == int(
                best_rescue["added_bp"]
            ):
                if float(
                    candidate_payload["meta"].get("overlap_identity", 0.0)
                ) > float(best_rescue["meta"].get("overlap_identity", 0.0)):
                    best_rescue = candidate_payload

        if best_rescue is None:
            break

        base_before = best_merged
        best_merged = str(best_rescue["merged"])
        best_contributors.add(str(best_rescue["contig_id"]))
        meta = dict(best_rescue["meta"])
        orientation = str(meta.get("orientation", "repr_suffix_vs_ext_prefix"))
        placement = (
            "right" if orientation == "repr_suffix_vs_ext_prefix" else "left"
        )
        best_steps.append(
            {
                "with_contig_id": str(best_rescue["contig_id"]),
                "placement": placement,
                "added_bp": len(best_merged) - len(base_before),
                "new_length": len(best_merged),
                "representative_before": base_before,
                "extender_sequence": str(meta.get("ext_sequence_used", "")),
                "merged_sequence": best_merged,
                "merge_meta": meta,
            }
        )
        rescue_steps += 1

    rescue_seconds = perf_counter() - rescue_start
    seed_label = (
        best_nascent_chain.split("->", 1)[0]
        if best_nascent_chain
        else seed_contig_id
    )
    best_nascent_chain = build_nascent_chain_label(seed_label, best_steps)
    total_seconds = perf_counter() - start_time

    return (
        best_merged,
        best_contributors,
        best_steps,
        {
            "cluster_size": len(cluster_rows),
            "candidate_pairs": candidate_pairs,
            "prefilter_passed": prefilter_passed,
            "prefilter_skipped": prefilter_skipped,
            "edges_built": len(edges),
            "pyopal_batch_calls": pyopal_batch_calls,
            "pyopal_targets_total": pyopal_targets_total,
            "edge_build_seconds": edge_build_seconds,
            "chain_seconds": chain_seconds,
            "rescue_steps": rescue_steps,
            "rescue_seconds": rescue_seconds,
            "total_seconds": total_seconds,
            "nascent_chain": best_nascent_chain,
        },
    )


# ANI clustering --------------


def cluster_contigs_by_ani(
    seq_rows: list[dict[str, Any]], min_identity: float, min_af: float, logger
) -> list[list[dict[str, Any]]]:
    """Cluster contigs by ANI using pyskani, returning groups of row dicts."""
    if len(seq_rows) < 2:
        return [[row] for row in seq_rows]

    try:
        import pyskani  # type: ignore[import-not-found]
    except ImportError as exc:
        raise click.ClickException(
            "ANI clustering requires pyskani. Install it or disable the ANI step."
        ) from exc

    for idx, row in enumerate(seq_rows):
        row["_ani_name"] = f"seq_{idx}"
    id_to_idx = {str(row["_ani_name"]): idx for idx, row in enumerate(seq_rows)}
    parent = list(range(len(seq_rows)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    try:
        database = pyskani.Database()
        for row in seq_rows:
            database.sketch(
                str(row["_ani_name"]), row["sequence"].encode("ascii")
            )
        for row in seq_rows:
            query_name = str(row["_ani_name"])
            hits = database.query(
                query_name,
                row["sequence"].encode("ascii"),
                cutoff=max(0.0, min_identity - 0.05),
            )
            for hit in hits:
                reference_name = str(getattr(hit, "reference_name", ""))
                if (
                    reference_name == query_name
                    or reference_name not in id_to_idx
                ):
                    continue
                identity = float(getattr(hit, "identity", 0.0))
                query_fraction = float(getattr(hit, "query_fraction", 0.0))
                reference_fraction = float(
                    getattr(hit, "reference_fraction", 0.0)
                )
                if (
                    identity >= min_identity
                    and min(query_fraction, reference_fraction) >= min_af
                ):
                    union(id_to_idx[query_name], id_to_idx[reference_name])
    except Exception as exc:
        raise click.ClickException(
            f"ANI clustering failed while querying pyskani: {exc}"
        ) from exc

    components: dict[int, list[dict[str, Any]]] = {}
    for idx, row in enumerate(seq_rows):
        root = find(idx)
        components.setdefault(root, []).append(row)

    clusters = list(components.values())
    clusters.sort(
        key=lambda cluster: min(int(item["_order"]) for item in cluster)
    )
    for cluster in clusters:
        cluster.sort(key=lambda item: int(item["_order"]))

    cluster_sizes = [len(cluster) for cluster in clusters if len(cluster) > 1]
    singleton_count = sum(1 for cluster in clusters if len(cluster) == 1)
    if cluster_sizes:
        logger.info(
            "ANI clustering formed %s multi-contig clusters (max size: %s; singletons: %s)",
            len(cluster_sizes),
            max(cluster_sizes),
            singleton_count,
        )
    else:
        logger.info(
            "ANI clustering formed no multi-contig clusters (singletons: %s)",
            singleton_count,
        )

    return clusters


#  Process-pool worker for pileup
def process_ani_cluster_pileup_worker(
    cluster: list[dict[str, Any]],
) -> dict[str, Any]:
    """Process one ANI cluster inside a worker process.

    Args:
            cluster: Rows belonging to one ANI cluster.

    Returns:
            dict[str, Any]: Worker payload containing representative output, optional
            pileup extension details, and per-cluster summary metadata.
    """
    if (
        ANI_WORKER_REPRESENTATIVE_MODE is None
        or ANI_WORKER_PILEUP_MIN_OVERLAP is None
        or ANI_WORKER_PILEUP_MIN_IDENTITY is None
        or ANI_WORKER_PREFIX_KMERS is None
        or ANI_WORKER_SUFFIX_KMERS is None
        or ANI_WORKER_ALIGNER_BACKEND is None
        or ANI_WORKER_THREADS is None
        or ANI_WORKER_PARASAIL_ALGORITHM is None
        or ANI_WORKER_PARASAIL_GAP_OPEN is None
        or ANI_WORKER_PARASAIL_GAP_EXTEND is None
        or ANI_WORKER_PARASAIL_MATCH is None
        or ANI_WORKER_PARASAIL_MISMATCH is None
    ):
        raise RuntimeError("ANI worker state was not initialized")

    if len(cluster) == 1:
        only = cluster[0]
        return {
            "is_multi": False,
            "representative_row": {
                "contig_id": only["contig_id"],
                "sequence": only["sequence"],
                "seq_length": only["seq_length"],
                "_order": only["_order"],
            },
        }

    representative = max(
        cluster, key=lambda row: (int(row["seq_length"]), -int(row["_order"]))
    )
    cluster_anchor = min(int(row["_order"]) for row in cluster) + 1
    cluster_id = f"ani_cluster_{cluster_anchor}"
    representative_sequence = str(representative["sequence"])
    representative_contig_id = str(representative["contig_id"])
    pileup_stats: dict[str, Any] = {}
    candidate_sequence = representative_sequence
    contributor_ids: set[str] = set()
    extension_steps: list[dict[str, Any]] = []

    if ANI_WORKER_REPRESENTATIVE_MODE == "pileup":
        candidate, contributor_ids, extension_steps, pileup_stats = (
            build_pileup_sequence(
                cluster,
                ANI_WORKER_PILEUP_MIN_OVERLAP,
                ANI_WORKER_PILEUP_MIN_IDENTITY,
                ANI_WORKER_PREFIX_KMERS,
                ANI_WORKER_SUFFIX_KMERS,
                ANI_WORKER_ALIGNER_BACKEND,
                threads=ANI_WORKER_THREADS,
                parasail_algorithm=ANI_WORKER_PARASAIL_ALGORITHM,
                parasail_gap_open=ANI_WORKER_PARASAIL_GAP_OPEN,
                parasail_gap_extend=ANI_WORKER_PARASAIL_GAP_EXTEND,
                parasail_match=ANI_WORKER_PARASAIL_MATCH,
                parasail_mismatch=ANI_WORKER_PARASAIL_MISMATCH,
            )
        )
        if candidate and len(candidate) > len(representative_sequence):
            candidate_sequence = candidate

    return {
        "is_multi": True,
        "cluster": cluster,
        "cluster_id": cluster_id,
        "representative": representative,
        "representative_contig_id": representative_contig_id,
        "representative_sequence": representative_sequence,
        "candidate_sequence": candidate_sequence,
        "contributor_ids": contributor_ids,
        "extension_steps": extension_steps,
        "pileup_stats": pileup_stats,
        "representative_row": {
            "contig_id": representative["contig_id"],
            "sequence": candidate_sequence,
            "seq_length": len(candidate_sequence),
            "_order": representative["_order"],
        },
    }


# Main extension orchestrator
def run_pileup_extension(
    seq_df: pl.DataFrame,
    min_identity: float,
    min_af: float,
    pileup_min_overlap: int,
    pileup_min_identity: float,
    pileup_aligner_backend: str,
    logger,
    threads: int = 4,
    parasail_algorithm: str = ANI_PILEUP_PARASAIL_ALGORITHM,
    parasail_gap_open: int = ANI_PILEUP_PARASAIL_GAP_OPEN,
    parasail_gap_extend: int = ANI_PILEUP_PARASAIL_GAP_EXTEND,
    parasail_match: int = ANI_PILEUP_PARASAIL_MATCH,
    parasail_mismatch: int = ANI_PILEUP_PARASAIL_MISMATCH,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run ANI clustering + pileup extension.

    Returns:
            extended_df: DataFrame with extended contig sequences (contig_id, sequence, seq_length).
            clusters_df: DataFrame describing ANI cluster membership
                    (cluster_id, contig_id, is_representative, was_extended, extended_bp).
    """
    if seq_df.is_empty() or seq_df.height < 2:
        clusters_df = pl.DataFrame(
            schema={
                "cluster_id": pl.String,
                "contig_id": pl.String,
                "is_representative": pl.Boolean,
                "was_extended": pl.Boolean,
                "extended_bp": pl.Int64,
            }
        )
        return seq_df.select("contig_id", "sequence", "seq_length"), clusters_df

    if pileup_aligner_backend not in {"parasail", "pyopal"}:
        raise click.ClickException(
            "--pileup-aligner must be one of: parasail, pyopal"
        )
    if parasail_algorithm not in {"ov", "sw"}:
        raise click.ClickException(
            "--pileup-parasail-algorithm must be one of: ov, sw"
        )

    seq_rows = seq_df.select("contig_id", "sequence", "seq_length").to_dicts()
    for idx, row in enumerate(seq_rows):
        row["_order"] = idx

    # Precompute terminal k-mer maps for prefiltering
    kmer_start = perf_counter()
    prefix_kmers_by_idx, suffix_kmers_by_idx = compute_terminal_kmer_maps(
        seq_rows, logger=logger
    )
    logger.debug(
        "Pileup global terminal kmer precompute: rows=%s elapsed_s=%.3f (k=%s, window=%s)",
        len(seq_rows),
        perf_counter() - kmer_start,
        ANI_PILEUP_PREFILTER_K,
        ANI_PILEUP_PREFILTER_WINDOW,
    )

    # ANI clustering
    clusters = cluster_contigs_by_ani(seq_rows, min_identity, min_af, logger)
    if not clusters:
        clusters_df = pl.DataFrame(
            schema={
                "cluster_id": pl.String,
                "contig_id": pl.String,
                "is_representative": pl.Boolean,
                "was_extended": pl.Boolean,
                "extended_bp": pl.Int64,
            }
        )
        return seq_df.select("contig_id", "sequence", "seq_length"), clusters_df

    # Process clusters with pileup
    representative_mode = "pileup"
    representatives: list[dict[str, Any]] = []
    cluster_membership_rows: list[dict[str, Any]] = []
    pileup_cluster_count = 0
    pileup_extended_count = 0
    pileup_max_extension = 0
    pileup_candidate_pairs = 0
    pileup_prefilter_passed = 0
    pileup_prefilter_skipped = 0
    pileup_edges_built = 0
    pileup_edge_seconds = 0.0
    pileup_total_seconds = 0.0

    def process_cluster(cluster: list[dict[str, Any]]) -> dict[str, Any]:
        if len(cluster) == 1:
            only = cluster[0]
            return {
                "is_multi": False,
                "representative_row": {
                    "contig_id": only["contig_id"],
                    "sequence": only["sequence"],
                    "seq_length": only["seq_length"],
                    "_order": only["_order"],
                },
            }

        representative = max(
            cluster,
            key=lambda row: (int(row["seq_length"]), -int(row["_order"])),
        )
        cluster_anchor = min(int(row["_order"]) for row in cluster) + 1
        cluster_id = f"ani_cluster_{cluster_anchor}"
        representative_sequence = str(representative["sequence"])
        representative_contig_id = str(representative["contig_id"])
        pileup_stats: dict[str, Any] = {}
        candidate_sequence = representative_sequence
        contributor_ids: set[str] = set()
        extension_steps: list[dict[str, Any]] = []

        candidate, contributor_ids, extension_steps, pileup_stats = (
            build_pileup_sequence(
                cluster,
                pileup_min_overlap,
                pileup_min_identity,
                prefix_kmers_by_idx,
                suffix_kmers_by_idx,
                pileup_aligner_backend,
                threads=threads,
                parasail_algorithm=parasail_algorithm,
                parasail_gap_open=parasail_gap_open,
                parasail_gap_extend=parasail_gap_extend,
                parasail_match=parasail_match,
                parasail_mismatch=parasail_mismatch,
            )
        )
        if candidate and len(candidate) > len(representative_sequence):
            candidate_sequence = candidate

        return {
            "is_multi": True,
            "cluster": cluster,
            "cluster_id": cluster_id,
            "representative": representative,
            "representative_contig_id": representative_contig_id,
            "representative_sequence": representative_sequence,
            "candidate_sequence": candidate_sequence,
            "contributor_ids": contributor_ids,
            "extension_steps": extension_steps,
            "pileup_stats": pileup_stats,
            "representative_row": {
                "contig_id": representative["contig_id"],
                "sequence": candidate_sequence,
                "seq_length": len(candidate_sequence),
                "_order": representative["_order"],
            },
        }

    # Decide whether to run in parallel
    run_parallel = pileup_aligner_backend == "parasail" and threads > 1
    if run_parallel:
        multi_cluster_count = sum(1 for cluster in clusters if len(cluster) > 1)
        logger.info(
            "Pileup parasail multiprocessing enabled: %s workers across %s multi-contig clusters",
            threads,
            multi_cluster_count,
        )
        with ProcessPoolExecutor(
            max_workers=threads,
            initializer=init_ani_cluster_pileup_worker,
            initargs=(
                representative_mode,
                pileup_min_overlap,
                pileup_min_identity,
                prefix_kmers_by_idx,
                suffix_kmers_by_idx,
                pileup_aligner_backend,
                threads,
                parasail_algorithm,
                parasail_gap_open,
                parasail_gap_extend,
                parasail_match,
                parasail_mismatch,
            ),
        ) as executor:
            cluster_results = list(
                executor.map(
                    process_ani_cluster_pileup_worker, clusters, chunksize=1
                )
            )
    else:
        cluster_results = [process_cluster(cluster) for cluster in clusters]

    # Collect results
    for cluster_result in cluster_results:
        representatives.append(cluster_result["representative_row"])
        cluster = cluster_result.get("cluster", [])
        representative_contig_id = str(
            cluster_result.get(
                "representative_contig_id",
                cluster_result["representative_row"]["contig_id"],
            )
        )
        cluster_id = str(
            cluster_result.get(
                "cluster_id", f"singleton_{representative_contig_id}"
            )
        )

        if not cluster_result.get("is_multi", False):
            # Singleton cluster
            cluster_membership_rows.append(
                {
                    "cluster_id": cluster_id,
                    "contig_id": cluster_result["representative_row"][
                        "contig_id"
                    ],
                    "is_representative": True,
                    "was_extended": False,
                    "extended_bp": 0,
                }
            )
            continue

        representative = cluster_result["representative"]
        representative_sequence = str(cluster_result["representative_sequence"])
        candidate_sequence = str(cluster_result["candidate_sequence"])
        contributor_ids = set(cluster_result["contributor_ids"])
        extension_steps = list(cluster_result["extension_steps"])
        pileup_stats = dict(cluster_result["pileup_stats"])
        was_extended = len(candidate_sequence) > len(representative_sequence)
        extension = (
            len(candidate_sequence) - len(representative_sequence)
            if was_extended
            else 0
        )

        # Build cluster membership rows
        for member in cluster:
            member_id = str(member["contig_id"])
            is_rep = member_id == representative_contig_id
            cluster_membership_rows.append(
                {
                    "cluster_id": cluster_id,
                    "contig_id": member_id,
                    "is_representative": is_rep,
                    "was_extended": was_extended and is_rep,
                    "extended_bp": extension if is_rep else 0,
                }
            )

        # Accumulate pileup stats
        pileup_cluster_count += 1
        pileup_candidate_pairs += int(pileup_stats.get("candidate_pairs", 0))
        pileup_prefilter_passed += int(pileup_stats.get("prefilter_passed", 0))
        pileup_prefilter_skipped += int(
            pileup_stats.get("prefilter_skipped", 0)
        )
        pileup_edges_built += int(pileup_stats.get("edges_built", 0))
        pileup_edge_seconds += float(
            pileup_stats.get("edge_build_seconds", 0.0)
        )
        pileup_total_seconds += float(pileup_stats.get("total_seconds", 0.0))

        logger.debug(
            "Pileup graph %s: cluster_size=%s candidate_pairs=%s prefilter_passed=%s prefilter_skipped=%s edges=%s edge_build_s=%.3f total_s=%.3f nascent=%s",
            cluster_id,
            int(pileup_stats.get("cluster_size", 0)),
            int(pileup_stats.get("candidate_pairs", 0)),
            int(pileup_stats.get("prefilter_passed", 0)),
            int(pileup_stats.get("prefilter_skipped", 0)),
            int(pileup_stats.get("edges_built", 0)),
            float(pileup_stats.get("edge_build_seconds", 0.0)),
            float(pileup_stats.get("total_seconds", 0.0)),
            str(pileup_stats.get("nascent_chain", representative_contig_id)),
        )

        if was_extended:
            pileup_extended_count += 1
            pileup_max_extension = max(pileup_max_extension, extension)
            contributor_list = sorted(
                cid
                for cid in contributor_ids
                if cid != representative_contig_id
            )
            logger.debug(
                "Pileup extension %s: representative=%s (%s bp -> %s bp, +%s) contributors=%s",
                cluster_id,
                representative_contig_id,
                len(representative_sequence),
                len(candidate_sequence),
                extension,
                contributor_list if contributor_list else ["none"],
            )

    # Build output DataFrames
    extended_df = pl.DataFrame(representatives).sort("_order").drop("_order")
    removed = seq_df.height - extended_df.height
    if removed > 0:
        logger.info(
            "Extend retained %s/%s representative contigs (removed %s redundant)",
            extended_df.height,
            seq_df.height,
            removed,
        )

    if pileup_cluster_count > 0:
        aligner_details = pileup_aligner_backend
        if pileup_aligner_backend == "parasail":
            aligner_details = (
                f"parasail/{parasail_algorithm}"
                f"/go{parasail_gap_open}"
                f"/ge{parasail_gap_extend}"
                f"/m{parasail_match}"
                f"/mm{parasail_mismatch}"
            )
        logger.info(
            "Pileup extended %s/%s multi-contig clusters (max extension: %s bp; min overlap: %s; min overlap identity: %.3f)",
            pileup_extended_count,
            pileup_cluster_count,
            pileup_max_extension,
            pileup_min_overlap,
            pileup_min_identity,
        )
        logger.info(
            "Pileup phase-1 metrics: candidate_pairs=%s prefilter_passed=%s prefilter_skipped=%s edges=%s edge_build_s=%.3f total_s=%.3f (k=%s, window=%s, min_shared=%s, aligner=%s)",
            pileup_candidate_pairs,
            pileup_prefilter_passed,
            pileup_prefilter_skipped,
            pileup_edges_built,
            pileup_edge_seconds,
            pileup_total_seconds,
            ANI_PILEUP_PREFILTER_K,
            ANI_PILEUP_PREFILTER_WINDOW,
            ANI_PILEUP_PREFILTER_MIN_SHARED,
            aligner_details,
        )

    clusters_df = pl.DataFrame(cluster_membership_rows)

    return extended_df, clusters_df


# CLI command -----------------


@click.command()
@click.option(
    "-i",
    "--input",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="Input contig FASTA/FASTQ file",
)
@click.option(
    "-o",
    "--output",
    default="extended_contigs.fasta",
    show_default=True,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Output path for extended contigs FASTA",
)
@click.option(
    "--clusters-output",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Output path for ANI cluster membership table (default: <output_stem>.clusters.tsv)",
)
@click.option(
    "--ani-min-identity",
    default=0.95,
    show_default=True,
    type=click.FloatRange(0.0, 1.0),
    help="Minimum ANI identity (0-1) for contigs to be placed in the same cluster",
)
@click.option(
    "--ani-min-af",
    default=0.80,
    show_default=True,
    type=click.FloatRange(0.0, 1.0),
    help="Minimum aligned fraction (0-1) for ANI clustering",
)
@click.option(
    "--pileup-min-overlap",
    default=50,
    show_default=True,
    type=click.IntRange(1, 1000000),
    help="Minimum overlap length for pileup extension",
)
@click.option(
    "--pileup-min-identity",
    default=0.98,
    show_default=True,
    type=click.FloatRange(0.0, 1.0),
    help="Minimum identity (0-1) within overlap windows when merging cluster members",
)
@click.option(
    "--pileup-aligner",
    default="parasail",
    show_default=True,
    type=click.Choice(["parasail", "pyopal"], case_sensitive=False),
    help="Pairwise aligner backend for overlap checks",
)
@click.option(
    "--pileup-parasail-algorithm",
    default=ANI_PILEUP_PARASAIL_ALGORITHM,
    show_default=True,
    type=click.Choice(["ov", "sw"], case_sensitive=False),
    help="Parasail algorithm: ov (semiglobal overlap) or sw (local Smith-Waterman)",
)
@click.option(
    "--pileup-parasail-gap-open",
    default=ANI_PILEUP_PARASAIL_GAP_OPEN,
    show_default=True,
    type=click.IntRange(1, 1000000),
    help="Parasail gap-open penalty",
)
@click.option(
    "--pileup-parasail-gap-extend",
    default=ANI_PILEUP_PARASAIL_GAP_EXTEND,
    show_default=True,
    type=click.IntRange(0, 1000000),
    help="Parasail gap-extend penalty",
)
@click.option(
    "--pileup-parasail-match",
    default=ANI_PILEUP_PARASAIL_MATCH,
    show_default=True,
    type=click.IntRange(1, 1000000),
    help="Parasail nucleotide match score for ov mode matrix",
)
@click.option(
    "--pileup-parasail-mismatch",
    default=ANI_PILEUP_PARASAIL_MISMATCH,
    show_default=True,
    type=click.IntRange(-1000000, 1000000),
    help="Parasail nucleotide mismatch score for ov mode matrix",
)
@click.option(
    "--output-format",
    default="tsv",
    show_default=True,
    type=click.Choice(["tsv", "csv", "parquet"], case_sensitive=False),
    help="Tabular output format for the cluster membership table",
)
@click.option(
    "--log-file",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Optional log file path",
)
@click.option(
    "-ll", "--log-level", default="INFO", show_default=True, hidden=True
)
@click.option(
    "-t",
    "--threads",
    default=4,
    show_default=True,
    type=click.IntRange(1, 1000),
    help="Number of worker processes for parallel pileup alignment",
)
def extend(
    input: Path,
    output: Path,
    clusters_output: Path | None,
    ani_min_identity: float,
    ani_min_af: float,
    pileup_min_overlap: int,
    pileup_min_identity: float,
    pileup_aligner: str,
    pileup_parasail_algorithm: str,
    pileup_parasail_gap_open: int,
    pileup_parasail_gap_extend: int,
    pileup_parasail_match: int,
    pileup_parasail_mismatch: int,
    output_format: str,
    log_file: Path | None,
    log_level: str,
    threads: int,
) -> None:
    """Extend contigs by ANI-guided overlap pileup.

    Workflow:
    1) Read input seqs (FASTA/FASTQ).
    2) Cluster input by ANI using pyskani.
    3) Within each multi-contig cluster, attempt overlap pileup extension
       to produce a longer representative contig.
    4) Write extended contigs (FASTA) and cluster membership table.

    Use this command to get more complete genomes from fragmented assemblies,
    for example when combining data from multiple experiments or samples.
    # TODO: support for multiple input files, e.g. one fiel is contigs from sample A, another file is reads from sample B.
    """
    logger = setup_logging(log_file, log_level)
    log_start_info(logger, locals())

    pileup_aligner = pileup_aligner.lower()
    pileup_parasail_algorithm = pileup_parasail_algorithm.lower()

    seq_df = load_sequences(input)
    if seq_df.is_empty():
        logger.warning("No sequences found in input file")
        return

    logger.info("Loaded %s contigs from %s", seq_df.height, input)

    extended_df, clusters_df = run_pileup_extension(
        seq_df=seq_df,
        min_identity=ani_min_identity,
        min_af=ani_min_af,
        pileup_min_overlap=pileup_min_overlap,
        pileup_min_identity=pileup_min_identity,
        pileup_aligner_backend=pileup_aligner,
        logger=logger,
        threads=threads,
        parasail_algorithm=pileup_parasail_algorithm,
        parasail_gap_open=pileup_parasail_gap_open,
        parasail_gap_extend=pileup_parasail_gap_extend,
        parasail_match=pileup_parasail_match,
        parasail_mismatch=pileup_parasail_mismatch,
    )

    # Write extended contigs FASTA
    output.parent.mkdir(parents=True, exist_ok=True)
    write_df = extended_df.select(
        pl.col("contig_id").alias("header"), pl.col("sequence")
    )
    frame_to_fastx(write_df, output)
    logger.info(
        "Extended contigs written to %s (%s sequences)",
        output,
        extended_df.height,
    )

    # Write cluster membership table
    if clusters_output is None:
        clusters_output = output.with_name(f"{output.stem}.clusters.tsv")
    clusters_output.parent.mkdir(parents=True, exist_ok=True)
    fmt = output_format.lower()
    if fmt == "tsv":
        clusters_df.write_csv(clusters_output, separator="\t")
    elif fmt == "csv":
        clusters_df.write_csv(clusters_output)
    elif fmt == "parquet":
        clusters_df.write_parquet(clusters_output)
    else:
        clusters_df.write_csv(clusters_output, separator="\t")
    logger.info(
        "Cluster membership table written to %s (%s rows)",
        clusters_output,
        clusters_df.height,
    )
