"""Regression tests for sequence hashing and interval hit selection."""

import polars as pl
import pytest

from rolypoly.utils.bio.interval_ops import (
    consolidate_hits,
    detect_polyprotein_pattern,
)
from rolypoly.utils.bio.polars_fastx import dereplicate_fasta, seq_hash_xxh3


@pytest.mark.parametrize("polyprotein", [False, True])
def test_adaptive_selection_is_query_local(polyprotein):
    rows = [("q", "a", 1, 500, 100, 1000), ("q", "b", 480, 979, 90, 1000)]
    if polyprotein:
        rows += [
            ("q", f"p{i}", start, start + 450, 10 + i, 1000)
            for i, start in enumerate([1] * 4 + [480] * 4)
        ]
    columns = ["qseqid", "sseqid", "q1", "q2", "score", "qlen"]
    base = pl.DataFrame(rows, schema=columns, orient="row")
    assert (
        detect_polyprotein_pattern(
            base,
            "q",
            query_id_col="qseqid",
            target_id_col="sseqid",
            q1_col="q1",
            q2_col="q2",
        )
        == polyprotein
    )
    extra = pl.DataFrame(
        [("other", "c", 1, 50, 80, 1000)], schema=columns, orient="row"
    )

    def select(data):
        return consolidate_hits(data, one_per_range=True, adaptive_overlap=True)

    alone = select(base).sort("sseqid")
    together = (
        select(pl.concat([base, extra]))
        .filter(pl.col("qseqid") == "q")
        .sort("sseqid")
    )
    assert alone.equals(together)
    assert alone["sseqid"].to_list() == (["a"] if polyprotein else ["a", "b"])


@pytest.mark.parametrize("ignore_case", [False, True])
def test_dereplication_case_policy(tmp_path, ignore_case):
    source = tmp_path / "input.fa"
    source.write_text(
        ">first description\nAACG\n>lower\naacg\n>duplicate\nAACG\n>reverse\nCGTT\n"
    )
    output = tmp_path / "output.fa"
    stats = dereplicate_fasta(
        source, output, ignore_case=ignore_case, batch_size=1
    )
    assert stats["old_id"].to_list() == (
        ["first", "reverse"] if ignore_case else ["first", "lower", "reverse"]
    )
    assert stats["redundancy"].to_list() == (
        [3, 1] if ignore_case else [2, 1, 1]
    )
    assert stats["members"][0] == (
        "first;lower;duplicate" if ignore_case else "first;duplicate"
    )
    assert stats["seq_hash"][0] == seq_hash_xxh3("AACG")
    if not ignore_case:
        assert stats["seq_hash"][1] == seq_hash_xxh3("aacg", ignore_case=False)
    assert output.read_text() == (
        ">first\nAACG\n>reverse\nCGTT\n"
        if ignore_case
        else ">first\nAACG\n>lower\naacg\n>reverse\nCGTT\n"
    )


def test_sequence_hash_case_policy():
    assert seq_hash_xxh3("aacg") == seq_hash_xxh3("AACG")
    assert seq_hash_xxh3("AaCg", ignore_case=True) == seq_hash_xxh3("AACG")
    assert seq_hash_xxh3("aacg", ignore_case=False) != seq_hash_xxh3(
        "AACG", ignore_case=False
    )


def test_one_per_range_treats_endpoint_touch_as_one_position_overlap():
    hits = pl.DataFrame(
        [("q", "best", 1, 10, 100), ("q", "endpoint_overlap", 10, 20, 50)],
        schema=["qseqid", "sseqid", "q1", "q2", "score"],
        orient="row",
    )

    result = consolidate_hits(
        hits, rank_columns="-score", one_per_range=True, min_overlap_positions=1
    )

    assert result["sseqid"].to_list() == ["best"]


def test_adaptive_threshold_uses_inclusive_alignment_length():
    hits = pl.DataFrame(
        [("q", "best", 1, 100, 100), ("q", "boundary", 87, 186, 50)],
        schema=["qseqid", "sseqid", "q1", "q2", "score"],
        orient="row",
    )

    result = consolidate_hits(
        hits, rank_columns="-score", one_per_range=True, adaptive_overlap=True
    ).sort("score", descending=True)

    assert result["sseqid"].to_list() == ["best", "boundary"]


def test_split_overlaps_clips_only_same_query_and_strand():
    hits = pl.DataFrame(
        [
            ("q1", "best", 1, 100, "+", 100),
            ("q1", "later", 80, 150, "+", 50),
            ("q1", "other_strand", 90, 140, "-", 40),
            ("q2", "other_query", 80, 150, "+", 30),
        ],
        schema=["qseqid", "sseqid", "q1", "q2", "strand", "score"],
        orient="row",
    )

    result = consolidate_hits(
        hits,
        rank_columns="-score",
        split=True,
        strand_col="strand",
        min_overlap_positions=1,
    ).sort(["qseqid", "sseqid"])

    assert list(result.select("qseqid", "sseqid", "q1", "q2").iter_rows()) == [
        ("q1", "best", 1, 100),
        ("q1", "later", 101, 150),
        ("q1", "other_strand", 90, 140),
        ("q2", "other_query", 80, 150),
    ]


def test_split_overlaps_preserves_reverse_coordinate_orientation():
    hits = pl.DataFrame(
        [("q", "best", 150, 100, 100), ("q", "later", 130, 80, 50)],
        schema=["qseqid", "sseqid", "q1", "q2", "score"],
        orient="row",
    )

    result = consolidate_hits(
        hits, rank_columns="-score", split=True, min_overlap_positions=1
    ).sort("score", descending=True)

    assert list(result.select("sseqid", "q1", "q2").iter_rows()) == [
        ("best", 150, 100),
        ("later", 99, 80),
    ]


def test_split_overlaps_respects_min_overlap_threshold():
    hits = pl.DataFrame(
        [
            ("q", "best", 1, 100, 100),
            ("q", "small_overlap", 96, 130, 50),
            ("q", "large_overlap", 90, 140, 40),
        ],
        schema=["qseqid", "sseqid", "q1", "q2", "score"],
        orient="row",
    )

    result = consolidate_hits(
        hits, rank_columns="-score", split=True, min_overlap_positions=10
    ).sort("score", descending=True)

    assert list(result.select("sseqid", "q1", "q2").iter_rows()) == [
        ("best", 1, 100),
        ("small_overlap", 96, 130),
        ("large_overlap", 131, 140),
    ]
