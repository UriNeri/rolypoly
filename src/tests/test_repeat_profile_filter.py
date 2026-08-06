import polars as pl

from rolypoly.utils.bio.interval_ops import filter_repeated_profile_regions


def test_repeat_profile_filter_removes_cid_88_pattern():
    hits = pl.DataFrame(
        {
            "query_full_name": ["CID_88_frame=-3", "CID_88_frame=-3", "CID_88_frame=-3"],
            "hmm_full_name": ["GENOMAD.025109.VV", "GENOMAD.025109.VV", "other"],
            "hmm_from": [343, 338, 1],
            "hmm_to": [385, 385, 90],
            "q1": [19, 117, 70],
            "q2": [62, 165, 159],
            "qlen": [170, 170, 170],
        }
    )

    kept, removed = filter_repeated_profile_regions(hits)

    assert kept["hmm_full_name"].to_list() == ["other"]
    assert removed.height == 2
    assert removed["repeat_occurrences"].to_list() == [2, 2]
    assert removed["repeat_filter_reason"].unique().to_list() == [
        "repeated_profile_region"
    ]


def test_repeat_profile_filter_keeps_distinct_profile_regions_and_same_query_locus():
    hits = pl.DataFrame(
        {
            "query_full_name": ["query", "query", "query"],
            "hmm_full_name": ["profile", "profile", "profile"],
            "hmm_from": [1, 101, 2],
            "hmm_to": [80, 180, 81],
            "q1": [1, 101, 2],
            "q2": [80, 180, 81],
            "qlen": [200, 200, 200],
        }
    )

    kept, removed = filter_repeated_profile_regions(hits)

    assert kept.height == 3
    assert removed.is_empty()


def test_repeat_profile_filter_supports_mmseqs_and_marks_broad_coverage():
    hits = pl.DataFrame(
        {
            "qseqid": ["query", "query"],
            "sseqid": ["profile", "profile"],
            "sstart": [20, 23],
            "send": [70, 72],
            "qstart": [1, 101],
            "qend": [80, 180],
            "qlen": [200, 200],
        }
    )

    kept, removed = filter_repeated_profile_regions(hits)

    assert kept.is_empty()
    assert removed.height == 2
    assert removed["repeat_query_coverage"].to_list() == [0.8, 0.8]
    assert removed["repeat_filter_reason"].unique().to_list() == [
        "repeated_profile_region_query_coverage"
    ]
