"""Scientific contracts for vectorized sequence transformations."""

import polars as pl
import pytest

from rolypoly.utils.bio.polars_fastx import count_kmers_df


@pytest.mark.filterwarnings("error::DeprecationWarning")
@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("canonical", [False, True])
def test_kmer_counts_preserve_boundaries_and_omit_incomplete_sequences(
    relative: bool, canonical: bool
) -> None:
    source = pl.DataFrame(
        {
            "seqid": ["empty", "null", "short", "exact", "repeat", "strand"],
            "seq": ["", None, "A", "AC", "AAAA", "ACGT"],
            "label": ["e", "n", "s", "x", "r", "t"],
        }
    )

    result = count_kmers_df(source, k=2, relative=relative, canonical=canonical)

    assert result.columns == ["seqid", "seq", "kmer_counts", "label"]
    assert result["seqid"].to_list() == ["exact", "repeat", "strand"]
    assert result["seq"].to_list() == ["AC", "AAAA", "ACGT"]
    assert result["label"].to_list() == ["x", "r", "t"]
    value_key = "proportion" if relative else "count"
    counts = {
        row["seqid"]: {
            entry["substrings"]: entry[value_key]
            for entry in row["kmer_counts"]
        }
        for row in result.to_dicts()
    }
    assert counts["exact"] == {"AC": 1}
    assert counts["repeat"] == {"AA": 1 if relative else 3}
    expected_strand = {"AC": 2, "CG": 1} if canonical else {"AC": 1, "CG": 1, "GT": 1}
    if relative:
        expected_strand = {kmer: count / 3 for kmer, count in expected_strand.items()}
    assert counts["strand"] == pytest.approx(expected_strand)


@pytest.mark.filterwarnings("error::DeprecationWarning")
@pytest.mark.parametrize("sequences", [[], ["", None, "A"]])
def test_kmer_counts_without_complete_kmers_preserve_schema(
    sequences: list[str | None],
) -> None:
    source = pl.DataFrame(
        {"seqid": [str(i) for i in range(len(sequences))], "seq": sequences},
        schema={"seqid": pl.String, "seq": pl.String},
    )

    result = count_kmers_df(source, k=2)

    assert result.is_empty()
    assert result.schema == {
        "seqid": pl.String,
        "seq": pl.String,
        "kmer_counts": pl.List(
            pl.Struct({"substrings": pl.String, "count": pl.UInt32})
        ),
    }
