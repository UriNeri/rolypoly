from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from click.testing import CliRunner

from rolypoly.rolypoly import rolypoly


@pytest.fixture(scope="module")
def runner() -> CliRunner:
    return CliRunner()


@pytest.mark.parametrize(
    "empty_database", ["rvmt.hmm", "pfam_rdrps_and_rts.hmm", "both"]
)
def test_marker_empty_database_preserves_numeric_hits(
    tmp_path, monkeypatch, runner, empty_database
):
    """An empty search must not break interval arithmetic or numeric ranking."""
    from rolypoly.utils.bio import alignments

    data = tmp_path / "data"
    dbs = data / "profiles" / "hmmdbs"
    dbs.mkdir(parents=True)
    for name in ("rvmt.hmm", "pfam_rdrps_and_rts.hmm"):
        (dbs / name).touch()
    monkeypatch.setenv("ROLYPOLY_DATA", str(data))
    query = tmp_path / "query.faa"
    query.write_text(">query1\n" + "ACDEFGHIKLMNPQRSTVWY" * 15 + "\n")

    def fake_search(**kwargs):
        # Ten profiles on a long query exercise adaptive polyprotein detection.
        rows = [
            dict(
                query_full_name="query1",
                hmm_full_name=f"profile{i}",
                profile_accession="",
                hmm_len=300,
                qlen=300,
                full_hmm_evalue=1e-20,
                full_hmm_score=float(score),
                full_hmm_bias=0.0,
                this_dom_score=float(score),
                this_dom_bias=0.0,
                hmm_from=1,
                hmm_to=250,
                q1=1,
                q2=250,
                env_from=1,
                env_to=250,
                hmm_cov=0.83,
                ali_len=250,
                dom_desc="fixture",
            )
            for i, score in enumerate([99, 100, 20, 21, 22, 23, 24, 25, 26, 27])
        ]
        hits = pl.DataFrame(rows)
        if (
            empty_database == "both"
            or Path(kwargs["db_path"]).name == empty_database
        ):
            hits = hits.head(0)
        hits.write_csv(kwargs["output"], separator="\t")
        return kwargs["output"]

    monkeypatch.setattr(alignments, "search_hmmdb", fake_search)
    output = tmp_path / "output"
    result = runner.invoke(
        rolypoly,
        [
            "marker-search",
            "--input",
            str(query),
            "--output",
            str(output),
            "--database",
            "RVMT,Pfam_RTs_RdRp",
            "--threads",
            "1",
            "--temp-dir",
            str(tmp_path / "temp"),
            "--log-file",
            str(tmp_path / "run.log"),
        ],
    )
    assert result.exit_code == 0, result.output + repr(result.exception)
    hits = pl.read_csv(output / "marker_search_results.tsv", separator="\t")
    if empty_database == "both":
        assert hits.is_empty()
    else:
        assert hits["hmm_full_name"].to_list() == ["profile1"]
        assert hits["q1"].to_list() == [1]
        assert hits["q2"].to_list() == [250]
        assert hits["full_hmm_score"].to_list() == [100.0]
