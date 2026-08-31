from pathlib import Path

import polars as pl

from rolypoly.commands.identify_virus.marker_search import (
    project_marker_features,
    search_mmseqs_marker_db,
)
from rolypoly.utils.bio import alignments


def test_search_mmseqs_marker_db_writes_marker_schema(tmp_path, monkeypatch):
    def fake_mmseqs_easy_search(**kwargs):
        assert kwargs["format_mode"] == 4
        assert "theader" in kwargs["format_output"]
        Path(kwargs["output"]).write_text(
            "query\tqheader\ttarget\ttheader\tevalue\tbits\tqstart\tqend\t"
            "tstart\ttend\tqlen\ttlen\talnlen\tqcov\ttcov\tqaln\ttaln\tqseq\n"
            "query1\tquery1 description\tPF04197_seed\t"
            "PF04197_seed representative/1-4\t1e-20\t80\t1\t4\t2\t5\t4\t6\t"
            "4\t1.0\t0.667\tAC-D\tACED\tACD\n"
            "query1\tquery1 description\tPF00078_seed\t"
            "PF00078_seed representative/1-4\t1e-3\t10\t1\t4\t2\t5\t4\t6\t"
            "4\t1.0\t0.667\tAC-D\tACED\tACD\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        alignments, "mmseqs_easy_search", fake_mmseqs_easy_search
    )
    output = tmp_path / "marker_hits.tsv"

    search_mmseqs_marker_db(
        amino_file=tmp_path / "query.faa",
        db_path=tmp_path / "profiles",
        output=output,
        temp_dir=tmp_path / "tmp",
        threads=1,
        inc_evalue=0.001,
        score=20,
        min_ali_len=4,
        include_aligned_region=True,
        include_alignment_string=True,
        include_full_query=True,
        logger=None,
        include_alignment_path=True,
    )

    hits = pl.read_csv(output, separator="\t")
    assert hits.height == 1
    assert hits.row(0, named=True) == {
        "query_full_name": "query1 description",
        "hmm_full_name": "PF04197",
        "profile_accession": "PF04197",
        "hmm_len": 6,
        "qlen": 4,
        "full_hmm_evalue": 1e-20,
        "full_hmm_score": 80,
        "full_hmm_bias": 0.0,
        "this_dom_score": 80,
        "this_dom_bias": 0.0,
        "hmm_from": 2,
        "hmm_to": 5,
        "q1": 1,
        "q2": 4,
        "env_from": 1,
        "env_to": 4,
        "hmm_cov": 0.667,
        "ali_len": 4,
        "dom_desc": "representative/1-4",
        "aligned_region": "AC-D",
        "domain_index": 1,
        "profile_alignment": "ACED",
        "query_alignment": "AC-D",
        "full_qseq": "ACD",
        "identity_str": "|| |",
    }


def test_marker_feature_projection_uses_backend_state_maps(tmp_path):
    feature_path = tmp_path / "marker_features.tsv"
    pl.DataFrame(
        {
            "marker_db": ["rvmt"],
            "profile_id": ["profile_a"],
            "feature_id": ["feature_a"],
            "canonical_term_id": ["CDD:feature_a"],
            "raw_source_label": ["A"],
            "hmmer_states_1based": ["2;4"],
            "mmseqs_states_1based": ["1;3"],
        }
    ).write_csv(feature_path, separator="\t")

    def hit(profile_alignment, query_alignment):
        return pl.DataFrame(
            {
                "database_id": ["RVMT"],
                "query_full_name": ["query1"],
                "hmm_full_name": ["profile_a"],
                "profile_accession": [""],
                "domain_index": ["1"],
                "hmm_len": [4],
                "hmm_from": [1],
                "hmm_to": [4],
                "q1": [1],
                "q2": [4],
                "profile_alignment": [profile_alignment],
                "query_alignment": [query_alignment],
            }
        )

    hmm_output = tmp_path / "hmm.tsv"
    mmseqs_output = tmp_path / "mmseqs.tsv"
    project_marker_features(
        hit("ACDE", "ACDE"), feature_path, hmm_output, "hmmsearch"
    )
    project_marker_features(
        hit("ACDE", "ACDE"), feature_path, mmseqs_output, "mmseqs2"
    )

    hmm_row = pl.read_csv(hmm_output, separator="\t").row(0, named=True)
    mmseqs_row = pl.read_csv(mmseqs_output, separator="\t").row(0, named=True)
    assert "profile_alignment" not in pl.read_csv(hmm_output, separator="\t").columns
    assert (
        hmm_row["projection_status"]
        == mmseqs_row["projection_status"]
        == "complete"
    )
    assert hmm_row["projected_query_positions_1based"] == "2;4"
    assert hmm_row["projected_query_residues"] == "CE"
    assert mmseqs_row["projected_query_positions_1based"] == "1;3"
    assert mmseqs_row["projected_query_residues"] == "AD"


def test_marker_feature_projection_states_and_profile_gap_encodings(tmp_path):
    feature_path = tmp_path / "marker_features.tsv"
    pl.DataFrame(
        {
            "marker_db": ["rvmt"] * 5,
            "profile_id": ["profile_a"] * 5,
            "feature_id": ["gap", "deleted", "out", "ambiguous", "mixed"],
            "canonical_term_id": [
                "CDD:gap",
                "CDD:deleted",
                "CDD:out",
                "CDD:ambiguous",
                "CDD:mixed",
            ],
            "raw_source_label": ["G", "D", "O", "U", "M"],
            "hmmer_states_1based": ["2;3", "2", "1", "2", "1;2"],
            "mmseqs_states_1based": ["2;3", "2", "1", "2", "1;2"],
        }
    ).write_csv(feature_path, separator="\t")

    def hit(
        query_name,
        profile_alignment,
        query_alignment,
        profile_start,
        profile_end,
        query_end,
    ):
        return pl.DataFrame(
            {
                "database_id": ["RVMT"],
                "query_full_name": [query_name],
                "hmm_full_name": ["profile_a"],
                "profile_accession": [""],
                "domain_index": ["1"],
                "hmm_len": [4],
                "hmm_from": [profile_start],
                "hmm_to": [profile_end],
                "q1": [1],
                "q2": [query_end],
                "profile_alignment": [profile_alignment],
                "query_alignment": [query_alignment],
            }
        )

    for backend, profile_gap in (("hmmsearch", "."), ("mmseqs2", "-")):
        hits = pl.concat(
            [
                hit("gap", f"AC{profile_gap}DE", "ACXDE", 1, 4, 5),
                hit("deleted", "ACDE", "A-DE", 1, 4, 3),
                hit("out", "DE", "DE", 3, 4, 2),
                hit("ambiguous", "AC", "AX", 1, 2, 2),
                hit("mixed", "AC", "A-", 1, 2, 1),
            ]
        )
        output = tmp_path / f"{backend}.states.tsv"
        project_marker_features(hits, feature_path, output, backend)
        projected = pl.read_csv(output, separator="\t")
        statuses = {
            (row["query_full_name"], row["feature_id"]): row[
                "projection_status"
            ]
            for row in projected.iter_rows(named=True)
        }
        assert statuses[("gap", "gap")] == "complete"
        assert statuses[("deleted", "deleted")] == "absent"
        assert statuses[("out", "out")] == "absent"
        assert statuses[("ambiguous", "ambiguous")] == "absent"
        assert statuses[("mixed", "mixed")] == "partial"
        gap_row = projected.filter(
            (pl.col("query_full_name") == "gap")
            & (pl.col("feature_id") == "gap")
        ).row(0, named=True)
        assert gap_row["insertion_query_positions_1based"] == "3"
        assert gap_row["feature_query_start_1based"] == "2"
        assert gap_row["feature_query_end_1based"] == "4"
        deleted_row = projected.filter(
            (pl.col("query_full_name") == "deleted")
            & (pl.col("feature_id") == "deleted")
        ).row(0, named=True)
        assert deleted_row["deleted_profile_states_1based"] == "2"
        out_row = projected.filter(
            (pl.col("query_full_name") == "out")
            & (pl.col("feature_id") == "out")
        ).row(0, named=True)
        assert out_row["out_of_span_profile_states_1based"] == "1"
        ambiguous_row = projected.filter(
            (pl.col("query_full_name") == "ambiguous")
            & (pl.col("feature_id") == "ambiguous")
        ).row(0, named=True)
        assert ambiguous_row["unresolved_profile_states_1based"] == "2"
        assert ambiguous_row["unresolved_query_positions_1based"] == "2"
        assert ambiguous_row["unresolved_query_residues"] == "X"
        mixed_row = projected.filter(
            (pl.col("query_full_name") == "mixed")
            & (pl.col("feature_id") == "mixed")
        ).row(0, named=True)
        assert mixed_row["mapped_profile_states_1based"] == "1"
        assert mixed_row["deleted_profile_states_1based"] == "2"
