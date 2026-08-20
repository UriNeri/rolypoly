from pathlib import Path

import polars as pl

from rolypoly.commands.identify_virus.marker_search import (
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
        "full_qseq": "ACD",
        "identity_str": "|| |",
    }
