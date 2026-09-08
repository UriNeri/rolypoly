import polars as pl

from rolypoly.utils.viz.genome_maps import (
    fasta_lengths,
    load_contig_lengths,
    load_run_stats,
    write_report_for_dir,
)


def test_report_uses_contig_id_map_length_for_renamed_contig(tmp_path):
    output_dir = tmp_path / "run"
    protein_dir = output_dir / "annotation_results" / "protein_annotation"
    assembly_dir = output_dir / "assembly"
    protein_dir.mkdir(parents=True)
    assembly_dir.mkdir()
    marker_table = protein_dir / "combined_annotations.tsv"
    mapping_path = assembly_dir / "contigs_id_map.tsv"
    report_path = output_dir / "genome_maps.html"

    pl.DataFrame(
        {
            "query_full_name": ["CID_01_1 # 5 # 80 # 1 # ID=CID_01_1"],
            "hmm_full_name": ["RdRp"],
            "source": ["rvmt"],
            "env_from": [1],
            "env_to": [20],
            "qlen": [25],
            "hmm_from": [1],
            "hmm_to": [20],
            "hmm_len": [20],
            "full_hmm_evalue": [1e-20],
            "full_hmm_score": [50.0],
            "hmm_cov": [1.0],
            "ali_len": [20],
            "q1": [1],
            "q2": [20],
        }
    ).write_csv(marker_table, separator="\t")
    pl.DataFrame(
        {
            "old_id": ["NODE_1_length_100_cov_5"],
            "new_id": ["CID_01"],
            "length": [100],
        }
    ).write_csv(mapping_path, separator="\t")

    assert load_contig_lengths(mapping_path) == {"CID_01": 100}
    write_report_for_dir(
        output_dir,
        report_path,
        marker_table=marker_table,
        with_stats=False,
        mark_best=False,
    )

    html = report_path.read_text()
    assert '"contig": "CID_01"' in html
    assert '"length": 100' in html
    assert '"raw_id": "NODE_1_length_100_cov_5"' in html


def test_fasta_lengths_streams_valid_records(tmp_path):
    fasta = tmp_path / "contigs.fa"
    fasta.write_text(">a\nACGT\n>b\nAA\n")

    assert fasta_lengths(fasta) == [4, 2]


def test_load_run_stats_falls_back_to_id_map_when_endpoint_fasta_unreadable(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "run"
    assembly_dir = output_dir / "assembly"
    assembly_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "old_id": ["raw1", "raw2"],
            "new_id": ["CID_1", "CID_2"],
            "assembler": ["input", "input"],
            "length": [100, 50],
        }
    ).write_csv(assembly_dir / "contigs_id_map.tsv", separator="\t")
    (assembly_dir / "length_filtered.fasta").write_text(">CID_1\nACGT\n")

    def fail_parse(_path):
        raise ValueError("bad fasta")

    monkeypatch.setattr(
        "rolypoly.utils.viz.genome_maps.parse_fastx_file", fail_parse
    )

    stats = load_run_stats(output_dir)
    assert stats["assembly"]["source"] == "contigs_id_map.tsv"
    assert stats["assembly"]["n_contigs"] == 2
    assert stats["assembly"]["total_bp"] == 150
