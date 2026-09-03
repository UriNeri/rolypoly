import polars as pl

from rolypoly.utils.viz.genome_maps import (
    build_report_file_catalog,
    load_contig_lengths,
    load_original_contig_ids,
    render_html,
    write_report_for_dir,
)


def test_report_ui_scopes_controls_and_embeds_provenance(tmp_path):
    mapping_path = tmp_path / "contigs_id_map.tsv"
    pl.DataFrame(
        {"old_id": ["assembler_contig_42"], "new_id": ["CID_01"]}
    ).write_csv(mapping_path, separator="\t")
    original_ids = load_original_contig_ids(mapping_path)
    contig = {
        "contig": "CID_01",
        "raw_id": original_ids["CID_01"],
        "short": "CID_01",
        "length": 100,
        "orfs": [],
        "n_orfs": 0,
        "n_hits": 0,
        "n_best": {"score": 0},
        "n_source": 0,
        "sources": [],
        "best_score": 0,
        "top_profile": "",
        "rna": None,
        "nucleic": None,
        "motifs": None,
    }

    html = render_html(
        [contig],
        nucleic={"__all__": [{"contig": "CID_01"}]},
        extra_tabs=[
            {
                "id": "taxonomy",
                "label": "Taxonomy",
                "columns": ["query", "family"],
                "rows": [["CID_01", "Exampleviridae"]],
            }
        ],
        command_line="rolypoly roll -i input.fasta -o output",
        log_text="pipeline started\npipeline finished",
    )

    assert "▤ Contig Table" in html
    assert html.count("Show original contig IDs") >= 3
    assert "tabHasContigColumn(t)" in html
    assert "renderExtraCell(t.columns[i],v)" in html
    assert "showRawIds&&c&&c.raw_id?c.raw_id:h.contig" in html
    assert "name==='maps'?'flex':'none'" in html
    assert '"command_line": "rolypoly roll -i input.fasta -o output"' in html
    assert '"raw_id": "assembler_contig_42"' in html
    assert "▧ Log" in html
    assert "pipeline started\\npipeline finished" in html
    assert "Export shown TSV" in html
    assert "Load referenced FASTA" in html
    assert "Choose FASTA" in html
    assert "Sequences are not embedded in this report" in html


def test_original_id_mapping_omits_inert_cid_to_same_cid_entries(tmp_path):
    mapping_path = tmp_path / "contigs_id_map.tsv"
    pl.DataFrame(
        {
            "old_id": ["CID_01", "assembler_contig_42"],
            "new_id": ["CID_01", "CID_02"],
        }
    ).write_csv(mapping_path, separator="\t")

    assert load_original_contig_ids(mapping_path) == {
        "CID_02": "assembler_contig_42"
    }


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


def test_report_file_catalog_uses_relative_external_paths(tmp_path):
    output_dir = tmp_path / "run"
    protein_dir = output_dir / "annotation_results" / "protein_annotation"
    protein_dir.mkdir(parents=True)
    report = output_dir / "genome_maps.html"
    marker_table = protein_dir / "combined_annotations.tsv"
    nucleic_table = output_dir / "results_vs_reference.tab"
    orfs = protein_dir / "predicted_orfs.faa"
    contigs = output_dir / "all_matched_contigs.fasta"
    marker_table.write_text("query\tprofile\n")
    nucleic_table.write_text("query\ttarget\n")
    orfs.write_text(">CID_01_1\nMPEPTIDE\n")
    contigs.write_text(">CID_01\nACGT\n")

    catalog = build_report_file_catalog(
        output_dir,
        report,
        marker_table=marker_table,
        nucleic_tables=[("reference", nucleic_table)],
    )

    assert catalog["tables"] == [
        {
            "label": "Original protein-hit table",
            "kind": "protein",
            "path": "annotation_results/protein_annotation/combined_annotations.tsv",
        },
        {
            "label": "Original nucleic-hit table — reference",
            "kind": "nucleic",
            "path": "results_vs_reference.tab",
        },
    ]
    assert {item["kind"] for item in catalog["fastas"]} == {"contigs", "orfs"}
    assert all(not item["path"].startswith("/") for item in catalog["fastas"])
