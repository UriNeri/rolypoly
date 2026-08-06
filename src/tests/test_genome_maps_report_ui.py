import polars as pl

from rolypoly.utils.viz.genome_maps import (
    build_report_file_catalog,
    load_original_contig_ids,
    render_html,
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
        command_line="rolypoly roll -i input.fasta -o output",
        log_text="pipeline started\npipeline finished",
    )

    assert "▤ Contig Table" in html
    assert "Show raw contig ID" in html
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
