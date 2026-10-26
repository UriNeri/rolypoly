"""Scientific contracts for accession-based Rfam typing and ontology attributes."""

import polars as pl
import pytest

from rolypoly.utils.bio.polars_fastx import (
    RNA_FEATURE_TYPES,
    annotate_rfam_hits,
    classify_rfam_feature,
    convert_record_to_gff3_record,
    ensure_unified_schema,
)
from rolypoly.utils.viz.genome_maps import build_rna_by_contig, classify_rna_feature


@pytest.mark.parametrize("category,expected,display", [
    ("Gene; rRNA;", "rRNA", "rRNA"),
    ("Gene; tRNA;", "tRNA", "tRNA"),
    ("Cis-reg; riboswitch;", "riboswitch", "riboswitch"),
    ("Cis-reg; IRES;", "internal_ribosome_entry_site", "IRES"),
    ("Cis-reg; frameshift_element;", "cis_regulatory_frameshift_element", "frameshift"),
    ("Cis-reg; thermoregulator;", "RNA_thermometer", "CRE"),
    ("Gene; ribozyme;", "ribozyme", "ribozyme"),
    ("Gene; snRNA; snoRNA; CD-box;", "C_D_box_snoRNA", "ncRNA"),
    ("Gene; snRNA; snoRNA; HACA-box;", "H_ACA_box_snoRNA", "ncRNA"),
    ("Gene; snRNA; snoRNA; scaRNA;", "scaRNA", "ncRNA"),
    ("Gene; snRNA;", "snRNA", "ncRNA"),
    ("Gene; miRNA;", "miRNA", "ncRNA"),
    ("Gene; lncRNA;", "lncRNA", "ncRNA"),
    ("Gene; antisense;", "antisense_RNA", "ncRNA"),
    ("Gene; sRNA;", "ncRNA", "ncRNA"),
    ("Intron;", "intron", "other"),
    ("", "nucleotide_match", "other"),
])
def test_curated_rfam_categories(category, expected, display):
    assert classify_rfam_feature("RF99998", "LSU_rRNA_eukarya", category) == (
        expected, display,
    )
    assert expected in RNA_FEATURE_TYPES


def test_metadata_identity_evidence_and_go_attributes():
    hits = pl.DataFrame({
        "sequence_id": ["contig"], "rfam_id": ["RF99998.1"],
        "profile_name": ["original_model"], "ribozyme_description": ["original evidence"],
        "start": [10], "end": [20], "score": [42.0], "strand": ["-"],
        "evalue": [1e-6], "type": ["ribozyme"],
        "mdl_from": [1], "gc": [0.5],
    })
    metadata = pl.DataFrame({
        "rfam_id": ["RF99998"], "profile_name": ["curated_model"],
        "rfam_type": ["Cis-reg; frameshift_element;"],
        "rfam_description": ["curated description"], "rfam_release": ["test"],
        "metadata_source": ["synthetic:curated"],
        "Ontology_term": ["GO:0003723"],
        "score": [9999.0],
    })
    typed = annotate_rfam_hits(hits, metadata)
    assert typed["type"].to_list() == ["cis_regulatory_frameshift_element"]
    assert typed["rna_class"].to_list() == ["frameshift"]
    for column in set(hits.columns) - {"type"}:
        assert typed[column].equals(hits[column])
    row = typed.row(0, named=True)
    gff = convert_record_to_gff3_record(row).split("\t")
    assert gff[2] == "cis_regulatory_frameshift_element"
    assert "Ontology_term=GO:0003723" in gff[8]
    stacked = ensure_unified_schema([
        ("rna", typed),
        ("trna", hits.drop("rfam_id", "ribozyme_description", "mdl_from", "gc")),
    ])
    assert pl.concat(stacked).height == 2
    assert stacked[0]["Ontology_term"].to_list() == ["GO:0003723"]
    assert stacked[1]["mdl_from"].to_list() == [None]
    assert build_rna_by_contig(typed)["contig"]["features"][0]["klass"] == "frameshift"


def test_legacy_and_unknown_report_types():
    assert classify_rna_feature("ribozyme", "LSU_rRNA_eukarya") == "rRNA"
    assert classify_rna_feature("ribozyme", "Hammerhead_3") == "ribozyme"
    assert classify_rna_feature("ribozyme", "unknown", "hairpin ribosomal IRES") == "other"
    assert classify_rna_feature("nucleotide_match", "unknown", "ribozyme") == "other"
    assert classify_rfam_feature("RF99999", "tRNA") == ("nucleotide_match", "other")


def test_invalid_metadata_is_an_error():
    hits = pl.DataFrame({"rfam_id": ["RF00005"], "profile_name": ["tRNA"]})
    with pytest.raises(ValueError, match="missing columns"):
        annotate_rfam_hits(hits, pl.DataFrame({"rfam_id": ["RF00005"]}))


@pytest.mark.parametrize("mode", [
    "simple", "none", "one_per_range", "one_per_query", "drop_contained", "split", "merge",
])
def test_rna_typing_preserves_overlap_selection(tmp_path, mode):
    from rolypoly.commands.annotation.annotate_RNA import (
        RNAAnnotationConfig,
        process_ribozymes_data,
        resolve_rna_element_overlaps,
    )
    from rolypoly.utils.bio.interval_ops import consolidate_hits

    config = RNAAnnotationConfig(
        input=tmp_path / "input.fna", output_dir=tmp_path, threads=1,
        log_file=None, log_level="WARNING", memory="1g", resolve_mode=mode,
    )
    config.rfam_metadata = None
    config.rfam_metadata_loaded = True
    raw = tmp_path / "ribozymes.out"
    raw.write_text(
        "LSU_rRNA_eukarya RF02543 contig - cm 1 80 1 80 + no 1 0.5 0 80 1e-20 ! LSU\n"
        "Hammerhead_3 RF00008 contig - cm 1 60 20 79 + no 1 0.5 0 40 1e-8 ! Hammerhead\n"
        "unknown RF99999 contig - cm 1 60 100 159 + no 1 0.5 0 30 1e-6 ! unknown\n"
    )
    original = process_ribozymes_data(config, raw).with_columns(
        pl.lit("ribozyme").alias("type")
    )
    if mode == "none":
        expected = original
    else:
        options = {
            "merge": False, "split": False, "one_per_range": False,
            "one_per_query": False, "drop_contained": False,
        }
        options["one_per_range" if mode == "simple" else mode] = True
        expected = consolidate_hits(
            input=original,
            column_specs="sequence_id,profile_name" if mode == "simple" else "sequence_id,type",
            rank_columns="-score,+evalue", min_overlap_positions=10,
            alphabet="nucl", adaptive_overlap=False if mode == "simple" else True,
            **options,
        )
    resolve_rna_element_overlaps(config)
    resolved = pl.read_csv(tmp_path / "ribozymes_resolved.out", separator="\t")
    columns = ["sequence_id", "start", "end", "score", "rfam_id", "strand"]
    assert resolved.select(columns).sort("start").equals(expected.select(columns).sort("start"))
    assert set(resolved["type"]).issubset(RNA_FEATURE_TYPES)
