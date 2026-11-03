from pathlib import Path
import polars as pl
from rolypoly.utils.viz.genome_maps import (
    attach_marker_evidence,
    attach_projected_marker_motifs,
    write_report_for_dir,
)


def marker_table(tmp_path, **changes):
    row = dict(query_full_name='CID_frame=-2 ', source_seq_id='CID', frame_id=-2,
        hmm_full_name='RVMT_marker', database_id='RVMT', qlen=33, hmm_len=10,
        full_hmm_evalue=1e-12, full_hmm_score=50., hmm_cov=1., ali_len=10,
        q1=2, q2=11, env_from=1, env_to=12, hmm_from=1, hmm_to=10)
    row.update(changes)
    path=tmp_path/'marker_search_results.tsv'
    pl.DataFrame([row]).write_csv(path, separator='\t')
    return path


def test_legacy_reverse_frame_uses_aligned_span_and_retains_scores(tmp_path):
    path=marker_table(tmp_path)
    tabs=[]
    models=attach_marker_evidence([], [path], tabs, {'CID': 101})
    orf=models[0]['orfs'][0]; hit=orf['hits'][0]
    assert (hit['nt_from'],hit['nt_to']) == (68,97)
    assert hit['evalue'] == 1e-12
    assert orf['orf_id'] == 'CID_frame=-2'
    assert orf['evidence_stage'] == 'marker-search'
    assert not tabs
    models[0]['orfs'][0].pop('evidence_stage')  # An annotation query with the same ID.
    attach_marker_evidence(models,[path],tabs,{'CID':101})
    assert len(models[0]['orfs']) == 2  # Separate evidence, never grouped by ID.


def test_unmapped_gene_is_not_guessed_or_labelled_six_frame(tmp_path):
    path=marker_table(tmp_path, query_full_name='CID_orf_1', frame_id=None)
    tabs=[]
    assert not attach_marker_evidence([], [path], tabs, {'CID':101})
    assert tabs


def test_marker_only_report_discovers_explicit_gene_coordinates(tmp_path):
    marker_table(tmp_path, query_full_name='CID_orf_1', translation_id='CID_orf_1',
        translation_method='pyrodigal', translation_nt_start=5, translation_nt_end=100,
        nt_start=8, nt_end=37, aa_start=2, aa_end=11, strand=1, contig_length=101)
    result=write_report_for_dir(tmp_path,with_stats=False)
    text=Path(result).read_text()
    assert 'marker-search' in text and 'CID_orf_1' in text
    assert 'Original marker-search hits' in text
    assert '"translation_method": "pyrodigal"' in text


def test_supporting_marker_grouping_preserves_extensions_and_reading_phase():
    from copy import deepcopy
    from rolypoly.utils.viz.genome_maps import mark_supporting_marker_hits
    base = dict(source='RVMT', profile='RdRp', nt_from=10, nt_to=99, hmm_from=1, hmm_to=30, score=50)
    annotation = dict(orf_id='CID_orf_1', strand=1, hits=[base])
    marker = dict(orf_id='CID_frame_p1', strand=1, evidence_stage='marker-search', hits=[deepcopy(base)])
    marker['hits'][0]['score'] = 40
    extension=deepcopy(marker); extension['hits'][0]['nt_to']=102
    phase=deepcopy(marker); phase['hits'][0]['nt_from']=11
    other_profile=deepcopy(marker); other_profile['hits'][0]['profile']='other'
    reverse=deepcopy(marker); reverse['strand']=-1
    contigs=[dict(orfs=[annotation,marker,extension,phase,other_profile,reverse])]
    mark_supporting_marker_hits(contigs)
    assert marker['hits'][0]['supporting_annotation']=='CID_orf_1'
    assert marker['hits'][0]['score']==40
    assert annotation['hits'][0]['marker_support']==['CID_frame_p1']
    for o in (extension,phase,other_profile,reverse):
        assert 'supporting_annotation' not in o['hits'][0]


def test_marker_matched_sequence_is_embedded_without_alignment_gaps(tmp_path):
    path=marker_table(tmp_path, aligned_region='Ma-k.K')
    models=attach_marker_evidence([], [path], [], {'CID':101})
    assert models[0]['orfs'][0]['hits'][0]['matched_sequence']=='MAKK'


def test_explicit_marker_coordinates_are_parsed_before_sequence_slice(tmp_path):
    path=marker_table(tmp_path, query_full_name='CID_orf_1', translation_id='CID_orf_1',
        translation_method='pyrodigal', translation_nt_start=1, translation_nt_end=30,
        nt_start=4, nt_end=15, aa_start='2', aa_end='5', strand=1, contig_length=30)
    (tmp_path/'predicted_orfs.faa').write_text('>CID_orf_1\nMAKLMNQ\n')

    models=attach_marker_evidence([], [path], [], {'CID':30})

    assert models[0]['orfs'][0]['hits'][0]['matched_sequence']=='AKLM'


def test_projected_motifs_use_marker_path_and_reverse_translation(tmp_path):
    marker = marker_table(
        tmp_path, marker_path_id=1, source_seq_id="CID", frame_id=-2,
        translation_nt_start=2, translation_nt_end=100, strand=-1,
        nt_start=68, nt_end=97, aligned_region="ABCDEFGHIJ",
    )
    pl.DataFrame({
        "marker_path_id": [1, 1, 1],
        "canonical_term_id": ["modern_structural:C", "modern_structural:A", "modern_structural:B"],
        "projected_query_positions_1based": ["3;5", "6;7", None],
        "projected_query_residues": ["GD", "DD", None],
        "projection_status": ["complete", "partial", "absent"],
    }).write_csv(tmp_path / "marker_evidence.tsv", separator="\t")
    contigs = [{"contig": "CID", "length": 101, "motifs": None}]
    attach_projected_marker_motifs(contigs, [marker])
    motifs = contigs[0]["motifs"]
    assert motifs["conformation"] == "A,C"
    by_letter = {m["letter"]: m for m in motifs["motifs"]}
    assert (by_letter["C"]["nt_from"], by_letter["C"]["nt_to"]) == (86, 94)
    assert (by_letter["C"]["context_nt_from"], by_letter["C"]["context_nt_to"]) == (68, 97)
    assert by_letter["C"]["projected_positions"] == "3;5"
    assert by_letter["C"]["alignment"] == "ABCDEFGHIJ"
    assert by_letter["A"]["projection_status"] == "partial"

    from rolypoly.utils.bio.motif_gff import write_motif_gff
    gff = tmp_path / "motifs.gff3"
    write_motif_gff(contigs, gff)
    lines = gff.read_text().splitlines()
    assert lines[0] == "##gff-version 3"
    assert any("\tprotein_match\t68\t97\t" in line for line in lines)
    assert any("\tpolypeptide_motif\t86\t94\t" in line and "query_aa_positions=3%3B5" in line for line in lines)
    assert any("projection_status=partial" in line for line in lines)


def test_flat_motif_table_preserves_projection_and_context(tmp_path):
    from rolypoly.utils.viz.genome_maps import build_motifs_by_contig, attach_motifs

    path = tmp_path / "flat.tsv"
    pl.DataFrame({
        "sequence_id": ["CID_frame=-2"], "motif_conformation": ["C"],
        "motif_a_start": [None], "motif_c_start": [3], "motif_c_end": [9],
        "motif_c_score": [25.0], "motif_c_evalue": [1e-5], "motif_c_profile": ["C1"],
        "motif_c_alignment": ["ABCDEFG"], "motif_c_feature_from": [5],
        "motif_c_feature_to": [6], "motif_c_projected_positions": ["5;6"],
        "motif_c_projected_residues": ["GD"], "motif_c_projection_status": ["complete"],
    }).write_csv(path, separator="\t")
    contigs = [{"contig": "CID", "length": 101, "motifs": None}]
    attach_motifs(contigs, build_motifs_by_contig(path))
    motif = contigs[0]["motifs"]["motifs"][0]
    assert (motif["nt_from"], motif["nt_to"]) == (83, 88)
    assert (motif["context_nt_from"], motif["context_nt_to"]) == (74, 94)
    assert motif["projected_residues"] == "GD"


def test_directory_report_exports_marker_feature_gff(tmp_path):
    marker_table(
        tmp_path, marker_path_id=1, source_seq_id="CID", frame_id=-2,
        translation_nt_start=2, translation_nt_end=100, strand=-1,
        nt_start=68, nt_end=97, aligned_region="ABCDEFGHIJ", contig_length=101,
    )
    pl.DataFrame({
        "marker_path_id": [1], "canonical_term_id": ["modern_structural:C"],
        "projected_query_positions_1based": ["3;5"],
        "projected_query_residues": ["GD"], "projection_status": ["complete"],
    }).write_csv(tmp_path / "marker_evidence.tsv", separator="\t")
    report = write_report_for_dir(tmp_path, with_stats=False)
    assert '"source": "marker projection"' in report.read_text()
    assert "\tpolypeptide_motif\t86\t94\t" in (tmp_path / "genome_maps_motifs.gff3").read_text()


def test_changed_alignment_does_not_inherit_stale_projection(tmp_path, caplog):
    marker = marker_table(
        tmp_path, marker_path_id=1, source_seq_id="CID", frame_id=-2,
        translation_nt_start=2, translation_nt_end=100, strand=-1,
        nt_start=68, nt_end=97,
    )
    pl.DataFrame({
        "marker_path_id": [1], "canonical_term_id": ["modern_structural:C"],
        "projected_query_positions_1based": ["3;5"],
        "projected_query_residues": ["GD"], "projection_status": ["complete"],
        "q1": [3], "q2": [11],
    }).write_csv(tmp_path / "marker_evidence.tsv", separator="\t")
    contigs = [{"contig": "CID", "length": 101, "motifs": None}]
    attach_projected_marker_motifs(contigs, [marker])
    assert contigs[0]["motifs"] is None
    assert "coordinates differ" in caplog.text
