from pathlib import Path
import polars as pl
from rolypoly.utils.viz.genome_maps import attach_marker_evidence, write_report_for_dir


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
