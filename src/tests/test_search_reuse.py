import json
import logging

import polars as pl
import pytest

from rolypoly.utils.bio import search_reuse as reuse


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(reuse, 'hmm_signature', lambda db, params: {
        'tool': 'hmmsearch', 'version': 'test',
        'database_sha256': reuse.translation_file_digest(db), 'parameters': params,
    })
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.mkdir(); target.mkdir()
    manifest = {'signature': {'method': 'six-frame', 'versions': {'implementation': 'test'},
                              'parameters': {}},
                'inputs': {'CID': {'header': 'CID original description', 'sha256': 'dna'}}}
    for directory in (source, target):
        (directory / 'translation_manifest.json').write_text(json.dumps(manifest))
        (directory / 'predicted_orfs.faa').write_text('>CID_frame_p1\nMKK\n>CID_frame_m1\nMLL\n')
    db = tmp_path / 'markers.hmm'; db.write_text('profile contents')
    raw = tmp_path / 'raw.tsv'
    pl.DataFrame({'query_full_name': ['CID_frame_p1 ', 'CID_frame_m1 '],
                  'hmm_full_name': ['marker', 'marker'], 'this_dom_score': [40., 30.]}).write_csv(raw, separator='\t')
    return source, target, db, raw


def attempt(bundle, parameters=None, fields=None):
    source, target, db, _ = bundle
    return reuse.reuse_hmm_search(source, target, db, target / 'hits.tsv',
                                  parameters or {}, fields or [], logging.getLogger(__name__))


def test_identical_search_reused_before_resolution(bundle):
    source, target, db, raw = bundle
    reuse.save_hmm_search(source, db, raw, {}, ['alignment_strings'])
    assert attempt(bundle, fields=['alignment_strings'])
    assert pl.read_csv(target / 'hits.tsv', separator='\t').height == 2
    assert (target / 'hits.tsv.reuse.json').exists()


@pytest.mark.parametrize('change', ['database', 'parameters', 'fields', 'sequence', 'header', 'version', 'result'])
def test_mismatch_runs_fresh_search(bundle, change):
    source, target, db, raw = bundle
    reuse.save_hmm_search(source, db, raw, {}, [])
    parameters, fields = {}, []
    if change == 'database': db.write_text('changed profile')
    if change == 'parameters': parameters = {'inc_e': 0.1}
    if change == 'fields': fields = ['alignment_strings']
    if change == 'sequence': (target / 'predicted_orfs.faa').write_text('>CID_frame_p1\nMNN\n')
    if change in ('header', 'version'):
        path = target / 'translation_manifest.json'
        data = json.loads(path.read_text())
        if change == 'header': data['inputs']['CID']['header'] = 'CID changed description'
        else: data['signature']['versions']['implementation'] = 'changed'
        path.write_text(json.dumps(data))
    if change == 'result': next((source / 'search_cache').glob('*/hits.tsv')).write_text('tampered')
    assert not attempt(bundle, parameters, fields)
    assert not (target / 'hits.tsv').exists()


def test_subset_requires_fixed_search_spaces(bundle):
    source, target, db, raw = bundle
    reuse.save_hmm_search(source, db, raw, {}, [])
    (target / 'predicted_orfs.faa').write_text('>CID_frame_p1\nMKK\n')
    assert not attempt(bundle)
    params = {'Z': 10, 'domZ': 10}
    reuse.save_hmm_search(source, db, raw, params, [])
    assert attempt(bundle, params)
    assert pl.read_csv(target / 'hits.tsv', separator='\t')['query_full_name'].to_list() == ['CID_frame_p1 ']


def test_nucleotide_superset_with_identical_proteins_is_reusable(bundle):
    source, target, db, raw = bundle
    path = source / 'translation_manifest.json'
    data = json.loads(path.read_text())
    data['inputs']['no_proteins'] = {'header': 'no_proteins', 'sha256': 'other_dna'}
    path.write_text(json.dumps(data))
    reuse.save_hmm_search(source, db, raw, {}, [])
    assert attempt(bundle)


def test_empty_results_are_reusable(bundle):
    source, target, db, raw = bundle
    pl.DataFrame(schema={'query_full_name': pl.String, 'hmm_full_name': pl.String}).write_csv(raw, separator='\t')
    reuse.save_hmm_search(source, db, raw, {}, [])
    assert attempt(bundle)
    assert pl.read_csv(target / 'hits.tsv', separator='\t').is_empty()
