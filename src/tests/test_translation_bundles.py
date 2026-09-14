"""Canonical ID, complete provenance, and guarded translation reuse contracts."""
import json
from pathlib import Path

import polars as pl
import pytest

from rolypoly.utils.bio import translation
from rolypoly.utils.bio.polars_fastx import enrich_protein_coordinates


@pytest.fixture(autouse=True)
def fixed_tool_versions(monkeypatch):
    monkeypatch.setattr(translation, 'translation_signature', lambda method, parameters: {
        'method': method.replace('six_frame', 'six-frame'), 'parameters': parameters, 'versions': {'tool': 'fixture-1'}
    })


def make_bundle(tmp_path, method='pyrodigal'):
    dna = tmp_path / 'input.fa'
    dna.write_text('>CID_4 original contig description\n' + 'A' * 80 + '\n>other\n' + 'T' * 80 + '\n')
    native = tmp_path / 'native.faa'
    if method == 'pyrodigal':
        native.write_text('>CID_4_97 # 41 # 70 # -1 # ID=1_97;partial=01;note=second token\nMAKPEPTIDE\n'
                          '>CID_4_33 # 11 # 40 # 1 # ID=1_33;partial=00\nMAKPEPTIDE\n'
                          '>other_8 # 11 # 40 # 1 # ID=2_8\nMAKPEPTIDE\n')
    else:
        native.write_text('>CID_4_frame=-2\toriginal second token\n' + 'K' * 26 + '\n')
    bundle = tmp_path / 'bundle'
    meta = translation.normalize_translation_output(dna, native, bundle, method, {'minimum_length': 30})
    return dna, native, bundle, meta


def test_normalized_ids_are_position_ordered_and_preserve_complete_headers(tmp_path):
    dna, native, bundle, meta = make_bundle(tmp_path)
    mapping = {r['original_translation_id']: r for r in meta.iter_rows(named=True)}
    assert mapping['CID_4_33']['translation_id'] == 'CID_4_orf_1'
    assert mapping['CID_4_97']['translation_id'] == 'CID_4_orf_2'
    assert mapping['CID_4_97']['translation_label'] == 'CID_4_orf_2'
    assert mapping['CID_4_97']['original_header'].endswith('note=second token')
    assert json.loads(mapping['CID_4_97']['prediction_attributes'])['note'] == 'second token'
    assert mapping['CID_4_97']['original_source_header'] == 'CID_4 original contig description'
    assert (bundle / 'tool_outputs/predicted_orfs.faa').read_bytes() == native.read_bytes()
    assert {h for h, _ in translation.translation_records(bundle / 'predicted_orfs.faa')} == set(meta['translation_id'])
    gff = (bundle / 'predicted_orfs.gff').read_text()
    assert 'ID=CID_4_orf_2' in gff
    assert '\t41\t70\t.\t-\t0\t' in gff
    # Idempotent preparation must not overwrite native provenance with canonical headers.
    again = translation.normalize_translation_output(dna, bundle / 'predicted_orfs.faa', bundle, 'pyrodigal', {'minimum_length': 30})
    assert again.to_dicts() == meta.to_dicts()


def test_frame_ids_preserve_full_tabbed_header_in_report_reader(tmp_path):
    from rolypoly.utils.viz.genome_maps import read_hit_table
    _, native, bundle, meta = make_bundle(tmp_path, 'six-frame')
    row = meta.row(0, named=True)
    assert row['translation_id'] == 'CID_4_frame_m2'
    assert row['translation_label'] == 'CID_4_frame_m2'
    assert row['orf_nt_start'] is None and row['orf_ordinal'] is None
    assert row['original_header'] == next(translation.translation_records(native))[0]
    read = read_hit_table(bundle / 'translation_metadata.tsv')
    assert read['original_header'][0] == row['original_header']
    assert '\ttranslated_region\t' in (bundle / 'predicted_orfs.gff').read_text()


def test_reuse_subset_retains_ids_and_native_mapping(tmp_path):
    dna, _, bundle, meta = make_bundle(tmp_path)
    subset = tmp_path / 'subset.fa'
    subset.write_text('>CID_4 original contig description\n' + 'A' * 80 + '\n')
    reused = translation.reuse_translation_bundle(bundle, subset, tmp_path / 'reused', 'pyrodigal', {'minimum_length': 30})
    assert set(reused['translation_id']) == {'CID_4_orf_1', 'CID_4_orf_2'}
    assert set(reused['original_translation_id']) == {'CID_4_33', 'CID_4_97'}
    assert set(h for h, _ in translation.translation_records(tmp_path / 'reused/predicted_orfs.faa')) == set(reused['translation_id'])


@pytest.mark.parametrize('change', ['sequence', 'header', 'method', 'parameters', 'version', 'tampered'])
def test_reuse_rejects_incompatible_or_modified_bundle(tmp_path, change):
    dna, _, bundle, _ = make_bundle(tmp_path)
    method, params = 'pyrodigal', {'minimum_length': 30}
    if change == 'sequence':
        dna.write_text(dna.read_text().replace('AAAA', 'CCCC', 1))
    elif change == 'header':
        dna.write_text(dna.read_text().replace('original contig', 'different contig'))
    elif change == 'method':
        method = 'six-frame'
    elif change == 'parameters':
        params = {'minimum_length': 60}
    elif change == 'version':
        path = bundle / 'translation_manifest.json'
        manifest = json.loads(path.read_text()); manifest['signature']['versions'] = {'tool': 'fixture-2'}
        path.write_text(json.dumps(manifest))
    else:
        with (bundle / 'predicted_orfs.faa').open('a') as handle:
            handle.write('A\n')
    with pytest.raises(ValueError, match='reuse rejected'):
        translation.reuse_translation_bundle(bundle, dna, tmp_path / 'reused', method, params)
    assert not (tmp_path / 'reused').exists()


def test_old_hit_ids_can_be_migrated_without_losing_provenance(tmp_path):
    _, _, _, meta = make_bundle(tmp_path)
    old = pl.DataFrame({'sequence_id': ['CID_4_97'], 'start': [1], 'end': [5]})
    row = enrich_protein_coordinates(old, meta, allow_original_ids=True).row(0, named=True)
    assert row['translation_id'] == 'CID_4_orf_2'
    assert row['search_query_id'] == 'CID_4_97'
    assert row['original_header'].endswith('note=second token')


def test_empty_bundle_is_valid(tmp_path):
    dna, native = tmp_path / 'dna.fa', tmp_path / 'native.faa'
    dna.write_text('>no_genes\nAAAAAA\n'); native.write_text('')
    meta = translation.normalize_translation_output(dna, native, tmp_path / 'bundle', 'pyrodigal', {'minimum_length': 30})
    assert meta.is_empty()
    assert (tmp_path / 'bundle/predicted_orfs.gff').read_text() == '##gff-version 3\n'


def test_canonical_search_does_not_guess_using_colliding_native_ids(tmp_path):
    source = tmp_path / 'proteins.fa'
    source.write_text('>protein\nMAKPEPTIDE\n>protein_protein_1 second description\nMAKPEPTIDE\n')
    meta = translation.normalize_translation_output(source, source, tmp_path / 'bundle', 'input_protein', {})
    hits = pl.DataFrame({'sequence_id': ['protein_protein_1'], 'start': [1], 'end': [3]})
    row = enrich_protein_coordinates(hits, meta).row(0, named=True)
    assert row['original_translation_id'] == 'protein'
    with pytest.raises(ValueError, match='Ambiguous'):
        enrich_protein_coordinates(hits, meta, allow_original_ids=True)


def test_matched_input_export_uses_mapping_not_normalized_id_suffixes(tmp_path):
    from rolypoly.commands.identify_virus.marker_search import write_matched_input_seqs_fasta
    dna, _, bundle, meta = make_bundle(tmp_path)
    hits = enrich_protein_coordinates(pl.DataFrame({
        'query_full_name': ['CID_4_orf_2'], 'q1': [1], 'q2': [3],
    }), meta)
    output = tmp_path / 'matched.fna'
    write_matched_input_seqs_fasta(hits, str(dna), output, 'nucl', 'pyrodigal')
    assert len(list(translation.translation_records(output))) == 1
    assert next(translation.translation_records(output))[0] == 'CID_4 original contig description'
