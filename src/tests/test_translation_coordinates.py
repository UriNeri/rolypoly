"""Coordinate contracts across translation methods and protein-search schemas."""
from types import SimpleNamespace
from urllib.parse import unquote

import polars as pl
import pytest

from rolypoly.utils.bio.interval_ops import (
    amino_to_nucleotide, nucleotide_to_amino, normalize_oriented_interval,
)
from rolypoly.utils.bio.polars_fastx import enrich_protein_coordinates
from rolypoly.utils.bio.translation import build_translation_metadata


@pytest.mark.parametrize('strand', [1, -1])
@pytest.mark.parametrize('aa', [(1, 1), (2, 5), (1, 10)])
def test_codon_projection_roundtrip(strand, aa):
    nt = amino_to_nucleotide(*aa, 11, 40, strand)
    assert nt[1] - nt[0] + 1 == (aa[1] - aa[0] + 1) * 3
    assert nucleotide_to_amino(*nt, 11, 40, strand) == aa


def test_orientation_is_not_guessed_from_sorted_bounds():
    assert normalize_oriented_interval(10, 30) == (10, 30, None)
    assert normalize_oriented_interval(10, 30, '-') == (10, 30, -1)
    assert normalize_oriented_interval(30, 10, descending_encodes_strand=True) == (10, 30, -1)
    with pytest.raises(ValueError, match='conflicts'):
        normalize_oriented_interval(30, 10, '+', descending_encodes_strand=True)
    with pytest.raises(ValueError, match='complete codons'):
        nucleotide_to_amino(12, 20, 11, 40, 1)
    with pytest.raises(ValueError):
        amino_to_nucleotide(10, 11, 11, 40, 1)


@pytest.mark.parametrize('length', [30, 31, 32])
def test_all_six_frames_respect_contig_ends(tmp_path, length):
    dna, proteins = tmp_path / 'dna.fa', tmp_path / 'predicted_orfs.faa'
    dna.write_text('>contig_with_underscores\n' + 'A' * length + '\n')
    proteins.write_text(''.join(
        f'>contig_with_underscores_frame={f}\n' + 'K' * ((length - abs(f) + 1) // 3) + '\n'
        for f in [1, 2, 3, -1, -2, -3]
    ))
    metadata = build_translation_metadata(dna, proteins, 'six-frame')
    for row in metadata.iter_rows(named=True):
        frame = row['frame_id']
        assert row['source_seq_id'] == 'contig_with_underscores'
        assert row['orf_nt_start'] is None
        assert row['orf_nt_end'] is None
        if frame > 0:
            assert row['translation_nt_start'] == frame
        else:
            assert row['translation_nt_end'] == length - abs(frame) + 1
        assert amino_to_nucleotide(1, row['translation_length_aa'], row['translation_nt_start'], row['translation_nt_end'], row['strand']) == (row['translation_nt_start'], row['translation_nt_end'])


@pytest.mark.parametrize('method,header', [
    ('pyrodigal', 'contig_1 # 11 # 40 # -1 # ID=1_1'),
    ('ORFfinder', 'lcl|ORF1_contig:39:10 unnamed protein product'),
])
@pytest.mark.parametrize('backend', ['diamond', 'mmseqs2', 'hmmsearch'])
def test_search_hits_use_query_origin_not_subject_coordinates(tmp_path, method, header, backend):
    dna, proteins = tmp_path / 'dna.fa', tmp_path / 'predicted_orfs.faa'
    dna.write_text('>contig\n' + 'A' * 50 + '\n')
    proteins.write_text(f'>{header}\nMAKPEPTID\n')
    meta = build_translation_metadata(dna, proteins, method)
    query = header.split()[0]
    if backend == 'hmmsearch':
        hits = pl.DataFrame({'query_full_name': [header], 'q1': [2], 'q2': [4], 'hmm_from': [20], 'hmm_to': [22]})
    else:
        hits = pl.DataFrame({'sequence_id': [query], 'start': [2], 'end': [4], 'sstart': [20], 'send': [22]})
    enriched = enrich_protein_coordinates(hits, meta)
    row = enriched.row(0, named=True)
    assert (row['nt_start'], row['nt_end'], row['strand']) == (29, 37, -1)
    assert row['source_seq_id'] == 'contig'
    assert row['orf_nt_start'] == 11
    assert row['translation_nt_start'] == 14  # terminal stop omitted


def test_gff_mapping_and_export_escape_metadata(tmp_path):
    from rolypoly.commands.annotation.annotate_prot import write_combined_results_to_gff
    from rolypoly.utils.logging.loggit import get_logger
    dna, proteins = tmp_path / 'dna.fa', tmp_path / 'predicted_orfs.faa'
    dna.write_text('>contig\n' + 'A' * 50 + '\n')
    proteins.write_text('>arbitrary_id\nMAKPEPTID\n')
    proteins.with_suffix('.gff').write_text('##gff-version 3\ncontig\tcaller\tCDS\t11\t40\t.\t-\t0\tID=arbitrary_id\n')
    meta = build_translation_metadata(dna, proteins, 'pyrodigal')
    hits = pl.DataFrame({'sequence_id': ['arbitrary_id'], 'start': [2], 'end': [4], 'description': ['name;other=x,y\tline\nnext%']})
    enriched = enrich_protein_coordinates(hits, meta)
    write_combined_results_to_gff(SimpleNamespace(output_dir=tmp_path, input=dna, logger=get_logger()), enriched)
    text = (tmp_path / 'combined_annotations.gff3').read_text()
    record = text.splitlines()[1].split('\t')
    assert len(record) == 9
    assert record[:1] == ['contig']
    assert record[3:5] == ['29', '37']
    assert record[6:8] == ['-', '.']
    attrs = dict(item.split('=', 1) for item in record[8].split(';'))
    assert unquote(attrs['description']) == hits['description'][0]
    assert text.split('##FASTA', 1)[1].lstrip().startswith('>contig')


def test_protein_only_does_not_invent_genomic_mapping(tmp_path):
    proteins = tmp_path / 'input.faa'
    proteins.write_text('>contig_1 # 11 # 40 # -1\nMAKPEPTID\n')
    meta = build_translation_metadata(proteins, proteins, 'input_protein')
    hits = pl.DataFrame({'sequence_id': ['contig_1'], 'start': [1], 'end': [4]})
    row = enrich_protein_coordinates(hits, meta).row(0, named=True)
    assert row['nt_start'] is None and row['strand'] is None and row['source_seq_id'] == 'contig_1'


def test_mmseqs_local_id_alias_preserves_search_id(tmp_path):
    dna, proteins = tmp_path / 'dna.fa', tmp_path / 'predicted_orfs.faa'
    dna.write_text('>contig\n' + 'A' * 50 + '\n')
    proteins.write_text('>lcl|ORF1_contig:39:10 unnamed protein product\nMAKPEPTID\n')
    meta = build_translation_metadata(dna, proteins, 'ORFfinder')
    hits = pl.DataFrame({'sequence_id': ['ORF1_contig:39:10'], 'start': [1], 'end': [9]})
    row = enrich_protein_coordinates(hits, meta).row(0, named=True)
    assert row['translation_id'] == 'lcl|ORF1_contig:39:10'
    assert row['search_query_id'] == 'ORF1_contig:39:10'
    assert row['nt_start'] == 14 and row['nt_end'] == 40


def test_empty_predictions_produce_typed_metadata(tmp_path):
    dna, proteins = tmp_path / 'dna.fa', tmp_path / 'predicted_orfs.faa'
    dna.write_text('>contig\nAAAAAA\n')
    proteins.write_text('')
    proteins.with_suffix('.gff').write_text('##gff-version 3\n')
    meta = build_translation_metadata(dna, proteins, 'pyrodigal')
    assert meta.is_empty()
    assert meta.schema['strand'] == pl.Int64


@pytest.mark.parametrize("backend", ["diamond", "hmmsearch"])
def test_report_prefers_explicit_six_frame_coordinates(tmp_path, backend):
    import json
    import re
    from rolypoly.utils.viz.genome_maps import write_genome_maps
    dna, proteins = tmp_path / 'dna.fa', tmp_path / 'predicted_orfs.faa'
    dna.write_text('>contig\n' + 'A' * 32 + '\n')
    proteins.write_text('>contig_frame=-2\nKKKKKKKKKK\n')
    meta = build_translation_metadata(dna, proteins, 'six-frame')
    hits = pl.DataFrame({'sequence_id': ['contig_frame=-2'], 'start': [2], 'end': [4], 'sseqid': ['profile'], 'source': ['test'], 'qlen': [10]})
    if backend == "hmmsearch":
        hits = hits.rename({"sequence_id": "query_full_name", "start": "q1", "end": "q2", "sseqid": "hmm_full_name", "source": "database_id"})
    enriched = enrich_protein_coordinates(hits, meta)
    report = write_genome_maps(enriched, tmp_path / 'report.html', mark_best=False)
    payload = json.loads(re.search(r'const DATA=(.*?);\s*\n', report.read_text()).group(1))
    contig = payload['contigs'][0]
    assert contig['contig'] == 'contig'
    assert contig['length'] == 32
    assert contig['query_label'] == 'translated frames'
    translation = contig['orfs'][0]
    assert (translation['start'], translation['end'], translation['strand']) == (2, 31, -1)
    assert (translation['hits'][0]['nt_from'], translation['hits'][0]['nt_to']) == (20, 28)


def test_unmapped_protein_report_shows_table_without_false_genome(tmp_path):
    import json
    import re
    from rolypoly.utils.viz.genome_maps import write_genome_maps
    proteins = tmp_path / 'input.faa'
    proteins.write_text('>protein\nMAKPEPTID\n')
    meta = build_translation_metadata(proteins, proteins, 'input_protein')
    hits = pl.DataFrame({'sequence_id': ['protein'], 'start': [1], 'end': [4], 'sseqid': ['profile'], 'source': ['test']})
    report = write_genome_maps(enrich_protein_coordinates(hits, meta), tmp_path / 'report.html')
    payload = json.loads(re.search(r'const DATA=(.*?);\s*\n', report.read_text()).group(1))
    assert payload['contigs'] == []
    assert payload['extra_tabs'][0]['id'] == 'unplaced_proteins'


def test_unresolved_diamond_export_keeps_first_hit(tmp_path, monkeypatch):
    import importlib
    from rolypoly.utils.logging.loggit import get_logger
    module = importlib.import_module('rolypoly.commands.annotation.annotate_prot')
    proteins = tmp_path / 'predicted_orfs.faa'
    proteins.write_text('>protein\nMAKPEPTID\n')
    raw = tmp_path / 'raw.tsv'
    raw.write_text('protein\ttarget\t100\t9\t0\t0\t1\t9\t1\t9\t1e-10\t50\t9\t9\n')
    monkeypatch.setattr(module, 'output_files', pl.DataFrame({
        'file': [str(raw)], 'description': ['protein domains for custom'],
        'db': ['custom'], 'tool': ['diamond'],
    }))
    config = SimpleNamespace(output_dir=tmp_path, output_format='tsv',
        search_tool='diamond', logger=get_logger(), keep_tmp=True,
        translation_metadata=build_translation_metadata(proteins, proteins, 'input_protein'))
    module.combine_results(config)
    result = pl.read_csv(tmp_path / 'combined_annotations.tsv', separator='\t')
    assert result.height == 1
    assert result['translation_id'][0] == 'protein'
    assert result['aa_end'][0] == 9
    assert result['nt_start'][0] is None
