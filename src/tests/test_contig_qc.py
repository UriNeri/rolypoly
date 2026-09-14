from pathlib import Path
from types import SimpleNamespace
import polars as pl
import pytest

from rolypoly.commands.assembly.filter_contigs import host_filter_evidence, rrna_removal_candidates, write_filter_evidence
from rolypoly.utils.bio.sequences import retain_contigs
from rolypoly.utils.viz.genome_maps import attach_filter_warnings


def test_rrna_union_coverage_does_not_double_count():
    rows=[dict(contig_id='c',contig_length=100,start=1,end=50),dict(contig_id='c',contig_length=100,start=30,end=60)]
    assert rrna_removal_candidates(rows,.8)==set()
    rows.append(dict(contig_id='c',contig_length=100,start=61,end=80))
    assert rrna_removal_candidates(rows,.8)=={'c'}


@pytest.mark.parametrize('flag_only', [False, True])
@pytest.mark.parametrize('kind', ['host_nucleotide','host_protein'])
def test_host_rules_capture_reverse_nucleotide_coordinates_and_preserve_headers(tmp_path, flag_only, kind):
    config=SimpleNamespace(flag_only=flag_only, filter_evidence=[],filter1_nuc='first',filter2_nuc='second',filter1_aa='first',filter2_aa='second')
    qcol='qheader' if kind=='host_nucleotide' else 'qtitle'
    hit=pl.DataFrame({qcol:['c full original description'],'qlen':[100],'qstart':[90],'qend':[31],'evalue':[1e-20],'bitscore':[50.],'sseqid':['host']})
    removed=host_filter_evidence(config,hit,hit.head(0),kind)
    row=config.filter_evidence[0]
    assert (row['start'],row['end'],row['strand'])==(31,90,'-')
    assert row['action']==('flagged' if flag_only else 'removed')
    fasta=tmp_path/'input.fa'; fasta.write_text('>c full original description\nACGT\n>unmatched kept description\nAAAA\n')
    out=tmp_path/'out.fa'
    retain_contigs(fasta,out,() if flag_only else removed)
    assert ('>c full original description' in out.read_text()) == flag_only
    assert '>unmatched kept description' in out.read_text()


def test_report_combines_rrna_warning_provenance(tmp_path):
    row=dict(contig_id='c',contig_length=100,start=10,end=80,strand='-',kind='rRNA',profile='5S_rRNA',accession='RF00001',source='cmscan rrna.cm',score=50.,evalue=1e-12,rule='cut_ga',action='flagged',description='rRNA')
    out=tmp_path/'out.fa';write_filter_evidence([row],out)
    feature=dict(klass='rRNA',profile='5S_rRNA',strand='-',start=10,end=80,note='annotate-rna')
    contigs=[dict(contig='c',orfs=[],rna=dict(features=[feature],structure=None))]
    attach_filter_warnings(contigs,[str(out)+'.filter_hits.tsv'])
    assert len(contigs[0]['rna']['features'])==1
    assert feature['filter_evidence'][0]['source']=='cmscan rrna.cm'
    assert 'annotate-rna' in feature['note'] and 'filter-contigs' in feature['note']


def test_report_does_not_reintroduce_discarded_contigs(tmp_path):
    row=dict(contig_id='gone',contig_length=100,start=10,end=80,strand='-',kind='rRNA',profile='5S_rRNA',accession='RF00001',source='cmscan rrna.cm',score=50.,evalue=1e-12,rule='cut_ga',action='removed',description='rRNA')
    out=tmp_path/'out.fa';write_filter_evidence([row],out)
    assert attach_filter_warnings([], [str(out)+'.filter_hits.tsv'])==[]


@pytest.mark.parametrize('flag_only', [False, True])
def test_rrna_scan_uses_model_thresholds_and_applies_only_coverage_removal(tmp_path, monkeypatch, flag_only):
    import importlib
    filtering = importlib.import_module("rolypoly.commands.assembly.filter_contigs")
    fasta=tmp_path/'input.fa';fasta.write_text('>whole original\n'+'A'*100+'\n>local description\n'+'A'*100+'\n')
    db=tmp_path/'models.cm';db.write_text('test')
    for suffix in ('.i1f','.i1i','.i1m','.i1p'): (tmp_path/('models.cm'+suffix)).touch()
    config=SimpleNamespace(rrna_db=db, temp_dir=tmp_path, evidence_output=tmp_path/'out.fa',threads=1,rrna_min_fraction=.8,flag_only=flag_only,filter_evidence=[])
    (tmp_path/'out.fa.rrna.tblout').write_text('mock hit\n')
    calls=[]
    monkeypatch.setattr(filtering.subprocess,'run',lambda args, **kwargs: calls.append(args))
    monkeypatch.setattr('rolypoly.utils.various.read_cmscan_tblout',lambda path: pl.DataFrame([
        dict(query_name='whole',seq_from=95,seq_to=6,strand='-',inc='!',target_name='SSU',target_accession='RF00001',score=50.,e_value=1e-10,description='rRNA'),
        dict(query_name='local',seq_from=20,seq_to=40,strand='+',inc='!',target_name='SSU',target_accession='RF00001',score=50.,e_value=1e-10,description='rRNA'),
        dict(query_name='local',seq_from=1,seq_to=100,strand='+',inc='?',target_name='SSU',target_accession='RF00001',score=1.,e_value=1.,description='weak'),
    ]))
    filtering.rrna_filter(config,fasta,tmp_path/'out.fa')
    assert '--cut_ga' in calls[0]
    assert len(config.filter_evidence)==2
    assert ('>whole original' in (tmp_path/'out.fa').read_text())==flag_only
    assert '>local description' in (tmp_path/'out.fa').read_text()
    assert config.filter_evidence[0]['strand']=='-'


def test_failed_rrna_search_does_not_filter_sequences(tmp_path, monkeypatch):
    import subprocess
    import importlib
    filtering = importlib.import_module("rolypoly.commands.assembly.filter_contigs")
    fasta=tmp_path/'in.fa';fasta.write_text('>c\nACGT\n')
    db=tmp_path/'db.cm';db.touch()
    for suffix in ('.i1f','.i1i','.i1m','.i1p'): Path(str(db)+suffix).touch()
    config=SimpleNamespace(rrna_db=db,temp_dir=tmp_path,evidence_output=tmp_path/'out.fa',threads=1)
    def fail(*args,**kwargs): raise subprocess.CalledProcessError(1,'cmscan')
    monkeypatch.setattr(filtering.subprocess,'run',fail)
    with pytest.raises(subprocess.CalledProcessError):
        filtering.rrna_filter(config,fasta,tmp_path/'out.fa')
    assert not (tmp_path/'out.fa').exists()


def test_empty_rrna_scan_retains_input(tmp_path, monkeypatch):
    import importlib
    filtering = importlib.import_module("rolypoly.commands.assembly.filter_contigs")
    fasta=tmp_path/'input.fa';fasta.write_text('>c original description\nACGT\n')
    db=tmp_path/'db.cm';db.touch()
    for suffix in ('.i1f','.i1i','.i1m','.i1p'): Path(str(db)+suffix).touch()
    (tmp_path/'out.fa.rrna.tblout').write_text('# Infernal no matches\n')
    config=SimpleNamespace(rrna_db=db,temp_dir=tmp_path,evidence_output=tmp_path/'out.fa',threads=1,filter_evidence=[])
    monkeypatch.setattr(filtering.subprocess,'run',lambda *a, **kw: None)
    filtering.rrna_filter(config,fasta,tmp_path/'out.fa')
    assert (tmp_path/'out.fa').read_text()==fasta.read_text()
    assert config.filter_evidence==[]
