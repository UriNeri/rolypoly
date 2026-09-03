from pathlib import Path

from rolypoly.commands.identify_virus.search_viruses import (
    get_builtin_virus_db_paths,
)
from rolypoly.utils.bio.library_detection import (
    find_fasta_files,
    identify_fasta_files,
    is_fasta_file,
    resolve_sequence_inputs,
)
from rolypoly.utils.bio import sequences
from rolypoly.utils.bio.sequences import (
    filter_fasta_by_headers,
)


def test_builtin_virus_db_paths_include_non_riboviria(tmp_path):
    paths = get_builtin_virus_db_paths(tmp_path)

    assert paths == {
        "ncbi_ribovirus": tmp_path
        / "reference_seqs/ncbi_virus/mmseqs/refseq_ribovirus_genomes_cleaned",
        "ncbi_non_riboviria": tmp_path
        / "reference_seqs/ncbi_virus/mmseqs/refseq_non_riboviria_genomes",
        "rvmt": tmp_path / "reference_seqs/RVMT/mmseqs/RVMT_cleaned",
    }
    assert all(isinstance(path, Path) for path in paths.values())


def test_nucleic_search_resolves_comma_separated_inputs(tmp_path):
    first = tmp_path / "first.fasta"
    second = tmp_path / "second.fastq.gz"
    first.touch()
    second.touch()

    assert resolve_sequence_inputs(f"{first},{second}") == [first, second]


def test_nucleic_search_recognizes_compressed_custom_fasta():
    assert is_fasta_file("cleaned_reference.fasta.gz")


def test_nucleic_search_discovers_sequence_files_in_directory(tmp_path):
    fasta = tmp_path / "a.fasta"
    fastq = tmp_path / "b.fastq"
    ignored = tmp_path / "notes.txt"
    fasta.touch()
    fastq.touch()
    ignored.touch()

    assert resolve_sequence_inputs(str(tmp_path)) == [fasta, fastq]


def test_legacy_fasta_discovery_contract_is_preserved(tmp_path):
    nucleotide = tmp_path / "contigs.fasta"
    protein = tmp_path / "proteins.faa"
    nucleotide.touch()
    protein.touch()

    assert find_fasta_files(tmp_path) == [nucleotide]
    assert identify_fasta_files(tmp_path) == {"fasta_files": [nucleotide]}


def test_sequence_module_reexports_detection_helpers():
    assert sequences.find_fasta_files is find_fasta_files
    assert sequences.is_fasta_file is is_fasta_file
    assert sequences.resolve_sequence_inputs is resolve_sequence_inputs


def test_filter_multi_fastas_matches_full_headers_across_inputs(tmp_path):
    fasta = tmp_path / "first.fasta"
    fastq = tmp_path / "second.fastq"
    output = tmp_path / "matched.fasta.gz"
    fasta.write_text(">shared description\nACGT\n>ignored\nAAAA\n")
    fastq.write_text("@shared duplicate\nTTTT\n+\nIIII\n@unique\nCCCC\n+\nIIII\n")

    counts = filter_fasta_by_headers(
        [fasta, fastq],
        {"shared duplicate", "unique"},
        output,
        return_counts=True,
    )

    import gzip

    assert counts == {"records_processed": 4, "records_written": 2}
    with gzip.open(output, "rt") as handle:
        assert handle.read() == ">shared duplicate\nTTTT\n>unique\nCCCC\n"


def test_single_fasta_filter_retains_legacy_exact_match_contract(tmp_path):
    fasta = tmp_path / "input.fasta"
    output = tmp_path / "selected.fasta"
    fasta.write_text(">first description\nAAAA\n>second\nCCCC\n")

    counts = filter_fasta_by_headers(
        fasta,
        ["first description"],
        output,
        return_counts=True,
    )

    assert counts == {"records_processed": 1, "records_written": 1}
    assert output.read_text() == ">first description\nAAAA\n"
