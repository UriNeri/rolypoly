from pathlib import Path

from rolypoly.commands.identify_virus.search_viruses import (
    get_builtin_virus_db_paths,
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
