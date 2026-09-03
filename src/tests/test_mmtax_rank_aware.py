import polars as pl
import pytest
import taxopy

from rolypoly.commands.virotype.mmtax import (
    assign_contig_taxonomy,
    rank_vote_metrics,
    weighted_majority_taxid,
)


def make_taxdb(tmp_path):
    """Create two neutral, competing lineages for rank-aware voting tests."""
    nodes = tmp_path / "nodes.dmp"
    names = tmp_path / "names.dmp"
    nodes.write_text(
        "1\t|\t1\t|\tno rank\t|\n"
        "2\t|\t1\t|\trealm\t|\n"
        "3\t|\t2\t|\tfamily\t|\n"
        "4\t|\t3\t|\tgenus\t|\n"
        "5\t|\t4\t|\tspecies\t|\n"
        "6\t|\t2\t|\tfamily\t|\n"
        "7\t|\t6\t|\tgenus\t|\n"
        "8\t|\t7\t|\tspecies\t|\n"
    )
    names.write_text(
        "1\t|\troot\t|\t\t|\tscientific name\t|\n"
        "2\t|\tExample realm\t|\t\t|\tscientific name\t|\n"
        "3\t|\tAlpha family\t|\t\t|\tscientific name\t|\n"
        "4\t|\tAlpha genus\t|\t\t|\tscientific name\t|\n"
        "5\t|\tAlpha species\t|\t\t|\tscientific name\t|\n"
        "6\t|\tBeta family\t|\t\t|\tscientific name\t|\n"
        "7\t|\tBeta genus\t|\t\t|\tscientific name\t|\n"
        "8\t|\tBeta species\t|\t\t|\tscientific name\t|\n"
    )
    return taxopy.TaxDb(
        nodes_dmp=str(nodes), names_dmp=str(names), keep_files=True
    )


def test_shallow_taxa_are_neutral_at_deeper_ranks(tmp_path):
    taxdb = make_taxdb(tmp_path)

    taxid, support = weighted_majority_taxid(
        ["1", "2", "5"], [1000.0, 900.0, 100.0], taxdb, majority=0.5
    )

    assert taxid == "5"
    assert support == 1.0
    family_support, family_informative = rank_vote_metrics(
        ["1", "2", "5"], [1000.0, 900.0, 100.0], taxid, "family", taxdb
    )
    assert family_support == 1.0
    assert family_informative == 0.05


def test_rank_votes_cannot_cross_the_selected_parent_lineage(tmp_path):
    taxdb = make_taxdb(tmp_path)

    taxid, support = weighted_majority_taxid(
        ["5", "8"], [40.0, 60.0], taxdb, majority=0.5
    )

    assert taxid == "8"
    assert support == 1.0
    assert taxopy.Taxon(taxid, taxdb).name_lineage == [
        "Beta species",
        "Beta genus",
        "Beta family",
        "Example realm",
        "root",
    ]


def test_tied_resolved_children_stop_at_their_parent(tmp_path):
    taxdb = make_taxdb(tmp_path)

    taxid, support = weighted_majority_taxid(
        ["5", "8"], [50.0, 50.0], taxdb, majority=0.5
    )

    assert taxid == "2"
    assert support == 1.0


def test_assignment_reports_resolved_and_informative_support(tmp_path):
    taxdb = make_taxdb(tmp_path)
    hits = pl.DataFrame(
        {
            "protein": ["p1", "p1", "p1"],
            "target": ["broad-root", "broad-realm", "resolved-species"],
            "identity": [0.9, 0.9, 0.9],
            "alignment_length": [50, 41, 11],
            "bitscore": [100.0, 99.0, 95.0],
            "evalue": [1e-20, 1e-20, 1e-20],
            "taxid": ["1", "2", "5"],
            "query_start": [1, 40, 90],
            "query_end": [50, 80, 100],
            "query_length": [120, 120, 120],
        }
    )
    protein_map = pl.DataFrame(
        {
            "protein": ["p1"],
            "contig": ["c1"],
            "protein_length": [120],
            "contig_length": [1000],
            "cds_start": [101],
            "cds_end": [460],
            "cds_strand": ["+"],
        }
    )

    result = assign_contig_taxonomy(
        hits,
        protein_map,
        taxdb,
        top=10.0,
        backend="synthetic",
    )

    row = result.row(0, named=True)
    assert row["taxid"] == "5"
    assert row["lineage"] == "1;2;3;4;5"
    assert row["method"] == "rank-aware-weighted:bitscore"
    assert row["support"] == 1.0
    assert row["family_support"] == 1.0
    assert row["informative_fraction"] == pytest.approx(95.0 / 294.0)
    assert row["best_match_target"] == "broad-root"
    assert row["best_match_taxon"] == "root"
    assert row["best_match_bitscore"] == 100.0
    assert row["proteins_assigned"] == 1
    assert row["total_proteins"] == 1
    assert row["aligned_residues"] == 91
    assert row["residue_alignment_fraction"] == pytest.approx(91 / 120)
    assert row["projected_aligned_nt"] == 273
    assert row["projected_alignment_genome_fraction"] == pytest.approx(
        273 / 1000
    )
