import logging
from pathlib import Path

import polars as pl
import pytest
import taxopy
import rich_click as click

from rolypoly.commands.virotype.mmtax import (
    assign_contig_taxonomy,
    rank_vote_metrics,
    read_single_cds_coordinates,
    search_mmseqs,
    weighted_majority_taxid,
)
from rolypoly.utils.bio.polars_fastx import read_protein_gff_map


@pytest.mark.parametrize(
    "memory,expected_bytes",
    [
        ("30g", 30 * 1024**3 * 95 // 100),
        ("20GB", 19 * 1024**3),
        ("10g", 9 * 1024**3),
        ("8g", 7 * 1024**3),
        ("1536m", 512 * 1024**2),
        ("2g", 1024**3),
        (str(30 * 1024**3 + 1), 30 * 1024**3 * 95 // 100),
    ],
)
def test_mmseqs_taxonomy_reserves_memory(monkeypatch, caplog, memory, expected_bytes):
    calls = []

    def run_command(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr("rolypoly.utils.various.run_command_comp", run_command)
    with caplog.at_level(logging.INFO):
        search_mmseqs(
            Path("proteins.faa"), Path("reference"), Path("hits.tsv"),
            Path("work"), 2, memory, 1e-5, 0.1, 30, "normal",
            logging.getLogger(__name__),
        )
    assert len(calls) == 1
    assert calls[0]["params"]["split-memory-limit"] == f"{expected_bytes}B"
    assert "not a hard process memory limit" in caplog.text


@pytest.mark.parametrize("memory", ["0", "512m", "1g", "1024mb"])
def test_mmseqs_taxonomy_rejects_insufficient_memory(monkeypatch, memory):
    def unexpected_search(**kwargs):
        pytest.fail("MMseqs2 must not start without room for the safety reserve")

    monkeypatch.setattr("rolypoly.utils.various.run_command_comp", unexpected_search)
    with pytest.raises(click.BadParameter, match="greater than 1"):
        search_mmseqs(
            Path("proteins.faa"), Path("reference"), Path("hits.tsv"),
            Path("work"), 2, memory, 1e-5, 0.1, 30, "normal",
            logging.getLogger(__name__),
        )


@pytest.mark.parametrize("command_name", ["mmtax", "roll"])
@pytest.mark.parametrize("explicit_backend", [None, "mmseqs"])
def test_taxonomy_cli_defaults_to_diamond(tmp_path, command_name, explicit_backend):
    from rolypoly.rolypoly import rolypoly

    input_path = tmp_path / "input.fasta"
    input_path.write_text(">protein\nMPEPTIDE\n")
    output_flag = "--output" if command_name == "mmtax" else "--output-dir"
    backend_flag = "--backend" if command_name == "mmtax" else "--taxonomy-backend"
    backend_name = "backend" if command_name == "mmtax" else "taxonomy_backend"
    command = rolypoly.get_command(click.Context(rolypoly), command_name)
    assert command is not None
    args = ["--input", str(input_path), output_flag, str(tmp_path / "output")]
    if explicit_backend:
        args += [backend_flag, explicit_backend]
    with command.make_context(command_name, args) as ctx:
        assert ctx.params[backend_name] == (explicit_backend or "diamond")


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


def test_gff_protein_map_and_single_cds_coordinates_share_id_parsing(tmp_path):
    gff = tmp_path / "proteins.gff"
    gff.write_text(
        "contig one\trp\tCDS\t5\t20\t.\t+\t0\tID=prot1 description\n"
        "contig2\trp\tCDS\t30\t60\t.\t-\t0\tprotein_id=prot2 extra\n"
        'contig3\trp\tCDS\t70\t90\t.\t+\t0\ttranscript_id "prot3 note"\n'
        'contig3\trp\tCDS\t95\t120\t.\t+\t0\ttranscript_id "prot3 note"\n'
    )

    mapping = read_protein_gff_map(gff).sort("protein")
    assert mapping.to_dict(as_series=False) == {
        "protein": ["prot1", "prot2", "prot3"],
        "contig": ["contig", "contig2", "contig3"],
    }

    coordinates = read_single_cds_coordinates(gff).sort("protein")
    assert coordinates.to_dict(as_series=False) == {
        "protein": ["prot1", "prot2"],
        "cds_start": [5, 30],
        "cds_end": [20, 60],
        "cds_strand": ["+", "-"],
    }


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


def test_assignment_accepts_rank_metrics_first_resolved_after_100_proteins(tmp_path):
    taxdb = make_taxdb(tmp_path)
    proteins = [f"p{i}" for i in range(101)]
    hits = pl.DataFrame({
        "protein": proteins,
        "target": proteins,
        "identity": [0.9] * 101,
        "alignment_length": [30] * 101,
        "bitscore": [100.0] * 101,
        "evalue": [1e-20] * 101,
        "taxid": ["1"] * 100 + ["5"],
    })
    mapping = pl.DataFrame({"protein": proteins, "contig": proteins})

    result = assign_contig_taxonomy(hits, mapping, taxdb, backend="diamond")

    assert result.height == 101
    resolved = result.filter(pl.col("query") == "p100").row(0, named=True)
    assert resolved["taxid"] == "5"
    assert resolved["family_support"] == 1.0
    assert resolved["informative_fraction"] == 1.0
    assert result.filter(pl.col("query") != "p100")["family_support"].null_count() == 100


def test_assignment_with_no_mapped_hits_returns_typed_empty_table(tmp_path):
    taxdb = make_taxdb(tmp_path)
    hits = pl.DataFrame({
        "protein": ["unmapped"], "target": ["target"], "identity": [0.9],
        "alignment_length": [30], "bitscore": [100.0],
        "evalue": [1e-20], "taxid": ["5"],
    })
    mapping = pl.DataFrame({"protein": ["other"], "contig": ["c1"]})

    result = assign_contig_taxonomy(hits, mapping, taxdb)

    assert result.is_empty()
    assert result.schema["family_support"] == pl.Float64
    assert result.schema["query"] == pl.String
