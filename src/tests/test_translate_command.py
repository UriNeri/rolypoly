"""Contracts for the standalone NumPy translation command and ORF mode."""

from click.testing import CliRunner

from rolypoly.rolypoly import rolypoly
from rolypoly.utils.bio.translation import (
    find_six_frame_orfs_numpy,
    translate_6frx_numpy,
    translation_records,
)


def test_orf_finder_reports_complete_and_both_partial_boundaries():
    complete = find_six_frame_orfs_numpy("ATGAAATAA")
    forward = next(orf for orf in complete if orf["frame"] == 1)
    assert forward == {
        "frame": 1,
        "strand": "+",
        "start": 1,
        "end": 9,
        "protein": "MKX",
        "nucleotide": "ATGAAATAA",
        "protein_length": 2,
        "partial_5prime": False,
        "partial_3prime": False,
    }

    edge_to_stop = find_six_frame_orfs_numpy("AAATAA")
    forward = next(orf for orf in edge_to_stop if orf["frame"] == 1)
    assert forward["protein"] == "KX"
    assert forward["partial_5prime"] is True
    assert forward["partial_3prime"] is False

    start_to_edge = find_six_frame_orfs_numpy("ATGAAA")
    forward = next(orf for orf in start_to_edge if orf["frame"] == 1)
    assert forward["protein"] == "MK"
    assert forward["partial_5prime"] is False
    assert forward["partial_3prime"] is True


def test_orf_finder_uses_genetic_code_starts_and_reverse_coordinates():
    alternative = find_six_frame_orfs_numpy(
        "GTGAAATAA", genetic_code=11, include_partial=False
    )
    assert (
        next(orf for orf in alternative if orf["frame"] == 1)["protein"]
        == "MKX"
    )
    atg_only = find_six_frame_orfs_numpy(
        "GTGAAATAA",
        genetic_code=11,
        alternative_starts=False,
        include_partial=False,
    )
    assert not any(orf["frame"] == 1 for orf in atg_only)

    reverse = find_six_frame_orfs_numpy("TTATTTCAT", include_partial=False)
    reverse_orf = next(orf for orf in reverse if orf["frame"] == -1)
    assert (reverse_orf["start"], reverse_orf["end"]) == (1, 9)
    assert reverse_orf["nucleotide"] == "ATGAAATAA"


def test_orf_finder_can_emit_nested_starts_and_preserve_stop_symbols():
    longest = find_six_frame_orfs_numpy(
        "ATGATGAAATAA", include_partial=False, stops_as_x=False
    )
    assert [
        orf["protein"] for orf in longest if orf["frame"] == 1
    ] == ["MMK*"]

    nested = find_six_frame_orfs_numpy(
        "ATGATGAAATAA",
        include_partial=False,
        all_starts=True,
        stops_as_x=False,
    )
    assert [orf["protein"] for orf in nested if orf["frame"] == 1] == [
        "MMK*",
        "MK*",
    ]


def test_translate_command_default_matches_library_output(tmp_path):
    source = tmp_path / "input.fa"
    expected = tmp_path / "expected.faa"
    observed = tmp_path / "observed.faa"
    source.write_text(">sequence description\nATGTAARYT\n")
    translate_6frx_numpy(source, expected)

    result = CliRunner().invoke(
        rolypoly,
        [
            "translate",
            "--input",
            str(source),
            "--output",
            str(observed),
            "--log-file",
            str(tmp_path / "translate.log"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert observed.read_bytes() == expected.read_bytes()


def test_translate_command_writes_orf_faa_fna_and_gff3_fasta(tmp_path):
    source = tmp_path / "input.fa"
    proteins = tmp_path / "orfs.faa"
    nucleotides = tmp_path / "orfs.fna"
    gff = tmp_path / "orfs.gff3"
    source.write_text(">contig:1 description\nATGAAATAA\n")

    result = CliRunner().invoke(
        rolypoly,
        [
            "translate",
            "-i",
            str(source),
            "-o",
            str(proteins),
            "--mode",
            "orfs",
            "--complete-orfs-only",
            "--header-format",
            "canonical",
            "--fna-output",
            str(nucleotides),
            "--gff-output",
            str(gff),
            "--gff-include-fasta",
            "--log-file",
            str(tmp_path / "translate.log"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert list(translation_records(proteins)) == [("contig%3A1_orf_1", "MKX")]
    assert list(translation_records(nucleotides)) == [
        ("contig%3A1_orf_1", "ATGAAATAA")
    ]
    gff_text = gff.read_text()
    assert "contig:1\trolypoly-translate\tCDS\t1\t9\t.\t+\t0\t" in gff_text
    assert "ID=contig%253A1_orf_1" in gff_text
    assert "##FASTA" in gff_text
    assert ">contig:1 description\nATGAAATAA" in gff_text
