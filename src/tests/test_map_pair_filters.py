"""Stable contracts for paired-read SAM filtering in ``rolypoly map``."""

import gzip

import pytest
from click.testing import CliRunner

from rolypoly.utils.bio.alignments import (
    filter_sam_by_pair_status,
    sam_record_matches_pair_filters,
)
from rolypoly.rolypoly import rolypoly


def sam_fields(
    flag: int,
    reference: str = "contig_a",
    position: int = 100,
    mate_reference: str = "=",
    mate_position: int = 200,
) -> list[str]:
    """Build the mandatory fields of a small SAM record."""
    return [
        "pair",
        str(flag),
        reference,
        str(position),
        "60",
        "50M",
        mate_reference,
        str(mate_position),
        "150",
        "A" * 50,
        "I" * 50,
    ]


def test_map_exposes_mapper_independent_compression_flag():
    result = CliRunner().invoke(rolypoly, ["map", "--help"])

    assert result.exit_code == 0
    assert "--compressed" in result.output


def test_concordant_requires_same_reference_inward_facing_pair():
    forward_read1 = sam_fields(0x1 | 0x20 | 0x40)
    reverse_read2 = sam_fields(
        0x1 | 0x10 | 0x80, position=200, mate_position=100
    )
    outward_read1 = sam_fields(
        0x1 | 0x20 | 0x40, position=250, mate_position=100
    )
    different_contig = sam_fields(0x1 | 0x20 | 0x40, mate_reference="contig_b")

    assert sam_record_matches_pair_filters(forward_read1, concordant=True)
    assert sam_record_matches_pair_filters(reverse_read2, concordant=True)
    assert not sam_record_matches_pair_filters(outward_read1, concordant=True)
    assert not sam_record_matches_pair_filters(
        different_contig, concordant=True
    )


def test_proper_uses_mapper_defined_sam_flag():
    proper_pair = sam_fields(0x1 | 0x2 | 0x20 | 0x40)
    same_reference_without_proper_flag = sam_fields(0x1 | 0x20 | 0x40)

    assert sam_record_matches_pair_filters(proper_pair, proper=True)
    assert not sam_record_matches_pair_filters(
        same_reference_without_proper_flag, proper=True
    )


@pytest.mark.parametrize("flag", [0, 8, 73, 256, 2048, 4, 77, 141, 260])
def test_mapped_only_uses_read_flag_not_mate_flag(flag):
    assert sam_record_matches_pair_filters(
        sam_fields(flag), mapped_only=True
    ) == (not bool(flag & 0x4))


@pytest.mark.parametrize(
    "flag,retained",
    [(0, True), (73, True), (133, True), (69, True),
     (256, True), (2048, True), (4, False), (77, False), (141, False)],
)
def test_mapped_templates_retain_either_mapped_mate(flag, retained):
    assert sam_record_matches_pair_filters(
        sam_fields(flag), mapped_templates_only=True
    ) == retained


def test_mapped_templates_preserve_unmapped_mate_and_tags(tmp_path):
    sam_path = tmp_path / "paired.sam.gz"
    header = "@HD\tVN:1.6\n@SQ\tSN:contig_a\tLN:500\n"
    mapped = "\t".join(sam_fields(73)) + "\tZZ:Z:custom\n"
    mate = "\t".join(sam_fields(133)) + "\tRG:Z:library\n"
    removed = "\t".join(sam_fields(77)) + "\n"
    with gzip.open(sam_path, "wt") as handle:
        handle.write(header + mapped + mate + removed)
    assert filter_sam_by_pair_status(
        sam_path, mapped_templates_only=True
    ) == (3, 2)
    with gzip.open(sam_path, "rt") as handle:
        assert handle.read() == header + mapped + mate
    assert not sam_record_matches_pair_filters(
        sam_fields(133), concordant=True, mapped_templates_only=True
    )


def test_mapped_only_preserves_headers_and_mapped_records(tmp_path):
    sam_path = tmp_path / "reads.sam"
    header = "@HD\tVN:1.6\n@SQ\tSN:contig_a\tLN:500\n"
    kept = "\t".join(sam_fields(73)) + "\n"
    secondary = "\t".join(sam_fields(256)) + "\n"
    unmapped = "\t".join(sam_fields(77)) + "\n"
    sam_path.write_text(header + kept + unmapped + secondary)
    assert filter_sam_by_pair_status(sam_path, mapped_only=True) == (3, 2)
    assert sam_path.read_text() == header + kept + secondary


def test_invalid_sam_preserves_original_when_filtering(tmp_path):
    sam_path = tmp_path / "invalid.sam"
    content = "@HD\tVN:1.6\ninvalid\n"
    sam_path.write_text(content)
    with pytest.raises(ValueError, match="Invalid SAM record"):
        filter_sam_by_pair_status(sam_path, mapped_only=True)
    assert sam_path.read_text() == content
    assert not list(tmp_path.glob("*.tmp"))


def test_pair_filter_preserves_header_and_supports_gzip(tmp_path):
    sam_path = tmp_path / "reads.sam.gz"
    kept = "\t".join(sam_fields(0x1 | 0x20 | 0x40))
    removed = "\t".join(
        sam_fields(0x1 | 0x20 | 0x40, mate_reference="contig_b")
    )
    with gzip.open(sam_path, "wt") as handle:
        handle.write("@HD\tVN:1.6\n")
        handle.write(f"{kept}\n{removed}\n")

    assert filter_sam_by_pair_status(sam_path, concordant=True) == (2, 1)
    with gzip.open(sam_path, "rt") as handle:
        filtered = handle.read()

    assert filtered == f"@HD\tVN:1.6\n{kept}\n"


def test_map_accepts_comma_pair_before_rejecting_mmseqs_pair_filter(tmp_path):
    reference = tmp_path / "reference.fasta"
    read1 = tmp_path / "sample_R1.fastq"
    read2 = tmp_path / "sample_R2.fastq"
    reference.write_text(">ref\nACGTACGT\n")
    read1.write_text("@pair/1\nACGT\n+\nIIII\n")
    read2.write_text("@pair/2\nACGT\n+\nIIII\n")

    result = CliRunner().invoke(
        rolypoly,
        [
            "map",
            "--input",
            f"{read1},{read2}",
            "--reference",
            str(reference),
            "--output",
            str(tmp_path / "mapping"),
            "--mapper",
            "mmseqs",
            "--concordant",
        ],
    )

    assert result.exit_code != 0
    assert "mmseqs mapper searches reads independently" in result.output
