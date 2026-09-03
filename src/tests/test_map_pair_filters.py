"""Stable contracts for paired-read SAM filtering in ``rolypoly map``."""

import gzip
from unittest.mock import patch

from click.testing import CliRunner

from rolypoly.commands.reads.map import build_bwa_mem2_call
from rolypoly.utils.bio.alignments import (
    filter_sam_by_pair_status,
    sam_record_matches_pair_filters,
)
from rolypoly.rolypoly import rolypoly
from rolypoly.utils.various import run_command_comp


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


def test_bwa_mem2_flags_use_run_command_comp_params():
    positional_args, params = build_bwa_mem2_call(
        "index/ref",
        ["reads_R1.fq", "reads_R2.fq"],
        "mapped.sam",
        4,
        report_all=True,
        interleaved=True,
        extra_flags="-Y -R '@RG ID:sample'",
    )

    assert positional_args == [
        "-Y -R '@RG ID:sample'",
        "index/ref",
        "reads_R1.fq",
        "reads_R2.fq",
    ]
    assert params == {"t": 4, "o": "mapped.sam", "a": True, "p": True}

    with patch("subprocess.run"):
        command = run_command_comp(
            "bwa-mem2 mem",
            positional_args=positional_args,
            params=params,
            check_output=False,
            return_final_cmd=True,
            prefix_style="single",
        )

    assert " ".join(command.split()) == (
        "bwa-mem2 mem -t 4 -o mapped.sam -a -p "
        "-Y -R '@RG ID:sample' index/ref reads_R1.fq reads_R2.fq"
    )


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
