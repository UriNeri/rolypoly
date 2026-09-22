"""Regression coverage for native assembler inputs and filter-reads handoff."""

import gzip
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rolypoly.commands.assembly.assemble import (
    LibraryInfo,
    handle_input_files,
    run_megahit,
    run_penguin,
    run_with_read_stream,
    run_spades,
)
from rolypoly.utils.bio.library_detection import (
    determine_fastq_type,
    identify_fastq_files,
)


def fastq(path, headers):
    text = "".join(f"@{h}\nACGTACGT\n+\nIIIIIIII\n" for h in headers)
    if str(path).endswith(".gz"):
        with gzip.open(path, "wt") as stream:
            stream.write(text)
    else:
        path.write_text(text)
    return str(path)


def config(tmp_path):
    return SimpleNamespace(
        output_dir=tmp_path,
        temp_dir=tmp_path,
        threads=1,
        memory="2g",
        raw_fasta=[],
        long_read_type="nanopore",
        logger=logging.getLogger(__name__),
        step_params={
            "spades": {"mode": "meta", "k": "21"},
            "megahit": {
                "k-min": 21,
                "k-max": 31,
                "k-step": 10,
                "min-contig-len": 100,
            },
            "penguin": {"min-contig-len": 100, "num-iterations": 1},
        },
    )


@pytest.mark.parametrize(
    "headers,expected",
    [
        (["a/1", "a/2", "b/1", "b/2"], "interleaved"),
        (["a 1:N:0:AT", "a 2:N:0:AT"], "interleaved"),
        (["a/1", "b/1"], "paired_R1"),
        (["a 2:N:0:AT", "b 2:N:0:AT"], "paired_R2"),
        (["a/1", "b/2"], "unknown"),
        (["a/1", "b/1", "a/2", "b/2"], "unknown"),
        (["a", "b"], "single"),
    ],
)
def test_short_fastq_layout(tmp_path, headers, expected):
    path = fastq(tmp_path / "reads.fq.gz", headers)
    assert determine_fastq_type(path)["file_type"] == expected


def test_explicit_same_library_keeps_both_mates_and_orphans():
    info = LibraryInfo()
    info.add_paired(7, "left.fq", "right.fq")
    info.add_single(7, "orphans.fq")
    info.add_merged(7, "merged.fq")
    libs = info.to_assembly_dict()
    assert len(libs) == 1
    assert libs["lib_7"] == dict(
        r1="left.fq",
        r2="right.fq",
        single="orphans.fq",
        merged="merged.fq",
        interleaved=None,
    )


def test_filter_reads_folder_and_raw_inputs(tmp_path):
    for sample in ("a", "b"):
        fastq(
            tmp_path / f"dedupe_final_interleaved_{sample}.fq.gz",
            ["x/1", "x/2"],
        )
        fastq(tmp_path / f"dedupe_final_merged_{sample}.fq.gz", ["merged"])
    fastq(tmp_path / "raw_R1.fq", ["r/1"])
    fastq(tmp_path / "raw_R2.fq", ["r/2"])
    detected = identify_fastq_files(tmp_path)
    assert set(detected["rolypoly_data"]) == {"a", "b"}
    assert len(detected["R1_R2_pairs"]) == 1
    libs, count = handle_input_files(tmp_path)
    assert count == 3
    assert any(lib.get("r2", "").endswith("raw_R2.fq") for lib in libs.values())
    for sample in ("a", "b"):
        assert str(libs[sample]["merged"]).endswith(f"merged_{sample}.fq.gz")


def test_detected_interleaved_stays_paired(tmp_path):
    path = fastq(tmp_path / "external.fq", ["x/1", "x/2"])
    libs, _ = handle_input_files(tmp_path)
    assert list(libs.values()) == [{"interleaved": path, "merged": None}]


def test_spades_dataset_associations_and_memory(tmp_path):
    cfg = config(tmp_path)
    cfg.raw_fasta = ["trusted one.fa", "trusted two.fa"]
    libs = {
        "one": {
            "r1": "left.fq",
            "r2": "right.fq",
            "single": "orphans.fq",
            "merged": "merged.fq",
        },
        "long1": {"long_read": "ont1.fq"},
        "long2": {"long_read": "ont2.fq.gz"},
    }
    with patch("subprocess.run") as run:
        run_spades(cfg, libs)
    cmd = run.call_args.args[0]
    assert cmd[cmd.index("-m") + 1] == "2"
    dataset = json.loads(Path(cmd[cmd.index("--dataset") + 1]).read_text())
    assert dataset[0]["right reads"] == [str(Path("right.fq").resolve())]
    assert "single reads" in dataset[0] and "merged reads" in dataset[0]
    assert dataset[0]["orientation"] == "fr"
    assert len(dataset[1]["single reads"]) == 2
    assert len(dataset[2]["single reads"]) == 2
    assert cfg.raw_fasta == ["trusted one.fa", "trusted two.fa"]
    assert (
        len(list(tmp_path.iterdir())) == 1
    )  # Only a manifest, no read copies.


def test_spades_many_libraries_no_concatenation(tmp_path):
    libs = {str(i): {"interleaved": f"{i}.fq"} for i in range(12)}
    with patch("subprocess.run"):
        run_spades(config(tmp_path), libs, mode="sc")
    assert (
        len(json.loads((tmp_path / "spades_sc_dataset.yaml").read_text())) == 12
    )
    assert len(list(tmp_path.iterdir())) == 1


def test_meta_pools_paired_libraries_without_copying_reads(tmp_path, caplog):
    cfg = config(tmp_path)
    cfg.raw_fasta = ["trusted.fa"]
    libs = {
        "a": {"r1": "a1.fq", "r2": "a2.fq", "single": "a_orphan.fq", "merged": "a_merged.fq"},
        "b": {"interleaved": "b.fq", "single": "b_orphan.fq", "merged": "b_merged.fq"},
        "c": {"r1": "c1.fq", "r2": "c2.fq"},
        "long": {"long_read": "ont.fq"},
    }
    with patch("subprocess.run"):
        run_spades(cfg, libs, mode="meta")
    dataset = json.loads((tmp_path / "spades_meta_dataset.yaml").read_text())
    paired, = [entry for entry in dataset if entry["type"] == "paired-end"]
    for field, expected in {
        "left reads": ["a1.fq", "c1.fq"],
        "right reads": ["a2.fq", "c2.fq"],
        "interlaced reads": ["b.fq"],
        "single reads": ["a_orphan.fq", "b_orphan.fq"],
        "merged reads": ["a_merged.fq", "b_merged.fq"],
    }.items():
        assert paired[field] == [str(Path(path).resolve()) for path in expected]
    assert [entry["type"] for entry in dataset] == ["paired-end", "nanopore", "trusted-contigs"]
    assert "pooling 3 input libraries" in caplog.text
    assert len(list(tmp_path.iterdir())) == 1  # Only the dataset, no read copies.
    assert libs["a"]["r1"] == "a1.fq"  # Other assembler inputs remain intact.


@pytest.mark.parametrize("libs", [
    {"single": {"single": "s.fq"}},
    {"a": {"r1": "a1.fq", "r2": "a2.fq"}, "single": {"single": "s.fq"}},
])
def test_meta_rejects_independent_single_libraries(tmp_path, libs):
    with pytest.raises(Exception, match="metaSPAdes requires paired-end input"):
        run_spades(config(tmp_path), libs, mode="meta")


def test_megahit_native_lists(tmp_path):
    libs = {
        "a": {"r1": "a 1.fq", "r2": "a 2.fq"},
        "b": {"r1": "b1.fq.gz", "r2": "b2.fq.gz", "single": "s.fq"},
        "c": {"interleaved": "i.fq", "merged": "m.fq"},
    }
    (tmp_path / "megahit_custom_out").mkdir()
    with (
        patch("subprocess.run") as run,
        patch("glob.glob", return_value=["k31.final.contigs.fa"]),
    ):
        run_megahit(config(tmp_path), libs)
    cmd = run.call_args_list[0].args[0]
    assert cmd[cmd.index("-1") + 1] == "a 1.fq,b1.fq.gz"
    assert cmd[cmd.index("-2") + 1] == "a 2.fq,b2.fq.gz"
    assert cmd[cmd.index("--12") + 1] == "i.fq"
    assert cmd[cmd.index("-r") + 1] == "s.fq,m.fq"
    assert not list(tmp_path.glob("all_*"))


def test_penguin_native_pairs_and_mixed_compression(tmp_path):
    cfg = config(tmp_path)
    with patch("subprocess.run") as run:
        run_penguin(cfg, {"a": {"r1": "a 1.fq", "r2": "a 2.fq"}})
    assert "'a 1.fq' 'a 2.fq'" in run.call_args.args[0]
    assert not (tmp_path / "all_penguin_input.fq").exists()
    left = fastq(tmp_path / "plain.fq", ["a"])
    right = fastq(tmp_path / "compressed.fq.gz", ["b"])
    with patch("rolypoly.commands.assembly.assemble.run_with_read_stream") as stream:
        run_penguin(cfg, {"a": {"single": left}, "b": {"merged": right}})
    command, paths = stream.call_args.args
    assert command[:3] == ["penguin", "guided_nuclassemble", "stdin"]
    assert paths == [left, right]
    assert not (tmp_path / "all_penguin_input.fq").exists()


def test_read_stream_preserves_plain_and_gzip_records(tmp_path):
    import sys
    left = fastq(tmp_path / "plain.fq", ["a", "b"])
    right = fastq(tmp_path / "compressed.fq.gz", ["c"])
    output = tmp_path / "received.fq"
    command = [sys.executable, "-c",
               "import sys,pathlib; pathlib.Path(sys.argv[1]).write_bytes(sys.stdin.buffer.read())",
               str(output)]
    run_with_read_stream(command, [left, right])
    with gzip.open(right, "rb") as stream:
        assert output.read_bytes() == Path(left).read_bytes() + stream.read()


def test_read_stream_reports_consumer_failure(tmp_path):
    import subprocess
    import sys
    reads = fastq(tmp_path / "reads.fq", [str(i) for i in range(10000)])
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_with_read_stream([sys.executable, "-c", "raise SystemExit(7)"], [reads])
    assert error.value.returncode == 7


def test_penguin_raw_fasta_only_passed_directly(tmp_path):
    cfg = config(tmp_path)
    cfg.raw_fasta = [str(tmp_path / "raw contigs.fa")]
    with patch("subprocess.run") as run:
        run_penguin(cfg, {})
    import shlex
    assert shlex.split(run.call_args.args[0])[2] == cfg.raw_fasta[0]


def test_penguin_raw_fasta_prevents_accidental_file_pairing(tmp_path):
    cfg = config(tmp_path)
    cfg.raw_fasta = ["one.fa", "two.fa.gz"]
    for libraries, expected in [({}, cfg.raw_fasta),
                                ({"a": {"r1": "r1.fq", "r2": "r2.fq"}},
                                 ["r1.fq", "r2.fq", *cfg.raw_fasta])]:
        with patch("rolypoly.commands.assembly.assemble.run_with_read_stream") as stream:
            run_penguin(cfg, libraries)
        command, paths = stream.call_args.args
        assert command[:3] == ["penguin", "guided_nuclassemble", "stdin"]
        assert paths == expected


def test_read_stream_preserves_mixed_format_boundaries(tmp_path):
    import sys
    fasta = tmp_path / "first.fa"
    fasta.write_bytes(b">contig\nAACGT")
    reads = fastq(tmp_path / "reads.fq.gz", ["read"])
    output = tmp_path / "received"
    command = [sys.executable, "-c",
               "import sys,pathlib; pathlib.Path(sys.argv[1]).write_bytes(sys.stdin.buffer.read())",
               str(output)]
    run_with_read_stream(command, [str(fasta), reads])
    with gzip.open(reads, "rb") as source:
        assert output.read_bytes() == fasta.read_bytes() + b"\n" + source.read()
