from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from contextlib import contextmanager
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tomllib

import click
import polars as pl
import pytest
from click.testing import CliRunner

from rolypoly.rolypoly import rolypoly


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def runner() -> CliRunner:
    return CliRunner()


def sequence_fixture(name: str) -> Path:
    return repo_root() / "testing_folder/inputs/sequences" / name


@pytest.mark.parametrize("mapper", ["minimap2", "bwa-mem2"])
@pytest.mark.parametrize("include_unmapped", [False, True])
@pytest.mark.parametrize(
    "options,compressed",
    [([], True), (["-cp", "sam_only"], True),
     (["-cp", "none"], False), (["--compressed"], True)],
)
def test_map_sam_compression(
    runner: CliRunner, tmp_path: Path, monkeypatch, mapper: str,
    options: list[str], compressed: bool, include_unmapped: bool,
) -> None:
    from rolypoly.commands.reads import map as mapping

    if not shutil.which("pigz") or (mapper == "bwa-mem2" and not shutil.which(mapper)):
        pytest.skip("Mapping/compression executable unavailable")
    rng = random.Random(218)
    sequence = "".join(rng.choices("ACGT", k=600))
    reference = tmp_path / "ref.fa"
    reference.write_text(f">contig\n{sequence}\n")
    reads = tmp_path / "reads.fq"
    reads.write_text(
        f"@mapped\n{sequence[100:250]}\n+\n{'I' * 150}\n"
        f"@unmapped\n{'N' * 150}\n+\n{'I' * 150}\n"
    )
    output = tmp_path / "mapping output"
    args = ["map", "-i", str(reads), "-r", str(reference), "-o", str(output),
            "-m", mapper, "-t", "1", "-g", str(tmp_path / "map.log")]
    if include_unmapped:
        args.append("--include-unmapped")
    originals = []
    open_output = mapping.sam_output

    @contextmanager
    def capture_output(path, threads):
        with open_output(path, threads) as destination:
            buffer = io.StringIO()
            yield buffer
            content = buffer.getvalue()
            originals.append(content.encode())
            destination.write(content)

    monkeypatch.setattr(mapping, "sam_output", capture_output)
    result = runner.invoke(rolypoly, args + options, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    files = list(output.rglob("*.sam.gz" if compressed else "*.sam"))
    assert len(files) == 1
    assert not list(output.rglob("*.sam" if compressed else "*.sam.gz"))
    opener = gzip.open if compressed else open
    with opener(files[0], "rt") as handle:
        content = handle.read()
    assert "@SQ\tSN:contig" in content
    assert "mapped\t" in content
    if mapper == "bwa-mem2":
        assert any(line.startswith("unmapped\t") and int(line.split("\t")[1]) & 4
                   for line in content.splitlines()) == include_unmapped
    if not include_unmapped:
        assert all(not int(line.split("\t")[1]) & 4
                   for line in content.splitlines() if not line.startswith("@"))
    if compressed:
        assert originals == [content.encode()]


@pytest.mark.parametrize("mapper", ["bbmap", "bwa-mem2", "minimap2"])
@pytest.mark.parametrize("layout", ["paired", "interleaved"])
@pytest.mark.parametrize("include_unmapped", [False, True])
def test_map_retains_both_mates_when_either_maps(
    runner: CliRunner, tmp_path: Path, mapper: str,
    layout: str, include_unmapped: bool,
) -> None:
    required = ["pigz"]
    if mapper == "bbmap":
        required.append("java")
    elif mapper == "bwa-mem2":
        required.append(mapper)
    if any(not shutil.which(tool) for tool in required):
        pytest.skip("Mapping/compression executable unavailable")
    rng = random.Random(981)
    sequence = "".join(rng.choices("ACGT", k=1200))
    reference = tmp_path / "ref.fa"
    reference.write_text(f">contig\n{sequence}\n")
    mapped1 = sequence[100:250]
    mapped2 = sequence[400:550].translate(str.maketrans("ACGT", "TGCA"))[::-1]
    libraries = [
        ("left_only", mapped1, "N" * 150),
        ("right_only", "N" * 150, mapped2),
        ("neither", "N" * 150, "N" * 150),
    ]
    read1, read2 = tmp_path / "R1.fq", tmp_path / "R2.fq"
    interleaved = tmp_path / "interleaved.fq"
    first, second, combined = [], [], []
    for name, seq1, seq2 in libraries:
        r1 = f"@{name}/1\n{seq1}\n+\n{'I' * 150}\n"
        r2 = f"@{name}/2\n{seq2}\n+\n{'I' * 150}\n"
        first.append(r1)
        second.append(r2)
        combined.extend([r1, r2])
    read1.write_text("".join(first))
    read2.write_text("".join(second))
    interleaved.write_text("".join(combined))
    output = tmp_path / "mapped"
    args = [
        "map", "-r", str(reference), "-o", str(output), "-m", mapper,
        "-t", "1", "-M", "1g", "-cp", "sam_only",
        "-g", str(tmp_path / "map.log"),
    ]
    if layout == "paired":
        args.extend(["--paired-end", "1", str(read1), str(read2)])
    else:
        args.extend(["-i", str(interleaved)])
    if include_unmapped:
        args.append("--include-unmapped")
    result = runner.invoke(rolypoly, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    with gzip.open(next(output.rglob("*.sam.gz")), "rt") as handle:
        records = [line.rstrip("\n").split("\t") for line in handle
                   if not line.startswith("@")]
    assert not list(output.rglob("*.sam"))
    assert not list(output.rglob("*.tmp"))
    expected = {"left_only", "right_only", "neither"} if include_unmapped else {
        "left_only", "right_only",
    }
    assert {fields[0] for fields in records} == expected
    for name, seq1, seq2 in libraries:
        if name not in expected:
            continue
        pair = [fields for fields in records if fields[0] == name]
        assert len(pair) == 2
        flags = [int(fields[1]) for fields in pair]
        assert all(flag & 1 for flag in flags)
        assert sum(bool(flag & 64) for flag in flags) == 1
        assert sum(bool(flag & 128) for flag in flags) == 1
        if name != "neither":
            assert sum(bool(flag & 4) for flag in flags) == 1
            mapped = next(fields for fields in pair if not int(fields[1]) & 4)
            unmapped = next(fields for fields in pair if int(fields[1]) & 4)
            assert int(mapped[1]) & 8
            assert not int(unmapped[1]) & 8
            assert unmapped[6] in {"=", mapped[2]}
            assert unmapped[7] == mapped[3]
            assert unmapped[9] == ("N" * 150)
            # BBMap sets qualities of N bases to zero.
            assert unmapped[10] == ("!" if mapper == "bbmap" else "I") * 150
        else:
            assert all(flag & 4 and flag & 8 for flag in flags)


@pytest.mark.parametrize("command", ["map", "roll"])
def test_mapping_rejects_removed_compression_policy(runner: CliRunner, command: str) -> None:
    required = ["-r", "reference.fa"] if command == "map" else ["-i", "reads.fq"]
    result = runner.invoke(
        rolypoly, [command, *required, "-cp", "over_10gb"],
    )
    assert result.exit_code != 0
    assert "Invalid value" in result.output and "over_10gb" in result.output


@pytest.mark.parametrize(
    "mapper,option",
    [("bbmap", "--concordant"), ("bbmap", "--proper"),
     ("bwa-mem2", "--concordant"), ("bwa-mem2", "--proper"),
     ("minimap2", "--concordant")],
)
def test_map_pair_filters_on_gzip(
    runner: CliRunner, tmp_path: Path, mapper: str, option: str,
) -> None:
    required = ["pigz"]
    if mapper == "bbmap":
        required.append("java")
    elif mapper == "bwa-mem2":
        required.append(mapper)
    if any(not shutil.which(tool) for tool in required):
        pytest.skip("Mapping/compression executable unavailable")
    rng = random.Random(14)
    sequence = "".join(rng.choices("ACGT", k=1200))
    reference = tmp_path / "ref.fa"
    reference.write_text(f">ref\n{sequence}\n")
    read1, read2 = tmp_path / "R1.fq", tmp_path / "R2.fq"
    read1.write_text(f"@pair/1\n{sequence[100:250]}\n+\n{'I' * 150}\n")
    read2.write_text(f"@pair/2\n{'N' * 150}\n+\n{'I' * 150}\n")
    output = tmp_path / "mapped"
    result = runner.invoke(rolypoly, [
        "map", "-r", str(reference), "-o", str(output), "-m", mapper,
        "--paired-end", "1", str(read1), str(read2), option,
        "--include-unmapped", "-t", "1", "-M", "1g",
        "-g", str(tmp_path / "map.log"),
    ], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    with gzip.open(next(output.rglob("*.sam.gz")), "rt") as handle:
        content = handle.read()
    assert "@SQ\tSN:ref" in content
    assert all(line.startswith("@") for line in content.splitlines())
    assert not list(output.rglob("*.sam"))


@pytest.mark.parametrize("failure", ["mapper", "invalid-sam", "compressor", "broken-pipe"])
def test_mapping_stream_failure_does_not_publish_output(
    tmp_path: Path, monkeypatch, failure: str,
) -> None:
    from rolypoly.commands.reads.map import run_sam_command

    if not shutil.which("pigz"):
        pytest.skip("pigz unavailable")
    output = tmp_path / "output.sam.gz"
    content = "@HD\tVN:1.6\nmapped\t0\tref\t1\t60\t4M\t*\t0\t0\tACGT\tIIII\n"
    code = f"import sys; sys.stdout.write({content!r}); sys.exit(7)"
    if failure == "invalid-sam":
        code = "print('invalid SAM')"
    elif failure in {"compressor", "broken-pipe"}:
        fake = tmp_path / "bin"
        fake.mkdir()
        compressor = fake / "pigz"
        read_input = "sys.stdin.buffer.read(); " if failure == "compressor" else ""
        compressor.write_text(
            f"#!{sys.executable}\nimport sys\n{read_input}sys.exit(8)\n"
        )
        compressor.chmod(0o755)
        monkeypatch.setenv("PATH", str(fake) + os.pathsep + os.environ["PATH"])
        code = f"import sys; sys.stdout.write({content!r} * 10000)"
    with pytest.raises(click.ClickException):
        run_sam_command([sys.executable, "-c", code], output, 1, mapped_templates_only=True)
    assert not output.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_mapping_stream_preserves_existing_staging_file(tmp_path: Path) -> None:
    from rolypoly.commands.reads.map import sam_output

    output = tmp_path / "output.sam"
    staging = tmp_path / ".output.sam.tmp"
    staging.write_text("previous work")
    with pytest.raises(click.ClickException, match="Failed to write"):
        with sam_output(output, 1):
            pytest.fail("Existing staging file should not be opened.")
    assert staging.read_text() == "previous work"
    assert not output.exists()


@pytest.mark.parametrize("compressed", [False, True])
def test_mapping_stream_filters_losslessly(tmp_path: Path, compressed: bool) -> None:
    from rolypoly.commands.reads.map import run_sam_command

    if compressed and not shutil.which("pigz"):
        pytest.skip("pigz unavailable")
    header = "@HD\tVN:1.6\n@SQ\tSN:ref\tLN:100\n@CO\tcustom provenance\n"
    mapped = "pair\t73\tref\t1\t60\t4M\t*\t0\t0\tACGT\tIIII\tZZ:Z:custom\n"
    mate = "pair\t133\tref\t1\t0\t*\t=\t1\t0\tNNNN\tIIII\tRG:Z:library\n"
    secondary = "other\t256\tref\t2\t10\t4M\t*\t0\t0\tACGT\tIIII\tAS:i:4\n"
    removed = "unmapped\t77\t*\t0\t0\t*\t*\t0\t0\tNNNN\tIIII\n"
    content = header + mapped + mate + removed + secondary
    output = tmp_path / ("out.sam.gz" if compressed else "out.sam")
    run_sam_command(
        [sys.executable, "-c", f"import sys; sys.stdout.write({content!r})"],
        output, 1, mapped_templates_only=True,
    )
    opener = gzip.open if compressed else open
    with opener(output, "rt") as handle:
        assert handle.read() == header + mapped + mate + secondary
    assert not list(tmp_path.glob("*.tmp"))
    if compressed:
        assert not list(tmp_path.glob("*.sam"))


@pytest.mark.parametrize("flags", ["-o redirected.sam", "'unclosed"])
def test_map_rejects_bwa_output_flags_before_indexing(
    runner: CliRunner, tmp_path: Path, monkeypatch, flags: str,
) -> None:
    from rolypoly.commands.reads import map as mapping

    reference, reads = tmp_path / "ref.fa", tmp_path / "reads.fq"
    reference.write_text(">ref\nACGTACGT\n")
    reads.write_text("@read\nACGT\n+\nIIII\n")

    def unexpected_index(*args, **kwargs):
        pytest.fail("Invalid extra flags must be rejected before indexing.")

    monkeypatch.setattr(mapping, "run_command_comp", unexpected_index)
    result = runner.invoke(rolypoly, [
        "map", "-i", str(reads), "-r", str(reference),
        "-o", str(tmp_path / "output"), "-m", "bwa-mem2", "-cp", "none",
        "--bwa-mem2-extra-flags", flags, "-g", str(tmp_path / "map.log"),
    ])
    assert result.exit_code != 0
    assert "--bwa-mem2-extra-flags" in result.output


@pytest.mark.parametrize("policy", [None, "none"])
def test_map_mmseqs_gzip_output(
    runner: CliRunner, tmp_path: Path, monkeypatch, policy: str | None,
) -> None:
    from rolypoly.commands.reads import map as mapping

    if not shutil.which("mmseqs") or not shutil.which("pigz"):
        pytest.skip("MMseqs or pigz unavailable")
    rng = random.Random(14)
    sequence = "".join(rng.choices("ACGT", k=1200))
    reference = tmp_path / "ref.fa"
    reference.write_text(f">ref\n{sequence}\n")
    reads = tmp_path / "reads.fq"
    reads.write_text(f"@mapped\n{sequence[100:250]}\n+\n{'I' * 150}\n")
    original_run = mapping.run_command_comp

    def tiny_index(base_cmd, **kwargs):
        if base_cmd == "mmseqs":
            # Keep the synthetic nucleotide index small.
            kwargs["positional_args"] = [*kwargs["positional_args"], "-k", "8"]
        return original_run(base_cmd, **kwargs)

    monkeypatch.setattr(mapping, "run_command_comp", tiny_index)
    output = tmp_path / "mapping"
    args = [
        "map", "-i", str(reads), "-r", str(reference), "-o", str(output),
        "-m", "mmseqs", "-t", "1", "-M", "1g",
        "-g", str(tmp_path / "map.log"),
    ]
    if policy:
        args.extend(["-cp", policy])
    result = runner.invoke(rolypoly, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    suffix = "*.sam" if policy == "none" else "*.sam.gz"
    files = list(output.rglob(suffix))
    assert len(files) == 1
    opener = open if policy == "none" else gzip.open
    with opener(files[0], "rt") as handle:
        content = handle.read()
    assert content.startswith("@HD\t")
    assert "\nmapped\t" in content
    assert "\tAS:i:" in content and "\tNM:i:" in content
    assert not list(output.glob("mmseqs/*tmp*"))
    if policy != "none":
        assert not list(output.rglob("*.sam"))


def test_map_bbmap_appends_gzip_libraries(runner: CliRunner, tmp_path: Path) -> None:
    if not shutil.which("java") or not shutil.which("pigz"):
        pytest.skip("Java or pigz unavailable")
    rng = random.Random(22)
    sequence = "".join(rng.choices("ACGT", k=1200))
    reference = tmp_path / "ref.fa"
    reference.write_text(f">ref\n{sequence}\n")
    output = tmp_path / "mapping"
    args = [
        "map", "-r", str(reference), "-o", str(output), "-m", "bbmap",
        "-t", "1", "-M", "1g", "-g", str(tmp_path / "map.log"),
    ]
    for library in ("A", "B"):
        read1, read2 = tmp_path / f"{library}_R1.fq", tmp_path / f"{library}_R2.fq"
        read1.write_text(f"@{library}/1\n{sequence[100:250]}\n+\n{'I' * 150}\n")
        read2.write_text(f"@{library}/2\n{'N' * 150}\n+\n{'I' * 150}\n")
        args.extend(["--paired-end", library, str(read1), str(read2)])
    result = runner.invoke(rolypoly, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    files = list(output.rglob("*.sam.gz"))
    assert len(files) == 1
    with gzip.open(files[0], "rt") as handle:
        content = handle.read()
    records = [line.split("\t") for line in content.splitlines() if not line.startswith("@")]
    assert len(records) == 4
    assert {fields[0] for fields in records} == {"A", "B"}
    assert sum(bool(int(fields[1]) & 4) for fields in records) == 2
    assert content.count("@HD\t") == 1
    assert not list(output.rglob("*.sam"))
    assert not (output / "bbmap/.bbmap.sam.gz").exists()


@pytest.mark.parametrize("mapper", ["bbmap", "bwa-mem2", "mmseqs"])
def test_map_backend_failure_is_explicit(
    runner: CliRunner, tmp_path: Path, monkeypatch, mapper: str,
) -> None:
    import bbmapy
    from rolypoly.commands.reads import map as mapping

    if not shutil.which("pigz"):
        pytest.skip("pigz unavailable")
    reference, reads = tmp_path / "ref.fa", tmp_path / "reads.fq"
    reference.write_text(">ref\nACGTACGT\n")
    reads.write_text("@read\nACGT\n+\nIIII\n")
    output = tmp_path / "output"

    def fail_bbmap(**kwargs):
        Path(kwargs["outm"]).write_bytes(b"partial gzip")
        raise RuntimeError("synthetic tool failure")

    monkeypatch.setattr(bbmapy, "bbmap", fail_bbmap)
    monkeypatch.setattr(mapping, "run_command_comp", lambda *a, **kw: False)
    result = runner.invoke(rolypoly, [
        "map", "-i", str(reads), "-r", str(reference), "-o", str(output),
        "-m", mapper, "-g", str(tmp_path / "map.log"),
    ])
    assert result.exit_code != 0
    assert "failed" in result.output
    assert not list(output.rglob("*.sam.gz"))


@pytest.mark.parametrize("failure", ["command", "conflict", "missing-pigz"])
def test_sam_compression_retains_original_on_failure(
    tmp_path: Path, monkeypatch, failure: str,
) -> None:
    from rolypoly.commands.reads import map as mapping

    sam = tmp_path / "original.sam"
    sam.write_text("@HD\tVN:1.6\n")
    if failure == "missing-pigz":
        monkeypatch.setattr(mapping.shutil, "which", lambda name: None)
        message = "requires pigz"
    else:
        monkeypatch.setattr(mapping.shutil, "which", lambda name: "/pigz")
        if failure == "command":
            monkeypatch.setattr(mapping, "run_command_comp", lambda *a, **kw: False)
            message = "original SAM retained"
        else:
            Path(str(sam) + ".gz").write_bytes(b"existing archive")
            message = "Refusing to overwrite"
    with pytest.raises(click.ClickException, match=message):
        mapping.compress_sam_outputs([str(sam)], "sam_only", 1)
    assert sam.read_text() == "@HD\tVN:1.6\n"
    if failure == "conflict":
        assert Path(str(sam) + ".gz").read_bytes() == b"existing archive"


@pytest.mark.parametrize("mapper_name", ["bbmap", "mmseqs"])
def test_mapper_specs_route_invocation_and_stream_filter(
    runner: CliRunner, tmp_path: Path, monkeypatch, mapper_name: str,
) -> None:
    from rolypoly.commands.reads import map as mapping

    if not shutil.which("pigz"):
        pytest.skip("pigz unavailable")
    monkeypatch.setitem(
        mapping.MAPPER_SPECS, mapper_name, mapping.MAPPER_SPECS["bwa-mem2"].copy(),
    )
    reference = tmp_path / "reference.fa"
    reference.write_text(">ref\nACGTACGT\n")
    read1, read2 = tmp_path / "R1.fq", tmp_path / "R2.fq"
    read1.write_text("@pair/1\nACGT\n+\nIIII\n")
    read2.write_text("@pair/2\nNNNN\n+\nIIII\n")
    output = tmp_path / "mapping"
    header = "@HD\tVN:1.6\n@SQ\tSN:ref\tLN:8\n"
    mapped = "pair\t73\tref\t1\t60\t4M\t*\t0\t0\tACGT\tIIII\tZZ:Z:custom\n"
    mate = "pair\t133\tref\t1\t0\t*\t=\t1\t0\tNNNN\tIIII\n"
    absent = "neither\t77\t*\t0\t0\t*\t*\t0\t0\tNNNN\tIIII\n"
    content = header + mapped + mate + absent
    real_stream = mapping.run_sam_command
    commands = []

    def stream(command, path, threads, **filters):
        commands.append(command)
        assert command[:2] == ["bwa-mem2", "mem"]
        real_stream(
            [sys.executable, "-c", f"import sys; sys.stdout.write({content!r})"],
            path, threads, **filters,
        )

    def no_posthoc(*args, **kwargs):
        pytest.fail("A stream-capable spec must not route through file compression.")

    monkeypatch.setattr(mapping, "run_command_comp", lambda *a, **kw: True)
    monkeypatch.setattr(mapping, "run_sam_command", stream)
    monkeypatch.setattr(mapping, "compress_sam_outputs", no_posthoc)
    args = [
        "map", "-r", str(reference), "-m", mapper_name, "-t", "1",
        "--paired-end", "1", str(read1), str(read2),
        "-g", str(tmp_path / "map.log"),
    ]
    for strict in (False, True):
        destination = output / ("strict" if strict else "default")
        options = ["--concordant"] if strict else []
        result = runner.invoke(
            rolypoly, [*args, "-o", str(destination), *options],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        files = list((destination / mapper_name).glob("*.sam.gz"))
        assert len(files) == 1
        with gzip.open(files[0], "rt") as handle:
            assert handle.read() == (header if strict else header + mapped + mate)
        assert not list(destination.rglob("*.sam"))
        assert not (destination / mapper_name / "bwa_index").exists()
    assert len(commands) == 2


@pytest.mark.parametrize("output_mode", ["native_gzip", "sam_stream", "sam_file"])
@pytest.mark.parametrize("policy", ["none", "sam_only"])
@pytest.mark.parametrize("strict", [False, True])
def test_mapper_specs_compose_custom_invocation_processing(
    runner: CliRunner, tmp_path: Path, monkeypatch,
    output_mode: str, policy: str, strict: bool,
) -> None:
    from rolypoly.commands.reads import map as mapping

    if policy == "sam_only" and not shutil.which("pigz"):
        pytest.skip("pigz unavailable")
    header = "@HD\tVN:1.6\n@SQ\tSN:ref\tLN:8\n"
    mapped = "pair\t73\tref\t1\t60\t4M\t*\t0\t0\tACGT\tIIII\tZZ:Z:custom\n"
    mate = "pair\t133\tref\t1\t0\t*\t=\t1\t0\tNNNN\tIIII\n"
    calls: list[mapping.MapperRun] = []

    def invoke(run: mapping.MapperRun, spec: mapping.MapperSpec) -> list[str]:
        calls.append(run)
        path = run["output"] / f"custom{run['suffix']}"
        with mapping.sam_output(path, run["threads"]) as handle:
            handle.write(header + mapped + mate)
        (run["output"] / "custom_index").mkdir()
        (run["output"] / "retained_directory").mkdir()
        return [str(path)]

    spec = mapping.MAPPER_SPECS["bbmap"].copy()
    spec["run"] = invoke
    if output_mode == "native_gzip":
        spec["output_mode"] = "native_gzip"
    elif output_mode == "sam_stream":
        spec["output_mode"] = "sam_stream"
    else:
        spec["output_mode"] = "sam_file"
    spec["pair_filter_stage"] = "posthoc"
    spec["temporary_patterns"] = ("custom_index",)
    monkeypatch.setitem(mapping.MAPPER_SPECS, "bbmap", spec)
    reference = tmp_path / "reference.fa"
    reference.write_text(">ref\nACGTACGT\n")
    read1, read2 = tmp_path / "R1.fq", tmp_path / "R2.fq"
    read1.write_text("@pair/1\nACGT\n+\nIIII\n")
    read2.write_text("@pair/2\nNNNN\n+\nIIII\n")
    output = tmp_path / "mapping"
    result = runner.invoke(rolypoly, [
        "map", "-r", str(reference), "-o", str(output), "-m", "bbmap",
        "--paired-end", "1", str(read1), str(read2), "-cp", policy,
        "-g", str(tmp_path / "map.log"), *(["--concordant"] if strict else []),
    ], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    suffix = ".sam.gz" if policy == "sam_only" else ".sam"
    path = output / "bbmap" / f"custom{suffix}"
    if policy == "sam_only":
        with gzip.open(path, "rt") as handle:
            content = handle.read()
        assert not list(output.rglob("*.sam"))
    else:
        content = path.read_text()
    assert content == (header if strict else header + mapped + mate)
    assert len(calls) == 1
    assert calls[0]["suffix"] == (".sam" if output_mode == "sam_file" else suffix)
    assert not (output / "bbmap" / "custom_index").exists()
    assert (output / "bbmap" / "retained_directory").is_dir()


@pytest.mark.parametrize("stage", ["stream", "posthoc"])
@pytest.mark.parametrize("layout", ["paired", "interleaved"])
def test_mapper_specs_select_mappy_filter_stage(
    runner: CliRunner, tmp_path: Path, monkeypatch, stage: str, layout: str,
) -> None:
    from rolypoly.commands.reads import map as mapping
    from rolypoly.utils.bio import alignments

    spec = mapping.MAPPER_SPECS["minimap2"].copy()
    spec["pair_filter_stage"] = "stream" if stage == "stream" else "posthoc"
    monkeypatch.setitem(mapping.MAPPER_SPECS, "minimap2", spec)
    rng = random.Random(14)
    sequence = "".join(rng.choices("ACGT", k=1200))
    reference = tmp_path / "reference.fa"
    reference.write_text(f">ref\n{sequence}\n")
    first = f"@pair/1\n{sequence[100:250]}\n+\n{'I' * 150}\n"
    second = f"@pair/2\n{'N' * 150}\n+\n{'I' * 150}\n"
    output = tmp_path / "mapping"
    read1, read2 = tmp_path / "R1.fq", tmp_path / "R2.fq"
    if layout == "paired":
        read1.write_text(first)
        read2.write_text(second)
        inputs = ["--paired-end", "1", str(read1), str(read2)]
    else:
        # Include an odd trailing read to preserve interleaved serialization.
        read1.write_text(first + second + first.replace("@pair/1", "@trailing"))
        monkeypatch.setattr(
            mapping, "handle_input_fastq",
            lambda *args, **kwargs: {"interleaved_files": [read1]},
        )
        inputs = ["-i", str(read1)]

    posthoc = alignments.filter_sam_by_pair_status
    calls = []

    def filter_file(*args, **kwargs):
        calls.append(args)
        return posthoc(*args, **kwargs)

    monkeypatch.setattr(alignments, "filter_sam_by_pair_status", filter_file)
    result = runner.invoke(rolypoly, [
        "map", "-r", str(reference), "-o", str(output), "-m", "minimap2",
        *inputs, "--concordant", "--include-unmapped", "-cp", "none",
        "-g", str(tmp_path / "map.log"),
    ], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    path = next(output.rglob("*.sam"))
    content = path.read_text()
    assert "@SQ\tSN:ref" in content
    assert all(line.startswith("@") for line in content.splitlines())
    assert len(calls) == (0 if stage == "stream" else 1)


@pytest.mark.parametrize(
    "capability,option",
    [("supports_concordant", "--concordant"), ("supports_proper", "--proper")],
)
def test_mapper_specs_drive_capability_validation(
    runner: CliRunner, tmp_path: Path, monkeypatch, capability: str, option: str,
) -> None:
    from rolypoly.commands.reads import map as mapping

    spec = mapping.MAPPER_SPECS["bbmap"].copy()
    if capability == "supports_concordant":
        spec["supports_concordant"] = False
    else:
        spec["supports_proper"] = False
    monkeypatch.setitem(mapping.MAPPER_SPECS, "bbmap", spec)
    reference = tmp_path / "ref.fa"
    reference.write_text(">ref\nACGTACGT\n")
    read1, read2 = tmp_path / "R1.fq", tmp_path / "R2.fq"
    read1.write_text("@pair/1\nACGT\n+\nIIII\n")
    read2.write_text("@pair/2\nACGT\n+\nIIII\n")
    output = tmp_path / "mapping"
    result = runner.invoke(rolypoly, [
        "map", "-r", str(reference), "-o", str(output), "-m", "bbmap",
        "--paired-end", "1", str(read1), str(read2), option, "-cp", "none",
        "-g", str(tmp_path / "map.log"),
    ])
    assert result.exit_code != 0
    assert option in result.output and "bbmap adapter" in result.output
    assert not (output / "bbmap").exists()


def test_mapper_specs_define_map_and_roll_cli_choices() -> None:
    from rolypoly.commands.misc.end_2_end import roll
    from rolypoly.commands.reads.map import MAPPER_SPECS, map as map_reads

    for command, expected in (
        (map_reads, tuple(MAPPER_SPECS)), (roll, (*MAPPER_SPECS, "none")),
    ):
        parameter = next(param for param in command.params if param.name == "mapper")
        assert isinstance(parameter.type, click.Choice)
        assert tuple(parameter.type.choices) == expected


@pytest.mark.parametrize("resume_mapping", [False, True])
@pytest.mark.parametrize("include_unmapped", [False, True])
def test_roll_sam_compression(
    runner: CliRunner, tmp_path: Path, resume_mapping: bool,
    include_unmapped: bool,
) -> None:
    if not shutil.which("pigz") or not shutil.which("bwa-mem2"):
        pytest.skip("pigz or bwa-mem2 unavailable")
    rng = random.Random(218)
    sequence = "".join(rng.choices("ACGT", k=600))
    reads = tmp_path / "reads.fq"
    reads.write_text(
        f"@mapped\n{sequence[100:250]}\n+\n{'I' * 150}\n"
        f"@unmapped\n{'N' * 150}\n+\n{'I' * 150}\n"
    )
    output = tmp_path / "roll"
    assembly = output / "assembly"
    assembly.mkdir(parents=True)
    (assembly / "final_assembly.fasta").write_text(f">contig\n{sequence}\n")
    sam = output / "read_mapping/bwa-mem2/previous.sam"
    previous = b"@HD\tVN:1.6\nunmapped\t4\t*\t0\t0\t*\t*\t0\t0\tACGT\tIIII\n"
    if resume_mapping:
        sam.parent.mkdir(parents=True)
        sam.write_bytes(previous)
    args = [
        "roll", "-i", str(reads), "-o", str(output), "-t", "1",
        "--skip-existing", "--cluster-backend", "none", "-m", "bwa-mem2",
        "-g", str(tmp_path / "roll.log"),
        "--skip-steps",
        "filter_reads,filter_contigs,marker_search,nucleic_search,"
        "annotate,rdrp_motif_search,taxonomy,report",
    ]
    if include_unmapped:
        args.append("--include-unmapped")
    result = runner.invoke(rolypoly, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    archives = list((output / "read_mapping").rglob("*.sam.gz"))
    assert len(archives) == 1
    assert not list((output / "read_mapping").rglob("*.sam"))
    with gzip.open(archives[0], "rb") as handle:
        content = handle.read()
    assert content == previous if resume_mapping else b"mapped\t" in content
    if not resume_mapping:
        assert (b"\nunmapped\t" in content) == include_unmapped
    resumed = runner.invoke(rolypoly, args, catch_exceptions=False)
    assert resumed.exit_code == 0, resumed.output
    with gzip.open(archives[0], "rb") as handle:
        assert handle.read() == content


@pytest.mark.parametrize("output_format", ["tsv", "csv", "gff3"])
@pytest.mark.parametrize("resolved", [False, True])
@pytest.mark.parametrize("with_metadata", [False, True])
def test_annotate_rna_family_types(
    runner: CliRunner, tmp_path: Path, monkeypatch,
    output_format: str, resolved: bool, with_metadata: bool,
) -> None:
    from urllib.parse import unquote

    from rolypoly.utils import various
    from rolypoly.utils.viz.genome_maps import attach_rna, build_rna_by_contig, render_html

    fixture = repo_root() / "testing_folder/inputs/annotations/rfam_typing.tblout"
    output = tmp_path / "annotation"
    fasta = tmp_path / "input.fna"
    fasta.write_text(">contig1\n" + "ACGT" * 200 + "\n")
    bundle = tmp_path / "bundle"
    monkeypatch.setenv("ROLYPOLY_DATA", str(bundle))
    if with_metadata:
        profiles = bundle / "profiles"
        profiles.mkdir(parents=True)
        metadata = pl.DataFrame({
            "rfam_id": ["RF99999"], "profile_name": ["curated_model"],
            "rfam_type": ["Gene; sRNA;"], "rfam_description": ["Curated noncoding RNA"],
            "rfam_release": ["15.1"], "metadata_source": ["synthetic:curated"],
            "Ontology_term": ["GO:0003723"],
        })
        with gzip.open(profiles / "rfam_family.tsv.gz", "wb") as handle:
            metadata.write_csv(handle, separator="\t")

    def infernal(**kwargs):
        assert kwargs["base_cmd"] == "cmscan"
        Path(kwargs["output_file"]).write_text(fixture.read_text())
        return True

    monkeypatch.setattr(various, "run_command_comp", infernal)
    skips = "predict_secondary_structure,predict_trnas"
    if not resolved:
        skips += ",resolve_rna_element_overlaps"
    result = runner.invoke(rolypoly, [
        "annotate-rna", "-i", str(fasta), "-o", str(output),
        "--output-format", output_format, "--skip-steps", skips,
        "-g", str(tmp_path / "rna.log"),
    ], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    expected = [
        "cytosolic_LSU_rRNA", "cytosolic_LSU_rRNA", "cytosolic_5_8S_rRNA",
        "cytosolic_SSU_rRNA", "tRNA", "riboswitch", "ribozyme", "nucleotide_match",
    ]
    if with_metadata:
        expected[-1] = "ncRNA"
    raw = pl.read_csv(output / "ribozymes_annotations.tsv", separator="\t")
    assert raw["type"].to_list() == expected
    assert raw["rfam_id"].to_list()[-1] == "RF99999"
    reverse = raw.filter(pl.col("rfam_id") == "RF00002").row(0, named=True)
    assert (reverse["start"], reverse["end"], reverse["strand"]) == (191, 250, "-")
    assert (reverse["seq_from_raw"], reverse["seq_to_raw"]) == (250, 191)
    if resolved:
        selected = pl.read_csv(output / "ribozymes_resolved.out", separator="\t")
        assert selected.sort("start")["type"].to_list() == expected
    if output_format == "gff3":
        lines = (output / "combined_annotations.gff3").read_text().split("##FASTA")[0]
        records = [line.split("\t") for line in lines.splitlines() if not line.startswith("#")]
        assert sorted(record[2] for record in records) == sorted(expected)
        attributes = [
            dict(unquote(item).split("=", 1) for item in record[8].split(";"))
            for record in records
        ]
        assert {attrs["rfam_id"] for attrs in attributes} == set(raw["rfam_id"])
        assert all(attrs["so_accession"].startswith("SO:") for attrs in attributes)
        if with_metadata:
            assert next(attrs for attrs in attributes if attrs["rfam_id"] == "RF99999")[
                "Ontology_term"
            ] == "GO:0003723"
    else:
        table = pl.read_csv(
            output / f"combined_annotations.{output_format}",
            separator="\t" if output_format == "tsv" else ",",
        )
        assert table.sort("start")["type"].to_list() == expected
        assert set(raw.columns).issubset(table.columns)
        raw = table
    rna = build_rna_by_contig(raw)["contig1"]
    assert [feature["klass"] for feature in sorted(rna["features"], key=lambda f: f["start"])] == [
        "rRNA", "rRNA", "rRNA", "rRNA", "tRNA", "riboswitch", "ribozyme",
        "ncRNA" if with_metadata else "other",
    ]
    contigs = []
    attach_rna(contigs, {"contig1": rna})
    html = render_html(contigs)
    assert "cytosolic_LSU_rRNA" in html and expected[-1] in html


def multi_library_reads(directory: Path) -> Path:
    """Create tiny paired libraries with within- and cross-library duplicates."""
    directory.mkdir()
    rng = random.Random(145)
    shared = "".join(rng.choices("ACGT", k=160))
    for library in ("A", "B"):
        independent = "".join(rng.choices("ACGT", k=300))
        fragments = [shared, shared, independent]
        for mate in (1, 2):
            with (directory / f"{library}_R{mate}.fastq").open("w") as handle:
                for index, fragment in enumerate(fragments):
                    sequence = fragment[:100] if mate == 1 else fragment[-100:].translate(
                        str.maketrans("ACGT", "TGCA")
                    )[::-1]
                    handle.write(
                        f"@{library}_{index}/{mate}\n{sequence}\n+\n{'I' * len(sequence)}\n"
                    )
    return directory


@pytest.mark.parametrize("quality", [24, 25, 29, 30, 35])
def test_roll_and_filter_reads_default_parameters(
    runner: CliRunner, tmp_path: Path, monkeypatch, quality: int,
) -> None:
    from copy import deepcopy

    from rolypoly.commands.reads import filter_reads

    inputs = multi_library_reads(tmp_path / "inputs")
    output = tmp_path / "roll"
    assembly = output / "assembly"
    assembly.mkdir(parents=True)
    (assembly / "final_assembly.fasta").write_text(">CID_1\n" + "ACGT" * 100 + "\n")
    captured = []

    def process_reads(config, tracker):
        filter_reads.auto_tune_params(
            {"average_read_length": 100, "average_read_quality": quality},
            config, config.protected_step_params,
        )
        captured.append({
            "preset": config.preset,
            "params": deepcopy(config.step_params),
            "skips": sorted(config.skip_steps),
            "policy": config.multi_library_policy,
            "speed": config.speed,
            "trim_polya": config.trim_polya,
            "artifacts": config.remove_synthetic_artifacts_enabled,
        })
        config.cleanup_temp()

    monkeypatch.setattr(filter_reads, "process_reads", process_reads)
    standalone = runner.invoke(
        rolypoly,
        ["filter-reads", "-i", str(inputs), "-o", str(tmp_path / "filtered"),
         "-g", str(tmp_path / "filter.log")],
        catch_exceptions=False,
    )
    assert standalone.exit_code == 0, standalone.output
    rolled = runner.invoke(
        rolypoly,
        ["roll", "-i", str(inputs), "-o", str(output), "--skip-existing",
         "--cluster-backend", "none", "--skip-steps",
         "filter_contigs,marker_search,nucleic_search,map_reads,annotate,"
         "rdrp_motif_search,taxonomy,report",
         "-g", str(tmp_path / "roll.log")],
        catch_exceptions=False,
    )
    assert rolled.exit_code == 0, rolled.output
    assert len(captured) == 2
    assert captured[0] == captured[1]
    assert captured[0]["params"]["quality_trim_unmerged"] == {
        "qtrim": "rl", "trimq": 5, "minlen": 25,
    }
    for log in ("filter.log", "roll.log"):
        assert "effective quality_trim_unmerged trimq=5 minlen=25" in (
            tmp_path / log
        ).read_text()


@pytest.mark.parametrize("saved_preset", [None, "strict"])
@pytest.mark.parametrize("explicit_preset", [None, "auto"])
def test_filter_reads_config_preset_precedence(
    runner: CliRunner, tmp_path: Path, monkeypatch,
    saved_preset: str | None, explicit_preset: str | None,
) -> None:
    from rolypoly.commands.reads import filter_reads

    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({
        "input": str(tmp_path), "output": str(tmp_path / "out"),
        "preset": saved_preset, "log_file": str(tmp_path / "filter.log"),
    }))
    captured = []

    def process_reads(config, tracker):
        captured.append(config.preset)
        config.cleanup_temp()

    monkeypatch.setattr(filter_reads, "process_reads", process_reads)
    args = ["filter-reads", "--config-file", str(config_file)]
    if explicit_preset:
        args.extend(["--preset", explicit_preset])
    result = runner.invoke(rolypoly, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert captured == [explicit_preset or saved_preset]


def test_filter_reads_warns_about_inactive_speed(
    runner: CliRunner, tmp_path: Path, monkeypatch,
) -> None:
    from rolypoly.commands.reads import filter_reads

    monkeypatch.setattr(filter_reads, "process_reads", lambda config, tracker: config.cleanup_temp())
    log = tmp_path / "filter.log"
    result = runner.invoke(
        rolypoly,
        ["filter-reads", "-i", str(tmp_path), "-o", str(tmp_path / "out"),
         "--speed", "15", "-g", str(log)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "--speed is deprecated and has no effect" in log.read_text()


@pytest.mark.parametrize(
    "policy,preset,zip_reports",
    [
        ("keep-separate", None, False),
        ("single-output", None, False),
        ("pre-concatenate", None, False),
        ("keep-separate", "fast", False),
        ("keep-separate", None, True),
    ],
)
def test_filter_reads_multi_library_policy(
    runner: CliRunner, tmp_path: Path, monkeypatch,
    policy: str, preset: str | None, zip_reports: bool,
) -> None:
    import bbmapy
    from rolypoly.commands.assembly.assemble import handle_input_files
    from rolypoly.commands.reads import filter_reads

    if not shutil.which("java"):
        pytest.skip("BBMap runtime requires Java")
    inputs = multi_library_reads(tmp_path / "inputs")
    output = tmp_path / "roll" / "filtered_reads"
    calls = []
    original_clumpify = bbmapy.clumpify
    library_steps = []

    def clumpify(**kwargs):
        calls.append(kwargs)
        return original_clumpify(**kwargs)

    monkeypatch.setattr(bbmapy, "clumpify", clumpify)
    monkeypatch.setattr(filter_reads, "generate_reports", lambda *args, **kwargs: False)
    for name in ("discover_merge_adapters", "error_correct_1", "error_correct_2"):
        def step(path, config, tracker, step_name=name):
            library_steps.append((step_name, config.file_name))
            return path
        step.__name__ = name
        monkeypatch.setattr(filter_reads, name, step)
    args = [
        "filter-reads", "-i", str(inputs), "-o", str(output),
        "-mlp", policy, "--disable-auto", "-M", "1g", "-t", "1",
        "-g", str(tmp_path / "reads.log"), "-ll", "DEBUG",
        "--skip-steps",
        "filter_known_dna,decontaminate_rrna,filter_identified_dna,"
        "trim_adapters,trim_polya_tails,"
        "remove_synthetic_artifacts,entropy_filter,"
        "quality_trim_unmerged",
    ]
    if preset:
        args.extend(["--preset", preset])
    if zip_reports:
        args.append("--zip-reports")
    result = runner.invoke(rolypoly, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    if preset == "fast":
        policy = "pre-concatenate"
    libraries, count = handle_input_files(output)
    assert count == (2 if policy == "keep-separate" else 1)
    counts = {"merged": 0, "interleaved": 0}
    for library in libraries.values():
        for kind in counts:
            assert library[kind] is not None
            with gzip.open(library[kind], "rt") as handle:
                headers = handle.read().splitlines()[::4]
            counts[kind] += len(headers)
            if kind == "interleaved":
                assert len(headers) % 2 == 0
                for index in range(0, len(headers), 2):
                    assert headers[index].rsplit("/", 1)[0] == headers[index + 1].rsplit("/", 1)[0]
    assert counts == {"merged": 2 if policy == "keep-separate" else 1, "interleaved": 4}
    first_calls = [call for call in calls if "dedupe_first_" in call["out"]]
    assert len(first_calls) == (1 if policy == "pre-concatenate" else 2)
    assert len(library_steps) == (0 if policy == "pre-concatenate" else 6)
    if policy == "pre-concatenate":
        assert all(call.get("optical") == "f" for call in calls)
    if policy == "single-output":
        assert all("pooled_" in call["in_file"] for call in calls if "dedupe_final_" in call["out"])
    assert all(call["interleaved"] == "f" for call in calls if "dedupe_final_merged_" in call["out"])
    if zip_reports:
        import tarfile
        with tarfile.open(next(output.glob("*_run_info.tar.gz"))) as archive:
            with archive.extractfile("./libraries.tsv") as handle:
                manifest = pl.read_csv(handle, separator="\t")
            configs = [name for name in archive.getnames() if "library_config_" in name]
            assert len(configs) == 2
        assert not (output / "run_info").exists()
    else:
        manifest = pl.read_csv(output / "run_info/libraries.tsv", separator="\t")
    assert manifest.height == 2
    assert set(manifest["multi_library_policy"]) == {policy}
    assert all(Path(path).exists() for column in ("output_interleaved", "output_merged")
               for path in manifest[column].drop_nulls())
    assert not list(output.glob("rolypoly_tmp_*"))

    calls.clear()
    resumed = runner.invoke(rolypoly, args + ["--skip-existing"], catch_exceptions=False)
    if zip_reports:
        assert resumed.exit_code != 0
        assert "matching completed library manifest" in resumed.output
        assert not calls
        return
    assert resumed.exit_code == 0, resumed.output
    assert not calls
    from rolypoly.commands.assembly import assemble

    backends = []

    def backend(config, libraries):
        backends.append(len(libraries))
        contigs = config.output_dir / "backend.fa"
        contigs.write_text(">contig\n" + "ACGT" * 100 + "\n")
        return contigs

    def unexpected_filtering(*args, **kwargs):
        pytest.fail("roll reprocessed completed standalone filtered libraries")

    monkeypatch.setattr(filter_reads, "process_read_library", unexpected_filtering)
    monkeypatch.setattr(assemble, "run_megahit", backend)
    filtering_only = runner.invoke(
        rolypoly,
        ["roll", "-i", str(inputs), "-o", str(output.parent),
         "--skip-existing", "-mlp", policy,
         "-g", str(tmp_path / "filter_only.log"), "--skip-steps",
         "assemble,filter_contigs,cluster,length_filter,marker_search,"
         "nucleic_search,map_reads,annotate,rdrp_motif_search,taxonomy,report",
         *(["--filter-preset", preset] if preset else [])],
        catch_exceptions=False,
    )
    assert filtering_only.exit_code == 0, filtering_only.output
    assert not (output.parent / "assembly").exists()
    assert not list(output.parent.glob("rolypoly__*"))
    assert not backends and not calls
    roll_args = [
        "roll", "-i", str(inputs), "-o", str(output.parent),
        "--skip-existing", "-mlp", policy, "--assembler", "megahit",
        "--no-rmdup", "--cluster-backend", "none", "-M", "1g", "-t", "1",
        "-g", str(tmp_path / "roll.log"), "--skip-steps",
        "filter_contigs,marker_search,nucleic_search,map_reads,annotate,"
        "rdrp_motif_search,taxonomy,report",
    ]
    if preset:
        roll_args.extend(["--filter-preset", preset])
    for _ in range(2):
        rolled = runner.invoke(rolypoly, roll_args, catch_exceptions=False)
        assert rolled.exit_code == 0, rolled.output
    assert backends == [2 if policy == "keep-separate" else 1]
    assert not calls
    assert (output.parent / "assembly/final_assembly.fasta").is_file()
    assert "Reusing completed filtered libraries" in (tmp_path / "roll.log").read_text()
    changed = runner.invoke(
        rolypoly, args + ["--skip-existing", "--preset", "strict", "-mlp",
                         "single-output" if policy != "single-output" else "keep-separate"],
    )
    assert changed.exit_code != 0
    assert "matching completed library manifest" in changed.output
    transferred_inputs = inputs.rename(tmp_path / "transferred_inputs")
    transferred_output = tmp_path / "transferred_roll"
    shutil.copytree(output.parent, transferred_output, symlinks=False)
    transferred_args = [
        str(transferred_inputs) if arg == str(inputs)
        else str(transferred_output) if arg == str(output.parent)
        else arg
        for arg in roll_args
    ]
    automatic = runner.invoke(rolypoly, transferred_args)
    assert automatic.exit_code != 0
    assert "matching completed library manifest" in automatic.output
    explicit = runner.invoke(
        rolypoly, transferred_args + ["--skip-steps", "filter_reads,"
        "filter_contigs,marker_search,nucleic_search,map_reads,annotate,"
        "rdrp_motif_search,taxonomy,report"],
        catch_exceptions=False,
    )
    assert explicit.exit_code == 0, explicit.output
    assert backends == [2 if policy == "keep-separate" else 1]
    assert not calls


@pytest.mark.parametrize("keep_tmp", [False, True])
@pytest.mark.parametrize("mini", [False, True])
def test_roll_filtering_only_fresh_run(
    runner: CliRunner, tmp_path: Path, monkeypatch, keep_tmp: bool, mini: bool,
) -> None:
    from rolypoly.commands.reads import filter_reads

    inputs = multi_library_reads(tmp_path / "inputs")
    output = tmp_path / "roll"
    seen = []

    def process_reads(config, tracker):
        seen.append(Path(config.input))
        assert config.skip_steps == ["trim_polya_tails", "filter_known_dna"]
        (config.output_dir / "completed.txt").write_text("filtering completed")
        config.cleanup_temp()

    monkeypatch.setattr(filter_reads, "process_reads", process_reads)
    args = [
        "roll", "-i", str(inputs), "-o", str(output),
        "-g", str(tmp_path / "roll.log"), "--skip-steps",
        "subsample,assemble,filter_contigs,cluster,length_filter,marker_search,"
        "nucleic_search,map_reads,annotate,rdrp_motif_search,taxonomy,report",
    ]
    if keep_tmp:
        args.append("--keep-tmp")
    if mini:
        args.append("--mini")
    result = runner.invoke(rolypoly, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert seen == [inputs.resolve()]
    assert (output / "filtered_reads/completed.txt").exists()
    assert not (output / "assembly").exists()
    assert not (output / "mini_input").exists()
    assert bool(list(output.glob("rolypoly__*"))) == keep_tmp
    assert "Filtering-only roll completed" in (tmp_path / "roll.log").read_text()
    assert "ROLYPOLY_SUPPRESS_CITATIONS_WRITE" not in os.environ


@pytest.mark.parametrize("skipped", [
    set(), {"filter_contigs"}, {"cluster"}, {"length_filter"},
    {"filter_contigs", "cluster", "length_filter"},
])
@pytest.mark.parametrize("cached", [False, True])
def test_roll_contig_processing_skip_steps(
    runner: CliRunner, tmp_path: Path, monkeypatch, skipped: set[str], cached: bool,
) -> None:
    from rolypoly.commands.misc import end_2_end

    query = tmp_path / "contigs.fa"
    query.write_text(">short\n" + "ACGT" * 50 + "\n>long\n" + "ACGT" * 100 + "\n")
    output = tmp_path / "roll"
    assembly = output / "assembly"
    assembly.mkdir(parents=True)
    paths = {
        "filter_contigs": assembly / "filtered_assembly.fasta",
        "cluster": assembly / "clustered_assembly.fasta",
        "length_filter": assembly / "length_filtered.fasta",
    }
    if cached:
        (assembly / "final_assembly.fasta").write_text(">previous_input\nAAAA\n")
        for stage in skipped:
            paths[stage].write_text(">stale\n" + "G" * 500 + "\n")
    calls = []
    annotation_input = []

    class CommandContext:
        def invoke(self, command, **kwargs):
            calls.append(command.name)
            if command.name == "filter-contigs":
                shutil.copyfile(kwargs["input"], kwargs["output"])
            elif command.name == "cluster":
                shutil.copyfile(kwargs["input_path"], kwargs["representatives_fasta"])
            elif command.name == "annotate":
                annotation_input.append(Path(kwargs["input"]))
            else:
                pytest.fail(f"Unexpected command {command.name}")

    monkeypatch.setattr(end_2_end, "shared_command_context", lambda command: CommandContext())
    result = runner.invoke(
        rolypoly,
        ["roll", "-i", str(query), "-o", str(output), "--host", str(query),
         "--min-len", "300", "--skip-steps",
         ",".join(["filter_reads", "assemble", "marker_search", "nucleic_search",
                   "map_reads", "rdrp_motif_search", "taxonomy", "report", *sorted(skipped)]),
         "-g", str(tmp_path / "roll.log"), *(["--skip-existing"] if cached else [])],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    log_text = (tmp_path / "roll.log").read_text()
    assert ("without fully validating input" in log_text) == cached
    assert ("Contigs mode restages the supplied assembly" in log_text) == cached
    assert ("filter-contigs" in calls) == ("filter_contigs" not in skipped)
    assert ("cluster" in calls) == ("cluster" not in skipped)
    expected = assembly / "final_assembly.fasta"
    for stage in ("filter_contigs", "cluster", "length_filter"):
        if stage not in skipped:
            expected = paths[stage]
    assert annotation_input == [expected]
    text = expected.read_text()
    assert ">CID_2" in text and ">stale" not in text
    assert (">CID_1" in text) == ("length_filter" in skipped)
    for stage in skipped:
        assert paths[stage].exists() == cached
        if cached:
            assert ">stale" in paths[stage].read_text()


def test_roll_rejects_unknown_skip_step(runner: CliRunner, tmp_path: Path) -> None:
    output = tmp_path / "roll"
    result = runner.invoke(
        rolypoly, ["roll", "-i", str(tmp_path), "-o", str(output),
                   "--skip-steps", "length_filer"],
    )
    assert result.exit_code == 2, result.output
    assert "Unknown steps: length_filer" in result.output
    assert "length_filter" in result.output
    assert not output.exists()


def test_filter_reads_input_append_retains_every_library(tmp_path: Path) -> None:
    from rolypoly.commands.reads.filter_reads import ReadFilterConfig, process_input_fastq

    if not shutil.which("java"):
        pytest.skip("BBMap runtime requires Java")
    inputs = multi_library_reads(tmp_path / "inputs")
    config = ReadFilterConfig(
        input=str(inputs), output=str(tmp_path / "output"), memory="1g",
        log_file=str(tmp_path / "reads.log"),
    )
    combined, _, _ = process_input_fastq(config)
    with gzip.open(combined, "rt") as handle:
        headers = handle.read().splitlines()[::4]
    assert len(headers) == 12
    assert sum(header.startswith("@A_") for header in headers) == 6
    assert sum(header.startswith("@B_") for header in headers) == 6


@pytest.mark.parametrize("layout", ["single", "mixed", "interleaved", "all-merged"])
@pytest.mark.parametrize("policy", ["keep-separate", "single-output", "pre-concatenate"])
def test_filter_reads_library_layouts(
    runner: CliRunner, tmp_path: Path, monkeypatch, layout: str, policy: str
) -> None:
    from rolypoly.commands.assembly.assemble import handle_input_files
    from rolypoly.commands.reads import filter_reads

    if not shutil.which("java"):
        pytest.skip("BBMap runtime requires Java")
    inputs = multi_library_reads(tmp_path / "inputs")
    if layout == "single":
        for path in inputs.glob("*R2*"):
            path.unlink()
        for path in inputs.glob("*R1*"):
            text = path.read_text().replace("/1", "")
            path.rename(path.with_name(path.name.replace("_R1", "")))
            path.with_name(path.name.replace("_R1", "")).write_text(text)
    elif layout == "mixed":
        path = inputs / "B_R2.fastq"
        path.unlink()
        path = inputs / "B_R1.fastq"
        single = inputs / "single.fastq"
        single.write_text(path.read_text().replace("/1", ""))
        path.unlink()
    else:
        for library in ("A", "B"):
            r1, r2 = inputs / f"{library}_R1.fastq", inputs / f"{library}_R2.fastq"
            first, second = r1.read_text().splitlines(), r2.read_text().splitlines()
            if layout == "all-merged":
                first, second = first[:8], second[:8]
            combined = inputs / f"{library}.fastq"
            combined.write_text("".join(
                "\n".join(first[index:index + 4] + second[index:index + 4]) + "\n"
                for index in range(0, len(first), 4)
            ))
            r1.unlink()
            r2.unlink()
    monkeypatch.setattr(filter_reads, "generate_reports", lambda *args, **kwargs: False)
    output = tmp_path / "out"
    result = runner.invoke(
        rolypoly,
        ["filter-reads", "-i", str(inputs), "-o", str(output), "-mlp", policy,
         "--memory", "1g", "--disable-auto", "-g", str(tmp_path / "reads.log"),
         "--skip-steps",
         "filter_known_dna,decontaminate_rrna,filter_identified_dna,"
         "discover_merge_adapters,trim_adapters,trim_polya_tails,entropy_filter,"
         "error_correct_1,error_correct_2,quality_trim_unmerged"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    libraries, count = handle_input_files(output)
    assert count == (2 if policy == "keep-separate" else 1)
    tracker = pl.read_csv(output / "run_info/output_tracker.csv")
    assert not tracker.filter(pl.col("is_merged"))["interleaved"].any()
    if layout == "single" or (layout == "mixed" and policy == "pre-concatenate"):
        assert not tracker["interleaved"].any()
    elif layout != "mixed":
        assert tracker.filter(~pl.col("is_merged"))["interleaved"].all()
    total = 0
    for library in libraries.values():
        for path in library.values():
            if path:
                with gzip.open(path, "rt") as handle:
                    total += len(handle.read().splitlines()) // 4
    if layout == "single":
        assert total == (4 if policy == "keep-separate" else 3)
        assert all(library["interleaved"] is None for library in libraries.values())
    elif layout == "all-merged":
        assert total == (2 if policy == "keep-separate" else 1)
    elif layout == "interleaved":
        assert total == (6 if policy == "keep-separate" else 5)
    else:
        # Pre-concatenate mixed layouts must not pair unrelated single reads.
        assert total == 5
        if policy == "pre-concatenate":
            assert all(library["interleaved"] is None for library in libraries.values())


@pytest.mark.parametrize(
    "options,expected_policy,expected_assemblers",
    [
        ([], "keep-separate", ["spades_rnaviral", "megahit"]),
        (["--assembler", "megahit", "--no-rmdup"], "keep-separate", ["megahit"]),
        (["--assembler", "megahit,penguin", "-mlp", "single-output"],
         "single-output", ["megahit", "penguin"]),
        (["--preset", "fast"], "pre-concatenate", ["megahit"]),
        (["--mini", "--mini-subset-type", "first"], "pre-concatenate", ["megahit"]),
        (["--filter-preset", "fast", "-mlp", "single-output", "--assembler", "megahit"],
         "pre-concatenate", ["megahit"]),
        (["--assembly-preset", "fast", "--assembler", "spades_rnaviral"],
         "pre-concatenate", ["spades_rnaviral"]),
    ],
)
def test_roll_library_policy_and_explicit_assembler(
    runner: CliRunner, tmp_path: Path, monkeypatch, options: list[str],
    expected_policy: str, expected_assemblers: list[str],
) -> None:
    from rolypoly.commands.assembly import assemble
    from rolypoly.commands.reads import filter_reads
    from rolypoly.utils.bio.library_detection import handle_input_fastq

    inputs = multi_library_reads(tmp_path / "inputs")
    output = tmp_path / "roll"
    captured = {}
    backends = []

    def process_reads(config, tracker):
        info = handle_input_fastq(config.input, strict=True)
        assert len(info["R1_R2_pairs"]) == 2
        captured["policy"] = config.multi_library_policy
        for index in range(2 if expected_policy == "keep-separate" else 1):
            for kind in ("interleaved", "merged"):
                path = config.output_dir / f"dedupe_final_{kind}_library{index}.fq.gz"
                headers = ["x/1", "x/2"] if kind == "interleaved" else ["merged"]
                with gzip.open(path, "wt") as handle:
                    for header in headers:
                        handle.write(f"@{header}\n{'ACGT' * 25}\n+\n{'I' * 100}\n")

    def backend(config, libraries, mode=None, output_label=None):
        name = "spades_rnaviral" if mode == "rnaviral" else output_label
        backends.append(name)
        assert len(libraries) == (2 if expected_policy == "keep-separate" else 1)
        assert all(library["interleaved"] and library["merged"] for library in libraries.values())
        captured["dereplicate"] = config.dereplicate
        path = config.output_dir / f"{name}.fa"
        path.write_text(f">{name}\n{'ACGT' * 150}\n")
        return path

    monkeypatch.setattr(filter_reads, "process_reads", process_reads)
    monkeypatch.setattr(assemble, "run_spades", backend)
    monkeypatch.setattr(assemble, "run_megahit", lambda config, libs: backend(config, libs, output_label="megahit"))
    monkeypatch.setattr(assemble, "run_penguin", lambda config, libs: backend(config, libs, output_label="penguin"))
    result = runner.invoke(
        rolypoly,
        ["roll", "-i", str(inputs), "-o", str(output), "-t", "1",
         "--memory", "1g", "--sample-size", "6", "--cluster-backend", "none",
         "--skip-steps",
         "filter_contigs,marker_search,nucleic_search,map_reads,annotate,rdrp_motif_search,taxonomy,report",
         "--log-file", str(tmp_path / "roll.log"), *options],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert captured["policy"] == expected_policy
    assert backends == expected_assemblers
    assert captured["dereplicate"] == ("--no-rmdup" not in options)
    saved = json.loads((output / "assembly/assembly_config.json").read_text())
    assert saved["assembler"] == str(expected_assemblers)


def test_assemble_preserves_working_directory_tmp(
    runner: CliRunner, tmp_path: Path, monkeypatch,
) -> None:
    from rolypoly.commands.assembly import assemble

    monkeypatch.chdir(tmp_path)
    inputs = multi_library_reads(tmp_path / "inputs")
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    sentinel = scratch / "unrelated.txt"
    sentinel.write_text("preserve unrelated scratch data")
    output = scratch / "assembly"

    def backend(config, libraries):
        contigs = config.output_dir / "backend.fa"
        contigs.write_text(">contig\n" + "ACGT" * 100 + "\n")
        return contigs

    monkeypatch.setattr(assemble, "run_megahit", backend)
    result = runner.invoke(
        rolypoly,
        ["assemble", "--input-dir", str(inputs), "--assembler", "megahit",
         "-o", str(output), "--memory", "1g", "-g", str(tmp_path / "assembly.log")],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert sentinel.read_text() == "preserve unrelated scratch data"
    assert (output / "final_assembly.fasta").is_file()


def test_filter_reads_fetched_references_do_not_leak_between_libraries(
    tmp_path: Path, monkeypatch,
) -> None:
    from rolypoly.commands.reads.filter_reads import ReadFilterConfig, fetch_and_mask_genomes
    from rolypoly.utils.bio import genome_fetch

    def fetch(stats_file, output_file, **kwargs):
        if "lib001" in stats_file:
            Path(output_file).write_text(">host\n" + "A" * 100 + "\n")

    monkeypatch.setattr(genome_fetch, "fetch_genomes_from_stats_file", fetch)
    outputs = []
    for name in ("lib001", "lib002"):
        config = ReadFilterConfig(
            output=str(tmp_path / "out"), temp_dir=str(tmp_path / "work"),
            file_name=name, log_file=str(tmp_path / "reads.log"),
            adapters=str(tmp_path / "adapters.fa"), skip_steps=["mask_fetched_dna"],
        )
        (config.temp_dir / f"stats_decontaminate_rrna_{name}.txt").write_text("stats\n")
        outputs.append(fetch_and_mask_genomes(config))
    assert Path(outputs[0]).is_file()
    assert outputs[1] == "host_empty"


@pytest.mark.parametrize(
    "options,expected",
    [
        ([], ["first description", "reverse", "ambiguous", "ambiguous_rc",
              "palindrome", "unique", "first other description"]),
        (["--revcomp-as-different"], ["first description", "reverse", "ambiguous",
                                      "ambiguous_rc", "palindrome", "unique",
                                      "first other description"]),
        (["--revcomp-as-same"], ["first description", "ambiguous", "palindrome",
                                 "unique", "first other description"]),
        (["--case-sensitive"], ["first description", "duplicate", "reverse", "lower",
                                "ambiguous", "ambiguous_rc", "palindrome", "unique",
                                "first other description"]),
    ],
)
def test_dereplicate_seqs_policies(
    runner: CliRunner, tmp_path: Path, options: list[str], expected: list[str]
) -> None:
    result = runner.invoke(
        rolypoly,
        ["dereplicate-seqs", "-i", str(sequence_fixture("utilities.fasta")),
         "-g", str(tmp_path / "dedup.log"), "-ll", "DEBUG", *options],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert [line[1:] for line in result.stdout.splitlines() if line.startswith(">")] == expected
    assert result.stdout.startswith(">first description\nAaCG\n")
    assert "duplicates removed" in result.stderr


@pytest.mark.parametrize("by,count", [("id", 2), ("name", 3), ("seq", 3)])
def test_dereplicate_seqs_multiple_inputs(
    runner: CliRunner, tmp_path: Path, by: str, count: int
) -> None:
    first, second = tmp_path / "first.fa", tmp_path / "second.fa"
    first.write_text(">id description\nAAAA\n>other\nCCCC\n")
    second.write_text(">id other description\nGGGG\n>id description\nAAAA\n")
    result = runner.invoke(
        rolypoly, ["dereplicate-seqs", "-i", f"{first},{second}", "--by", by,
                   "-o", "-", "-g", str(tmp_path / "dedup.log")],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert result.stdout.count(">") == count
    assert result.stdout.count(">id description\nAAAA\n") == 1


def test_dereplicate_seqs_gzip_outputs(runner: CliRunner, tmp_path: Path) -> None:
    source = tmp_path / "reads.fq.gz"
    with gzip.open(source, "wt") as handle:
        handle.write(sequence_fixture("utilities.fastq").read_text())
    output, duplicates, groups = (tmp_path / "nested/unique.fa.gz",
                                  tmp_path / "duplicates.fa.gz", tmp_path / "groups.tsv.gz")
    result = runner.invoke(
        rolypoly, ["dereplicate-seqs", "-i", str(source), "--revcomp-as-same",
                   "-o", str(output), "-d", str(duplicates), "-m", str(groups),
                   "-g", str(tmp_path / "dedup.log")], catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    for path in (output, duplicates, groups):
        assert path.read_bytes().startswith(b"\x1f\x8b")
    with gzip.open(output, "rt") as handle:
        assert handle.read() == ">first description\nAACG\n"
    with gzip.open(duplicates, "rt") as handle:
        assert handle.read() == ">duplicate\nAACG\n>reverse\nCGTT\n"
    with gzip.open(groups, "rt") as handle:
        assert handle.read() == "3\tfirst description, duplicate, reverse\n"


@pytest.mark.parametrize(
    "options,expected",
    [
        (["-s", "first description"], ">first description\nAaCG\n"),
        (["-s", "first"], ""),
        (["-s", "first", "--match", "substring"],
         ">first description\nAaCG\n>first other description\nGGGG\n"),
        (["-H", str(sequence_fixture("headers.txt")), "-s", "palindrome"],
         ">first description\nAaCG\n>palindrome\nACGT\n>unique\nTTAA\n"),
        (["-s", "ambiguous", "--match", "substring", "--invert"],
         ">first description\nAaCG\n>duplicate\nAACG\n>reverse\nCGTT\n>lower\naacg\n"
         ">palindrome\nACGT\n>palindrome_copy\nACGT\n>unique\nTTAA\n"
         ">first other description\nGGGG\n"),
    ],
)
def test_extract_seqs_selection(
    runner: CliRunner, tmp_path: Path, options: list[str], expected: str
) -> None:
    result = runner.invoke(
        rolypoly, ["extract-seqs", "-i", str(sequence_fixture("utilities.fasta")),
                   "-g", str(tmp_path / "extract.log"), "-ll", "DEBUG", *options],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert result.stdout == expected
    if not expected:
        assert "No records were selected" in result.stderr


@pytest.mark.parametrize("directory", [False, True])
def test_extract_seqs_multiple_inputs_and_gzip(
    runner: CliRunner, tmp_path: Path, directory: bool
) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    first, second = inputs / "a.fa", inputs / "b.fq.gz"
    first.write_text(">shared description\nAAAA\n")
    with gzip.open(second, "wt") as handle:
        handle.write("@shared description\nCCCC\n+\nIIII\n@unique\nGGGG\n+\nJJJJ\n")
    output = tmp_path / "nested/selected.fa.gz"
    args = ["extract-seqs", "-i", str(inputs) if directory else f"{first},{second}",
            "-s", "shared description", "-s", "unique", "-g", str(tmp_path / "extract.log")]
    streamed = runner.invoke(rolypoly, args + ["-o", "-"], catch_exceptions=False)
    saved = runner.invoke(rolypoly, args + ["-o", str(output)], catch_exceptions=False)
    assert streamed.exit_code == saved.exit_code == 0
    assert streamed.stdout == ">shared description\nAAAA\n>unique\nGGGG\n"
    assert saved.stdout == ""
    assert output.read_bytes().startswith(b"\x1f\x8b")
    with gzip.open(output, "rt") as handle:
        assert handle.read() == streamed.stdout


def test_extract_seqs_progress_stays_off_stdout(
    runner: CliRunner, tmp_path: Path
) -> None:
    source = tmp_path / "many.fa"
    source.write_text(">drop\nA\n" * 100000 + ">keep\nC\n")
    result = runner.invoke(
        rolypoly, ["extract-seqs", "-i", str(source), "-s", "keep",
                   "-g", str(tmp_path / "extract.log")], catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert result.stdout == ">keep\nC\n"
    assert "Processed 100000 records" in result.stderr


@pytest.mark.parametrize("command", ["dereplicate-seqs", "extract-seqs"])
@pytest.mark.parametrize("case", ["existing", "input_output", "input_log", "malformed",
                                  "missing_input", "missing_selection", "bad_revcomp"])
def test_sequence_utilities_errors(
    runner: CliRunner, tmp_path: Path, command: str, case: str
) -> None:
    if (command == "extract-seqs" and case == "bad_revcomp") or (
        command == "dereplicate-seqs" and case == "missing_selection"
    ):
        pytest.skip("Other command's option")
    source = tmp_path / "input.fa"
    source.write_text(">first\nAACG\n")
    output, log = tmp_path / "out.fa.gz", tmp_path / "command.log"
    options = ["-s", "first"] if command == "extract-seqs" else []
    if case == "existing":
        output.write_bytes(b"preserve existing output")
    elif case == "input_output":
        output = source
    elif case == "input_log":
        log = source
    elif case == "malformed":
        source = tmp_path / "input.fq"
        source.write_text("@first\nAACG\n+\n")
    elif case == "missing_input":
        source.unlink()
    elif case == "missing_selection":
        options = []
    elif case == "bad_revcomp":
        options += ["--revcomp-as-same", "--case-sensitive"]
    before = {path: path.read_bytes() for path in (source, output) if path.exists()}
    result = runner.invoke(rolypoly, [command, "-i", str(source), "-o", str(output),
                                     "-g", str(log), *options])
    assert result.exit_code != 0, result.output
    assert result.stdout == ""
    for path, content in before.items():
        assert path.read_bytes() == content
    if output not in before:
        assert not output.exists()


@pytest.mark.parametrize("format", ["json", "toml"])
def test_detect_seq_layout_stdout_and_file_contract(
    runner: CliRunner, tmp_path: Path, format: str
) -> None:
    source = repo_root() / "testing_folder/inputs/reads/layout"
    inputs = tmp_path / 'inputs with "quotes" \u03b1 \U0001f41b \x7f'
    inputs.mkdir()
    for name in ("sample_R1.fq", "sample_R2.fq", "single.fq"):
        shutil.copyfile(source / name, inputs / name)
    with gzip.open(inputs / "interleaved.fq.gz", "wt") as handle:
        handle.write((source / "interleaved.fq").read_text())
    shutil.copyfile(source / "single.fq", inputs / "dedupe_final_merged_filtered.fq")
    shutil.copyfile(source / "single.fq", inputs / "dedupe_final_merged_orphan.fq")
    shutil.copyfile(
        source / "interleaved.fq", inputs / "dedupe_final_interleaved_filtered.fq"
    )
    fasta = inputs / "contigs.fa"
    fasta.write_text(">contig\nACGT\n")
    # Unknown sampled layout must remain visible despite single-end fallback.
    (inputs / "ambiguous.fq").write_text(
        "@a/1\nACGT\n+\nIIII\n@b/2\nACGT\n+\nIIII\n"
    )
    args = [
        "detect-seq-layout", "-i", str(inputs), "--format", format,
        "-g", str(tmp_path / "layout.log"), "-ll", "DEBUG",
    ]
    result = runner.invoke(rolypoly, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    parse = json.loads if format == "json" else tomllib.loads
    report = parse(result.stdout)
    assert "Detected Casava" in result.stderr
    assert report["schema_version"] == 1
    assert report["sample_size"] == 1000
    assert report["header_sample_size"] == 100
    assert report["fasta_files"] == [str(fasta)]
    info = report["fastq"]
    assert info["R1_R2_pairs"] == [
        [str(inputs / "sample_R1.fq"), str(inputs / "sample_R2.fq")]
    ]
    assert info["interleaved_files"] == [str(inputs / "interleaved.fq.gz")]
    assert info["single_end_files"] == [
        str(inputs / "ambiguous.fq"), str(inputs / "single.fq")
    ]
    assert info["rolypoly_data"]["filtered"] == {
        "interleaved": str(inputs / "dedupe_final_interleaved_filtered.fq"),
        "merged": str(inputs / "dedupe_final_merged_filtered.fq"),
    }
    assert info["rolypoly_data"]["orphan"] == {
        "merged": str(inputs / "dedupe_final_merged_orphan.fq")
    }
    details = info["file_details"][str(inputs / "interleaved.fq.gz")]
    assert details["file_type"] == "interleaved"
    assert details["is_gzipped"] is True
    assert details["pair_1_count"] == details["pair_2_count"] == 1
    assert details["average_read_length"] == 8
    assert details["average_read_quality"] == 40
    assert details["header_analysis"]["format"] == "casava"
    assert details["header_analysis"]["barcodes"] == ["AGGCTTCT"]
    assert info["file_details"][str(inputs / "ambiguous.fq")]["file_type"] == "unknown"

    output = tmp_path / "nested" / f"layout.{format}"
    saved = runner.invoke(rolypoly, args + ["-o", str(output)], catch_exceptions=False)
    assert saved.exit_code == 0, saved.output
    assert saved.stdout == ""
    assert parse(output.read_text()) == report
    original = output.read_bytes()
    repeated = runner.invoke(rolypoly, args + ["-o", str(output)], catch_exceptions=False)
    assert repeated.exit_code != 0
    assert output.read_bytes() == original


@pytest.mark.parametrize(
    "names,expected_types",
    [
        (["single.fq"], ["single"]),
        (["interleaved.fq"], ["interleaved"]),
        (["sample_R1.fq", "sample_R2.fq"], ["paired_R1", "paired_R2"]),
        (
            ["single.fq", "sample_R1.fq", "sample_R2.fq"],
            ["single", "paired_R1", "paired_R2"],
        ),
        (
            ["sample_R1.fq", "contigs.fa", "sample_R2.fq"],
            ["paired_R1", "paired_R2"],
        ),
    ],
)
def test_detect_seq_layout_explicit_inputs(
    runner: CliRunner, tmp_path: Path, names: list[str], expected_types: list[str]
) -> None:
    source = repo_root() / "testing_folder/inputs/reads/layout"
    fasta = tmp_path / "contigs.fa"
    fasta.write_text(">contig\nACGT\n")
    paths = [fasta if name == "contigs.fa" else source / name for name in names]
    fastq_paths = [path for path in paths if path != fasta]
    result = runner.invoke(
        rolypoly,
        ["detect-layout", "-i", ",".join(map(str, paths)), "-o", "-",
         "-g", str(tmp_path / "layout.log")],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["fasta_files"] == ([str(fasta)] if fasta in paths else [])
    info = report["fastq"]
    assert [
        info["file_details"][str(path)]["file_type"] for path in fastq_paths
    ] == expected_types
    if len(fastq_paths) == 2:
        assert info["R1_R2_pairs"] == [list(map(str, fastq_paths))]
    else:
        key = "interleaved_files" if expected_types == ["interleaved"] else "single_end_files"
        assert info[key] == list(map(str, fastq_paths))
        assert info["R1_R2_pairs"] == []


@pytest.mark.parametrize(
    "case", [
        "missing", "empty", "malformed", "non_fastq", "unsupported", "empty_dir",
        "mixed_directories", "invalid_format", "empty_input", "bad_directory",
        "bad_pair", "bad_filtered", "bad_gzip",
    ]
)
def test_detect_seq_layout_invalid_inputs(
    runner: CliRunner, tmp_path: Path, case: str
) -> None:
    path = tmp_path / "reads.fq"
    input_value = str(path)
    extra = []
    if case == "empty":
        path.touch()
    elif case == "malformed":
        path.write_text("@read\nACGT\n+\n")
    elif case == "non_fastq":
        path.write_text(">contig\nACGT\n")
    elif case == "unsupported":
        path = tmp_path / "notes.txt"
        path.write_text("not sequence data\n")
        input_value = str(path)
    elif case == "empty_dir":
        input_value = str(tmp_path)
    elif case == "mixed_directories":
        input_value = f"{tmp_path},{tmp_path}"
    elif case == "invalid_format":
        extra = ["--format", "yaml"]
    elif case == "empty_input":
        input_value = ""
    elif case == "bad_directory":
        path.write_text("@read\nACGT\n+\n")
        input_value = str(tmp_path)
    elif case == "bad_pair":
        path.write_text("@read\nACGT\n+\n")
        mate = repo_root() / "testing_folder/inputs/reads/layout/sample_R1.fq"
        input_value = f"{mate},{path}"
    elif case == "bad_filtered":
        path = tmp_path / "dedupe_final_merged_sample.fq"
        path.write_text("@read\nACGT\n+\n")
        input_value = str(tmp_path)
    elif case == "bad_gzip":
        path = tmp_path / "reads.fq.gz"
        path.write_bytes(b"\x1f\x8bnot a gzip stream")
        input_value = str(path)
    result = runner.invoke(
        rolypoly,
        ["detect-seq-layout", "-i", input_value, "-g", str(tmp_path / "layout.log"),
         "-o", str(tmp_path / "output.json"), *extra],
        catch_exceptions=False,
    )
    assert result.exit_code != 0, result.output
    assert result.stdout == ""
    assert not (tmp_path / "output.json").exists()


def render(value: str, tmp_path: Path) -> str:
    return (
        value.replace("{tmp}", str(tmp_path))
        .replace("{data}", os.environ["ROLYPOLY_DATA"])
    )


def render_values(values: list[str], tmp_path: Path) -> list[str]:
    return [render(str(value), tmp_path) for value in values]


def pick_log_file_option(command_name: str) -> str | None:
    ctx = click.Context(rolypoly)
    command = rolypoly.get_command(ctx, command_name)
    if command is None:
        return None

    for parameter in command.params:
        if not isinstance(parameter, click.Option):
            continue
        if "--log-file" in parameter.opts:
            return "--log-file"
    return None


def inject_log_file_arg(
    args: list[str], tmp_path: Path, scenario_id: str
) -> list[str]:
    if not args:
        return args

    command_name = args[0]
    if "--log-file" in args:
        return args

    log_option = pick_log_file_option(command_name)
    if not log_option:
        return args

    log_file_path = tmp_path / f"{command_name}_{scenario_id}.log"
    return args + [log_option, str(log_file_path)]


def inject_debug_log_level(args: list[str]) -> list[str]:
    if not args:
        return args

    command_name = args[0]
    option_tokens = {"--log-level", "-ll", "-l"}
    if any(token in option_tokens for token in args[1:]):
        return args

    suffix = debug_log_suffix_for_command(command_name)
    if not suffix:
        return args
    return args + suffix


def pick_debug_value(option: click.Option) -> str:
    option_type = getattr(option, "type", None)
    choices = getattr(option_type, "choices", None)
    if not choices:
        return "DEBUG"

    for candidate in choices:
        if str(candidate).lower() == "debug":
            return str(candidate)
    return "DEBUG"


def debug_log_suffix_for_command(command_name: str) -> list[str]:
    ctx = click.Context(rolypoly)
    command = rolypoly.get_command(ctx, command_name)
    if command is None:
        return []

    for parameter in command.params:
        if not isinstance(parameter, click.Option):
            continue

        if "--log-level" in parameter.opts:
            return ["--log-level", pick_debug_value(parameter)]
        if "-ll" in parameter.opts:
            return ["-ll", pick_debug_value(parameter)]
        if "-l" in parameter.opts:
            return ["-l", pick_debug_value(parameter)]

    return []


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_table(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return pl.read_csv(path, separator="\t")
    if suffix == ".csv":
        return pl.read_csv(path)
    if suffix == ".parquet":
        return pl.read_parquet(path)
    raise ValueError(f"Unsupported table format for schema checks: {path}")


def apply_preconditions(scenario: dict, tmp_path: Path) -> None:
    for command_name in scenario.get("required_commands", []):
        if shutil.which(command_name) is None:
            pytest.skip(
                f"Scenario '{scenario['id']}' skipped: required command '{command_name}' not found"
            )

    for group in scenario.get("required_any_commands", []):
        if not any(shutil.which(command_name) for command_name in group):
            pretty = ", ".join(group)
            pytest.skip(
                f"Scenario '{scenario['id']}' skipped: none of [{pretty}] are available"
            )

    for required_path in scenario.get("required_paths", []):
        path_obj = Path(render(required_path, tmp_path))
        if not path_obj.exists():
            pytest.skip(
                f"Scenario '{scenario['id']}' skipped: required path missing ({path_obj})"
            )

    for fixture_name in scenario.get("fixtures", []):
        source = repo_root() / "testing_folder" / "inputs" / "mmtax"
        fixture_dir = tmp_path / "mmtax_fixture"
        fixture_dir.mkdir(parents=True, exist_ok=True)
        if fixture_name == "mmtax_mmseqs_db":
            database = fixture_dir / "ncbi_virus"
            subprocess.run(
                [
                    "mmseqs",
                    "createdb",
                    str(source / "reference.faa"),
                    str(database),
                    "--dbtype",
                    "1",
                ],
                check=True,
            )
            subprocess.run(
                [
                    "mmseqs",
                    "createtaxdb",
                    str(database),
                    str(fixture_dir / "taxonomy_tmp"),
                    "--ncbi-tax-dump",
                    str(source / "taxonomy"),
                    "--tax-mapping-file",
                    str(source / "accession2taxid.tsv"),
                ],
                check=True,
            )
        elif fixture_name == "mmtax_diamond_db":
            subprocess.run(
                [
                    "diamond",
                    "makedb",
                    "--in",
                    str(source / "reference.faa"),
                    "--db",
                    str(fixture_dir / "ncbi_virus"),
                    "--taxonmap",
                    str(source / "accession2taxid_diamond.tsv"),
                    "--taxonnodes",
                    str(source / "taxonomy" / "nodes.dmp"),
                    "--taxonnames",
                    str(source / "taxonomy" / "names.dmp"),
                    "--threads",
                    "1",
                ],
                check=True,
            )
        elif fixture_name == "nucleic_search_inputs":
            contig_source = repo_root() / "testing_folder/inputs/contigs"
            shutil.copyfile(
                contig_source / "test_contigs.fasta",
                tmp_path / "nucleic_queries_1.fasta",
            )
            shutil.copyfile(
                contig_source / "test_fasta_for_masking.fasta",
                tmp_path / "nucleic_queries_2.fasta",
            )
        else:
            raise ValueError(f"Unknown CLI fixture: {fixture_name}")


def load_cli_scenarios() -> list[dict]:
    scenario_path = Path(__file__).with_name("cli_scenarios.json")
    return json.loads(scenario_path.read_text())


def parse_csv_values(raw_value: str | None) -> set[str]:
    if not raw_value:
        return set()
    return {item.strip() for item in raw_value.split(",") if item.strip()}


def should_skip_scenario(
    scenario: dict, request: pytest.FixtureRequest
) -> str | None:
    scenario_ids = parse_csv_values(
        request.config.getoption("--cli-scenarios")
        or os.environ.get("RP_CLI_SCENARIOS")
    )
    command_names = parse_csv_values(
        request.config.getoption("--cli-commands")
        or os.environ.get("RP_CLI_COMMANDS")
    )
    match_tokens = parse_csv_values(
        request.config.getoption("--cli-match")
        or os.environ.get("RP_CLI_MATCH")
    )

    scenario_id = str(scenario.get("id", ""))
    command_name = (
        str(scenario.get("args", [""])[0]) if scenario.get("args") else ""
    )
    searchable = " ".join(
        [scenario_id, str(scenario.get("description", "")), command_name]
    ).lower()

    if scenario_ids and scenario_id not in scenario_ids:
        return f"scenario id '{scenario_id}' not selected"

    if command_names and command_name not in command_names:
        return f"command '{command_name}' not selected"

    if match_tokens and not any(
        token.lower() in searchable for token in match_tokens
    ):
        return "no cli-match token matched"

    return None


@pytest.mark.parametrize(
    "scenario", load_cli_scenarios(), ids=lambda row: row["id"]
)
def test_cli_scenarios(
    runner: CliRunner,
    tmp_path: Path,
    scenario: dict,
    request: pytest.FixtureRequest,
) -> None:
    skip_reason = should_skip_scenario(scenario, request)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    apply_preconditions(scenario, tmp_path)

    args = render_values(scenario["args"], tmp_path)
    args = inject_log_file_arg(
        args, tmp_path, str(scenario.get("id", "scenario"))
    )
    args = inject_debug_log_level(args)

    result = runner.invoke(rolypoly, args, catch_exceptions=False)

    assert result.exit_code == 0, (
        f"Scenario '{scenario['id']}' failed with args: {args}\n{result.output}"
    )

    for expected in scenario.get("expected_files", []):
        expected_path = Path(render(expected, tmp_path))
        assert expected_path.exists(), (
            f"Expected output file missing: {expected_path}"
        )
        assert expected_path.stat().st_size > 0, (
            f"Output file is empty: {expected_path}"
        )

    for expected_dir in scenario.get("expected_dirs", []):
        expected_dir_path = Path(render(expected_dir, tmp_path))
        assert expected_dir_path.exists(), (
            f"Expected output directory missing: {expected_dir_path}"
        )
        assert expected_dir_path.is_dir(), (
            f"Expected directory path is not a directory: {expected_dir_path}"
        )

    for file_path, required_tokens in scenario.get(
        "expected_contains", {}
    ).items():
        rendered_path = Path(render(file_path, tmp_path))
        content = rendered_path.read_text()
        for token in required_tokens:
            assert token in content, (
                f"Expected token '{token}' not found in {rendered_path}"
            )

    for file_path, expected_columns in scenario.get(
        "expected_table_columns", {}
    ).items():
        rendered_path = Path(render(file_path, tmp_path))
        frame = read_table(rendered_path)
        for column_name in expected_columns:
            assert column_name in frame.columns, (
                f"Expected column '{column_name}' not found in {rendered_path}. "
                f"Observed columns: {frame.columns}"
            )

    for file_path, min_rows in scenario.get(
        "expected_table_min_rows", {}
    ).items():
        rendered_path = Path(render(file_path, tmp_path))
        frame = read_table(rendered_path)
        assert frame.height >= int(min_rows), (
            f"Expected at least {min_rows} rows in {rendered_path}, found {frame.height}"
        )

    for file_path, expected_prefix in scenario.get(
        "expected_checksum_prefix", {}
    ).items():
        rendered_path = Path(render(file_path, tmp_path))
        digest = sha256(rendered_path)
        assert digest.startswith(expected_prefix), (
            f"Checksum mismatch for {rendered_path}: expected prefix {expected_prefix}, got {digest}"
        )
