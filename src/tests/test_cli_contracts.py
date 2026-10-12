from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
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
    output = tmp_path / "output"
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
    changed = runner.invoke(
        rolypoly, args + ["--skip-existing", "--preset", "strict", "-mlp",
                         "single-output" if policy != "single-output" else "keep-separate"],
    )
    assert changed.exit_code != 0
    assert "matching completed library manifest" in changed.output


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
