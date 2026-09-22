"""Exercise skip flags through Click and preserve all assembler results."""

import importlib

import pytest
from click.testing import CliRunner

from rolypoly.rolypoly import rolypoly

assemble = importlib.import_module("rolypoly.commands.assembly.assemble")


@pytest.mark.parametrize(
    "skip_steps",
    [
        ("rename", "dereplicate"),
        ["rename", "dereplicate"],
        "rename, dereplicate",
    ],
)
def test_config_preserves_skip_steps(tmp_path, skip_steps):
    cfg = assemble.AssemblyConfig(
        output=tmp_path / "out",
        log_file=tmp_path / "assembly.log",
        skip_steps=skip_steps,
    )
    assert cfg.skip_steps == ["rename", "dereplicate"]


@pytest.mark.parametrize(
    "flags,expected_count,renamed",
    [
        (["--skip-steps", "dereplicate"], 4, True),
        (["--no-rmdup"], 4, True),
        (["--skip-steps", "rename"], 3, False),
        (["--skip-steps", "rename", "--skip-steps", "dereplicate"], 4, False),
        (["--skip-steps", "rename", "--no-rmdup"], 4, False),
    ],
)
def test_skip_flags_preserve_all_assemblers(
    tmp_path, monkeypatch, flags, expected_count, renamed
):
    def fake_spades(config, libraries, **kwargs):
        folder = config.output_dir / "spades_meta_output"
        folder.mkdir()
        path = folder / "contigs.fasta"
        path.write_text(">spades_shared\nAAAACCCC\n>spades_unique\nAACCAACC\n")
        return path

    def fake_megahit(config, libraries):
        folder = config.output_dir / "megahit_custom_out"
        folder.mkdir()
        path = folder / "final.contigs.fa"
        path.write_text(
            ">megahit_shared\nAAAACCCC\n>megahit_unique\nAAGGAAGG\n"
        )
        return path

    monkeypatch.setattr(assemble, "run_spades", fake_spades)
    monkeypatch.setattr(assemble, "run_megahit", fake_megahit)
    reads = tmp_path / "reads.fq"
    reads.write_text("@read\nAACCAACC\n+\nIIIIIIII\n")
    output = tmp_path / "output"
    result = CliRunner().invoke(
        rolypoly,
        [
            "assemble",
            "--single-end",
            "1",
            str(reads),
            "--output",
            str(output),
            "--log-file",
            str(tmp_path / "assembly.log"),
            *flags,
        ],
    )
    assert result.exit_code == 0, result.output + repr(result.exception)
    final = output / "final_assembly.fasta"
    assert final.is_file(), "Cleanup must preserve the final symlink target"
    text = final.read_text()
    assert text.count(">") == expected_count
    assert "AACCAACC" in text and "AAGGAAGG" in text
    assert (">CID_" in text) == renamed
    if expected_count == 4:
        assert not (output / "dereplicated_contigs.fasta").exists()
        assert text.count("AAAACCCC") == 2
