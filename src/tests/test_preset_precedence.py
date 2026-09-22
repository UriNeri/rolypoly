"""Explicit per-tool options survive presets and automatic read-length tuning."""

import json

import pytest
import rich_click as click

from rolypoly.commands.assembly.assemble import (
    AssemblyConfig,
    apply_assembly_preset,
    tune_assembly_kmers,
)
from rolypoly.commands.reads.filter_reads import (
    ReadFilterConfig,
    apply_filter_reads_preset,
    auto_tune_params,
)


@pytest.mark.parametrize("as_json", [False, True])
def test_filter_overrides_survive_preset_and_auto_tuning(tmp_path, as_json):
    overrides = {
        "quality_trim_unmerged": {"trimq": 17},
        "decontaminate_rrna": {"mincovfraction": 0.83},
    }
    config = ReadFilterConfig(
        input=str(tmp_path),
        output=str(tmp_path / "out"),
        log_file=str(tmp_path / "test.log"),
        adapters=str(tmp_path / "adapters.fa"),
        override_parameters=json.dumps(overrides) if as_json else overrides,
    )
    protected = apply_filter_reads_preset(
        "rna_virus_metat", click.Context(click.Command("test")), config
    )
    config.protected_step_params.update(protected)
    auto_tune_params(
        {"average_read_length": 150, "average_read_quality": 35},
        config,
        config.protected_step_params,
    )
    assert config.step_params["quality_trim_unmerged"]["trimq"] == 17
    assert config.step_params["quality_trim_unmerged"]["minlen"] == 25
    assert config.step_params["decontaminate_rrna"]["mincovfraction"] == 0.83


@pytest.mark.parametrize("as_json", [False, True])
def test_assembly_overrides_survive_preset_and_auto_tuning(tmp_path, as_json):
    overrides = {
        "spades": {"k": "21,55,127", "mode": "isolate"},
        "megahit": {"k-max": 127, "k-step": 18},
    }
    config = AssemblyConfig(
        output=tmp_path / "out",
        log_file=tmp_path / "test.log",
        override_parameters=json.dumps(overrides) if as_json else overrides,
    )
    apply_assembly_preset(
        "metatranscriptome", click.Context(click.Command("test")), config
    )
    tune_assembly_kmers(config, 75)
    assert config.step_params["spades"] == overrides["spades"]
    assert config.step_params["megahit"]["k-max"] == 127
    assert config.step_params["megahit"]["k-step"] == 18
    assert config.assembler == ["spades", "megahit"]


def test_assembly_defaults_still_tune_to_read_length(tmp_path):
    config = AssemblyConfig(
        output=tmp_path / "out", log_file=tmp_path / "test.log"
    )
    apply_assembly_preset(
        "rna_virus", click.Context(click.Command("test")), config
    )
    tune_assembly_kmers(config, 75)
    assert config.step_params["spades"]["k"] == "21,33,55"
    assert config.step_params["megahit"]["k-max"] == 73
    assert config.step_params["megahit"]["k-step"] == 10


def test_explicit_assembly_flags_survive_preset(tmp_path):
    config = AssemblyConfig(
        output=tmp_path / "out",
        log_file=tmp_path / "test.log",
        assembler=["penguin"],
        dereplicate=False,
        spades_mode="isolate",
    )
    command = click.Command(
        "test",
        params=[
            click.Option(["--assembler"]),
            click.Option(["--dereplicate"]),
            click.Option(["--spades-mode"]),
        ],
    )
    ctx = click.Context(command)
    ctx.params = {
        "assembler": ["penguin"],
        "dereplicate": False,
        "spades_mode": "isolate",
    }
    for name in ctx.params:
        ctx.set_parameter_source(name, click.core.ParameterSource.COMMANDLINE)
    apply_assembly_preset("metatranscriptome", ctx, config)
    assert config.assembler == ["penguin"]
    assert config.dereplicate is False
    assert config.step_params["spades"]["mode"] == "isolate"


def test_filter_auto_tunes_only_unprotected_parameters(tmp_path):
    config = ReadFilterConfig(
        input=str(tmp_path),
        output=str(tmp_path / "out"),
        log_file=str(tmp_path / "test.log"),
        adapters=str(tmp_path / "adapters.fa"),
        override_parameters={"quality_trim_unmerged": {"minlen": 42}},
    )
    auto_tune_params(
        {"average_read_length": 150, "average_read_quality": 35},
        config,
        config.protected_step_params,
    )
    assert config.step_params["quality_trim_unmerged"]["minlen"] == 42
    assert config.step_params["quality_trim_unmerged"]["trimq"] == 15
