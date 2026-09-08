from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from click.testing import CliRunner

from rolypoly.rolypoly import rolypoly


@pytest.fixture(scope="module")
def runner() -> CliRunner:
    return CliRunner()


def test_roll_skipped_nucleic_search_ignores_cached_nucleic_hits(
    tmp_path: Path, runner: CliRunner
) -> None:
    query = tmp_path / "input.fa"
    query.write_text(">first\n" + "A" * 250 + "\n>second\n" + "C" * 250 + "\n")

    output = tmp_path / "roll_out"
    marker_output = output / "marker_search_results"
    marker_output.mkdir(parents=True)
    pl.DataFrame(
        [{"source_seq_id": "CID_1", "marker_role": "candidate", "score": 100.0}]
    ).write_csv(marker_output / "marker_search_results.tsv", separator="\t")

    stale_nucleic_output = output / "nucleic_search_results"
    stale_nucleic_output.mkdir()
    pl.DataFrame([{"qheader": "CID_2"}]).write_csv(
        stale_nucleic_output / "cached_vs_db.tab", separator="\t"
    )

    result = runner.invoke(
        rolypoly,
        [
            "roll",
            "--input",
            str(query),
            "--output-dir",
            str(output),
            "--skip-existing",
            "--skip-steps",
            (
                "filter_reads,assemble,nucleic_search,map_reads,annotate,"
                "rdrp_motif_search,taxonomy,report"
            ),
            "--cluster-backend",
            "none",
            "--min-len",
            "1",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    matched = (output / "all_matched_contigs.fasta").read_text()
    assert ">CID_1" in matched
    assert ">CID_2" not in matched


def test_roll_active_nucleic_search_reuses_cached_hits_with_skip_existing(
    tmp_path: Path, runner: CliRunner
) -> None:
    query = tmp_path / "input.fa"
    query.write_text(">first\n" + "A" * 250 + "\n>second\n" + "C" * 250 + "\n")

    output = tmp_path / "roll_out"
    nucleic_output = output / "nucleic_search_results"
    nucleic_output.mkdir(parents=True)
    pl.DataFrame([{"qheader": "CID_2"}]).write_csv(
        nucleic_output / "cached_vs_db.tab", separator="\t"
    )

    result = runner.invoke(
        rolypoly,
        [
            "roll",
            "--input",
            str(query),
            "--output-dir",
            str(output),
            "--skip-existing",
            "--skip-steps",
            (
                "filter_reads,assemble,marker_search,map_reads,annotate,"
                "rdrp_motif_search,taxonomy,report"
            ),
            "--cluster-backend",
            "none",
            "--min-len",
            "1",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    matched = (output / "all_matched_contigs.fasta").read_text()
    assert ">CID_1" not in matched
    assert ">CID_2" in matched


def test_roll_skipping_both_discovery_steps_uses_final_assembly(
    tmp_path: Path, monkeypatch, runner: CliRunner
) -> None:
    from rolypoly.commands.misc import end_2_end

    query = tmp_path / "input.fa"
    query.write_text(">first\n" + "A" * 250 + "\n>second\n" + "C" * 250 + "\n")

    output = tmp_path / "roll_out"
    stale_nucleic_output = output / "nucleic_search_results"
    stale_nucleic_output.mkdir(parents=True)
    pl.DataFrame([{"qheader": "CID_2"}]).write_csv(
        stale_nucleic_output / "cached_vs_db.tab", separator="\t"
    )

    captured = {}

    class FakeCommandContext:
        def invoke(self, _command, **kwargs):
            captured["input"] = kwargs["input"]
            Path(kwargs["output"]).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        end_2_end,
        "shared_command_context",
        lambda _command: FakeCommandContext(),
    )

    result = runner.invoke(
        rolypoly,
        [
            "roll",
            "--input",
            str(query),
            "--output-dir",
            str(output),
            "--skip-existing",
            "--skip-steps",
            (
                "filter_reads,assemble,marker_search,nucleic_search,map_reads,"
                "rdrp_motif_search,taxonomy,report"
            ),
            "--cluster-backend",
            "none",
            "--min-len",
            "1",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    annotate_input = Path(captured["input"])
    assert annotate_input == output / "assembly" / "length_filtered.fasta"
    assembly_text = annotate_input.read_text()
    assert ">CID_1" in assembly_text
    assert ">CID_2" in assembly_text
    assert not (output / "all_matched_contigs.fasta").exists()
