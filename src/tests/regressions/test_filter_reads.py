import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from rolypoly.commands.reads import filter_reads


@pytest.mark.parametrize("read_names", [
    ["merged_final_sample.fq.gz"],
    ["merged_final_sample.fq.gz", "unmerged_final_sample.fq.gz"],
])
@pytest.mark.parametrize("completed_location", ["temporary", "run_info"])
def test_falco_resume_requires_all_outputs(
    tmp_path, monkeypatch, read_names, completed_location
):
    work = tmp_path / "work"
    work.mkdir()
    output = tmp_path / "output"
    monkeypatch.setattr(filter_reads, "config", SimpleNamespace(
        temp_dir=work, output_dir=output,
    ))
    monkeypatch.setattr(filter_reads, "tools", [])
    for name in read_names:
        (work / name).write_bytes(b"input placeholder; backend is stubbed")
    filenames = [
        f"{prefix}{suffix}"
        for prefix in ([""] if len(read_names) == 1 else [f"{name}_" for name in read_names])
        for suffix in ("fastqc_data.txt", "summary.txt", "fastqc_report.html")
    ]
    completed = (
        work if completed_location == "temporary" else output / "run_info"
    ) / "falco_post_trim_reads"
    completed.mkdir(parents=True)
    for name in filenames:
        (completed / name).write_text("previous Falco output")

    calls = []

    def run_falco(**kwargs):
        calls.append(kwargs)
        destination = Path(kwargs["params"]["outdir"])
        for name in filenames:
            (destination / name).write_text("regenerated Falco output")
        return True

    monkeypatch.setattr(filter_reads, "run_command_comp", run_falco)
    logger = logging.getLogger("test_falco_resume")

    filter_reads.generate_reports("sample", 1, True, logger)
    assert not calls
    assert "falco" in filter_reads.tools
    if completed_location == "run_info":
        # An empty temporary QC folder would overwrite completed reports during cleanup.
        assert not (work / "falco_post_trim_reads").exists()

    # A missing summary for any input must force a new run.
    (completed / filenames[-2]).unlink()
    filter_reads.generate_reports("sample", 1, True, logger)
    assert len(calls) == 1
    assert calls[-1]["positional_args"] == [str(work / name) for name in read_names]

    # Empty files are not reusable either.
    generated = work / "falco_post_trim_reads"
    (generated / filenames[-1]).write_text("")
    filter_reads.generate_reports("sample", 1, True, logger)
    assert len(calls) == 2

    filter_reads.generate_reports("sample", 1, False, logger)
    assert len(calls) == 3

    if completed_location == "run_info":
        for name in filenames:
            (completed / name).write_text("completed report from earlier run")
        (generated / filenames[-1]).write_text("")
        # Partial temporary output must not replace complete run_info on cleanup.
        filter_reads.generate_reports("sample", 1, True, logger)
        assert len(calls) == 4


def test_falco_without_final_reads_does_not_launch(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(filter_reads, "config", SimpleNamespace(
        temp_dir=tmp_path, output_dir=tmp_path / "output",
    ))
    calls = []
    monkeypatch.setattr(filter_reads, "run_command_comp", lambda **kwargs: calls.append(kwargs))

    filter_reads.generate_reports("sample", 1, True, logging.getLogger("test_falco_resume"))

    assert not calls
    assert "no final FASTQ" in caplog.text


@pytest.mark.parametrize("failure", ["exit_status", "missing_output", "empty_output"])
@pytest.mark.parametrize("previous", ["none", "temporary", "run_info", "partial_temporary"])
def test_falco_failure_preserves_reads_and_complete_reports(
    tmp_path, monkeypatch, caplog, failure, previous
):
    work = tmp_path / "work"
    work.mkdir()
    output = tmp_path / "output"
    logger = logging.getLogger("test_falco_failure")
    config = SimpleNamespace(
        temp_dir=work, output_dir=output, logger=logger, keep_tmp=False,
        zip_reports=False, save=lambda path: path.write_text("{}"),
    )
    monkeypatch.setattr(filter_reads, "config", config)
    monkeypatch.setattr(filter_reads, "tools", [])
    reads = work / "merged_final_sample.fq.gz"
    reads.write_bytes(b"filtered reads; backend is stubbed")
    tracker = SimpleNamespace(
        to_csv=lambda path: path.write_text("filename\n"),
        get_latest_merged_file=lambda: str(reads),
        get_latest_non_merged_file=lambda: None,
    )
    filenames = ("fastqc_data.txt", "summary.txt", "fastqc_report.html")
    final_qc = output / "run_info" / "falco_post_trim_reads"
    if previous != "none":
        complete = (
            work / "falco_post_trim_reads" if previous == "temporary" else final_qc
        )
        complete.mkdir(parents=True)
        for name in filenames:
            (complete / name).write_text("previous complete report")
    if previous == "partial_temporary":
        incomplete = work / "falco_post_trim_reads"
        incomplete.mkdir()
        (incomplete / "fastqc_data.txt").write_text("old partial report")

    def fail_falco(**kwargs):
        destination = Path(kwargs["params"]["outdir"])
        for name in filenames:
            if failure == "missing_output" and name == "summary.txt":
                continue
            contents = "" if failure == "empty_output" and name == "summary.txt" else "new report"
            (destination / name).write_text(contents)
        return failure != "exit_status"

    monkeypatch.setattr(filter_reads, "run_command_comp", fail_falco)

    ready = filter_reads.generate_reports("sample", 1, False, logger)

    assert "Falco reports generated" not in caplog.text
    assert any(record.levelname == "WARNING" and "Falco" in record.message for record in caplog.records)
    filter_reads.cleanup_and_move_files(config, tracker, falco_ready=ready)
    assert (output / reads.name).read_bytes() == b"filtered reads; backend is stubbed"
    if previous == "none":
        assert not final_qc.exists()
    else:
        assert {path.name: path.read_text() for path in final_qc.iterdir()} == {
            name: "previous complete report" for name in filenames
        }
