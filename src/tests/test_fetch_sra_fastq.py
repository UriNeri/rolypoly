from rolypoly.commands.misc import fetch_sra_fastq
from rolypoly.commands.misc.fetch_sra_fastq import (
    is_valid_accession,
    is_valid_run_id,
    resolve_accession_to_run_ids,
)


def test_is_valid_run_id_accepts_srr_example() -> None:
    assert is_valid_run_id("SRR10307479")


def test_is_valid_run_id_accepts_err_and_drr() -> None:
    assert is_valid_run_id("ERR123456")
    assert is_valid_run_id("DRR987654")


def test_is_valid_run_id_rejects_non_accession_values() -> None:
    assert not is_valid_run_id("/dev/null")
    assert not is_valid_run_id("SRX10307479")
    assert not is_valid_run_id("")


def test_is_valid_accession_accepts_experiment_sample_study() -> None:
    assert is_valid_accession("SRX7018852")
    assert is_valid_accession("SRS123456")
    assert is_valid_accession("SRP123456")


def test_resolve_accession_to_run_ids_passthrough_for_run_id() -> None:
    assert resolve_accession_to_run_ids("SRR10307479") == ["SRR10307479"]


def test_resolve_accession_to_run_ids_maps_experiment(monkeypatch) -> None:
    class DummyResponse:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            return None

    def fake_get(url):
        assert "accession=SRX7018852" in url
        return DummyResponse("run_accession\nSRR10307479\n")

    monkeypatch.setattr(fetch_sra_fastq.requests, "get", fake_get)
    resolved = resolve_accession_to_run_ids("SRX7018852")
    assert resolved == ["SRR10307479"]
