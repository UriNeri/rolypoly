import tarfile
import zipfile

from rolypoly.commands.misc.get_external_data import (
    DATA_URLS,
    NERSC_DATA_URL,
    ZENODO_DATA_URL,
)
from rolypoly.utils.various import extract


def test_data_source_order_and_release_url():
    assert DATA_URLS == (ZENODO_DATA_URL, NERSC_DATA_URL)
    assert "/records/22636256/files/data.tar.gz" in ZENODO_DATA_URL


def test_active_extract_handles_zip_archive(tmp_path):
    archive = tmp_path / "payload.zip"
    source = tmp_path / "source.txt"
    source.write_text("zip payload\n")
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.write(source, arcname="source.txt")

    output = tmp_path / "out"
    extract(archive, output)

    assert (output / "source.txt").read_text() == "zip payload\n"


def test_active_extract_handles_tar_gz_archive(tmp_path):
    archive = tmp_path / "payload.tar.gz"
    source = tmp_path / "source.txt"
    source.write_text("tar payload\n")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname="source.txt")

    output = tmp_path / "out"
    extract(archive, output)

    assert (output / "source.txt").read_text() == "tar payload\n"
