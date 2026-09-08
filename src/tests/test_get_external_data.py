from rolypoly.commands.misc.get_external_data import (
    DATA_URLS,
    NERSC_DATA_URL,
    ZENODO_DATA_URL,
)


def test_data_source_order_and_release_url():
    assert DATA_URLS == (ZENODO_DATA_URL, NERSC_DATA_URL)
    assert "/records/22636256/files/data.tar.gz" in ZENODO_DATA_URL
