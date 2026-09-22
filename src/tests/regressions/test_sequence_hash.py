import builtins
import hashlib

import polars as pl
import pytest
import xxhash

from rolypoly.utils.bio.sequences import SEQUENCE_HASH_SEED, hash_bytes


def test_xxh3_external_digest_and_seed():
    assert SEQUENCE_HASH_SEED == 0
    # External XXH3-64 known value: changing the seed must not change stored IDs.
    assert hash_bytes(b"ACGT") == 17060466519357990319
    assert hash_bytes(b"ACGT", "xxh3") == xxhash.xxh3_64(
        b"ACGT", seed=0
    ).intdigest()


def test_missing_xxhash_falls_back_to_blake2b(monkeypatch):
    original_import = builtins.__import__

    def without_xxhash(name, *args, **kwargs):
        if name == "xxhash":
            raise ImportError("xxhash unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_xxhash)
    expected = hashlib.blake2b(b"ACGT", digest_size=8).hexdigest()
    assert format(hash_bytes(b"ACGT"), "016x") == expected
    with pytest.raises(ImportError):
        hash_bytes(b"ACGT", "xxh3")


def test_polars_alternative_matches_native_batch():
    values = [b"", b"ACGT", b"acgtn", b"N" * 10000]
    expected = pl.Series(values, dtype=pl.Binary).hash(seed=0).to_list()
    assert [hash_bytes(value, "polars") for value in values] == expected


def test_xxhash_runtime_errors_are_not_silently_rehashed(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("hash failed")

    monkeypatch.setattr(xxhash, "xxh3_64", broken)
    with pytest.raises(RuntimeError, match="hash failed"):
        hash_bytes(b"ACGT")
