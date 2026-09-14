"""Verified reuse of pre-resolution HMM searches between pipeline stages.

HMMER's inferred Z/domZ depend on the query protein population. Consequently
subset reuse requires explicit, identical Z and domZ; otherwise only identical
protein populations are eligible, even if their nucleotide input is a subset.
"""
import json
import shutil
from pathlib import Path

import polars as pl

from rolypoly.utils.bio.translation import (
    translation_file_digest,
    translation_input_fingerprints,
)


def hmm_signature(database, parameters):
    from importlib.metadata import version

    return {
        "tool": "pyhmmer.hmmsearch",
        "version": version("pyhmmer"),
        "database_sha256": translation_file_digest(database),
        "parameters": parameters,
    }


def save_hmm_search(bundle, database, output, parameters, fields):
    """Persist raw results and their search/translation provenance before filtering."""
    bundle = Path(bundle)
    translation = json.loads((bundle / "translation_manifest.json").read_text())
    signature = hmm_signature(database, parameters)
    cache = bundle / "search_cache" / signature["database_sha256"]
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output, cache / "hits.tsv")
    manifest = {
        "schema_version": 1,
        "signature": signature,
        "translation": translation,
        "proteins": translation_input_fingerprints(bundle / "predicted_orfs.faa"),
        "fields": fields,
        "result_sha256": translation_file_digest(cache / "hits.tsv"),
    }
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def reuse_hmm_search(source, destination_bundle, database, output, parameters, fields, logger):
    """Return True only after a verified result is written; mismatches rerun normally."""
    if not source:
        return False
    source, destination_bundle = Path(source), Path(destination_bundle)
    try:
        signature = hmm_signature(database, parameters)
        cache = source / "search_cache" / signature["database_sha256"]
        manifest = json.loads((cache / "manifest.json").read_text())
        current = json.loads((destination_bundle / "translation_manifest.json").read_text())
        previous = manifest["translation"]
        if manifest["schema_version"] != 1 or manifest["signature"] != signature:
            raise ValueError("search settings or tool version differ")
        if current["signature"] != previous["signature"] or current["signature"]["versions"] is None:
            raise ValueError("translation settings or version differ")
        if not all(previous["inputs"].get(k) == v for k, v in current["inputs"].items()):
            raise ValueError("input is not an identical subset of the original input")
        proteins = translation_input_fingerprints(destination_bundle / "predicted_orfs.faa")
        if not all(manifest["proteins"].get(k) == v for k, v in proteins.items()):
            raise ValueError("translated IDs or sequences differ")
        if proteins != manifest["proteins"] and not all(parameters.get(k) is not None for k in ("Z", "domZ")):
            raise ValueError("protein subset changes HMMER's inferred Z/domZ; a fresh search is required")
        if not set(fields).issubset(manifest["fields"]):
            raise ValueError("cached search lacks requested alignment fields")
        if translation_file_digest(cache / "hits.tsv") != manifest["result_sha256"]:
            raise ValueError("cached results changed")
        hits = pl.read_csv(cache / "hits.tsv", separator="\t", infer_schema_length=None)
        hits = hits.filter(pl.col("query_full_name").str.extract(r"^(\S+)").is_in(list(proteins)))
    except (OSError, ValueError, KeyError, TypeError, pl.exceptions.PolarsError) as error:
        logger.info("Search reuse unavailable for %s: %s", database, error)
        return False
    hits.write_csv(output, separator="\t")
    Path(str(output) + ".reuse.json").write_text(json.dumps({
        "source": str(cache.resolve()), "signature": signature,
        "result_sha256": translation_file_digest(output),
    }, indent=2) + "\n")
    logger.info("Reused verified raw marker search from %s", cache)
    return True
