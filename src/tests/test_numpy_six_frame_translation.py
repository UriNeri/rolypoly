"""Contracts for the native IUPAC-aware six-frame translators."""

import pytest

from rolypoly.utils.bio.polars_fastx import translate_6frx_numpy_polars
from rolypoly.utils.bio.translation import (
    CANONICAL_SIX_FRAME_DEFLINE,
    GENETIC_CODES_AA,
    SIX_FRAME_TRANSLATION_VERSION,
    build_translation_metadata,
    normalize_translation_output,
    numpy_translation_tables,
    translate_6frx_numpy,
    translate_sequence_numpy,
    translation_records,
    translation_signature,
    validate_six_frame_defline,
)


def test_numpy_translation_resolves_iupac_and_reverse_complement():
    assert translate_sequence_numpy("AAR") == "K"
    assert translate_sequence_numpy("YTT", frame=-1) == "K"
    assert translate_sequence_numpy("AUG", frame=-1) == "H"
    assert translate_sequence_numpy("NNN") == "X"
    assert translate_sequence_numpy("TAR") == "X"
    assert translate_sequence_numpy("TAR", clean=False) == "*"
    assert translate_sequence_numpy("---") == "-"
    assert translate_sequence_numpy("...") == "-"


def test_numpy_translation_table_is_cached():
    numpy_translation_tables.cache_clear()
    first = numpy_translation_tables(1, True)
    second = numpy_translation_tables(1, True)
    assert first[0] is second[0]
    assert first[1] is second[1]
    assert numpy_translation_tables.cache_info().hits == 1


def test_numpy_tables_support_every_bundled_genetic_code():
    for genetic_code in GENETIC_CODES_AA:
        base_codes, lookup = numpy_translation_tables(int(genetic_code))
        assert base_codes.shape == (256,)
        assert lookup.shape == (17**3,)


def test_direct_six_frame_output_uses_seqkit_naming(tmp_path):
    source = tmp_path / "input.fa"
    output = tmp_path / "output.faa"
    source.write_text(">one description\nATGTAACGTACN\n>two\nACGTACGTACGT\n")

    translate_6frx_numpy(source, output)

    records = list(translation_records(output))
    assert [header for header, _ in records] == [
        "one_frame=1 description",
        "one_frame=2 description",
        "one_frame=3 description",
        "one_frame=-1 description",
        "one_frame=-2 description",
        "one_frame=-3 description",
        "two_frame=1 ",
        "two_frame=2 ",
        "two_frame=3 ",
        "two_frame=-1 ",
        "two_frame=-2 ",
        "two_frame=-3 ",
    ]
    assert records[0][1] == "MXRT"


def test_minimum_length_filters_complete_frames(tmp_path):
    source = tmp_path / "input.fa"
    output = tmp_path / "output.faa"
    source.write_text(">sequence\nATGTAACGT\n")

    translate_6frx_numpy(source, output, min_orf_length=3)

    assert [header for header, _ in translation_records(output)] == [
        "sequence_frame=1 ",
        "sequence_frame=-1 ",
    ]


def test_polars_and_direct_numpy_paths_match(tmp_path):
    source = tmp_path / "input.fa"
    direct = tmp_path / "direct.faa"
    polars = tmp_path / "polars.faa"
    source.write_text(
        ">iupac ambiguous RNA\nAUGYTTACNTAR---...\n"
        ">canonical\nACGTACGTACGTACGT\n"
    )

    translate_6frx_numpy(source, direct)
    translate_6frx_numpy_polars(source, polars, streaming_chunk_size=1)

    assert polars.read_bytes() == direct.read_bytes()


@pytest.mark.parametrize(
    "translator", [translate_6frx_numpy, translate_6frx_numpy_polars]
)
def test_output_policy_supports_stops_and_canonical_deflines(
    tmp_path, translator
):
    source = tmp_path / "input.fa"
    output = tmp_path / "output.faa"
    source.write_text(">contig:1 description\nTAACCC\n")

    translator(
        source,
        output,
        stops_as_x=False,
        defline_template=CANONICAL_SIX_FRAME_DEFLINE,
    )

    records = list(translation_records(output))
    assert records[0] == ("contig%3A1_frame_p1", "*P")
    assert [header for header, _ in records][-1] == "contig%3A1_frame_m3"


def test_defline_template_validation():
    validate_six_frame_defline("{id}_fr{frame}")
    with pytest.raises(ValueError, match="uniquely"):
        validate_six_frame_defline("{id} frame={frame}")
    validate_six_frame_defline("{id} frame={frame}", require_unique_ids=False)
    with pytest.raises(ValueError, match="Unknown"):
        validate_six_frame_defline("{id}_{unknown}")
    with pytest.raises(ValueError, match="must not be empty"):
        validate_six_frame_defline("", require_unique_ids=False)


def test_empty_input_produces_empty_outputs(tmp_path):
    source = tmp_path / "empty.fa"
    direct = tmp_path / "direct.faa"
    polars = tmp_path / "polars.faa"
    source.write_text("")

    translate_6frx_numpy(source, direct)
    translate_6frx_numpy_polars(source, polars)

    assert direct.read_text() == polars.read_text() == ""


def test_threaded_long_contigs_preserve_order_and_output(tmp_path):
    source = tmp_path / "long.fa"
    sequential = tmp_path / "sequential.faa"
    threaded = tmp_path / "threaded.faa"
    source.write_text(
        ">first description\n" + "ACGTRYSWKMBDHVN" * 2_500 + "\n"
        ">second\n" + "TGCAYRSWMKVHDB" * 2_500 + "\n"
    )

    translate_6frx_numpy(source, sequential, threads=1)
    translate_6frx_numpy(source, threaded, threads=4)

    assert threaded.read_bytes() == sequential.read_bytes()


def test_native_output_builds_six_frame_metadata(tmp_path):
    source = tmp_path / "input.fa"
    output = tmp_path / "output.faa"
    source.write_text(">contig description\n" + "ACGT" * 8 + "\n")

    translate_6frx_numpy(source, output)
    metadata = build_translation_metadata(source, output, "six-frame")

    assert metadata.height == 6
    assert metadata["frame_id"].to_list() == [1, 2, 3, -1, -2, -3]
    assert set(metadata["source_seq_id"]) == {"contig"}
    assert metadata["original_source_header"].unique().to_list() == [
        "contig description"
    ]


def test_canonical_output_normalizes_without_changing_fasta(tmp_path):
    source = tmp_path / "input.fa"
    bundle = tmp_path / "bundle"
    output = bundle / "predicted_orfs.faa"
    bundle.mkdir()
    source.write_text(">contig:1 description\n" + "ACGT" * 8 + "\n")
    translate_6frx_numpy(
        source, output, defline_template=CANONICAL_SIX_FRAME_DEFLINE
    )
    before = output.read_bytes()

    metadata = normalize_translation_output(
        source,
        output,
        bundle,
        "six-frame",
        {
            "minimum_length": 0,
            "stops_as_x": True,
            "defline_template": CANONICAL_SIX_FRAME_DEFLINE,
        },
    )

    assert output.read_bytes() == before
    assert metadata["source_seq_id"].unique().to_list() == ["contig:1"]
    assert metadata["translation_id"][0] == "contig%3A1_frame_p1"


def test_six_frame_signature_tracks_native_backend():
    signature = translation_signature("six_frame", {"minimum_length": 0})
    assert (
        signature["versions"]["implementation"] == SIX_FRAME_TRANSLATION_VERSION
    )
    assert "numpy" in signature["versions"]
    assert "seqkit" not in signature["versions"]
