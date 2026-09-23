"""Translate nucleotide FASTA records with RolyPoly's NumPy backend."""

from contextlib import ExitStack, nullcontext
from pathlib import Path
from string import Formatter
from urllib.parse import quote

import rich_click as click
from needletail import parse_fastx_file

from rolypoly.utils.bio.translation import (
    CANONICAL_SIX_FRAME_DEFLINE,
    GENETIC_CODES_AA,
    SEQKIT_SIX_FRAME_DEFLINE,
    SIX_FRAMES,
    find_six_frame_orfs_numpy,
    format_six_frame_header,
    normalize_nucleotide_sequence,
    reverse_complement_iupac,
    translate_6frx_numpy,
    validate_six_frame_defline,
)


SEQKIT_ORF_DEFLINE = "{id}_frame={frame}_orf={orf} {description}"
CANONICAL_ORF_DEFLINE = "{canonical_id}_orf_{orf}"
ORF_DEFLINE_FIELDS = {
    "id",
    "canonical_id",
    "description",
    "orf",
    "frame",
    "frame_abs",
    "strand",
    "frame_token",
    "start",
    "end",
    "partial",
}


def _fasta_records(path: Path):
    for record in parse_fastx_file(str(path)):
        header = (
            record.id.decode()
            if isinstance(record.id, bytes)
            else str(record.id)
        )
        sequence = (
            record.seq.decode()
            if isinstance(record.seq, bytes)
            else str(record.seq)
        )
        yield header, sequence


def _select_defline(mode: str, header_format: str, custom: str | None) -> str:
    if custom is not None:
        return custom
    if mode == "orfs":
        return (
            CANONICAL_ORF_DEFLINE
            if header_format == "canonical"
            else SEQKIT_ORF_DEFLINE
        )
    return (
        CANONICAL_SIX_FRAME_DEFLINE
        if header_format == "canonical"
        else SEQKIT_SIX_FRAME_DEFLINE
    )


def _validate_orf_defline(template: str) -> None:
    fields = {
        name
        for _, name, _, _ in Formatter().parse(template)
        if name is not None
    }
    unknown = fields - ORF_DEFLINE_FIELDS
    if unknown:
        raise click.BadParameter(
            f"unknown ORF defline fields: {sorted(unknown)}",
            param_hint="--defline-template",
        )


def _format_orf_header(
    source_header: str, orf: dict[str, object], ordinal: int, template: str
) -> str:
    sequence_id, separator, description = source_header.partition(" ")
    if not separator:
        sequence_id, _, description = source_header.partition("\t")
    frame = int(orf["frame"])
    partial = (
        "both"
        if orf["partial_5prime"] and orf["partial_3prime"]
        else "5prime"
        if orf["partial_5prime"]
        else "3prime"
        if orf["partial_3prime"]
        else "none"
    )
    return template.format(
        id=sequence_id,
        canonical_id=quote(sequence_id, safe="_.-"),
        description=description,
        orf=ordinal,
        frame=frame,
        frame_abs=abs(frame),
        strand=orf["strand"],
        frame_token=f"{'p' if frame > 0 else 'm'}{abs(frame)}",
        start=orf["start"],
        end=orf["end"],
        partial=partial,
    )


def _write_gff_record(
    handle, source_id: str, feature_id: str, feature_type: str, feature: dict
) -> None:
    attributes = {
        "ID": feature_id,
        "frame_id": feature["frame"],
        "partial_5prime": str(feature.get("partial_5prime", False)).lower(),
        "partial_3prime": str(feature.get("partial_3prime", False)).lower(),
    }
    encoded = ";".join(
        f"{quote(str(key), safe='_.:-')}={quote(str(value), safe='_.:-|')}"
        for key, value in attributes.items()
    )
    handle.write(
        "\t".join(
            [
                quote(source_id, safe="_.:-"),
                "rolypoly-translate",
                feature_type,
                str(feature["start"]),
                str(feature["end"]),
                ".",
                str(feature["strand"]),
                "0" if feature_type == "CDS" else ".",
                encoded,
            ]
        )
        + "\n"
    )


def _write_six_frame_auxiliary(
    input_path: Path,
    nucleotide_output: Path | None,
    gff_output: Path | None,
    template: str,
    min_orf_length: int,
    append_fasta: bool,
) -> None:
    seen = set()
    with ExitStack() as stack:
        nucleotide_handle = (
            stack.enter_context(nucleotide_output.open("w"))
            if nucleotide_output
            else None
        )
        gff_handle = (
            stack.enter_context(gff_output.open("w")) if gff_output else None
        )
        if gff_handle:
            gff_handle.write("##gff-version 3\n")
        for header, raw_sequence in _fasta_records(input_path):
            source_id = header.split()[0]
            sequence = normalize_nucleotide_sequence(raw_sequence)
            reverse = reverse_complement_iupac(sequence)
            for frame in SIX_FRAMES:
                offset = abs(frame) - 1
                oriented = sequence if frame > 0 else reverse
                length = (len(oriented) - offset) // 3 * 3
                if length // 3 < min_orf_length:
                    continue
                nucleotide = oriented[offset : offset + length]
                defline = format_six_frame_header(header, frame, template)
                feature_id = defline.split()[0]
                if feature_id in seen:
                    raise click.ClickException(
                        f"duplicate output FASTA identifier: {feature_id}"
                    )
                seen.add(feature_id)
                if nucleotide_handle:
                    nucleotide_handle.write(f">{defline}\n{nucleotide}\n")
                if frame > 0:
                    start, end = offset + 1, offset + length
                else:
                    start = len(sequence) - (offset + length) + 1
                    end = len(sequence) - offset
                if gff_handle:
                    _write_gff_record(
                        gff_handle,
                        source_id,
                        feature_id,
                        "translated_region",
                        {
                            "frame": frame,
                            "strand": "+" if frame > 0 else "-",
                            "start": start,
                            "end": end,
                        },
                    )
    if gff_output and append_fasta:
        from rolypoly.utils.bio.polars_fastx import append_fasta_to_gff

        append_fasta_to_gff(input_path, gff_output)


def _write_orfs(
    input_path: Path,
    protein_output: Path,
    nucleotide_output: Path | None,
    gff_output: Path | None,
    template: str,
    threads: int,
    min_orf_length: int,
    genetic_code: int,
    stops_as_x: bool,
    alternative_starts: bool,
    include_partial: bool,
    all_starts: bool,
    append_fasta: bool,
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    def find(record):
        return find_six_frame_orfs_numpy(
            record[1],
            genetic_code=genetic_code,
            min_orf_length=min_orf_length,
            alternative_starts=alternative_starts,
            include_partial=include_partial,
            all_starts=all_starts,
            stops_as_x=stops_as_x,
        )

    executor_context = (
        ThreadPoolExecutor(max_workers=threads)
        if threads > 1
        else nullcontext()
    )
    with ExitStack() as stack:
        executor = stack.enter_context(executor_context)
        protein_handle = stack.enter_context(protein_output.open("w"))
        nucleotide_handle = (
            stack.enter_context(nucleotide_output.open("w"))
            if nucleotide_output
            else None
        )
        gff_handle = (
            stack.enter_context(gff_output.open("w")) if gff_output else None
        )
        if gff_handle:
            gff_handle.write("##gff-version 3\n")
        seen = set()

        def write_batch(records):
            use_threads = (
                executor is not None
                and len(records) > 1
                and sum(len(record[1]) for record in records)
                >= len(records) * 32_768
            )
            found = (
                executor.map(find, records)
                if use_threads
                else map(find, records)
            )
            for (header, _), orfs in zip(records, found):
                source_id = header.split()[0]
                for ordinal, orf in enumerate(orfs, start=1):
                    defline = _format_orf_header(header, orf, ordinal, template)
                    feature_id = defline.split()[0]
                    if not feature_id or feature_id in seen:
                        raise click.ClickException(
                            "ORF deflines must have unique, nonempty first tokens"
                        )
                    seen.add(feature_id)
                    protein_handle.write(f">{defline}\n{orf['protein']}\n")
                    if nucleotide_handle:
                        nucleotide_handle.write(
                            f">{defline}\n{orf['nucleotide']}\n"
                        )
                    if gff_handle:
                        _write_gff_record(
                            gff_handle, source_id, feature_id, "CDS", orf
                        )

        batch = []
        for record in _fasta_records(input_path):
            batch.append(record)
            if len(batch) == max(1, threads * 2):
                write_batch(batch)
                batch.clear()
        if batch:
            write_batch(batch)
    if gff_output and append_fasta:
        from rolypoly.utils.bio.polars_fastx import append_fasta_to_gff

        append_fasta_to_gff(input_path, gff_output)


@click.command(name="translate")
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Input nucleotide FASTA file.",
)
@click.option(
    "-o",
    "--output",
    "protein_output",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output protein FASTA file.",
)
@click.option(
    "--mode",
    type=click.Choice(["six-frame", "orfs"], case_sensitive=False),
    default="six-frame",
    help="Emit complete six-frame translations or start/stop-delimited ORFs.",
)
@click.option(
    "--min-length",
    "min_orf_length",
    type=click.IntRange(min=0),
    default=0,
    help="Minimum amino-acid length; ORF-mode terminal stops are excluded.",
)
@click.option(
    "--genetic-code",
    type=click.Choice(sorted(GENETIC_CODES_AA, key=int)),
    default="1",
    help="NCBI or bundled generic genetic code number.",
)
@click.option(
    "--stops-as-x/--stops-as-star",
    default=True,
    help="Write resolved stop codons as X or * in protein output.",
)
@click.option(
    "--header-format",
    type=click.Choice(["seqkit", "canonical"], case_sensitive=False),
    default="seqkit",
    help="Use SeqKit-compatible or canonical RolyPoly FASTA identifiers.",
)
@click.option(
    "--defline-template",
    default=None,
    help="Custom Python format template overriding --header-format.",
)
@click.option(
    "--alternative-starts/--atg-only",
    default=True,
    help="In ORF mode, recognize the selected genetic code's alternative starts.",
)
@click.option(
    "--partial-orfs/--complete-orfs-only",
    default=True,
    help="In ORF mode, include edge-to-stop and start-to-edge partial ORFs.",
)
@click.option(
    "--all-starts/--longest-only",
    default=False,
    help="In ORF mode, emit every nested start or only the first start per region.",
)
@click.option(
    "--fna-output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optionally write translated nucleotide regions in coding orientation.",
)
@click.option(
    "--gff-output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optionally write translated regions or ORFs as GFF3.",
)
@click.option(
    "--gff-include-fasta",
    is_flag=True,
    help="Append the input reference FASTA to --gff-output.",
)
def translate(
    input_path: Path,
    protein_output: Path,
    mode: str,
    min_orf_length: int,
    genetic_code: str,
    stops_as_x: bool,
    header_format: str,
    defline_template: str | None,
    alternative_starts: bool,
    partial_orfs: bool,
    all_starts: bool,
    fna_output: Path | None,
    gff_output: Path | None,
    gff_include_fasta: bool,
    threads: int,
    log_file: Path | None,
    log_level: str,
) -> None:
    """Translate nucleotide FASTA in six frames or extract translated ORFs."""
    from rolypoly.utils.logging.loggit import log_start_info, setup_logging

    logger = setup_logging(log_file, log_level)
    log_start_info(logger, locals())
    if gff_include_fasta and gff_output is None:
        raise click.UsageError("--gff-include-fasta requires --gff-output")
    output_paths = [
        path
        for path in (protein_output, fna_output, gff_output)
        if path is not None
    ]
    resolved_outputs = [path.resolve() for path in output_paths]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise click.UsageError("Protein, FNA, and GFF outputs must be distinct")
    if input_path.resolve() in resolved_outputs:
        raise click.UsageError("Output paths must differ from the input FASTA")
    template = _select_defline(mode, header_format, defline_template)
    if mode == "six-frame":
        try:
            validate_six_frame_defline(template)
        except ValueError as error:
            raise click.BadParameter(
                str(error), param_hint="--defline-template"
            ) from error
        translate_6frx_numpy(
            input_path,
            protein_output,
            threads=threads,
            min_orf_length=min_orf_length,
            genetic_code=int(genetic_code),
            stops_as_x=stops_as_x,
            defline_template=template,
        )
        if fna_output or gff_output:
            _write_six_frame_auxiliary(
                input_path,
                fna_output,
                gff_output,
                template,
                min_orf_length,
                gff_include_fasta,
            )
    else:
        _validate_orf_defline(template)
        _write_orfs(
            input_path,
            protein_output,
            fna_output,
            gff_output,
            template,
            threads,
            min_orf_length,
            int(genetic_code),
            stops_as_x,
            alternative_starts,
            partial_orfs,
            all_starts,
            gff_include_fasta,
        )
    logger.info(f"Wrote protein translations to {protein_output}")
