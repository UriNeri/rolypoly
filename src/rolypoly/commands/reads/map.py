"""Read mapping command: map reads to a reference using one or more aligners.

Supports bbmap (via bbmapy), minimap2 (via mappy Python bindings), bwa-mem2,
and mmseqs. Input discovery follows the same pattern as filter-reads and assemble:
--input / --input-dir for auto-detection, plus explicit --paired-end /
--single-end / --merged / --long-read library flags. Each input file is mapped
individually (no concatenation).
"""

import io
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from shlex import join, quote, split
from typing import TYPE_CHECKING, Callable, Iterator, Literal, TextIO, TypedDict

import rich_click as click

if TYPE_CHECKING:
    from mappy import Alignment

from rolypoly.utils.bio.library_detection import handle_input_fastq
from rolypoly.utils.logging.loggit import get_logger, setup_logging
from rolypoly.utils.various import (
    ensure_memory,
    get_reduced_memory,
    run_command_comp,
)

COMPRESSION_POLICIES = ("none", "sam_only")


class MapperRun(TypedDict):
    """Normalized inputs and spec-derived processing settings for one mapper."""

    reference: Path
    output: Path
    suffix: str
    paired_reads: list[tuple[str, str]]
    interleaved_reads: list[str]
    single_reads: list[str]
    threads: int
    memory: str
    include_unmapped: bool
    bwa_mem2_all: bool
    bwa_mem2_extra_flags: str
    stream_concordant: bool
    stream_proper: bool
    stream_mapped_templates_only: bool


class MapperSpec(TypedDict):
    """Invocation and processing capabilities, composed without name dispatch."""

    run: Callable[[MapperRun, "MapperSpec"], list[str]]  # Registered invocation; no adapter-name dispatcher.
    output_mode: Literal["native_gzip", "sam_stream", "sam_file"]  # Who writes/compresses SAM.
    needs_append_for_multi: bool  # Adapter combines libraries into one output.
    template_filter: Literal["builtin", "stream", "hits_only"]  # How fully unmapped templates are omitted.
    supports_unmapped_mates: bool  # Can retain both mates when either maps.
    supports_include_unmapped: bool  # Can emit fully unmapped reads/templates.
    supports_concordant: bool  # Has usable SAM mate fields for our FR predicate.
    supports_proper: bool  # Emits mapper-defined SAM proper-pair flag 0x2.
    pair_filter_stage: Literal["stream", "posthoc", "unsupported"]  # Where optional strict predicates run.
    temporary_patterns: tuple[str, ...]  # Mapper-relative directories removed unless --keep-tmp.


@contextmanager
def sam_output(path: Path, threads: int) -> Iterator[TextIO]:
    """Write SAM text directly to plain/gzip output, publishing only on success."""
    if path.exists():
        raise click.ClickException(f"Refusing to overwrite SAM output: {path}")
    staging = path.with_name(f".{path.name}.tmp")
    compressor = None
    completed = False
    owns_staging = False
    try:
        with staging.open("xb") as raw:
            owns_staging = True
            if path.suffix == ".gz":
                command = ["pigz", "-p", str(threads), "-c"]
                get_logger().info("Streaming compression: %s -> %s", join(command), path)
                compressor = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=raw)
                if compressor.stdin is None:
                    raise click.ClickException("Failed to open pigz input stream.")
                with io.TextIOWrapper(compressor.stdin, encoding="utf-8", newline="") as stream:
                    yield stream
                status = compressor.wait()
                if status != 0:
                    raise click.ClickException(f"pigz failed with exit status {status}: {path}")
            else:
                with io.TextIOWrapper(raw, encoding="utf-8", newline="") as stream:
                    yield stream
        staging.replace(path)
        completed = True
    except OSError as error:
        raise click.ClickException(f"Failed to write SAM output {path}: {error}") from error
    finally:
        if compressor is not None and compressor.poll() is None:
            compressor.terminate()
            try:
                compressor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                compressor.kill()
                compressor.wait()
        if owns_staging and not completed:
            staging.unlink(missing_ok=True)


def run_sam_command(
    command: list[str], output: Path, threads: int,
    concordant: bool = False, proper: bool = False,
    mapped_templates_only: bool = False,
) -> None:
    """Stream mapper stdout through SAM filters into plain/gzip output."""
    from rolypoly.utils.bio.alignments import filter_sam_stream

    logger = get_logger()
    logger.info("Streaming SAM command: %s", join(command))
    process = None
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True, encoding="utf-8")
        if process.stdout is None:
            raise click.ClickException("Failed to open mapper output stream.")
        with sam_output(output, threads) as destination:
            with process.stdout as source:
                read_count, written_count = filter_sam_stream(
                    source, destination, concordant=concordant, proper=proper,
                    mapped_templates_only=mapped_templates_only,
                )
            status = process.wait()
            if status != 0:
                raise click.ClickException(
                    f"Mapper failed with exit status {status}: {join(command)}"
                )
        logger.info("SAM stream %s: retained %d of %d records", output, written_count, read_count)
    except (OSError, ValueError) as error:
        raise click.ClickException(f"Streaming mapping failed: {error}") from error
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def compress_sam_outputs(
    outputs: list[str], policy: str, threads: int
) -> list[str]:
    """Losslessly gzip selected SAMs, removing originals only after success."""
    if policy not in COMPRESSION_POLICIES:
        raise click.ClickException(f"Unsupported compression policy: {policy}")
    if policy == "none":
        return outputs
    if not shutil.which("pigz"):
        raise click.ClickException("SAM compression requires pigz on PATH.")

    logger = get_logger()
    final_outputs = []
    for sam_output in outputs:
        sam_path = Path(sam_output)
        if sam_path.suffix != ".sam":
            final_outputs.append(sam_output)
            continue
        original_size = sam_path.stat().st_size
        compressed_path = sam_path.with_suffix(".sam.gz")
        if compressed_path.exists():
            raise click.ClickException(
                f"Refusing to overwrite compressed SAM: {compressed_path}. "
                f"Original retained: {sam_path}"
            )
        succeeded = run_command_comp(
            "pigz",
            params={"p": str(threads), "k": True},
            positional_args=["--", quote(str(sam_path))],
            output_file=str(compressed_path),
            logger=logger,
            check_status=True,
            prefix_style="single",
        )
        if not succeeded:
            raise click.ClickException(
                f"Failed to compress {sam_path}; original SAM retained."
            )
        sam_path.unlink()
        logger.info(
            "Compressed SAM %s -> %s (%d -> %d bytes; policy=%s)",
            sam_path, compressed_path, original_size,
            compressed_path.stat().st_size, policy,
        )
        final_outputs.append(str(compressed_path))
    return final_outputs


def mappy_hit_to_sam(
    hit: "Alignment | None", name: str, seq: str, qual: str | None,
    mate: "Alignment | None" = None, read1: bool | None = None,
) -> str:
    """Build a SAM record line from a mappy.Alignment hit.

    mappy.Alignment has no built-in SAM serialization, so the record is
    assembled manually from its fields. Handles:
      - strand-aware SEQ/QUAL (reverse-complemented for reverse-strand hits)
      - secondary-alignment flag (0x100) via hit.is_primary, so downstream
        tools (coverage/depth callers, dedup, etc.) don't treat every
        multi-mapping hit as an independent primary alignment
      - paired-end FLAG bits and RNEXT/PNEXT/TLEN when `mate` is provided

    Args:
        hit: mappy.Alignment for this read, or None for an unmapped read.
        name: read name (QNAME).
        seq: original read sequence, forward-strand orientation as read from
            the FASTQ (i.e. exactly as returned by mappy.fastx_read).
        qual: original quality string, forward-strand orientation, or "".
        mate: the mate's primary mappy.Alignment (or None if the mate did
            not map / this read is unpaired). Only used to populate
            RNEXT/PNEXT/TLEN and the mate-reverse-strand flag.
        read1: True if this is read 1 of a pair, False if read 2, None if
            this read is unpaired (single/merged/interleaved-orphan).

    Returns:
        Tab-separated SAM record string (no trailing newline).

    Note:
        The "properly paired" flag (0x2) is intentionally never set here,
        since validating expected orientation/insert size is out of scope;
        omitting it is SAM-compliant (most tools treat it as informational).
    """
    import mappy as mp

    if hit is None:
        flag = 0x4
        if read1 is not None:
            flag |= 0x1 | (0x40 if read1 else 0x80)
            flag |= 0x8 if mate is None else (0x20 if mate.strand == -1 else 0)
        reference = mate.ctg if mate is not None else "*"
        position = str(mate.r_st + 1) if mate is not None else "0"
        return "\t".join([
            name, str(flag), reference, position, "0", "*",
            "=" if mate is not None else "*", position, "0",
            seq, qual or "*",
        ])

    reverse = hit.strand == -1
    flag = 0x10 if reverse else 0
    if not hit.is_primary:
        flag |= 0x100

    if read1 is not None:
        flag |= 0x1
        flag |= 0x40 if read1 else 0x80
        if mate is not None:
            if mate.strand == -1:
                flag |= 0x20
        else:
            flag |= 0x8

    seq_out = seq if not reverse else mp.revcomp(seq)
    qual_out = (qual if not reverse else qual[::-1]) if qual else "*"

    if mate is not None:
        rnext = "=" if mate.ctg == hit.ctg else mate.ctg
        pnext = str(mate.r_st + 1)
        if mate.ctg == hit.ctg:
            span = max(hit.r_en, mate.r_en) - min(hit.r_st, mate.r_st)
            tlen = span if hit.r_st <= mate.r_st else -span
        else:
            tlen = 0
    else:
        rnext = "*"
        pnext = "0"
        tlen = 0

    tags = [f"NM:i:{hit.NM}"]
    if getattr(hit, "MD", ""):
        tags.append(f"MD:Z:{hit.MD}")

    fields = [
        name,
        str(flag),
        hit.ctg,
        str(hit.r_st + 1),
        str(hit.mapq),
        hit.cigar_str,
        rnext,
        pnext,
        str(tlen),
        seq_out,
        qual_out,
        *tags,
    ]
    return "\t".join(fields)


def write_paired_sam_records(
    sam_f: TextIO, name1: str, seq1: str, qual1: str | None,
    hits1: "list[Alignment]", name2: str, seq2: str, qual2: str | None,
    hits2: "list[Alignment]", include_unmapped: bool = False,
) -> None:
    """Write SAM records for one read pair, cross-referencing mate fields.

    Each mate's own primary alignment (first hit with is_primary=True, or
    the first hit if none is flagged) is used as the "mate" reference for
    the other mate's RNEXT/PNEXT/TLEN/mate-reverse-strand fields. All hits
    for both mates are written (including secondary ones, flagged 0x100).
    Both mates are retained if either maps. Fully unmapped pairs are skipped
    unless include_unmapped is set. The mapped mate of an unmapped read gets
    the mate-unmapped flag (0x8).
    """
    if not hits1 and not hits2 and not include_unmapped:
        return
    if name1.endswith("/1") and name2 == name1[:-2] + "/2":
        name1 = name2 = name1[:-2]
    primary1 = next(
        (h for h in hits1 if h.is_primary), hits1[0] if hits1 else None
    )
    primary2 = next(
        (h for h in hits2 if h.is_primary), hits2[0] if hits2 else None
    )

    for hit in hits1:
        sam_f.write(
            mappy_hit_to_sam(hit, name1, seq1, qual1, mate=primary2, read1=True)
            + "\n"
        )
    for hit in hits2:
        sam_f.write(
            mappy_hit_to_sam(
                hit, name2, seq2, qual2, mate=primary1, read1=False
            )
            + "\n"
        )
    if not hits1:
        sam_f.write(
            mappy_hit_to_sam(None, name1, seq1, qual1, mate=primary2, read1=True)
            + "\n"
        )
    if not hits2:
        sam_f.write(
            mappy_hit_to_sam(None, name2, seq2, qual2, mate=primary1, read1=False)
            + "\n"
        )


def run_bbmap(run: MapperRun, spec: MapperSpec) -> list[str]:
    """Invoke BBMap with native template retention, gzip, and library append."""
    from bbmapy import bbmap as bbmap_run

    logger = get_logger()
    bbmap_sam = run["output"] / f"bbmap{run['suffix']}"
    staging = bbmap_sam.with_name(f".{bbmap_sam.name}")
    if bbmap_sam.exists() or staging.exists():
        raise click.ClickException(f"Refusing to overwrite BBMap output: {bbmap_sam}")
    output_params = {"out" if run["include_unmapped"] else "outm": str(staging)}
    output_params["pigz"] = str(run["threads"]) if run["suffix"].endswith(".gz") else "f"
    memory_giga = get_reduced_memory(ensure_memory(run["memory"]), percentage=85)
    jobs = [{"in1": str(r1), "in2": str(r2)} for r1, r2 in run["paired_reads"]]
    jobs.extend({"in_file": str(fq), "interleaved": "t"} for fq in run["interleaved_reads"])
    jobs.extend({"in_file": str(fq)} for fq in run["single_reads"])
    try:
        for idx, inputs in enumerate(jobs):
            logger.info("Mapping BBMap library %s directly to %s", inputs, bbmap_sam)
            bbmap_run(
                ref=run["reference"],
                **output_params,
                threads=run["threads"],
                Xmx=memory_giga,
                minid=0.8,
                nodisk="t",
                overwrite="t",
                append="t" if idx > 0 and spec["needs_append_for_multi"] else "f",
                **inputs,
            )
        if not staging.exists() or staging.stat().st_size == 0:
            raise click.ClickException(f"BBMap did not produce SAM output: {bbmap_sam}")
        staging.replace(bbmap_sam)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        raise click.ClickException(f"BBMap mapping failed: {error}") from error
    finally:
        staging.unlink(missing_ok=True)
    return [str(bbmap_sam)]


def run_mappy(run: MapperRun, spec: MapperSpec) -> list[str]:
    """Generate mappy SAM output incrementally, retaining missing mate records."""
    import mappy as mp
    from rolypoly.utils.bio.alignments import filter_sam_stream

    reference, threads = run["reference"], run["threads"]
    include_unmapped = run["include_unmapped"]
    aligner = mp.Aligner(str(reference), preset="sr", n_threads=threads)
    if not aligner:
        raise click.ClickException(f"mappy: failed to load/build index from {reference}")
    sam_header = (
        "@HD\tVN:1.6\tSO:unsorted\n"
        + "\n".join(
            f"@SQ\tSN:{name}\tLN:{len(aligner.seq(name))}"
            for name in aligner.seq_names
        )
        + "\n@PG\tID:mappy\tPN:minimap2\n"
    )
    def records(r1: str, r2: str | None, interleaved: bool) -> Iterator[str]:
        yield from sam_header.splitlines(keepends=True)
        reads_iter = mp.fastx_read(r1)
        mate_iter = mp.fastx_read(r2) if r2 is not None else reads_iter if interleaved else None
        for name1, seq1, qual1 in reads_iter:
            mate = None
            if mate_iter is not None:
                try:
                    mate = next(mate_iter)
                except StopIteration:
                    if r2 is not None:
                        break  # Preserve lockstep truncation for separate mate files.
            hits1 = list(aligner.map(seq1, MD=True))
            if mate is not None:
                name2, seq2, qual2 = mate
                hits2 = list(aligner.map(seq2, MD=True))
                # Buffer only one template to reuse the existing mate serializer.
                with io.StringIO() as template:
                    write_paired_sam_records(
                        template, name1, seq1, qual1, hits1, name2, seq2, qual2, hits2,
                        include_unmapped=include_unmapped,
                    )
                    yield from template.getvalue().splitlines(keepends=True)
            else:
                for hit in hits1:
                    yield mappy_hit_to_sam(hit, name1, seq1, qual1) + "\n"
                if not hits1 and include_unmapped:
                    yield mappy_hit_to_sam(None, name1, seq1, qual1) + "\n"

    jobs: list[tuple[str, str, str | None, bool]] = [
        ("minimap2_paired", r1, r2, False) for r1, r2 in run["paired_reads"]
    ]
    jobs.extend(("minimap2_interleaved", fq, None, True) for fq in run["interleaved_reads"])
    jobs.extend(("minimap2", fq, None, False) for fq in run["single_reads"])
    outputs = []
    for idx, (prefix, r1, r2, interleaved) in enumerate(jobs):
        out_sam = run["output"] / f"{prefix}_{idx}{run['suffix']}"
        with sam_output(out_sam, threads) as sam_f:
            source = records(r1, r2, interleaved)
            if run["stream_concordant"] or run["stream_proper"] or run["stream_mapped_templates_only"]:
                filter_sam_stream(
                    source, sam_f, concordant=run["stream_concordant"],
                    proper=run["stream_proper"],
                    mapped_templates_only=run["stream_mapped_templates_only"],
                )
            else:
                sam_f.writelines(source)
        outputs.append(str(out_sam))
    return outputs


def run_bwa_mem2(run: MapperRun, spec: MapperSpec) -> list[str]:
    """Index the reference and stream BWA SAM using spec-derived predicates."""
    logger = get_logger()
    try:
        extra_flags = split(run["bwa_mem2_extra_flags"])
    except ValueError as error:
        raise click.ClickException(f"Invalid --bwa-mem2-extra-flags: {error}") from error
    if any(flag.startswith("-o") for flag in extra_flags):
        raise click.ClickException(
            "--bwa-mem2-extra-flags cannot set -o; SAM output is streamed."
        )
    bwa_index_dir = run["output"] / "bwa_index"
    bwa_index_dir.mkdir(parents=True, exist_ok=True)
    bwa_index_prefix = bwa_index_dir / "ref"
    index_succeeded = run_command_comp(
        "bwa-mem2",
        positional_args=[
            "index", "-p", quote(str(bwa_index_prefix)), quote(str(run["reference"])),
        ],
        positional_args_location="start",
        logger=logger,
        check_status=True,
        prefix_style="single",
    )
    if not index_succeeded:
        raise click.ClickException(f"bwa-mem2 index failed for {run['reference']}")

    mapping_jobs = [
        ("paired", [str(r1), str(r2)], False) for r1, r2 in run["paired_reads"]
    ]
    mapping_jobs.extend(
        ("interleaved", [str(fq)], True) for fq in run["interleaved_reads"]
    )
    mapping_jobs.extend(("single", [str(fq)], False) for fq in run["single_reads"])
    outputs = []
    for idx, (library_type, reads, is_interleaved) in enumerate(mapping_jobs):
        out_sam = run["output"] / f"bwa_mem2_{library_type}_{idx}{run['suffix']}"
        command = ["bwa-mem2", "mem", "-t", str(run["threads"])]
        if run["bwa_mem2_all"]:
            command.append("-a")
        if is_interleaved:
            command.append("-p")
        run_sam_command(
            command + [*extra_flags, str(bwa_index_prefix), *reads],
            out_sam, run["threads"],
            concordant=run["stream_concordant"],
            proper=run["stream_proper"],
            mapped_templates_only=run["stream_mapped_templates_only"],
        )
        outputs.append(str(out_sam))
    return outputs


def run_mmseqs(run: MapperRun, spec: MapperSpec) -> list[str]:
    """Write MMseqs hit-only SAM files for the shared gzip fallback."""
    logger = get_logger()
    all_fqs = (
        list(run["interleaved_reads"]) + list(run["single_reads"])
        + [f for pair in run["paired_reads"] for f in pair]
    )
    outputs = []
    for idx, fq in enumerate(all_fqs):
        run_tmp = run["output"] / f"mmseqs_tmp_{idx}"
        run_tmp.mkdir(parents=True, exist_ok=True)
        out_sam = run["output"] / f"mmseqs_reads_{idx}{run['suffix']}"
        succeeded = run_command_comp(
            "mmseqs",
            positional_args=[
                "easy-search", quote(str(fq)), quote(str(run["reference"])),
                quote(str(out_sam)), quote(str(run_tmp)),
            ],
            positional_args_location="start",
            params={
                "search-type": "3",
                "min-seq-id": "0.7",
                "threads": str(run["threads"]),
                "format-mode": "1",
                "split-memory-limit": run["memory"].upper(),
            },
            logger=logger,
            check_status=True,
            prefix_style="double",
        )
        if not succeeded:
            raise click.ClickException(f"MMseqs mapping failed for {fq}")
        outputs.append(str(out_sam))
    return outputs


# This is the policy source of truth for map and roll. The command composes
# invocation, stream predicates, post-hoc filtering, compression, and cleanup
# from these entries; it does not dispatch on mapper or adapter names.
MAPPER_SPECS: dict[str, MapperSpec] = {
    # BBMap decides pairing natively; outm keeps half-mapped pairs. Do not equate
    # pairedonly/killbadpairs with our predicates: they can alter mapping results.
    "bbmap": {
        "run": run_bbmap,  # Invoke the BBMap Python wrapper directly.
        "output_mode": "native_gzip",  # .sam.gz plus pigz=<threads>, no plain intermediate.
        "needs_append_for_multi": True,  # All libraries share one bbmap SAM.
        "template_filter": "builtin",  # outm omits fully unmapped templates.
        "supports_unmapped_mates": True,  # outm retains unmapped mates of mapped reads.
        "supports_include_unmapped": True,  # out emits all reads instead of outm.
        "supports_concordant": True,  # SAM mate coordinates/strand flags are available.
        "supports_proper": True,  # BBMap assigns flag 0x2 using its own pairing criteria.
        "pair_filter_stage": "posthoc",  # Preserve exact record-level predicates after mapping.
        "temporary_patterns": ("*tmp*",),  # Remove only mapper-local temporary directories.
    },
    # MMseqs searches each read independently. Output lacks unmapped records
    # and usable paired-template semantics; /dev/stdout fails as /dev/stdout.0.
    "mmseqs": {
        "run": run_mmseqs,  # Invoke easy-search with SAM --format-mode 1.
        "output_mode": "sam_file",  # Write real SAM then gzip; stdout is not a SAM channel.
        "needs_append_for_multi": False,  # A separate result file is written per input.
        "template_filter": "hits_only",  # Only hits exist; absent mates cannot be recovered.
        "supports_unmapped_mates": False,  # Paired input is searched as independent reads.
        "supports_include_unmapped": False,  # No unmapped records are emitted.
        "supports_concordant": False,  # Our paired-record predicate is unavailable.
        "supports_proper": False,  # No mapper-defined proper-pair status.
        "pair_filter_stage": "unsupported",  # Reject strict filters rather than silently ignoring them.
        "temporary_patterns": ("*tmp*",),  # Includes each easy-search work directory.
    },
    # BWA's SAM flags identify half-mapped/fully unmapped templates. Keep its
    # alignments and filter emitted stdout; native pairing heuristics are unchanged.
    "bwa-mem2": {
        "run": run_bwa_mem2,  # Index reference, then run mem without -o.
        "output_mode": "sam_stream",  # Mapper stdout -> predicates -> plain/pigz writer.
        "needs_append_for_multi": False,  # One SAM stream/file per input library.
        "template_filter": "stream",  # Omit unmapped single reads and pairs with both mates unmapped.
        "supports_unmapped_mates": True,  # Native SAM emits both mates and their flags.
        "supports_include_unmapped": True,  # Disable default template filtering to keep everything.
        "supports_concordant": True,  # Native SAM includes mate coordinates and strands.
        "supports_proper": True,  # BWA assigns flag 0x2; our filter respects it.
        "pair_filter_stage": "stream",  # Strict predicates run before compression/output.
        "temporary_patterns": ("*tmp*", "bwa_index"),  # Temporary index follows the spec, including aliases.
    },
    # This is our mappy serializer, not the minimap2 CLI: mates are mapped
    # separately, missing mate records are synthesized, and flag 0x2 is not set.
    "minimap2": {
        "run": run_mappy,  # Invoke Python bindings and RolyPoly SAM serialization.
        "output_mode": "sam_stream",  # Incremental records -> plain/pigz writer.
        "needs_append_for_multi": False,  # Separate SAM output per input library.
        "template_filter": "builtin",  # Serializer omits fully unmapped templates.
        "supports_unmapped_mates": True,  # Serializer writes missing mates alongside mapped partners.
        "supports_include_unmapped": True,  # Serializer can emit fully unmapped reads/pairs.
        "supports_concordant": True,  # Serializer supplies the mate fields needed for FR filtering.
        "supports_proper": False,  # Serializer intentionally does not infer proper-pair status.
        "pair_filter_stage": "posthoc",  # Strict concordance currently filters completed output.
        "temporary_patterns": ("*tmp*",),  # No disk index; preserve standard mapper-local cleanup.
    },
}


@click.command(name="map", no_args_is_help=True)
@click.option(
    "-i",
    "--input",
    default=None,
    help=(
        "Input FASTQ file, comma-separated FASTQ files, or directory to "
        "auto-detect libraries from"
    ),
)
@click.option(
    "-id",
    "--input-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Input directory to scan for FASTQ files (alias for --input)",
)
@click.option(
    "-r", "--reference", required=True, help="Reference FASTA to map against"
)
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=False, dir_okay=True),
    default="RP_mapping_output",
    help="Output directory",
)
@click.option(
    "--paired-end",
    multiple=True,
    nargs=3,
    default=(),
    help="Explicit paired-end library: <lib_num> <R1> <R2>",
)
@click.option(
    "--single-end",
    multiple=True,
    nargs=2,
    default=(),
    help="Explicit single-end library: <lib_num> <fastq>",
)
@click.option(
    "--merged",
    multiple=True,
    nargs=2,
    default=(),
    help="Explicit merged-read library: <lib_num> <fastq>",
)
@click.option(
    "--long-read",
    multiple=True,
    nargs=1,
    default=(),
    help="Long-read FASTQ file(s)",
)
@click.option(
    "-m",
    "--mapper",
    default=("bbmap",),
    multiple=True,
    type=click.Choice(tuple(MAPPER_SPECS)),
    help="Mapper (read aligner) choice. Use multiple -m flags to run multiple mappers.",
)
@click.option(
    "-ow",
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing output directory",
)
@click.option(
    "--concordant",
    "--concord",
    is_flag=True,
    default=False,
    help=(
        "Keep only paired alignments whose mates map to the same reference "
        "in inward-facing FR orientation. Does not enforce insert size."
    ),
)
@click.option(
    "--proper",
    is_flag=True,
    default=False,
    help=(
        "Keep only alignments marked proper-pair (SAM flag 0x2) by the mapper. "
        "Supported by " + ", ".join(
            name for name, spec in MAPPER_SPECS.items() if spec["supports_proper"]
        ) + "."
    ),
)
@click.option(
    "--include-unmapped",
    is_flag=True,
    default=False,
    help="Also retain fully unmapped reads/pairs where supported. By default, retain both mates if either maps.",
)
@click.option(
    "-z",
    "--compressed",
    is_flag=True,
    default=False,
    help="Alias for --compression-policy sam_only (the default).",
)
@click.option(
    "-cp",
    "--compression-policy",
    type=click.Choice(COMPRESSION_POLICIES),
    default="sam_only",
    show_default=True,
    help="SAM output compression: sam_only (gzip, default) or none (plain SAM).",
)
@click.option(
    "--bwa-mem2-all/--no-bwa-mem2-all",
    default=False,
    help=(
        "bwa-mem2: pass -a, which reports all alignments for single-end or "
        "unpaired paired-end reads."
    ),
)
@click.option(
    "--bwa-mem2-extra-flags",
    default="",
    help="Additional bwa-mem2 mem flags (e.g. '-Y -q'); -o is not allowed with streamed output.",
)
def map(
    input,
    input_dir,
    reference,
    threads,
    memory,
    output,
    keep_tmp,
    log_file,
    paired_end,
    single_end,
    merged,
    long_read,
    mapper,
    overwrite,
    concordant,
    proper,
    compressed,
    log_level,
    temp_dir=None,  # TBD
    bwa_mem2_all=False,
    bwa_mem2_extra_flags="",
    compression_policy: str = "sam_only",
    include_unmapped: bool = False,
):
    """Map reads to a reference FASTA with one or more aligners.

    Input libraries are auto-detected from a file, comma-separated file list, or
    directory supplied with --input / --input-dir, or declared explicitly via
    --paired-end / --single-end / --merged / --long-read. Each input file is
    mapped independently; outputs land in per-mapper sub-directories under
    --output.

    Pair filtering is deliberately optional. In fragmented assemblies, a real
    biological fragment may span an unassembled gap and place its mates on two
    different contigs; --concordant and --proper will exclude such evidence.

    Args:
        input: Path to a FASTQ file or directory for auto-detection.
        input_dir: Alias for --input (directory only).
        reference: Reference FASTA path.
        threads: Number of CPU threads.
        memory: RAM limit string (e.g. ``8g``).
        output: Output directory.
        keep_tmp: If set, keep intermediate files.
        log_file: Path to log file.
        paired_end: Explicit paired libraries as (lib_num, R1, R2) tuples.
        single_end: Explicit single-end libraries as (lib_num, fastq) tuples.
        merged: Explicit merged libraries as (lib_num, fastq) tuples.
        long_read: Long-read FASTQ files.
        mapper: Aligner(s) to run.
        overwrite: Overwrite output directory if it exists.
        concordant: Keep same-reference, inward-facing FR mate alignments.
        proper: Keep records carrying the mapper-defined SAM proper-pair flag.
        compressed: Compress final SAM output from every mapper with pigz.
        log_level: Logging verbosity.
        temp_dir: Optional stable temp directory (useful with --skip-existing).
        bwa_mem2_all: If True, pass -a to bwa-mem2.
        bwa_mem2_extra_flags: Additional bwa-mem2 arguments; output redirection is disallowed.
        compression_policy: Lossless gzip policy for final SAM outputs.
        include_unmapped: Retain unmapped records emitted by the backend.
    """
    setup_logging(log_file, log_level)
    logger = get_logger()
    if compressed:
        compression_policy = "sam_only"
    if compression_policy != "none" and not shutil.which("pigz"):
        raise click.ClickException("SAM compression requires pigz on PATH.")
    sam_suffix = ".sam.gz" if compression_policy == "sam_only" else ".sam"

    if input is None and input_dir is not None:
        input = input_dir

    has_explicit_inputs = any([paired_end, single_end, merged, long_read])
    if input is None and not has_explicit_inputs:
        raise click.ClickException(
            "No input reads provided. Use --input / --input-dir and/or "
            "explicit --paired-end / --single-end / --merged / --long-read."
        )

    outdir = Path(output).resolve()
    if outdir.exists() and not overwrite:
        raise click.ClickException(
            f"Output directory '{outdir}' already exists. Use --overwrite to replace it."
        )
    if outdir.exists() and overwrite:
        shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True, exist_ok=True)

    reference = Path(reference).resolve()
    if not reference.exists():
        raise click.ClickException(f"Reference FASTA not found: {reference}")

    ensure_memory(memory)

    # ── Input discovery (mirrors assemble.py: auto-detect then add explicit) ─
    paired_reads = []  # list of (r1_path_str, r2_path_str)
    interleaved_reads = []  # list of path_str
    single_reads = []  # list of path_str

    if input is not None:
        file_info = handle_input_fastq(input, logger=logger)
        paired_reads.extend(
            (str(r1), str(r2)) for r1, r2 in file_info.get("R1_R2_pairs", [])
        )
        interleaved_reads.extend(
            str(x) for x in file_info.get("interleaved_files", [])
        )
        single_reads.extend(
            str(x) for x in file_info.get("single_end_files", [])
        )
        for lib_data in file_info.get("rolypoly_data", {}).values():
            if lib_data.get("interleaved"):
                interleaved_reads.append(str(lib_data["interleaved"]))
            if lib_data.get("merged"):
                single_reads.append(str(lib_data["merged"]))

    for _, r1, r2 in paired_end:
        paired_reads.append((str(r1), str(r2)))
    for _, f in single_end:
        single_reads.append(str(f))
    for _, f in merged:
        single_reads.append(str(f))
    for (f,) in long_read:
        single_reads.append(str(f))

    # Deduplicate while preserving order
    seen: set = set()
    paired_reads = [x for x in paired_reads if not (x in seen or seen.add(x))]
    seen = set()
    interleaved_reads = [
        x for x in interleaved_reads if not (x in seen or seen.add(x))
    ]
    seen = set()
    single_reads = [x for x in single_reads if not (x in seen or seen.add(x))]

    if not paired_reads and not interleaved_reads and not single_reads:
        raise click.ClickException("No readable FASTQ inputs were detected.")

    logger.info(
        "Read inputs detected: paired=%d  interleaved=%d  single=%d",
        len(paired_reads),
        len(interleaved_reads),
        len(single_reads),
    )

    # Normalise mapper list (deduplicate, handle comma-separated tokens)
    requested_mappers: list = []
    seen = set()
    for m in mapper:
        for token in str(m).split(","):
            token = token.strip().lower()
            if token and token not in seen:
                seen.add(token)
                requested_mappers.append(token)

    if (concordant or proper) and not paired_reads and not interleaved_reads:
        raise click.ClickException(
            "--concordant/--proper require paired-end or interleaved input."
        )
    for mapper_name in requested_mappers:
        spec = MAPPER_SPECS.get(mapper_name)
        if spec is None:
            raise click.ClickException(f"Unsupported mapper: {mapper_name}")
        if (concordant or proper) and spec["pair_filter_stage"] == "unsupported":
            raise click.ClickException(
                f"The {mapper_name} mapper searches reads independently and cannot apply "
                "--concordant or --proper."
            )
        if concordant and not spec["supports_concordant"]:
            raise click.ClickException(f"The {mapper_name} adapter cannot apply --concordant.")
        if proper and not spec["supports_proper"]:
            raise click.ClickException(
                f"The {mapper_name} adapter does not assign SAM proper-pair flags; "
                "--proper is unsupported."
            )

    mapper_outputs: dict = {}

    for mapper_name in requested_mappers:
        spec = MAPPER_SPECS[mapper_name]
        backend_suffix = ".sam" if spec["output_mode"] == "sam_file" else sam_suffix
        mapper_outdir = outdir / mapper_name
        mapper_outdir.mkdir(parents=True, exist_ok=True)
        logger.info("Running mapper: %s", mapper_name)
        logger.debug("Mapper %s specification: %s", mapper_name, spec)
        if (
            ((paired_reads or interleaved_reads) and not spec["supports_unmapped_mates"])
            or (include_unmapped and not spec["supports_include_unmapped"])
        ):
            logger.warning(
                "%s adapter cannot retain all requested unmapped mates/reads "
                "(supports_unmapped_mates=%s, supports_include_unmapped=%s).",
                mapper_name, spec["supports_unmapped_mates"],
                spec["supports_include_unmapped"],
            )

        run: MapperRun = {
            "reference": reference,
            "output": mapper_outdir,
            "suffix": backend_suffix,
            "paired_reads": paired_reads,
            "interleaved_reads": interleaved_reads,
            "single_reads": single_reads,
            "threads": threads,
            "memory": memory,
            "include_unmapped": include_unmapped,
            "bwa_mem2_all": bwa_mem2_all,
            "bwa_mem2_extra_flags": bwa_mem2_extra_flags,
            "stream_concordant": concordant and spec["pair_filter_stage"] == "stream",
            "stream_proper": proper and spec["pair_filter_stage"] == "stream",
            "stream_mapped_templates_only": not include_unmapped and spec["template_filter"] == "stream",
        }
        outputs = spec["run"](run, spec)

        if spec["pair_filter_stage"] == "posthoc" and (concordant or proper):
            from rolypoly.utils.bio.alignments import filter_sam_by_pair_status

            for sam_file in outputs:
                read_count, written_count = filter_sam_by_pair_status(
                    sam_file, concordant=concordant, proper=proper,
                )
                logger.info(
                    "SAM-filtered %s: retained %d of %d records "
                    "(concordant=%s, proper=%s)",
                    sam_file,
                    written_count,
                    read_count,
                    concordant,
                    proper,
                )

        if spec["output_mode"] == "sam_file":
            outputs = compress_sam_outputs(outputs, compression_policy, threads)

        mapper_outputs[mapper_name] = outputs

    logger.info("Mapping completed. Output directory: %s", outdir)
    for mapper_name, files in mapper_outputs.items():
        logger.info("%s outputs: %s", mapper_name, files)

    if not keep_tmp:
        for mapper_name in requested_mappers:
            for pattern in MAPPER_SPECS[mapper_name]["temporary_patterns"]:
                for candidate in (outdir / mapper_name).glob(pattern):
                    if candidate.is_dir():
                        shutil.rmtree(candidate, ignore_errors=True)


if __name__ == "__main__":
    map()
