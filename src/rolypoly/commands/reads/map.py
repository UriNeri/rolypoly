"""Read mapping command: map reads to a reference using one or more aligners.

Supports bbmap (via bbmapy), minimap2 (via mappy Python bindings), bwa-mem2,
and mmseqs. Input discovery follows the same pattern as assemble:
--input / --input-dir for auto-detection, plus explicit --paired-end /
--single-end / --merged / --long-read library flags. Each input file is mapped
individually (no concatenation).
"""

import os
import shutil
from pathlib import Path

import rich_click as click

from rolypoly.utils.bio.library_detection import identify_fastq_files
from rolypoly.utils.logging.loggit import get_logger, setup_logging
from rolypoly.utils.various import (
    ensure_memory,
    get_reduced_memory,
    run_command_comp,
)


def mappy_hit_to_sam(hit, name, seq, qual, mate=None, read1=None):
    """Build a SAM record line from a mappy.Alignment hit.

    mappy.Alignment has no built-in SAM serialization, so the record is
    assembled manually from its fields. Handles:
      - strand-aware SEQ/QUAL (reverse-complemented for reverse-strand hits)
      - secondary-alignment flag (0x100) via hit.is_primary, so downstream
        tools (coverage/depth callers, dedup, etc.) don't treat every
        multi-mapping hit as an independent primary alignment
      - paired-end FLAG bits and RNEXT/PNEXT/TLEN when `mate` is provided

    Args:
        hit: mappy.Alignment for this read.
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
    sam_f, name1, seq1, qual1, hits1, name2, seq2, qual2, hits2
):
    """Write SAM records for one read pair, cross-referencing mate fields.

    Each mate's own primary alignment (first hit with is_primary=True, or
    the first hit if none is flagged) is used as the "mate" reference for
    the other mate's RNEXT/PNEXT/TLEN/mate-reverse-strand fields. All hits
    for both mates are written (including secondary ones, flagged 0x100).
    Mates with zero hits are skipped (unmapped reads are not written), and
    the surviving mate correctly gets the mate-unmapped flag (0x8).
    """
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


@click.command(name="map", no_args_is_help=True)
@click.option(
    "-i",
    "--input",
    default=None,
    help="Input FASTQ file or directory to auto-detect libraries from",
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
@click.option("-t", "--threads", default=1, type=int, help="Number of threads")
@click.option("-M", "--memory", default="6gb", help="RAM limit (e.g. 6gb)")
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=False, dir_okay=True),
    default="RP_mapping_output",
    help="Output directory",
)
@click.option(
    "-k", "--keep-tmp", is_flag=True, default=False, help="Keep temporary files"
)
@click.option(
    "-g",
    "--log-file",
    default=lambda: f"{os.getcwd()}/map_logfile.txt",
    help="Path to logfile",
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
    type=click.Choice(["bbmap", "mmseqs", "bwa-mem2", "minimap2"]),
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
    "-ll", "--log-level", default="INFO", hidden=True, help="Log level"
)
@click.option(
    "--temp-dir",
    default=None,
    help="Optional base directory for temporary files",
)
@click.option(
    "--bwa-mem2-all",
    is_flag=True,
    default=True,
    help="bwa-mem2: report all valid alignments (-a flag). Recommended for quasispecies/strain disentangling.",
)
@click.option(
    "--bwa-mem2-extra-flags",
    default="",
    help="Additional flags to pass verbatim to bwa-mem2 mem (e.g. '-Y -q').",
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
    log_level,
    temp_dir=None, # TBD
    bwa_mem2_all=True,
    bwa_mem2_extra_flags="",
):
    """Map reads to a reference FASTA with one or more aligners.

    Input libraries are auto-detected from --input / --input-dir (same detection
    logic as assemble) or declared explicitly via --paired-end / --single-end /
    --merged / --long-read. Each input file is mapped independently; outputs land
    in per-mapper sub-directories under --output.

    Args:
        input: Path to a FASTQ file or directory for auto-detection.
        input_dir: Alias for --input (directory only).
        reference: Reference FASTA path.
        threads: Number of CPU threads.
        memory: RAM limit string (e.g. '6gb').
        output: Output directory.
        keep_tmp: If set, keep intermediate files.
        log_file: Path to log file.
        paired_end: Explicit paired libraries as (lib_num, R1, R2) tuples.
        single_end: Explicit single-end libraries as (lib_num, fastq) tuples.
        merged: Explicit merged libraries as (lib_num, fastq) tuples.
        long_read: Long-read FASTQ files.
        mapper: Aligner(s) to run.
        overwrite: Overwrite output directory if it exists.
        log_level: Logging verbosity.
        temp_dir: Optional stable temp directory (useful with --skip-existing).
        bwa_mem2_all: If True, pass -a to bwa-mem2 to report all valid alignments.
        bwa_mem2_extra_flags: Additional verbatim flags for bwa-mem2 mem.
    """
    setup_logging(log_file, log_level)
    logger = get_logger()

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

    memory_giga = ensure_memory(memory)["giga"]

    # ── Input discovery (mirrors assemble.py: auto-detect then add explicit) ─
    paired_reads = []  # list of (r1_path_str, r2_path_str)
    interleaved_reads = []  # list of path_str
    single_reads = []  # list of path_str

    if input is not None:
        file_info = identify_fastq_files(
            Path(input).resolve(), return_rolypoly=True, logger=logger
        )
        paired_reads.extend(
            (str(r1), str(r2)) for r1, r2 in file_info.get("R1_R2_pairs", [])
        )
        interleaved_reads.extend(
            str(x) for x in file_info.get("interleaved_files", [])
        )
        single_reads.extend(str(x) for x in file_info.get("single_end", []))
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

    mapper_outputs: dict = {}

    for mapper_name in requested_mappers:
        mapper_outdir = outdir / mapper_name
        mapper_outdir.mkdir(parents=True, exist_ok=True)
        outputs: list = []
        logger.info("Running mapper: %s", mapper_name)

        # ── bbmap via bbmapy ─────────────────────────────────────────────────
        if mapper_name == "bbmap":
            memory_giga = get_reduced_memory(
                ensure_memory(memory), percentage=85
            )  # bbmapy overhead. this is now actually in mb but don't worry about it...
            from bbmapy import bbmap as bbmap_run

            bbmap_sam = mapper_outdir / "bbmap.sam"
            first = True
            for r1, r2 in paired_reads:
                bbmap_run(
                    ref=reference,
                    outm=str(bbmap_sam),
                    threads=threads,
                    Xmx=memory_giga,
                    minid=0.8,
                    nodisk="t",
                    overwrite="t",
                    append="f" if first else "t",
                    in1=str(r1),
                    in2=str(r2),
                )
                first = False
            for fq in interleaved_reads:
                bbmap_run(
                    ref=reference,
                    outm=str(bbmap_sam),
                    threads=threads,
                    Xmx=memory_giga,
                    minid=0.8,
                    nodisk="t",
                    overwrite="t",
                    append="f" if first else "t",
                    in_file=str(fq),
                    interleaved="t",
                )
                first = False
            for fq in single_reads:
                bbmap_run(
                    ref=reference,
                    outm=str(bbmap_sam),
                    threads=threads,
                    Xmx=memory_giga,
                    minid=0.8,
                    nodisk="t",
                    overwrite="t",
                    append="f" if first else "t",
                    in_file=str(fq),
                )
                first = False
            if bbmap_sam.exists() and bbmap_sam.stat().st_size > 0:
                run_command_comp(
                    "pigz",
                    params={"p": str(threads)},
                    positional_args=[str(bbmap_sam)],
                    logger=logger,
                    check_status=True,
                    prefix_style="single",
                )
                outputs.append(str(bbmap_sam) + ".gz")

        # ── minimap2 via mappy Python bindings ───────────────────────────────
        elif mapper_name == "minimap2":
            import mappy as mp

            aligner = mp.Aligner(str(reference), preset="sr", n_threads=threads)
            if not aligner:
                raise click.ClickException(
                    f"mappy: failed to load/build index from {reference}"
                )
            sam_header = (
                "@HD\tVN:1.6\tSO:unsorted\n"
                + "\n".join(
                    f"@SQ\tSN:{name}\tLN:{len(aligner.seq(name))}"
                    for name in aligner.seq_names
                )
                + "\n@PG\tID:mappy\tPN:minimap2\n"
            )

            idx = 0

            # Explicit/auto-detected R1+R2 pairs: read mates in lockstep so
            # RNEXT/PNEXT/TLEN and paired FLAG bits can be populated.
            for r1, r2 in paired_reads:
                out_sam = mapper_outdir / f"minimap2_paired_{idx}.sam"
                with open(out_sam, "w") as sam_f:
                    sam_f.write(sam_header)
                    for (name1, seq1, qual1), (name2, seq2, qual2) in zip(
                        mp.fastx_read(str(r1)), mp.fastx_read(str(r2))
                    ):
                        hits1 = list(aligner.map(seq1, MD=True))
                        hits2 = list(aligner.map(seq2, MD=True))
                        write_paired_sam_records(
                            sam_f,
                            name1,
                            seq1,
                            qual1,
                            hits1,
                            name2,
                            seq2,
                            qual2,
                            hits2,
                        )
                outputs.append(str(out_sam))
                idx += 1

            # Interleaved libraries: alternating R1/R2 records in one file.
            for fq in interleaved_reads:
                out_sam = mapper_outdir / f"minimap2_interleaved_{idx}.sam"
                with open(out_sam, "w") as sam_f:
                    sam_f.write(sam_header)
                    reads_iter = mp.fastx_read(str(fq))
                    for name1, seq1, qual1 in reads_iter:
                        hits1 = list(aligner.map(seq1, MD=True))
                        try:
                            name2, seq2, qual2 = next(reads_iter)
                        except StopIteration:
                            # Odd read out (unpaired trailing record): write as unpaired.
                            for hit in hits1:
                                sam_f.write(
                                    mappy_hit_to_sam(hit, name1, seq1, qual1)
                                    + "\n"
                                )
                            break
                        hits2 = list(aligner.map(seq2, MD=True))
                        write_paired_sam_records(
                            sam_f,
                            name1,
                            seq1,
                            qual1,
                            hits1,
                            name2,
                            seq2,
                            qual2,
                            hits2,
                        )
                outputs.append(str(out_sam))
                idx += 1

            # True single-end / merged reads: no mate to cross-reference.
            for fq in single_reads:
                out_sam = mapper_outdir / f"minimap2_{idx}.sam"
                with open(out_sam, "w") as sam_f:
                    sam_f.write(sam_header)
                    for name, seq, qual in mp.fastx_read(str(fq)):
                        for hit in aligner.map(seq, MD=True):
                            sam_f.write(
                                mappy_hit_to_sam(hit, name, seq, qual) + "\n"
                            )
                outputs.append(str(out_sam))
                idx += 1

        # ── bwa-mem2 ─────────────────────────────────────────────────────────
        elif mapper_name == "bwa-mem2":
            # Build the index into a temp subdirectory (cleaned up unless --keep-tmp).
            bwa_index_dir = mapper_outdir / "bwa_index"
            bwa_index_dir.mkdir(parents=True, exist_ok=True)
            bwa_index_prefix = bwa_index_dir / "ref"
            run_command_comp(
                "bwa-mem2",
                positional_args=[
                    "index",
                    "-p",
                    str(bwa_index_prefix),
                    str(reference),
                ],
                positional_args_location="start",
                logger=logger,
                check_status=True,
                prefix_style="single",
            )

            # -a: report all valid alignments (not just primary) so that
            # multi-mapping reads carry information about shared regions
            # between strains/quasispecies — critical for downstream
            # haplotyping/variant-phasing tools.
            # -o: use native output-file flag instead of shell redirection.
            mem_flags = ["-t", str(threads), "-a"]
            if not bwa_mem2_all:
                mem_flags.remove("-a")
            if bwa_mem2_extra_flags:
                mem_flags.extend(bwa_mem2_extra_flags.split())

            idx = 0
            for r1, r2 in paired_reads:
                out_sam = mapper_outdir / f"bwa_mem2_paired_{idx}.sam"
                run_command_comp(
                    "bwa-mem2",
                    positional_args=[
                        "mem",
                        *mem_flags,
                        "-o",
                        str(out_sam),
                        str(bwa_index_prefix),
                        str(r1),
                        str(r2),
                    ],
                    positional_args_location="start",
                    logger=logger,
                    check_status=True,
                    prefix_style="single",
                )
                outputs.append(str(out_sam))
                idx += 1
            for fq in interleaved_reads:
                # -p: smart pairing mode (reads interleaved paired-end from single file)
                out_sam = mapper_outdir / f"bwa_mem2_interleaved_{idx}.sam"
                run_command_comp(
                    "bwa-mem2",
                    positional_args=[
                        "mem",
                        *mem_flags,
                        "-p",
                        "-o",
                        str(out_sam),
                        str(bwa_index_prefix),
                        str(fq),
                    ],
                    positional_args_location="start",
                    logger=logger,
                    check_status=True,
                    prefix_style="single",
                )
                outputs.append(str(out_sam))
                idx += 1
            for fq in single_reads:
                out_sam = mapper_outdir / f"bwa_mem2_single_{idx}.sam"
                run_command_comp(
                    "bwa-mem2",
                    positional_args=[
                        "mem",
                        *mem_flags,
                        "-o",
                        str(out_sam),
                        str(bwa_index_prefix),
                        str(fq),
                    ],
                    positional_args_location="start",
                    logger=logger,
                    check_status=True,
                    prefix_style="single",
                )
                outputs.append(str(out_sam))
                idx += 1

        # ── mmseqs easy-search (SAM via format-mode 1) ───────────────────────
        elif mapper_name == "mmseqs":
            all_fqs = (
                list(interleaved_reads)
                + list(single_reads)
                + [f for pair in paired_reads for f in pair]
            )
            for idx, fq in enumerate(all_fqs):
                run_tmp = mapper_outdir / f"mmseqs_tmp_{idx}"
                run_tmp.mkdir(parents=True, exist_ok=True)
                out_sam = mapper_outdir / f"mmseqs_reads_{idx}.sam"
                run_command_comp(
                    "mmseqs",
                    positional_args=[
                        "easy-search",
                        str(fq),
                        str(reference),
                        str(out_sam),
                        str(run_tmp),
                    ],
                    positional_args_location="start",
                    params={
                        "search-type": "3",
                        "min-seq-id": "0.7",
                        "threads": str(threads),
                        "format-mode": "1",
                    },
                    logger=logger,
                    check_status=True,
                    prefix_style="double",
                )
                outputs.append(str(out_sam))

        else:
            raise click.ClickException(f"Unsupported mapper: {mapper_name}")

        mapper_outputs[mapper_name] = outputs

    logger.info("Mapping completed. Output directory: %s", outdir)
    for mapper_name, files in mapper_outputs.items():
        logger.info("%s outputs: %s", mapper_name, files)

    if not keep_tmp:
        for mapper_name in requested_mappers:
            for candidate in (outdir / mapper_name).glob("*tmp*"):
                if candidate.is_dir():
                    shutil.rmtree(candidate, ignore_errors=True)
        bwa_index_dir = outdir / "bwa-mem2" / "bwa_index"
        if bwa_index_dir.is_dir():
            shutil.rmtree(bwa_index_dir, ignore_errors=True)


if __name__ == "__main__":
    map()
