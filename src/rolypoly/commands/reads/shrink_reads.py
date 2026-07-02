import os
from pathlib import Path

import rich_click as click

from rolypoly.utils.bio.library_detection import (
    create_sample_file,
    handle_input_fastq,
)
from rolypoly.utils.logging.loggit import log_start_info, setup_logging


@click.command(name="shrink-reads", no_args_is_help=True)
@click.option(
    "-i",
    "-in",
    "--input",
    required=False,
    help="""Input raw reads file(s) or directory containing them. For paired-end reads, you can provide an interleaved file or the R1 and R2 files separated by comma. If a directory is provided, one output per input identified file/pair will be created. """,
)
@click.option(
    "-o",
    "-out",
    "--output",
    hidden=True,
    default=os.getcwd(),
    type=click.Path(),
    help="path to output directory",
)
@click.option(
    "-st",
    "--subset-type",
    default="top_reads",
    type=click.Choice(["top_reads", "random", "bbnorm"]),
    help="how to sample reads from input.",
)
@click.option(
    "-sz",
    "--sample-size",
    default=1000,
    type=click.FLOAT,
    help="For top_reads/random, at most this many reads (or proportion if <1). For bbnorm, this is the target k-mer depth.",
)
@click.option(
    "--bbnorm-min-depth",
    default=2,
    type=int,
    # hidden=True,
    help="Minimum depth threshold for bbnorm normalization (min in bbnorm.sh).",
)
@click.option(
    "-t",
    "--threads",
    default=1,
    type=int,
    help="Threads to use for bbnorm (if subset-type is bbnorm). No real threading support yet for random/top_reads methods.",
)
@click.option(
    "-g",
    "--log-file",
    type=click.Path(),
    default=lambda: f"{os.getcwd()}/rolypoly.log",
    help="Path to save loggging message to. defaults to current folder.",
)
@click.option(
    "-ll",
    "--log-level",
    default="info",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    help="Log level. Options: debug, info, warning, error, critical",
)   
def shrink_reads(
    input,
    output,
    subset_type,
    sample_size,
    bbnorm_min_depth,
    threads, # TODO: no real threading support yet (apart from bbnorm) maybe will have it if multiple input files are used (one per thread)
    log_file,
    log_level,
):
    """
    Subset FASTQ reads by count or fraction for lightweight test datasets.

    Supports deterministic head-style subsampling (`first_n`) and random
    sampling (`random`) for single-end, interleaved, and paired-end layouts.

    This command is intended for quick dry runs and resource-reduced tests,
    not as a full read-normalization strategy.
    """
    # Initialise logger
    logger = setup_logging(log_file=log_file, log_level=log_level)
    log_start_info(logger, locals())

    # Ensure output directory exists
    output_dir = Path(output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Starting read processing")
        # Detect and organise input FASTQ files
        file_info = handle_input_fastq(input, logger=logger)
        logger.debug(f"Detected file info: {file_info}")

        for file_path in file_info.get("single_end_files", []):
            file_path = Path(file_path)
            logger.info(f"Processing file: {file_path}")
            suffix = "bbnorm" if subset_type == "bbnorm" else "shrinked"
            output_file = output_dir / f"{file_path.stem}_{suffix}.fastq"
            create_sample_file(
                file_path=file_path,
                subset_type=subset_type,
                sample_size=sample_size,
                output_file=str(output_file),
                threads=threads,
                bbnorm_min_depth=bbnorm_min_depth,
                interleaved=False,
                logger=logger,
            )
            logger.info(f"Written sampled reads to {output_file}")

        for r1_path, r2_path in file_info.get("R1_R2_pairs", []):
            logger.info(f"Processing paired-end files: {r1_path} and {r2_path}")
            if subset_type != "bbnorm":
                logger.debug("""Note - to ensure paired reads are sampled, this will be slow (i.e. if reads_x/1 was selected from file R1, and his pair reads_x/2 is at the bottom of the R2 file, I can't think of a method to get it without going over all of R2 (if compressed). 
                             However, read order is usually assumed to be the same for R1 and R2...
                             """)

            suffix = "bbnorm" if subset_type == "bbnorm" else "shrinked"
            # Strip all FASTQ/gz extensions (Path.stem only removes one suffix, so
            # SRR26891210_1.fastq.gz would give SRR26891210_1.fastq, not SRR26891210_1).
            # Also use r1_stem for BOTH outputs so the base names match and
            # identify_fastq_files can pair them via the _R1/_R2 suffix alone.
            # (R1 and R2 inputs often differ only in _1/_2, so if we preserved
            # both stems, the output pair detector would fail to link them.)
            _fastq_exts = {".fastq", ".fq", ".gz", ".bz2"}
            r1_stem = Path(r1_path)
            while r1_stem.suffix.lower() in _fastq_exts:
                r1_stem = Path(r1_stem.stem)
            output_r1_file = output_dir / f"{r1_stem.name}_{suffix}_R1.fastq"
            output_r2_file = output_dir / f"{r1_stem.name}_{suffix}_R2.fastq"
            create_sample_file(
                file_path=f"{r1_path},{r2_path}",
                subset_type=subset_type,
                sample_size=sample_size,
                output_file=f"{output_r1_file},{output_r2_file}",
                threads=threads,
                bbnorm_min_depth=bbnorm_min_depth,
                logger=logger,
            )
            logger.info(
                f"Written sampled reads to {output_r1_file} and {output_r2_file}"
            )

        for file_path in file_info.get("interleaved_files", []):
            file_path = Path(file_path)
            logger.info(f"Processing file: {file_path}")
            suffix = "bbnorm" if subset_type == "bbnorm" else "shrinked"
            output_file = output_dir / f"{file_path.stem}_{suffix}.fastq"
            create_sample_file(
                file_path=file_path,
                subset_type=subset_type,
                sample_size=sample_size,
                output_file=str(output_file),
                threads=threads,
                bbnorm_min_depth=bbnorm_min_depth,
                interleaved=True,
                logger=logger,
            )
            logger.info(f"Written sampled reads to {output_file}")

        logger.info("Finished read processing")
        logger.info(f"Output: {output_dir}")

    except Exception as e:
        logger.error(f"An error occurred during read processing: {e}")
        raise
