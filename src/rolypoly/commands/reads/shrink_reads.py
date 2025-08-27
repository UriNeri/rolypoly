import os
import shutil
from pathlib import Path
from typing import Tuple, Union, Dict, Any, List
import rich_click as click
from rich.console import Console

from rolypoly.utils.logging.loggit import log_start_info, setup_logging
from rolypoly.utils.various import run_command_comp
from rolypoly.utils.bio.library_detection import handle_input_fastq, create_sample_file

console = Console()


@click.command(no_args_is_help=True)
@click.option(
    "-i",
    "-in",
    "--input",
    required=False,
    help="""Input raw reads file(s) or directory containing them. For paired-end reads, you can provide an interleaved file or the R1 and R2 files separated by comma.""",
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
    "-p",
    "-prop",
    "--proportion",
    default=1,
    type=float,
    help="Randomly sample reads from input. This will select <-p> * total reads",
)
@click.option(
    "-n",
    "-nth",
    "--n_reads",
    default=1000,
    type=int,
    help="Will only return (at most) this much reads",
)
@click.option(
    "-d",
    "-dir",
    "--direction",
    type=click.Choice(["first", "last", "both"]),
    default="both",
    help="if --n_reads is used, should the reads be extracted from the top of the file (first nth) or bottom of it (last nth) or both (first nth/2 + last nth/2)",
)
@click.option(
    "-kp",
    "--keep-matching-pairs",
    is_flag=True,
    help="If input is paired-end, output will also be paired-end (i.e., every R1 selected will have a matching R2).",
)
@click.option(
    "-t",
    "--threads",
    default=1,
    type=int,
    hidden=True,
    help="Number of threads to use. Example: -t 4",
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
    default="debug",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    help="Log level. Options: debug, info, warning, error, critical",
)
def shrink_reads(
    input,
    output,
    proportion,
    n_reads, 
    direction,
    keep_matching_pairs,
    threads,
    log_file,
    log_level,
):
    """
    Subsets data from an input FASTQ file(s) based on the provided options.
    Supports random sampling (when proportion < 1) and fixed number read sampling
    (first, last, or both) for single‑end FASTQ files.
    NOTE: UNLESS --keep-matching-pairs is used, the output may not be very useful in practice.
    TODO: no real threading support yet, maybe will have it if multiple input files are used (one per thread)
    """
    # Initialise logger
    logger = setup_logging(log_file=log_file, log_level=log_level)
    log_start_info(logger, locals())

    if keep_matching_pairs and proportion == 1 and n_reads == 0:
        logger.error("Please specify either --proportion, --nth or use --keep-matching-pairs with one of them.")
        raise click.ClickException("Invalid argument combination.")

    # Ensure output directory exists
    output_dir = Path(output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Starting read processing")
        # Detect and organise input FASTQ files
        file_info = handle_input_fastq(input, logger=logger)
        logger.debug(f"Detected file info: {file_info}")

        # Process single‑end files
        for file_path in file_info.get("single_end_files", []):
            file_path = Path(file_path)
            logger.info(f"Processing file: {file_path}")

            # Determine which sampling method to use
            sample_file = None
            if proportion < 1:
                # Random percentage sampling
                logger.debug(
                    f"Sampling {proportion*100}% of reads from {file_path}"
                )
                sample_file = create_sample_file(
                    file_path,
                    sample_type="random_percentage",
                    percentage=proportion/100,
                    logger=logger,
                )
            elif n_reads > 0:
                # Fixed‑read sampling based on direction
                if direction == "first":
                    logger.debug(f"Taking first {n_reads} reads from {file_path}")
                    sample_type = "first_n_lines"
                elif direction == "last":
                    logger.debug(f"Taking last {n_reads} reads from {file_path}")
                    sample_type = "last_n_lines"
                elif direction == "both":
                    logger.debug(
                        f"Taking first and last {n_reads // 2} reads from {file_path}"
                    )
                    sample_type = "first_last_n_lines"

                sample_file = create_sample_file(
                    file_path,
                    sample_type=sample_type,
                    n_lines=n_reads,
                    logger=logger,
                )
            else:
                # No sampling requested – just copy the original file
                logger.debug("No sampling requested; copying original file")
                sample_file = file_path

            # Prepare output filename
            output_file = output_dir / f"{file_path.stem}_shrinked.fastq"

            # Move (or copy) the sampled file to the output location
            if isinstance(sample_file, Path) and sample_file != output_file:
                shutil.move(str(sample_file), str(output_file))
            else:
                shutil.copy2(str(sample_file), str(output_file))

            logger.info(f"Written shrinked reads to {output_file}")

        # Process paired‑end files
        for r1_path, r2_path in file_info.get("R1_R2_pairs", []):
            r1_path = Path(r1_path)
            r2_path = Path(r2_path)
            logger.info(f"Processing paired-end files: {r1_path} and {r2_path}")

            # Determine which sampling method to use
            sample_r1 = None
            sample_r2 = None
            if proportion < 1:
                # Random percentage sampling
                logger.debug(
                    f"Sampling {proportion*100}% of reads from {r1_path} and {r2_path}"
                )
                sample_r1 = create_sample_file(
                    r1_path,
                    sample_type="random_percentage",
                    percentage=proportion/100,
                    logger=logger,
                )
                sample_r2 = create_sample_file(
                    r2_path,
                    sample_type="random_percentage",
                    percentage=proportion/100,
                    logger=logger,
                )
            elif n_reads > 0:
                if keep_matching_pairs:
                    # Fixed‑read sampling based on direction, keeping pairs
                    if direction == "first":
                        logger.debug(f"Taking first {n_reads} paired reads from {r1_path} and {r2_path}")
                        sample_type = "first_n_lines"
                    elif direction == "last":
                        logger.debug(f"Taking last {n_reads} paired reads from {r1_path} and {r2_path}")
                        sample_type = "last_n_lines"
                    else:  # both
                        logger.debug(
                            f"Taking first and last {n_reads // 2} paired reads from {r1_path} and {r2_path}"
                        )
                        sample_type = "first_last_n_lines"

                    sample_r1 = create_sample_file(
                        r1_path,
                        sample_type=sample_type,
                        n_lines=n_reads,
                        logger=logger,
                        keep_pairs=True,
                    )
                    sample_r2 = sample_r1  # sample_r2 will be derived from sample_r1
                else:
                    # Fixed‑read sampling based on direction, without keeping pairs
                    if direction == "first":
                        logger.debug(f"Taking first {n_reads} reads from {r1_path} and {r2_path}")
                        sample_type = "first_n_lines"
                    elif direction == "last":
                        logger.debug(f"Taking last {n_reads} reads from {r1_path} and {r2_path}")
                        sample_type = "last_n_lines"
                    else:  # both
                        logger.debug(
                            f"Taking first and last {n_reads // 2} reads from {r1_path} and {r2_path}"
                        )
                        sample_type = "first_last_n_lines"

                    sample_r1 = create_sample_file(
                        r1_path,
                        sample_type=sample_type,
                        n_lines=n_reads,
                        logger=logger,
                    )
                    sample_r2 = create_sample_file(
                        r2_path,
                        sample_type=sample_type,
                        n_lines=n_reads,
                        logger=logger,
                    )
            else:
                # No sampling requested – just copy the original files
                logger.debug("No sampling requested; copying original files")
                sample_r1 = r1_path
                sample_r2 = r2_path

            # Prepare output filenames
            output_r1 = output_dir / f"{r1_path.stem}_shrinked_R1.fastq"
            output_r2 = output_dir / f"{r1_path.stem}_shrinked_R2.fastq"

            # Move (or copy) the sampled files to the output location
            if isinstance(sample_r1, Path) and sample_r1 != output_r1:
                shutil.move(str(sample_r1), str(output_r1))
            else:
                shutil.copy2(str(sample_r1), str(output_r1))

            if isinstance(sample_r2, Path) and sample_r2 != output_r2:
                shutil.move(str(sample_r2), str(output_r2))
            else:
                shutil.copy2(str(sample_r2), str(output_r2))

            logger.info(f"Written shrinked reads to {output_r1} and {output_r2}")

        logger.info("Finished read processing")
        console.print("[bold green]✓[/bold green] Processed reads")
        console.print(f"[bold blue]Output:[/bold blue] {output_dir}")

    except Exception as e:
        logger.error(f"An error occurred during read processing: {e}")
        raise
