import os
import shutil
from pathlib import Path
from typing import Union
import rich_click as click

from rolypoly.utils.logging.loggit import log_start_info, setup_logging
from rolypoly.utils.bio.library_detection import handle_input_fastq, create_sample_file



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
    "-st",
    "-sty",
    "--subset_type",
    default="top_reads",
    type=click.Choice(["top_reads", "random"]),
    help="how to sample reads from input."
)
@click.option(
    "-s",
    "-smp",
    "--sample_size",
    default=1000,
    type=Union[int, float],
    help="Will only return (at most) this much reads (if <1, will be interpreted as a proportion of total reads, else as the exact number of reads to get)")
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
    subset_type,
    sample_size, 
    threads, # TODO: no real threading support yet, maybe will have it if multiple input files are used (one per thread)
    log_file,
    log_level,
):
    """
    Subsets data from an input FASTQ file(s) based on the provided options.
    Supports random sampling (when sample_size < 1) and fixed number read sampling
    (i.e. first n reads)
    NOTE: UNLESS --keep-matching-pairs is used, the output may not be very useful in practice.
    TODO: no real threading support yet, maybe will have it if multiple input files are used (one per thread)
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

        # Process single‑end files
        for file_path in file_info.get("single_end_files", []):
            file_path = Path(file_path)
            logger.info(f"Processing file: {file_path}")
            output_file = output_dir / f"{file_path.stem}_shrinked.fastq"
            create_sample_file(
                    file_path,
                    subset_type=subset_type,
                    sample_size=sample_size,
                    logger=logger,
                    output_file=output_file,
                )
            logger.info(f"Written shrinked reads to {output_file}")

        # Process paired‑end files
        for r1_path, r2_path in file_info.get("R1_R2_pairs", []):
            # print("a")
            logger.info(f"Processing paired-end files: {r1_path} and {r2_path}")
            logger.info(f"Note - to ensure paired reads are sampled, this will be slow (i.e. if reads_x/1 was selected from file R1, and his pair reads_x/2 is at the bottom of the R2 file, I can't think of a method to get it without going over all of R2 (if compressed")
            
            logger.info(f"Sampling {sample_size} from {r1_path}")
            output_R1_file = output_dir / f"{r1_path.stem}_shrinked_R1.fastq"
            output_R2_file = output_dir / f"{r1_path.stem}_shrinked_R2.fastq"
            output_file = output_R1_file+","+output_R2_file
            create_sample_file(
                    str(r1_path)+",",str(r2_path) ,
                    subset_type=subset_type,
                    sample_size=sample_size,
                    logger=logger,
                    output_file=output_file,
                )
                                         
            logger.info(f"Written shrinked reads to {output_R1_file} and {output_R2_file}")

        logger.info("Finished read processing")
        logger.info("[bold green]✓[/bold green] Processed reads")
        logger.info(f"[bold blue]Output:[/bold blue] {output_dir}")

    except Exception as e:
        logger.error(f"An error occurred during read processing: {e}")
        raise
