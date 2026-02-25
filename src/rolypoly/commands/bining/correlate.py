import os

import rich_click as click
from rich.console import Console

console = Console()


@click.command()
@click.option(
    "-i",
    "--input",
    required=True,
    help="Input directory containing rolypoly's virus identification and annotation results",
)
@click.option(
    "-o",
    "--output",
    default=lambda: f"{os.getcwd()}_correlate.tsv",
    help="output path",
)
@click.option("-t", "--threads", default=1, help="Number of threads")
@click.option(
    "-g",
    "--log-file",
    default=lambda: f"{os.getcwd()}/correlate_logfile.txt",
    help="Path to log file",
)
@click.option(
    "-ll", "--log-level", hidden=True, default="INFO", help="Log level"
)
def correlate(input, output, threads, log_file, log_level):
    """WIP WIP WIP correlate identified viral sequence across samples"""
    from rolypoly.utils.logging.loggit import setup_logging

    logger = setup_logging(log_file, log_level)
    logger.info("Starting to correlate      ")
    logger.info("Sorry! command not yet implemented!")


if __name__ == "main":
    correlate()
