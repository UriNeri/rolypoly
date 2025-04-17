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
    "-o", "--output", default=lambda: f"{os.getcwd()}_corrolate.tsv", help="output path"
)
@click.option("-t", "--threads", default=1, help="Number of threads")
@click.option(
    "-g",
    "--log-file",
    default=lambda: f"{os.getcwd()}/corrolate_logfile.txt",
    help="Path to log file",
)
def corrolate(input, output, threads, log_file):
    """WIP WIP WIP Corrolate identified viral sequence across samples"""
    from rolypoly.utils.loggit import setup_logging

    logger = setup_logging(log_file)
    logger.info("Starting to corrolate      ")
    logger.info("Sorry! command not yet implemented!")


if __name__ == "main":
    corrolate()
