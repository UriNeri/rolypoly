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
    default=lambda: f"{os.getcwd()}_virus_characteristics",
    help="Path to a tsv to save the summmary info of a RolyPoly run",
)
@click.option("-t", "--threads", default=1, help="Number of threads")
@click.option(
    "-g",
    "--log-file",
    default=lambda: f"{os.getcwd()}/summarise_logfile.txt",
    help="Path to log file",
)
def summarise(input, output, threads, log_file):
    """WIP WIP WIP Summarize RolyPoly results."""
    from rolypoly.utils.loggit import setup_logging

    logger = setup_logging(log_file)
    logger.info("Starting to summarise RolyPoly     ")
    logger.info("Sorry! command noit yet implemented!")


if __name__ == "main":
    summarise()
