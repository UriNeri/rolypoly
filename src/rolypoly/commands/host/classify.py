import os

import rich_click as click

# from rich.console import Console

# console = Console()


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
    default=lambda: f"{os.getcwd()}_predict_host_range.tsv",
    help="output path",
)
@click.option("-t", "--threads", default=1, help="Number of threads")
@click.option(
    "-g",
    "--log-file",
    default=lambda: f"{os.getcwd()}/predict_host_range_logfile.txt",
    help="Path to log file",
)
def predict_host_range(input, output, threads, log_file):
    """WIP WIP WIP Predict a viral seq host range - caution! this is not definitive"""
    from rolypoly.utils.loggit import setup_logging

    logger = setup_logging(log_file)
    logger.info("Starting to predict      ")
    logger.info("Sorry! command not yet implemented!")


if __name__ == "main":
    predict_host_range()
