import os
import click
from rich.console import Console
# from utils.logging import setup_logging
from rich.spinner import Spinner
import time
from rich.progress import track
from rich.spinner import Spinner,SPINNERS # type: ignore


console = Console()
tasks = [f"task {n}" for n in range(1, 11)]
# global SPINNERS
SPINNERS['myspinner'] = {'interval': 1000, 'frames': ['🦠 ', '🧬 ', '🔬 ']}
@click.command()
@click.option("-i", "--input", required=False, help="Input directory containing rolypoly's virus identification and annotation results")
@click.option("-o", "--output", default=lambda: f"{os.getcwd()}_predict_host_range.tsv", help="output path")
@click.option("-t", "--threads", default=1, help="Number of threads")
@click.option("-g", "--log-file", default=lambda: f"{os.getcwd()}/predict_host_range_logfile.txt", help="Path to log file")
def dummy(input, output,  threads, log_file):
    """WIP WIP WIP Predict a viral seq host range - caution! this is not definitive"""
    task = tasks.pop(0)
    with console.status(f"[bold green] Working on {task}    ",spinner="myspinner") as status: # 'myspinner'
        # while tasks:
            time.sleep(2.5)

            task = tasks.pop(0)
            status.update(f"[bold green] Working on {task}    ",spinner="dots2")
            time.sleep(2.5)

            # logger = setup_logging(log_file)
            # logger.info("Starting to predict      ")
            # logger.info("Sorry! command not yet implemented!")
            # for _ in track(range(1), description="Processing    "):
            # # Simulate some work

            #     spinner.render(12)
            #     time.sleep(10)
            console.log(f"{task} complete")
if __name__ == "main":
    dummy()

# spinner.render(time=1)

