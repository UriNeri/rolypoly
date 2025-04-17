import os

import rich_click as click
from rich.console import Console

console = Console()


@click.command()
@click.argument("update_type", type=click.Choice(["code", "data", "all"]))
@click.option(
    "-g", "--log-file", default="./update_logfile.txt", help="Path to log file"
)
def update(update_type, log_file):
    """Update RolyPoly code, data, or both components.

    This command handles updating either the RolyPoly codebase via git,
    the reference data and databases, or both components simultaneously.

    Args:
        update_type (str): Component to update. One of:
            - 'code': Update only the RolyPoly code
            - 'data': Update only the reference data and databases
            - 'all': Update both code and data
        log_file (str, optional): Path to write log messages.

    Note:
        - Code update requires git access and pip installation permissions
        - Data update requires the ROLYPOLY_DATA environment variable to be set
        - The code update will pull latest changes and reinstall the package
        - The data update will download latest reference data and databases

    Example:
             # Update both code and data
             update('all')
             # Update only reference data
             update('data', log_file='update.log')
    """
    import subprocess

    from rolypoly.utils.loggit import setup_logging

    logger = setup_logging(log_file)
    logger.info(f"Starting RolyPoly update for: {update_type}")

    # Get package location - go up 3 directories from utils/update.py to reach the root
    package_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    )
    data_dir = os.getenv("ROLYPOLY_DATA")

    if not data_dir:
        logger.error(
            "ROLYPOLY_DATA environment variable not set. Please set it before updating data."
        )
        return

    if update_type in ["code", "all"]:
        logger.info("Updating RolyPoly code    ")
        try:
            # Change to package directory
            os.chdir(package_path)

            # Pull latest code
            subprocess.run(["git", "pull"], check=True)

            # Reinstall package
            subprocess.run(["pip", "install", "-e", "."], check=True)

            logger.info("Code update completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error updating code: {str(e)}")
            return

    if update_type in ["data", "all"]:
        logger.info("Updating RolyPoly data    ")
        try:
            from rolypoly.commands.misc.prepare_external_data import (
                prepare_external_data,
            )

            prepare_external_data(
                try_hard=False, data_dir=data_dir, threads=4, log_file=log_file
            )
            logger.info("Data update completed successfully")
        except Exception as e:
            logger.error(f"Error updating data: {str(e)}")
            return

    logger.info("Update completed successfully!")


if __name__ == "__main__":
    update()
