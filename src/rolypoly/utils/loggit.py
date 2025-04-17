import logging
from pathlib import Path
from typing import Dict, Union

from rich.console import Console
from rich.logging import RichHandler


def get_version_info():
    """Get the current git version of RolyPoly. Returns short git hash or 'Unknown' if not in a git repository."""
    import os
    import subprocess
    from importlib import resources

    cwd = os.getcwd()
    try:
        os.chdir(str(resources.files("rolypoly")))
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("ascii")
            .strip()
        )
    except subprocess.CalledProcessError:
        return "Unknown"
    finally:
        os.chdir(cwd)


def setup_logging(
    log_file: Union[str, Path, logging.Logger],
    log_level: int = logging.INFO,
    logger_name: str = "RolyPoly",
) -> logging.Logger:
    """Setup logging configuration for RolyPoly with both file and console logging using rich formatting."""
    import subprocess

    # If log_file is already a logger, return it
    if isinstance(log_file, logging.Logger):
        return log_file

    # Get existing logger if it exists
    logger = logging.getLogger(logger_name)
    if logger.handlers:  # If logger already has handlers, it's already set up
        return logger

    if log_file is None:
        log_file = Path.cwd() / "rolypoly.log"

    # Convert log_file to Path if it's a string
    if isinstance(log_file, str):
        log_file = Path(log_file)

    # Create an empty log file if it doesn't exist
    if not log_file.exists():
        print(f"Creating log file: {log_file}")
        subprocess.call(f"echo ' ' > {log_file}", shell=True)

    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = (
        False  # Prevent log messages from being passed to the root logger
    )

    # Create console handler with rich formatting
    console = Console(width=150)
    console_handler = RichHandler(
        rich_tracebacks=True, console=console, show_time=False
    )
    console_handler.setLevel(log_level)

    console_formatter = logging.Formatter(
        "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        "%(asctime)s --- %(levelname)s --- %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


def log_start_info(logger: logging.Logger, config_dict: Dict):
    """Log initial information about the RolyPoly run including version, command line args, and config parameters."""
    import subprocess
    from sys import argv as sys_argv

    # Log command and launch location
    launch_command = " ".join(sys_argv)
    logger.debug(f"Original command called: {launch_command}")

    logger.debug(f"RolyPoly version: {get_version_info()}")
    logger.debug(f"Launch location: {Path.cwd()}")
    logger.debug(
        f"Submitter name: {subprocess.check_output('whoami', shell=True).decode().strip()}"
    )
    logger.debug(
        f"HOSTNAME: {subprocess.check_output('hostname', shell=True).decode().strip()}"
    )
    logger.debug(f"Config parameters:")
    for key, value in config_dict.items():
        logger.debug(f"{key}: {value}")
