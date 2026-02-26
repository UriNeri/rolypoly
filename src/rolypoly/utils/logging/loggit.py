import inspect
import logging
from pathlib import Path
from typing import Dict, Optional, Union

from rich.console import Console
from rich.logging import RichHandler


ROLYPOLY_HANDLER_ATTR = "rolypoly_handler_type"


def get_version_info() -> dict[str, str]:
    """Get the current version of RolyPoly (code and data).
    Returns a dictionary with the following keys:
    - "code": git hash (if available) or semver if installed via pip/uv.
    - "data": version of the data from the config file.
    """
    import json
    import os
    import subprocess
    from importlib import resources
    from importlib.metadata import version

    cwd = os.getcwd()
    version_info = {}
    # get code version
    try:
        os.chdir(str(resources.files("rolypoly")))
        # Try to get git hash first
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
        version_info["code"] = git_hash
    except subprocess.CalledProcessError:
        # If git fails, try to get installed package version assuming it was installed via pip
        try:
            version_info["code"] = version("rolypoly")
        except:
            version_info["code"] = "Unknown"
    finally:
        os.chdir(cwd)

    # get data version
    version_info["data"] = "Unknown"
    config_path = str(resources.files("rolypoly") / "rpconfig.json")
    config_path = Path(config_path)
    if config_path.exists():
        with config_path.open("r") as f:
            config = json.load(f)
        data_dir_path = Path(config["ROLYPOLY_DATA"])
        if data_dir_path.exists():
            with open(data_dir_path / "README.md", "r") as f:
                for line in f:
                    if "Date:" in line:
                        version_info["data"] = line.split("Date:")[1].strip()
                        break

    return version_info


def setup_logging(
    log_file: Union[str, Path, logging.Logger, None],
    log_level: Union[int, str] = logging.INFO,
    logger_name: str = "RolyPoly",  # Kept for compatibility, but now ignored (root logger is used)
) -> logging.Logger:
    """Setup logging configuration for RolyPoly with both file and console logging using rich formatting."""

    # If log_file is already a logger, return it
    if isinstance(log_file, logging.Logger):
        return log_file

    if log_file is None:
        log_file = Path.cwd() / "rolypoly.log"

    # Convert log_file to Path if it's a string
    if isinstance(log_file, str):
        log_file = Path(log_file)

    if isinstance(log_level, str):
        log_level = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }.get(log_level.lower(), logging.INFO)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    if not log_file.exists():
        Path.touch(log_file)

    logger = logging.getLogger()

    logger.setLevel(log_level)
    # No need to set propagate=False on root (it doesn't propagate anyway)

    rolypoly_console_handler = None
    rolypoly_file_handlers = []
    for handler in logger.handlers:
        handler_type = getattr(handler, ROLYPOLY_HANDLER_ATTR, None)
        if handler_type == "console":
            rolypoly_console_handler = handler
        elif handler_type == "file":
            rolypoly_file_handlers.append(handler)

    if rolypoly_console_handler is None:
        console = Console(width=150)
        try:
            rolypoly_console_handler = RichHandler(
                rich_tracebacks=True,
                console=console,
                show_time=False,
                show_path=True,
                enable_link_path=True,
            )
        except TypeError:
            rolypoly_console_handler = RichHandler(
                rich_tracebacks=True,
                console=console,
                show_time=False,
                show_path=True,
            )
        setattr(rolypoly_console_handler, ROLYPOLY_HANDLER_ATTR, "console")
        logger.addHandler(rolypoly_console_handler)

    rolypoly_console_handler.setLevel(log_level)
    rolypoly_console_handler.setFormatter(logging.Formatter("%(message)s"))

    existing_file_handler = None
    for file_handler in rolypoly_file_handlers:
        if Path(file_handler.baseFilename).resolve() == log_file.resolve():
            existing_file_handler = file_handler
        else:
            logger.removeHandler(file_handler)
            file_handler.close()

    if existing_file_handler is None:
        existing_file_handler = logging.FileHandler(log_file)
        setattr(existing_file_handler, ROLYPOLY_HANDLER_ATTR, "file")
        logger.addHandler(existing_file_handler)

    existing_file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        "%(asctime)s --- %(levelname)s --- %(message)s --- %(pathname)s:%(lineno)d",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    existing_file_handler.setFormatter(file_formatter)

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
    logger.debug("Config parameters:")
    for key, value in config_dict.items():
        logger.debug(f"{key}: {value}")


def get_logger(logger: Optional[logging.Logger] = None) -> logging.Logger:
    """Get a logger instance, creating a default one if none provided."""
    # If a logger object was provided, return it unchanged
    if isinstance(logger, logging.Logger):
        return logger

    # Determine the calling module so the logger reflects the caller
    caller_name = None
    for frame_info in inspect.stack()[1:]:
        module = inspect.getmodule(frame_info.frame)
        if module and module.__name__ != __name__:
            caller_name = module.__name__
            break

    if not caller_name:
        caller_name = __name__

    # Return the module-specific logger without adding handlers here.
    # This allows a single central configuration (via `setup_logging`) to
    # control formatting and handlers so that filename/line information
    # is consistently displayed for all modules.
    return logging.getLogger(caller_name)


def resolve_datadir() -> Path:
    """Resolve ROLYPOLY data directory at runtime.

    Order of resolution:
    1. `ROLYPOLY_DATA_DIR` environment variable
    2. `rpconfig.json` package resource key `ROLYPOLY_DATA`
    3. Fallback to a `data` directory at repo/package root or current working dir
    """
    import os

    # 1. environment
    env = os.environ.get("ROLYPOLY_DATA_DIR")
    if env:
        p = Path(env)
        if p.exists():
            return p

    # 2. package config
    try:
        import json
        from importlib import resources

        cfg_path = Path(str(resources.files("rolypoly") / "rpconfig.json"))
        if cfg_path.exists():
            with cfg_path.open() as fh:
                cfg = json.load(fh)
            data_dir = Path(cfg.get("ROLYPOLY_DATA", ""))
            if data_dir and data_dir.exists():
                return data_dir
    except Exception:
        pass

    # 3. fallback locations
    # try relative to package root
    pkg_root = Path(__file__).resolve().parents[3]
    candidate = pkg_root / "data"
    if candidate.exists():
        return candidate

    # final fallback to cwd --- Probably in ipython / creating a data release?
    return Path.cwd()
