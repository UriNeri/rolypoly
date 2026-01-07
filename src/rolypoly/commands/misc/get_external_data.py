import os
from pathlib import Path as pt

from rich.console import Console
from rich_click import command, option

from rolypoly.utils.various import (
    extract,
    find_most_recent_folder,
    move_contents_to_parent,
)

console = Console()
global tools
tools = []


@command(name="get-data")
@option(
    "--info",
    is_flag=True,
    default=False,
    help="Display current RolyPoly version, installation type, and configuration paths",
)
@option(
    "--ROLYPOLY_DATA",
    required=False,
    help="If you do not want to download the the data to same location as the rolypoly code, specify an alternative path. TODO: remind user to provide such alt path in other scripts? envirometnal variable maybe",
)
@option(
    "--log-file",
    default="./get_external_data_logfile.txt",
    help="Path to the log file",
)
def get_data(info, rolypoly_data, log_file):
    """Download or build external data required for RolyPoly.

    This command either downloads pre-built databases and reference data from
    a public repository, or builds them from scratch using the latest source data.

    Args:
        info (bool): If True, display version and configuration information and exit.
        rolypoly_data (str, optional): Alternative directory to store data. If None,
            uses the default RolyPoly data directory.
        log_file (str, optional): Path to write log messages. Defaults to
            "./get_external_data_logfile.txt".

    """
    import json
    from importlib import resources

    import requests

    from rolypoly.utils.logging.loggit import get_version_info, setup_logging

    logger = setup_logging(log_file)

    # probably a good time to verify java is present
    from bbmapy.update import ensure_java_availability

    ensure_java_availability()

    # Load configuration first
    config_path = str(resources.files("rolypoly") / "rpconfig.json")

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {"ROLYPOLY_DATA": ""}

    # Handle --info flag
    if info:
        version_info = get_version_info()
        for key, value in version_info.items():
            logger.info(f"{key}: {value}")
        return 0

    if rolypoly_data is None:
        ROLYPOLY_DATA = pt(str(resources.files("rolypoly"))) / "data"
    else:
        ROLYPOLY_DATA = pt(os.path.abspath(rolypoly_data))

    config["ROLYPOLY_DATA"] = str(ROLYPOLY_DATA)
    os.environ["ROLYPOLY_DATA"] = str(ROLYPOLY_DATA)
    print(os.environ["ROLYPOLY_DATA"])
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    logger.info(f"Starting data preparation to : {ROLYPOLY_DATA}")

    ROLYPOLY_DATA.mkdir(exist_ok=True)

    response = requests.get(
        "https://portal.nersc.gov/dna/microbial/prokpubs/rolypoly/data/data.tar.gz",
        stream=True,
    )
    with open(f"{ROLYPOLY_DATA}/data.tar.gz", "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    extract(
        archive_path=f"{ROLYPOLY_DATA}/data.tar.gz",
        extract_to=f"{ROLYPOLY_DATA}",
    )
    most_recent_folder = find_most_recent_folder(f"{ROLYPOLY_DATA}")
    move_contents_to_parent(most_recent_folder)
    os.remove(f"{ROLYPOLY_DATA}/data.tar.gz")
    logger.info(f"Finished fetching and extracting data to : {ROLYPOLY_DATA}")
    return 0
