import os
from pathlib import Path as pt

from rich.console import Console
from rich_click import command, option

from rolypoly.utils.various import extract

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
@option("--log-level", hidden=True, default="INFO", help="Log level")
def get_data(info, rolypoly_data, log_file, log_level):
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

    logger = setup_logging(log_file, log_level)

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

    ROLYPOLY_DATA.mkdir(parents=True, exist_ok=True)

    # Download the tarball
    logger.info("Downloading data tarball...")
    response = requests.get(
        "https://portal.nersc.gov/dna/microbial/prokpubs/rolypoly/data/data.tar.gz",
        stream=True,
    )
    tar_path = ROLYPOLY_DATA / "data.tar.gz"
    with open(str(tar_path), "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    
    logger.info("Extracting data tarball...")
    # Extract directly to ROLYPOLY_DATA
    # The tarball is created with relative paths (README.md, contam/, profiles/, etc.)
    # so it will extract directly to the target directory without wrapper folders
    extract(
        archive_path=str(tar_path),
        extract_to=str(ROLYPOLY_DATA),
    )
    
    # Clean up the tarball
    tar_path.unlink()
    
    logger.info(f"Finished fetching and extracting data to : {ROLYPOLY_DATA}")
    return 0
