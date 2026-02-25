import logging
import re
import shutil
from pathlib import Path

import requests
import rich_click as click

from rolypoly.utils.logging.loggit import setup_logging
from rolypoly.utils.various import run_command_comp

logger = logging.getLogger(__name__)


def is_valid_run_id(run_id: str) -> bool:
    """Validate ENA/SRA run accession format (e.g., SRR10307479)."""
    return bool(re.fullmatch(r"(SRR|ERR|DRR)\d+", run_id.strip()))


def is_valid_accession(accession: str) -> bool:
    """Validate common ENA/SRA accession formats (run/experiment/sample/study)."""
    return bool(
        re.fullmatch(
            r"(SRR|ERR|DRR|SRX|ERX|DRX|SRS|ERS|DRS|SRP|ERP|DRP)\d+",
            accession.strip(),
        )
    )


def resolve_accession_to_run_ids(accession: str) -> list[str]:
    """Resolve a run/experiment/sample/study accession to one or more run IDs."""
    accession = accession.strip()
    if is_valid_run_id(accession):
        return [accession]

    if not is_valid_accession(accession):
        return []

    url = (
        "https://www.ebi.ac.uk/ena/portal/api/filereport"
        f"?accession={accession}&result=read_run&fields=run_accession&format=tsv"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        logger.warning(f"Failed to resolve accession {accession}: {exc}")
        return []

    lines = [line.strip() for line in response.text.splitlines() if line.strip()]
    if len(lines) < 2:
        return []

    # first line is expected to be a header like "run_accession"
    run_ids = [line for line in lines[1:] if is_valid_run_id(line)]
    return list(dict.fromkeys(run_ids))


def get_downloader():
    """Check for available download tools and return the best one."""
    if shutil.which("aria2c"):
        return "aria2c"
    elif shutil.which("wget"):
        return "wget"
    else:
        logger.error(
            "Neither aria2c nor wget found. Please install one of them."
        )
        raise SystemExit(1)


def download_fastq(run_id, output_path):
    """Download FASTQ files for a given SRA run ID from **ENA**.

    Uses the ENA API to fetch FASTQ file URLs and downloads them using aria2c or wget.
    Handles both single-end and paired-end data (multiple FASTQ files).

    Args:
        run_id (str): SRA/ENA run accession (e.g., "SRR12345678")
        output_path (Path): Directory to save the downloaded files

    Note:
        - Uses ENA's portal API to get FASTQ file locations
        - Prefers aria2c for downloads, falls back to wget
    """
    import hashlib

    url = f"https://www.ebi.ac.uk/ena/portal/api/filereport?accession={run_id}&result=read_run&fields=fastq_ftp,fastq_aspera,fastq_md5,fastq_bytes"
    try:
        response = requests.get(url)
        response.raise_for_status()
        content = response.content.decode()

        # Parse the TSV response
        lines = content.strip().split("\n")
        if len(lines) < 2:
            logger.warning(f"No data found for {run_id}")
            return

        # Get headers and data
        headers = lines[0].split("\t")
        data = lines[1].split("\t")
        file_info = dict(zip(headers, data))

        if "fastq_ftp" not in file_info or not file_info["fastq_ftp"]:
            logger.warning(f"No FASTQ files found for {run_id}")
            return

        urls = file_info["fastq_ftp"].split(";")
        md5s = file_info.get("fastq_md5", "").split(";")
        sizes = file_info.get("fastq_bytes", "").split(";")

        # Add protocol prefix to URLs
        urls = [
            f"ftp://{url}"
            if not url.startswith(("ftp://", "http://", "https://"))
            else url
            for url in urls
            if url
        ]

        if not urls:
            logger.warning(f"No valid URLs found for {run_id}")
            return

        downloader = get_downloader()
        for i, url in enumerate(urls):
            # Get filename from URL and create output path
            filename = url.split("/")[-1]
            output_file = output_path / filename

            # Show file info if available
            if i < len(sizes) and sizes[i]:
                size_mb = float(sizes[i]) / (1024 * 1024)
                logger.info(f"Downloading {filename} ({size_mb:.1f} MB)")
            else:
                logger.info(f"Downloading {filename}")

            # Download file using the appropriate tool
            if downloader == "aria2c":
                success = run_command_comp(
                    "aria2c",
                    params={
                        "dir": str(output_path),
                        "out": filename,
                        "max-connection-per-server": "10",
                        "split": "16",
                        "summary-interval": "0",  # Disable download summary
                        "console-log-level": "warn",  # Only show warnings and errors
                    },
                    positional_args=[url],
                    check_output=True,
                    prefix_style="double",
                )
            else:  # wget
                success = run_command_comp(
                    "wget",
                    params={
                        "q": True,  # quiet
                        "O": str(output_file),
                    },
                    positional_args=[url],
                    check_output=True,
                    prefix_style="single",
                )

            if success:
                # Verify MD5 if available
                if i < len(md5s) and md5s[i]:
                    expected_md5 = md5s[i]

                    with open(output_file, "rb") as f:
                        actual_md5 = hashlib.md5(f.read()).hexdigest()
                    if actual_md5 == expected_md5:
                        logger.info(f"Downloaded and verified: {filename}")
                    else:
                        logger.error(f"Warning: MD5 mismatch for {filename}")
                        logger.error(f"Expected: {expected_md5}")
                        logger.error(f"Got: {actual_md5}")
                else:
                    logger.info(f"Downloaded: {filename}")
            else:
                logger.error(f"Failed to download {filename} for {run_id}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching information for {run_id}:")
        logger.error(str(e))
    except Exception as e:
        logger.error(f"Unexpected error processing {run_id}:")
        logger.error(str(e))


def download_xml(run_id, output_path):
    """Download XML metadata report for a given SRA run ID.

    Retrieves the detailed XML metadata report from ENA's browser API
    for the specified run accession.

    Args:
        run_id (str): SRA/ENA run accession (e.g., "SRR12345678")
        output_path (Path): Directory to save the XML file

    Note:
        - The XML file is saved as {run_id}.xml in the output directory
        - Contains detailed metadata about the run
    """
    url = f"https://www.ebi.ac.uk/ena/browser/api/xml/{run_id}"
    output_file = output_path / f"{run_id}.xml"
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(output_file, "wb") as f:
            f.write(response.content)
        logger.info(f"Downloaded XML: {output_file}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download XML for {run_id}:")
        logger.error(str(e))
    except Exception as e:
        logger.error(f"Error saving XML for {run_id}:")
        logger.error(str(e))


@click.command()
@click.option(
    "-i",
    "--input",
    required=True,
    type=str,
    help="SRA run ID or file containing run IDs (one per line)",
)
@click.option(
    "-o",
    "--output-dir",
    default="./",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory to save downloaded files",
)
@click.option(
    "-ll", "--log-level", hidden=True, default="INFO", help="Log level"
)
@click.option("--report", is_flag=True, help="Download XML report for each run")
def fetch_sra(input, output_dir, log_level, report):
    """Download SRA run FASTQ files and optional XML metadata from *ENA*

    Takes either a single SRA run ID (e.g., SRR12345678) or a file containing multiple run IDs (one per line).
    Downloads FASTQ files and optionally XML metadata reports to the specified output directory.

    Example usage:
    \n
    # Download single run:
    rolypoly fetch-sra -i SRR12345678 -o output_dir

    # Download multiple runs with metadata:
    rolypoly fetch-sra -i run_ids.txt -o output_dir --report

    * Note: The fastq headers may vary for the same SRA run/expriemtn based on the source and fetching method (s3, ftp, ena...)
    """

    global logger
    logger = setup_logging(None, log_level)

    # Validate input
    run_ids = []
    input_path = Path(input)
    if input_path.exists() and not input_path.is_dir():
        with open(input_path, "r") as f:
            accessions = [line.strip() for line in f if line.strip()]

        invalid_ids = [acc for acc in accessions if not is_valid_accession(acc)]
        if invalid_ids:
            logger.warning(
                f"Ignoring invalid accession(s): {', '.join(invalid_ids)}"
            )

        unresolved_ids = []
        for accession in accessions:
            if not is_valid_accession(accession):
                continue
            resolved = resolve_accession_to_run_ids(accession)
            if not resolved:
                unresolved_ids.append(accession)
                continue
            run_ids.extend(resolved)

        if unresolved_ids:
            logger.warning(
                "Could not resolve accession(s) to run IDs: "
                + ", ".join(unresolved_ids)
            )

        run_ids = list(dict.fromkeys(run_ids))
        if not run_ids:
            logger.error("Error: Input file is empty or has no resolvable accessions")
            return
    else:
        candidate = input.strip()
        if not is_valid_accession(candidate):
            logger.error(
                "Error: Input does not look like a supported ENA/SRA accession. "
                "Expected run/experiment/sample/study formats like "
                "SRR10307479, SRX7018852, SRS123456, or SRP123456."
            )
            return
        run_ids = resolve_accession_to_run_ids(candidate)
        if not run_ids:
            logger.error(
                f"Error: Could not resolve accession '{candidate}' to run IDs"
            )
            return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download files
    with click.progressbar(run_ids, label="Downloading SRA runs") as runs:
        for run_id in runs:
            if report:
                download_xml(run_id, output_dir)
            download_fastq(run_id, output_dir)


if __name__ == "__main__":
    fetch_sra()
