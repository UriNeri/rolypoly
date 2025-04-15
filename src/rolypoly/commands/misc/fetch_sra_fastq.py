import shutil
from pathlib import Path

import requests
import rich_click as click

from rolypoly.utils.various import console, run_command_comp


def get_downloader():
    """Check for available download tools and return the best one."""
    if shutil.which("aria2c"):
        return "aria2c"
    elif shutil.which("wget"):
        return "wget"
    else:
        console.print(
            "[bold red]Neither aria2c nor wget found. Please install one of them.[/bold red]"
        )
        raise SystemExit(1)


def download_fastq(run_id, output_path):
    """Download FASTQ files for a given SRA run ID from ENA.

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
            console.print(f"[yellow]No data found for {run_id}[/yellow]")
            return

        # Get headers and data
        headers = lines[0].split("\t")
        data = lines[1].split("\t")
        file_info = dict(zip(headers, data))

        if "fastq_ftp" not in file_info or not file_info["fastq_ftp"]:
            console.print(f"[yellow]No FASTQ files found for {run_id}[/yellow]")
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
            console.print(f"[yellow]No valid URLs found for {run_id}[/yellow]")
            return

        downloader = get_downloader()
        for i, url in enumerate(urls):
            # Get filename from URL and create output path
            filename = url.split("/")[-1]
            output_file = output_path / filename

            # Show file info if available
            if i < len(sizes) and sizes[i]:
                size_mb = float(sizes[i]) / (1024 * 1024)
                console.print(f"[blue]Downloading {filename} ({size_mb:.1f} MB)[/blue]")
            else:
                console.print(f"[blue]Downloading {filename}[/blue]")

            # Download file using the appropriate tool
            if downloader == "aria2c":
                success = run_command_comp(
                    "aria2c",
                    params={
                        "dir": str(output_path),
                        "out": filename,
                        "max-connection-per-server": "16",
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
                        console.print(
                            f"[green]Downloaded and verified: {filename}[/green]"
                        )
                    else:
                        console.print(
                            f"[red]Warning: MD5 mismatch for {filename}[/red]"
                        )
                        console.print(f"[red]Expected: {expected_md5}[/red]")
                        console.print(f"[red]Got: {actual_md5}[/red]")
                else:
                    console.print(f"[green]Downloaded: {filename}[/green]")
            else:
                console.print(f"[red]Failed to download {filename} for {run_id}[/red]")

    except requests.exceptions.RequestException as e:
        console.print(f"[red]Error fetching information for {run_id}:[/red]")
        console.print(f"[red]{str(e)}[/red]")
    except Exception as e:
        console.print(f"[red]Unexpected error processing {run_id}:[/red]")
        console.print(f"[red]{str(e)}[/red]")


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
        console.print(f"[green]Downloaded XML: {output_file}[/green]")
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Failed to download XML for {run_id}:[/red]")
        console.print(f"[red]{str(e)}[/red]")
    except Exception as e:
        console.print(f"[red]Error saving XML for {run_id}:[/red]")
        console.print(f"[red]{str(e)}[/red]")


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
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory to save downloaded files",
)
@click.option("--report", is_flag=True, help="Download XML report for each run")
def fetch_sra(input, output_dir, report):
    """Download SRA run FASTQ files and optional XML metadata.

    Takes either a single SRA run ID (e.g., SRR12345678) or a file containing multiple run IDs (one per line).
    Downloads FASTQ files and optionally XML metadata reports to the specified output directory.

    Example usage:
    \b
    # Download single run:
    rolypoly fetch-sra -i SRR12345678 -o output_dir

    # Download multiple runs with metadata:
    rolypoly fetch-sra -i run_ids.txt -o output_dir --report
    """

    # Validate input
    run_ids = []
    if Path(input).is_file():
        with open(input, "r") as f:
            run_ids = [line.strip() for line in f if line.strip()]
        if not run_ids:
            console.print("[red]Error: Input file is empty[/red]")
            return
    else:
        run_ids = [input]

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
