import argparse
import os
import subprocess
import requests
from pathlib import Path

def download_fastq(run_id):
    """Download FASTQ files for a given SRA run ID from ENA.

    Uses the ENA API to fetch FASTQ file URLs and downloads them using wget.
    Handles both single-end and paired-end data (multiple FASTQ files).

    Args:
        run_id (str): SRA/ENA run accession (e.g., "SRR12345678")

    Note:
        - Downloads files to the current working directory
        - Uses ENA's portal API to get FASTQ file locations
        - Wget output is redirected to /dev/null for cleaner output

    Example:
             download_fastq("SRR12345678")
        # Downloads: SRR12345678_1.fastq.gz, SRR12345678_2.fastq.gz
    """
    url1 = f"https://www.ebi.ac.uk/ena/portal/api/filereport?accession={run_id}&result=read_run&fields=fastq_ftp"
    response = requests.get(url1)
    if response.status_code == 200:
        url2=(response.content).decode().split(sep="\n")[1].split(sep="\t")[0]
        for link in url2.split(sep=";"):
                subprocess.run(f"wget -o /dev/null  { link}", check=True,shell=True)
    else:
        print(f"Failed to download {run_id}")

def download_xml(run_id, output_path):
    """Download XML metadata report for a given SRA run ID.

    Retrieves the detailed XML metadata report from ENA's browser API
    for the specified run accession.

    Args:
        run_id (str): SRA/ENA run accession (e.g., "SRR12345678")
        output_path (str): Directory to save the XML file

    Note:
        - The XML file is saved as {run_id}.xml in the output directory
        - Contains detailed metadata about the run, including:
            * Experiment details
            * Sample information
            * Run statistics
            * File locations

    Example:
             download_xml("SRR12345678", "metadata")
        Downloaded XML: metadata/SRR12345678.xml
    """
    url = f"https://www.ebi.ac.uk/ena/browser/api/xml/{run_id}"
    output_file = Path(output_path) / f"{run_id}.xml"
    # print(url)
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_file, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded XML: {output_file}")
    else:
        print(f"Failed to download XML for {run_id}")
    # metadata = xmltodict.parse(response.content.decode())["RUN_SET"]["RUN"]["RUN_LINKS"]["RUN_LINK"]
    # for i, item in enumerate(metadata):
    #     if item['XREF_LINK']['DB'].find("ENA-FASTQ-FILES") != -1:
    #         index = i
    #         break


    # metadata.str.find("ENA-FASTQ-FILES")
    return 

def main():
    """Command-line interface for downloading SRA data.

    Provides a command-line tool to download FASTQ files and optional
    XML metadata reports for one or more SRA run accessions.

    Arguments:
        input: SRA run ID or file containing run IDs (one per line)
        output_path: Directory to save downloaded files
        --report: Flag to download XML metadata reports

    Example usage:
        # Download single run
        python fetch_sra_fastq.py SRR12345678 output_dir

        # Download multiple runs with metadata
        python fetch_sra_fastq.py run_ids.txt output_dir --report
    """
    parser = argparse.ArgumentParser(description="Download SRA run fastq.gz files and optionally XML reports.")
    parser.add_argument("input", help="SRA run ID or file containing run IDs")
    parser.add_argument("output_path", help="Path to save downloaded files")
    parser.add_argument("--report", action="store_true", help="Download XML report for each run")
    
    args = parser.parse_args()
    
    run_ids = []
    if os.path.isfile(args.input):
        with open(args.input, 'r') as f:
            run_ids = [line.strip() for line in f if line.strip()]
    else:
        run_ids = [args.input]
    
    for run_id in run_ids:
        download_xml(run_id, args.output_path)
        download_fastq(run_id)

if __name__ == "__main__":
    main()

