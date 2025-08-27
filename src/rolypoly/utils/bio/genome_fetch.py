"""
Genome downloading and fetching functionality.
"""

import concurrent.futures
import os
import shutil
import subprocess as sp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from rolypoly.utils.various import extract_zip, run_command_comp
from rolypoly.utils.bio.sequences import filter_fasta_by_headers


global datadir
datadir = Path(os.environ.get("ROLYPOLY_DATA", ""))

def download_genome(taxid: str) -> None:
    """Download genome data from NCBI for a given taxon ID.

    Args:
        taxid (str): NCBI taxonomy ID for the organism

    Note:
        Uses the NCBI datasets command‑line tool to download genome data.
        Downloads RNA and genome data, excluding atypical sequences.
    """
    # Positional arguments required by `datasets` (sub‑commands + taxid)
    positional_args = ["download", "genome", "taxon", taxid]

    # Non‑positional (flag) parameters
    params = {
        "include": "rna,genome",
        "filename": f"{taxid}_fetched_genomes.zip",
        "assembly-version": "latest",
        "exclude-atypical": True,      # flag without a value
        "assembly-source": "RefSeq",
        "no-progressbar": True,       # flag without a value
    }

    # Run the command using the common wrapper from `various.py`
    run_command_comp(
        base_cmd="datasets",
        positional_args=positional_args,
        positional_args_location="start",
        params=params,
        logger=None,            # falls back to an internal logger
        output_file="",         # no output file to verify here
        skip_existing=False,
        check_status=True,      # raise if the command fails
        check_output=False,
        prefix_style="double",    # let the helper decide "-" vs "--"
        param_sep=" ",
        assign_operator=" ",
        resource_monitoring=False,
        return_final_cmd=True,
    )


def process_with_timeout(func: callable, arg: any, timeout: int) -> any: #type: ignore
    """Execute a function with a timeout.

    Args:
        func (callable): Function to execute
        arg (any): Argument to pass to the function
        timeout (int): Timeout in seconds

    Returns:
        any: Result of the function call, or None if timeout occurred

    Note:
        Uses ProcessPoolExecutor to run the function in a separate process.
    """
    from concurrent.futures import ProcessPoolExecutor, TimeoutError

    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, arg)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            print(f"Task for {arg} timed out after {timeout} seconds")
            return None


def fetch_genomes(
    input_file: str,
    output_file: str,
    threads: int = 1,
    max2take: int = 5,
    timeout: int = 600,
) -> None:
    """Fetch genomes from NCBI for masking purposes.

    Downloads and processes genomes from NCBI (via datasets) based on a BBMap stats file.
    Filters out metagenomic, uncultured, and synthetic sequences.

    Args:
        input_file (str): Path to BBMap stats file
        output_file (str): Path to save the processed sequences
        threads (int, optional): Number of threads to use.
        max2take (int, optional): Maximum number of genomes to process.
        timeout (int, optional): Timeout in seconds for each download.

    Note:
        - Excludes sequences with keywords indicating non-natural sources
        - Concatenates and deduplicates sequences
        - Removes viral sequences from the final output
    TODO:
        - replace taxonkit with taxopy
        - replace datasets.
    """

    with open(input_file, "r") as f:
        lines = f.readlines()[4:]

    taxons = set()
    for line in lines:
        taxon = line.split(sep=";")[-1]
        taxon = taxon.split(sep="\t")[0]
        if any(
            word in taxon.lower()
            for word in [
                "meta",
                "uncultured",
                "unidentified",
                "synthetic",
                "construct",
                "coli",
            ]
        ):
            print(
                f"{line} contains a keyword that indicates it is not an actual organism, skipping"
            )
        else:
            if "PREDICTED: " in taxon:
                taxon = taxon.split(sep=": ")[1]
                taxon = taxon.split(sep=" 28S")[0]
                taxon = taxon.split(sep=" 18S")[0]
                taxon = taxon.split(sep=" small subunit ")[0]
                taxon = taxon.split(sep=" large subunit ")[0]
            taxons.update([taxon])
            if len(taxons) > max2take:
                break

    if not taxons:
        print("No valid taxons found in the input file. Skipping genome fetching.")
        return

    # Use taxonkit to get taxids
    with open("tmp_gbs_50m_taxids.lst", "w") as f:
        sp.run(["taxonkit", "name2taxid", f"--data-dir {datadir}/taxdump"], input="\n".join(taxons).encode(), stdout=f)

    # Use datasets to download genomes
    with open("tmp_gbs_50m_taxids.lst", "r") as f:
        taxids = [
            line.split(sep="\t")[1].replace(" ", "_").strip()
            for line in f
            if line != ""
        ]
    taxids = list(set(taxids).difference(["", "562"]))  # Remove empty and E. coli

    if not taxids:
        print("No valid taxids found. Skipping genome fetching.")
        return

    with ProcessPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(
                process_with_timeout, download_genome, taxid, timeout
            )
            for taxid in taxids
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

    zip_files = list(Path(".").glob("*.zip"))
    if not zip_files:
        print("No genome zip files were downloaded. Skipping extraction.")
        return

    with ProcessPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(process_with_timeout, extract_zip, zip_file, timeout)
            for zip_file in zip_files
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred during extraction: {e}")

    # Concatenate and deduplicate sequences
    ref_seqs = set()
    ncbi_dataset_dir = Path("ncbi_dataset")
    if not ncbi_dataset_dir.exists():
        print("NCBI dataset directory not found. Skipping sequence processing.")
        return

    for folder in ncbi_dataset_dir.rglob("*/"):
        fna_files = list(folder.rglob("*.fna"))
        if not fna_files:
            continue
        rna_file = next((f for f in fna_files if "rna" in f.name.lower()), None)
        if rna_file:
            chosen_file = rna_file
        else:
            chosen_file = fna_files[0]  # choose the first file if no RNA file found
        ref_seqs.add(str(chosen_file))

    if not ref_seqs:
        print(
            "No FNA files found in the downloaded genomes. Skipping sequence processing."
        )
        return

    with open("tmp.lst", "w") as outf:
        for f in ref_seqs:
            outf.write(f + "\n")
    from shutil import which

    print(which("seqkit"))
    sp.call(
        ["seqkit", "rmdup", "-s", "--infile-list", "tmp.lst"],
        stdout=open("tmp.fasta", "w"),
    )

    # Remove any fasta entries that have "virus", "viral", "phage" in the header
    filter_fasta_by_headers(
        "tmp.fasta", ["virus", "viral", "phage"], output_file, invert=True
    )

    # Clean up
    for item in Path(".").glob("ncbi_dataset*"):
        if item.is_dir():
            shutil.rmtree(item)
        else:
            os.remove(item)
    remove_files = ["tmp.fasta", "tmp.lst"]
    for file in remove_files:
        if os.path.exists(file):
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error removing {file}: {e}") 