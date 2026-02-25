import datetime
import json
import logging
import os
import shutil
import subprocess
import tarfile
from pathlib import Path as pt

import polars as pl
import requests
from bbmapy import bbduk, bbmask, kcompress
from rich.console import Console
from rich_click import command, option

from rolypoly.utils.bio.alignments import (
    hmmdb_from_directory,
    mmseqs_profile_db_from_directory,
)
from rolypoly.utils.bio.sequences import (
    filter_fasta_by_headers,
    remove_duplicates,
    write_fasta_file,
)

from rolypoly.utils.bio.polars_fastx import from_fastx_eager
# import tqdm 
from rolypoly.utils.logging.citation_reminder import remind_citations
from rolypoly.utils.logging.loggit import get_version_info, setup_logging
from rolypoly.utils.various import fetch_and_extract, run_command_comp

console = Console()
global tools
tools = []

### DEBUG ARGS (for manually building, not entering via CLI):
threads = 6
log_file = "rolypoly_build_data.log"
data_dir = "/clusterfs/jgi/scratch/science/metagen/neri/code/rolypoly/data"


@command()
@option("--data-dir", required=True, help="Path to the data directory")
@option("--threads", default=4, help="Number of threads to use")
@option(
    "--log-file",
    default="./prepare_external_data_logfile.txt",
    help="Path to the log file",
)
@option("--log-level", hidden=True, default="INFO", help="Log level")
def build_data(data_dir, threads, log_file, log_level):
    """Build external data required for RolyPoly. This is an internal scratch script, wrapped in a click command for convenience (and logging), but exposed to users.
    1. Build geNomad RNA viral HMMs
    2. Build protein HMMs RdRp-scan, RVMT, Neordrp_v2.1, tsa_2018 and pfam_A_38
    3. Download and prepare rRNA databases SILVA_138.1_SSURef_NR99_tax_silva.fasta and SILVA_138.1_LSURef_NR99_tax_silva.fasta
    4. Download Rfam data.
    """

    global profile_dir  #
    global rrna_dir
    global hmmdb_dir
    global mmseqs_dbs
    global contam_dir
    global trna_dir

    logger = setup_logging(log_file, log_level)
    logger.info(f"Starting data preparation to : {data_dir}")

    contam_dir = os.path.join(data_dir, "contam")
    os.makedirs(contam_dir, exist_ok=True)

    rrna_dir = os.path.join(contam_dir, "rrna")
    os.makedirs(rrna_dir, exist_ok=True)

    trna_dir = os.path.join(contam_dir, "trna")
    os.makedirs(trna_dir, exist_ok=True)

    adapter_dir = os.path.join(contam_dir, "adapters")
    os.makedirs(adapter_dir, exist_ok=True)

    masking_dir = os.path.join(contam_dir, "masking")
    os.makedirs(masking_dir, exist_ok=True)

    # taxonomy_dir = os.path.join(data_dir, "taxdump")
    # os.makedirs(taxonomy_dir, exist_ok=True)

    reference_seqs = os.path.join(data_dir, "reference_seqs")
    os.makedirs(reference_seqs, exist_ok=True)

    mmseqs_ref_dir = os.path.join(reference_seqs, "mmseqs")
    os.makedirs(mmseqs_ref_dir, exist_ok=True)

    rvmt_dir = os.path.join(reference_seqs, "RVMT")
    os.makedirs(rvmt_dir, exist_ok=True)

    ncbi_ribovirus_dir = os.path.join(reference_seqs, "ncbi_ribovirus")
    os.makedirs(ncbi_ribovirus_dir, exist_ok=True)

    profile_dir = os.path.join(data_dir, "profiles")
    hmmdb_dir = os.path.join(profile_dir, "hmmdbs")
    mmseqs_dbs = os.path.join(profile_dir, "mmseqs_dbs")

    os.makedirs(hmmdb_dir, exist_ok=True)
    os.makedirs(mmseqs_dbs, exist_ok=True)

    genomad_dir = os.path.join(profile_dir, "genomad")
    os.makedirs(genomad_dir, exist_ok=True)

    # Add geNomad RNA viral markers
    prepare_genomad_rna_viral_markers(data_dir, threads, logger)
    shutil.rmtree(
        genomad_dir
    )  # the hmms and mmseqsdb would be moved into their dirs in data/profiles/..

    # RdRp-scan
    prepare_rdrp_scan(data_dir, threads, logger)

    # RVMT profiles
    prepare_RVMT_profiles(data_dir, threads, logger)

    # RVMT MMseqs database
    prepare_rvmt_mmseqs(data_dir, threads, logger)

    # neordrp2.1 profiles
    prepare_neordrp_profiles(data_dir, threads, logger)

    # NCBI ribovirus refseq
    prepare_ncbi_ribovirus(data_dir, threads, logger)

    # pfam RdRps and RTs
    prepare_pfam_rdrps_rt(data_dir, threads, logger)

    # RVMT motifs
    prepare_rvmt_motifs(data_dir, threads, logger)

    # UniRef50 viral
    prepare_uniref50_viral(data_dir, threads, logger)
    
    # contaminations
    prepare_contamination_seqs(data_dir, threads, logger)
    
    # plastid reference sequences
    prepare_plastid_data(data_dir, logger)
    # tRNA reference sequences
    prepare_trna_data(data_dir, logger)

    # Rfam
    download_and_extract_rfam(data_dir, logger)

    # subprocess.run(
    #     "cat NCBI_ribovirus/proteins/datasets_efetch_refseq_ribovirus_proteins_rmdup.faa RVMT/RVMT_allorfs_filtered_no_chimeras.faa | seqkit rmdup | seqkit seq -w0 > prots_for_masking.faa",
    #     shell=True,
    # )
    logger.info("Finished data preparation")


def prepare_rvmt_mmseqs(data_dir, threads, logger: logging.Logger):
    """Prepare RVMT database for seqs searches (mmseqs2 and diamond).

    Processes the RVMT database alignments
    and creates formatted databases for MMseqs2 searches.

    Args:
        data_dir (str): Base directory for data storage
        threads (int): Number of CPU threads to use
        logger: Logger object for recording progress and errors

    Note:
        Downloads RVMT contigs and metadata, filters out chimeric sequences,
        and creates MMseqs2 and compressed databases for fast searches.
    """

    logger.info("Preparing RVMT mmseqs database")

    # Create directories
    rvmt_dir = os.path.join(data_dir, "reference_seqs", "RVMT")
    mmdb_dir = os.path.join(rvmt_dir, "mmseqs")
    os.makedirs(rvmt_dir, exist_ok=True)
    os.makedirs(mmdb_dir, exist_ok=True)

    # Download RVMT contigs
    logger.info("Downloading RVMT contigs")
    contigs_fasta_path = fetch_and_extract(
        "https://portal.nersc.gov/dna/microbial/prokpubs/Riboviria/RiboV1.4/RiboV1.6_Contigs.fasta.gz",
        fetched_to=os.path.join(rvmt_dir, "RiboV1.6_Contigs.fasta.gz"),
        extract_to=rvmt_dir,
        expected_file="RiboV1.6_Contigs.fasta",
        logger=logger,
    )

    # Download and process RVMT info table to get chimeric sequences
    logger.info("Fetching RVMT metadata")
    chimera_ids = []

    info_df = pl.read_csv(
        "https://portal.nersc.gov/dna/microbial/prokpubs/Riboviria/RiboV1.4/RiboV1.6_Info.tsv",
        separator="\t",
        null_values=["NA", ""],
    )

    logger.info("Processing RVMT metadata to identify chimeric sequences")
    # Check for chimeric in `Note` column
    chimera_ids = (
        info_df.filter(
            pl.col("Note")
            .cast(pl.Utf8)
            .str.contains_any(
                ["chim", "rRNA", "cell"], ascii_case_insensitive=True
            )
        )
        .select(pl.col("ND"))
        .to_series()
        .to_list()
    )
    # Filter for chimeric sequences

    logger.info(f"Found {len(chimera_ids)} chimeric sequences to exclude")

    # Filter out chimeric sequences using rolypoly's filter function
    cleaned_path = os.path.join(rvmt_dir, "RVMT_cleaned_contigs.fasta")
    logger.info("Filtering out chimeric sequences")
    filter_fasta_by_headers(
        fasta_file=contigs_fasta_path,
        headers=chimera_ids,
        output_file=cleaned_path,
        invert=True,  # Keep sequences NOT in the chimera list
    )

    # Create MMseqs2 database
    logger.info("Creating MMseqs2 database")
    run_command_comp(
        base_cmd="mmseqs createdb",
        positional_args_location="start",
        positional_args=[cleaned_path, os.path.join(mmdb_dir, "RVMT_cleaned")],
        params={"dbtype": "2"},
        logger=logger,
    )

    # # Create entropy-masked temporary file before compression
    # logger.info("Creating entropy-masked sequences")
    # entropy_masked_path = os.path.join(rvmt_dir, "RVMT_entropy_masked.fasta")
    # from bbmapy import bbmask

    # bbmask(
    #     in1=cleaned_path,
    #     out=entropy_masked_path,
    #     entropy=0.05,
    #     entropywindow=140,
    #     threads=threads,
    # )

    # now similarly, but getting the ORFs
    all_orf_info = pl.read_csv(
        "https://portal.nersc.gov/dna/microbial/prokpubs/Riboviria/RiboV1.4/Simplified_AllORFsInfo.tsv",
        separator="\t",
        null_values=["NA", ""],
    )
    not_chimeric_orfs = (
        all_orf_info.filter(~all_orf_info["seqid"].is_in(chimera_ids))
        .select(pl.col("ORFID"))
        .to_series()
        .to_list()
    )

    rvmt_orfs = fetch_and_extract(
        "https://portal.nersc.gov/dna/microbial/prokpubs/Riboviria/RiboV1.4/RiboV1.5_AllORFs.faa",
        fetched_to=os.path.join(rvmt_dir, "rvmt_orfs.faa"),
        extract_to=rvmt_dir,
        expected_file="rvmt_orfs",
        logger=logger,
    )

    cleaned_orfs_path = os.path.join(rvmt_dir, "RVMT_cleaned_orfs.faa")
    logger.info("Filtering out ORFs from chimeric sequences")
    filter_fasta_by_headers(
        fasta_file=rvmt_orfs,
        headers=not_chimeric_orfs,
        output_file=cleaned_orfs_path,
        invert=False,  # Keep sequences in the non-chimeric ORF list
    )

    # Clean up temporary files, only keep the compressed
    try:
        os.remove(rvmt_dir + "/RiboV1.6_Contigs.fasta")
        os.remove(rvmt_dir + "/RiboV1.6_Contigs.fasta.gz")
        os.remove(rvmt_dir + "/rvmt_orfs.faa")
    except FileNotFoundError:
        logger.warning(
            "some temporary files for RVMT mmseqs preparation might not have been cleaned."
        )

    logger.info(
        f"RVMT databases created successfully in {rvmt_dir} and {mmdb_dir}"
    )


def download_and_extract_rfam(data_dir, logger):
    """Download and process Rfam database files.

    Retrieves Rfam database files and processes them for use in RNA
    family identification and annotation.

    Args:
        data_dir (str): Base directory for data storage
        logger: Logger object for recording progress and errors

    Note:
        Downloads both the sequence database and covariance models,
        and processes them for use with Infernal.
    """

    rfam_url = "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.cm.gz"
    rfam_cm_path = data_dir + "/Rfam.cm.gz"
    rfam_extract_path = data_dir + "/profiles/cm/"
    # subprocess.run("cmpress Rfam.cm", shell=True)

    logger.info("Downloading Rfam database")
    try:
        fetch_and_extract(
            rfam_url,
            extract_to=rfam_extract_path,

        )
        logger.info("Rfam database downloaded and extracted successfully.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading Rfam database: {e}")


def tar_everything_and_upload_to_NERSC(data_dir, version=""):
    """Package and upload prepared data to NERSC.

    Creates a tarball of all prepared databases and reference data,
    then uploads it to NERSC for distribution.

    Args:
        data_dir (str): Directory containing data to package
        version (str, optional): Version string to append to archive name.

    Note:
        Requires appropriate NERSC credentials and permissions to upload.
    """

    if version == "":
        version = get_version_info()
    with open(pt(data_dir) / "README.md", "w") as f_out:
        f_out.write(f"RolyPoly version: {version['code']}\n")
        f_out.write(f"Date: {datetime.datetime.now()}\n")
        f_out.write(f"Data dir: {data_dir}\n")
        f_out.write(
            "for more details see: https://pages.jgi.doe.gov/rolypoly/docs/\n"
        )
        f_out.write("Changes in this version: \n")
        f_out.write(
            " - Removed eukaryotic RdRp Pfam (see d1a0f1b3e2452253a4d47e20b81ac71652ccb944) \n"
        )
        f_out.write("Software / DBs used in the creation of this data: \n")
        tools.append("RolyPoly")
        tools.append("seqkit")
        tools.append("bbmap")
        tools.append("mmseqs2")
        tools.append("mmseqs")
        tools.append("hmmer")
        tools.append("pyhmmer")
        tools.append("eutils")
        tools.append("silva")
        tools.append("Rfam")
        tools.append("rvmt")
        tools.append("rdrp-scan")
        tools.append("neordrp_v2.1")
        tools.append("tsa_2018")
        tools.append("pfam_A_38")
        tools.append("refseq")
        f_out.write(remind_citations(tools, return_as_text=True) or "")

    
    tar_command = f"tar --use-compress-program='pigz -p 8 --best' -cf rpdb.tar.gz {data_dir}"  # threads

    subprocess.run(tar_command, shell=True)
    
    #check which .py files use files/stuff from datadir/data_dir
    # sp.run(f"find src/rolypoly -name '*.py' -exec grep  data_dir|datadir \;", shell=True)

    # # On NERSC
    # scp uneri@xfer.jgi.lbl.gov:/clusterfs/jgi/scratch/science/metagen/neri/code/rolypoly/data/rolypoly_data_slim_20251230.tar.gz /global/dna/projectdirs/microbial/prokpubs/www/rolypoly/data/data.tar.gz
    # chmod +777 -R /global/dna/projectdirs/microbial/prokpubs/www/rolypoly/data/
    # upload_command = f"gsutil cp {data_dir}.tar.gz gs://rolypoly-data/"
    # subprocess.run(upload_command, shell=True)

def analyze_data_dependencies(src_dir="src/rolypoly", data_dir=None):
    """Analyze Python files to determine required data paths for a slim tar.gz.

    Scans all .py files in the source directory for references to data_dir, datadir, or ROLYPOLY_DATA,
    extracts the data paths they use, and returns a set of required data files/directories.

    Also scans for exact filenames from the data directory that appear in the code.

    Args:
        src_dir (str): Source directory to scan for Python files
        data_dir (str): Data directory to scan for existing files

    Returns:
        set: Set of required data paths relative to data_dir
    """
    import re
    import os

    required_paths = set()

    # Get all files in data directory if provided
    data_files = set()
    if data_dir and os.path.exists(data_dir):
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, data_dir)
                data_files.add(rel_path)
                data_files.add(file)  # Also add just the filename for direct matches

    # Find all Python files that reference data_dir, datadir, or ROLYPOLY_DATA
    result = subprocess.run(
        f"find {src_dir} -name '*.py' -exec grep -l 'data_dir\\|datadir\\|ROLYPOLY_DATA' {{}} \\;",
        shell=True,
        capture_output=True,
        text=True
    )
    py_files = result.stdout.strip().split('\n') if result.stdout.strip() else []

    # Also find Python files that contain any of the data filenames
    if data_files:
        filename_pattern = '|'.join(re.escape(f) for f in data_files if f)
        if filename_pattern:
            result2 = subprocess.run(
                f"find {src_dir} -name '*.py' -exec grep -l '{filename_pattern}' {{}} \\;",
                shell=True,
                capture_output=True,
                text=True
            )
            py_files.extend(result2.stdout.strip().split('\n') if result2.stdout.strip() else [])

    # Remove duplicates and empty strings
    py_files = list(set(py_files))
    py_files = [f for f in py_files if f]

    for py_file in py_files:
        if not py_file or 'build_data.py' in py_file:
            continue
        with open(py_file, 'r') as f:
            content = f.read()

        # Find patterns like data_dir / "path" or Path(data_dir) / "path"
        # Also os.path.join(data_dir, "path")
        # And data_dir + "/path"
        # And similar for ROLYPOLY_DATA environment variables

        # Pattern 1: data_dir / "path" or data_dir / 'path'
        path_pattern1 = r'data_dir\s*/\s*["\']([^"\']+)["\']'
        matches1 = re.findall(path_pattern1, content)
        print(f"Pattern 1 matches in {py_file}: {matches1}")
        for match in matches1:
            required_paths.add(match)

        # Pattern 2: os.path.join(data_dir, "path", ...)
        join_pattern = r'os\.path\.join\(\s*data_dir\s*,\s*["\']([^"\']+)["\'](?:\s*,\s*["\']([^"\']+)["\'])*'
        matches2 = re.findall(join_pattern, content)
        print(f"Pattern 2 matches in {py_file}: {matches2}")
        for match in matches2:
            # match is a tuple of path components
            path = os.path.join(*[m for m in match if m])
            required_paths.add(path)

        # Pattern 3: data_dir + "/path"
        plus_pattern = r'data_dir\s*\+\s*["\']([^"\']+)["\']'
        matches3 = re.findall(plus_pattern, content)
        print(f"Pattern 3 matches in {py_file}: {matches3}")
        for match in matches3:
            # Remove leading / if present
            path = match.lstrip('/')
            required_paths.add(path)

        # Pattern 4: Path(data_dir) / "path"
        path_pattern2 = r'Path\(\s*data_dir\s*\)\s*/\s*["\']([^"\']+)["\']'
        matches4 = re.findall(path_pattern2, content)
        print(f"Pattern 4 matches in {py_file}: {matches4}")
        for match in matches4:
            required_paths.add(match)

        # Pattern 5: datadir / "path" (for cases where datadir is used)
        datadir_pattern = r'datadir\s*/\s*["\']([^"\']+)["\']'
        matches5 = re.findall(datadir_pattern, content)
        print(f"Pattern 5 matches in {py_file}: {matches5}")
        for match in matches5:
            required_paths.add(match)

        # Pattern 6: os.path.join(os.environ.get("ROLYPOLY_DATA", ""), "path", ...)
        rolypoly_join_pattern = r'os\.path\.join\(\s*os\.environ\.get\(\s*["\']ROLYPOLY_DATA["\']\s*,\s*["\'][^"\']*["\']\s*\)\s*,\s*["\']([^"\']+)["\'](?:\s*,\s*["\']([^"\']+)["\'])*'
        matches6 = re.findall(rolypoly_join_pattern, content)
        print(f"Pattern 6 matches in {py_file}: {matches6}")
        for match in matches6:
            # match is a tuple of path components
            path = os.path.join(*[m for m in match if m])
            required_paths.add(path)

        # Pattern 7: Path(os.environ.get("ROLYPOLY_DATA", "")) / "path"
        rolypoly_path_pattern = r'Path\(\s*os\.environ\.get\(\s*["\']ROLYPOLY_DATA["\']\s*,\s*["\'][^"\']*["\']\s*\)\s*\)\s*/\s*["\']([^"\']+)["\']'
        matches7 = re.findall(rolypoly_path_pattern, content)
        print(f"Pattern 7 matches in {py_file}: {matches7}")
        for match in matches7:
            required_paths.add(match)

        # Pattern 7b: Path(os.environ.get("ROLYPOLY_DATA", ""), "path") - multiple args to Path
        rolypoly_path_multi_pattern = r'Path\(os\.environ\.get\(\"ROLYPOLY_DATA\",\s*\"\"\),\s*\"([^\"]+)\"'
        matches7b = re.findall(rolypoly_path_multi_pattern, content)
        print(f"Pattern 7b matches in {py_file}: {matches7b}")
        for match in matches7b:
            required_paths.add(match)

        # Pattern 7c: os.path.join(Path(os.environ.get("ROLYPOLY_DATA", "")), "path")
        rolypoly_join_path_pattern = r'os\.path\.join\(\s*Path\(os\.environ\.get\(\"ROLYPOLY_DATA\",\s*\"\"\)\),\s*\"([^\"]+)\"'
        matches7c = re.findall(rolypoly_join_path_pattern, content)
        print(f"Pattern 7c matches in {py_file}: {matches7c}")
        for match in matches7c:
            required_paths.add(match)

        # Pattern 8: os.path.join(os.environ.get("ROLYPOLY_DATA_DIR", ""), "path", ...)
        rolypoly_dir_join_pattern = r'os\.path\.join\(\s*os\.environ\.get\(\s*["\']ROLYPOLY_DATA_DIR["\']\s*,\s*["\'][^"\']*["\']\s*\)\s*,\s*["\']([^"\']+)["\'](?:\s*,\s*["\']([^"\']+)["\'])*'
        matches8 = re.findall(rolypoly_dir_join_pattern, content)
        print(f"Pattern 8 matches in {py_file}: {matches8}")
        for match in matches8:
            # match is a tuple of path components
            path = os.path.join(*[m for m in match if m])
            required_paths.add(path)

        # Pattern 9: Path(os.environ["ROLYPOLY_DATA"]) / "path" (when using os.environ["ROLYPOLY_DATA"])
        rolypoly_env_path_pattern = r'Path\(\s*os\.environ\[\s*["\']ROLYPOLY_DATA["\']\s*\]\s*\)\s*/\s*["\']([^"\']+)["\']'
        matches9 = re.findall(rolypoly_env_path_pattern, content)
        print(f"Pattern 9 matches in {py_file}: {matches9}")
        for match in matches9:
            required_paths.add(match)

        # Pattern 10: Direct filename matches from data directory
        if data_files:
            for data_file in data_files:
                if data_file in content and '/' in data_file:  # Only add if it's a full path
                    required_paths.add(data_file)

    # Clean up paths (remove any empty or invalid)
    required_paths = {p for p in required_paths if p and not p.startswith('/')}

    # Get all existing paths in data_dir
    all_paths = set()
    for root, dirs, files in os.walk(data_dir):
        rel_root = os.path.relpath(root, data_dir)
        if rel_root != '.':
            all_paths.add(rel_root)
        for d in dirs:
            all_paths.add(os.path.relpath(os.path.join(root, d), data_dir))
        for f in files:
            all_paths.add(os.path.relpath(os.path.join(root, f), data_dir))

    # Filter required_paths to actual existing paths that contain the required string
    filtered_paths = set()
    for req in required_paths:
        matching = [p for p in all_paths if req in p]
        filtered_paths.update(matching)

    return filtered_paths


def create_slim_tarball(data_dir, required_paths, version=datetime.datetime.now().strftime("%Y%m%d")):
    """Create a slim tarball containing only the required data paths.

    Args:
        data_dir (str): Base data directory
        required_paths (set): Set of required relative paths
        version (str): Version string for the archive name
    """
    import tempfile
    import subprocess

    # Deduplicate paths: remove subpaths if parent directory is included
    def deduplicate_paths(paths):
        sorted_paths = sorted(paths)
        deduped = []
        for path in sorted_paths:
            # Check if this path is a subpath of any already added path
            is_subpath = False
            for added in deduped:
                if path.startswith(added + os.sep) or path == added:
                    is_subpath = True
                    break
            if not is_subpath:
                deduped.append(path)
        return deduped

    required_paths = [
        'README.md',
        'contam/adapters/AFire_illuminatetritis1223.fa',
        'contam/adapters/bbmap_adapters.fa',
        'contam/masking/combined_deduplicated_orfs.faa',
        'contam/masking/combined_entropy_masked.fasta',
        'contam/rrna/ncbi_rRNA_all_sequences_masked_entropy.fasta',
        'contam/rrna/rrna_to_genome_mapping.parquet',
        'contam/rrna/silva_rRNA_all_sequences_masked_entropy.fasta',
        'profiles/genomad_rna_viral_markers_with_annotation.csv',
        'profiles/motif_metadata.json',
        'profiles/NVPC_descriptions.csv',
        'profiles/vfam.annotations.tsv',
        'profiles/cm/Rfam.cm',
        'profiles/hmmdbs/genomad_rna_viral_markers.hmm',
        'profiles/hmmdbs/neordrp2.1.hmm',
        'profiles/hmmdbs/nvpc.hmm',
        'profiles/hmmdbs/pfam_rdrps_and_rts.hmm',
        'profiles/hmmdbs/Pfam-A.hmm',
        'profiles/hmmdbs/rdrp_scan.hmm',
        'profiles/hmmdbs/rvmt_motifs.hmm',
        'profiles/hmmdbs/rvmt.hmm',
        'profiles/hmmdbs/vfam.hmm',
        'profiles/mmseqs_dbs/genomad',
        'profiles/mmseqs_dbs/nvpc',
        'profiles/mmseqs_dbs/rdrp_scan',
        'profiles/mmseqs_dbs/RVMT',
        'profiles/mmseqs_dbs/rvmt_motifs',
        'profiles/mmseqs_dbs/vfam',
        # 'reference_seqs/mito_refseq/combined_mito_refseq.fasta',
        # 'reference_seqs/plastid_refseq/combined_plastid_refseq.fasta',
        'reference_seqs/ncbi_ribovirus/mmseqs',
        'reference_seqs/ncbi_ribovirus/refseq_ribovirus_genomes_orfs.faa',
        'reference_seqs/ncbi_ribovirus/refseq_ribovirus_genomes.fasta',
        'reference_seqs/RVMT/mmseqs',
        'reference_seqs/RVMT/RVMT_cleaned_contigs.fasta',
        'reference_seqs/RVMT/RVMT_cleaned_orfs.faa',
        'reference_seqs/uniref/uniref50_viral.tsv',
        'reference_seqs/uniref/uniref50_viral.fasta'
    ]
    required_paths.append('README.md')  # always include README
    required_paths = deduplicate_paths(required_paths)
    # get size info
    total_size = 0
    for rel_path in required_paths:
        full_path = os.path.join(data_dir, rel_path)
        if os.path.exists(full_path):
            if os.path.isfile(full_path):
                total_size += os.path.getsize(full_path)
            elif os.path.isdir(full_path):
                for dirpath, dirnames, filenames in os.walk(full_path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        total_size += os.path.getsize(fp)
    print(f"Total size of selected data for slim tarball: {total_size / (1024*1024):.2f} MB")

    tarball_name = f"rolypoly_data_slim_{version}.tar.gz"
    
    print(f"Creating slim tarball: {tarball_name}")
    print(f"Required paths: {sorted(required_paths)}")

    # Create a temporary file with the list of paths
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        for rel_path in required_paths:
            full_path = os.path.join(data_dir, rel_path)
            if os.path.exists(full_path):
                f.write(rel_path + '\n')
            else:
                print(f"Warning: {rel_path} does not exist, skipping")
        listfile = f.name

    try:
        # Use tar with pigz for faster compression
        tar_command = f"tar --use-compress-program='pigz -p 18 --best' -cf {tarball_name} -C {data_dir} -T {listfile}"
        subprocess.run(tar_command, shell=True, check=True)
        print(f"Slim tarball created: {tarball_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating tarball: {e}")
    finally:
        # Clean up temporary file
        os.unlink(listfile)

    return tarball_name


def prepare_uniref50_viral(data_dir, threads, logger):
    """Download and prepare UniRef50 viral protein sequences."""
    logger.info("Downloading UniRef50 viral protein sequences")
    os.makedirs(os.path.join(
        data_dir, "reference_seqs/uniref"), exist_ok=True)
    uniref50_viral_fasta = os.path.join(
        data_dir, "reference_seqs/uniref/uniref50_viral.fasta")
    fetch_and_extract(
        "https://rest.uniprot.org/uniref/stream?compressed=true&format=fasta&query=%28%28identity%3A0.5%29+AND+%28taxonomy_id%3A10239%29+AND+%28count%3A%5B1+TO+192133%5D%29%29",
        fetched_to=uniref50_viral_fasta + ".gz",
        extract_to=os.path.dirname(uniref50_viral_fasta),
        expected_file=os.path.basename(uniref50_viral_fasta),
        logger=logger,
        debug=True,
    )
    uniref50_viral_data = os.path.join(
        data_dir, "reference_seqs/uniref/uniref50_viral.tsv")
    fetch_and_extract(
        "https://rest.uniprot.org/uniref/stream?compressed=true&fields=id%2Cname%2Ctypes%2Ccount%2Corganism%2Clength%2Cidentity%2Cmembers&format=tsv&query=%28%28identity%3A0.5%29+AND+%28taxonomy_id%3A10239%29+AND+%28count%3A%5B1+TO+192133%5D%29%29",
        fetched_to=uniref50_viral_data+ ".gz",
        extract_to=os.path.dirname(uniref50_viral_fasta),
        extract=True,
        logger=logger,
        debug=True,
    )
    # why the hell is it still compressed??? 2 times gzipped???
    from rolypoly.utils.various import extract
    extract(uniref50_viral_data+".gz", uniref50_viral_data)

    # remove sequences of uncharacterized/hypothetical proteins, or non-informative such as "polyprotein fragment"
    logger.info("Filtering UniRef50 viral protein sequences")
    uniref_df = pl.read_csv(
        uniref50_viral_data,
        separator="\t",
        null_values=["NA", ""],
    )
    to_remove_terms = [
        "uncharacterized protein",
        "hypothetical protein",
        "putative protein",
        "predicted protein",
        "unnamed protein product",
        "polyprotein fragment",
        "genome polyprotein",
        "fragment",        
    ]
    pattern = "|".join(to_remove_terms)
    filtered_uniref_df = uniref_df.filter(
        ~pl.col("Cluster Name").str.to_lowercase().str.contains(pattern)
    )
    filtered_ids = filtered_uniref_df.select(pl.col("Cluster ID")).to_series().to_list()
    logger.info(f"Removing {uniref_df.height - filtered_uniref_df.height} non-informative sequences from UniRef50 viral dataset")
    # filter fasta
    # first load the fasta to a dataframe to get the original headers containing the Cluster IDs + defline and remove based on only the IDs
    unidf = from_fastx_eager(uniref50_viral_fasta).with_columns(
        pl.col("header").str.split(" ").list.first().alias("Cluster ID")
    )
    unidf_filtered = unidf.filter(
        pl.col("Cluster ID").is_in(filtered_ids)
    )
    write_fasta_file(format="fasta",headers=unidf_filtered["header"].to_list(), seqs=unidf_filtered["sequence"].to_list(), output_file=uniref50_viral_fasta)

    # clean up temporary gz files
    try:    
        os.remove(uniref50_viral_fasta + ".gz")
        os.remove(uniref50_viral_data + ".gz")
    except FileNotFoundError:
        logger.warning(
            "some temporary files for UniRef50 viral preparation might not have been cleaned."
        )

    # https://rest.uniprot.org/uniref/stream?compressed=true&fields=id%2Cname%2Ctypes%2Ccount%2Corganism%2Clength%2Cidentity%2Cmembers&format=tsv&query=%28%28identity%3A0.5%29+AND+%28taxonomy_id%3A10239%29+AND+%28count%3A%5B1+TO+192133%5D%29%29

    # https://rest.uniprot.org/uniref/stream?compressed=true&format=fasta&query=%28%28identity%3A0.5%29+AND+%28taxonomy_id%3A10239%29+AND+%28count%3A%5B1+TO+192133%5D%29%29



def prepare_genomad_rna_viral_markers(
    data_dir, threads, logger: logging.Logger):
    """Download and prepare RNA viral HMMs from geNomad markers.

    Downloads the geNomad database, analyzes the marker metadata to identify
    RNA viral specific markers, and creates an HMM database from their alignments.

    Args:
        data_dir (str): Base directory for data storage
        threads (int): Number of CPU threads to use
        logger: Logger object for recording progress and errors
    """

    logger.info("Starting geNomad RNA viral HMM preparation")

    # Create directories
    genomad_dir = os.path.join(data_dir, "profiles/genomad")
    genomad_db_dir = os.path.join(genomad_dir, "genomad_db")
    genomad_markers_dir = os.path.join(genomad_db_dir, "markers")
    genomad_alignments_dir = os.path.join(genomad_markers_dir, "alignments")
    os.makedirs(genomad_dir, exist_ok=True)
    os.makedirs(genomad_db_dir, exist_ok=True)
    os.makedirs(genomad_markers_dir, exist_ok=True)
    os.makedirs(genomad_alignments_dir, exist_ok=True)

    # Download metadata and database
    genomad_data = "https://zenodo.org/api/records/14886553/files-archive"  # noqa
    db_url = "https://zenodo.org/records/14886553/files/genomad_msa_v1.9.tar.gz?download=1"
    metadata_url = "https://zenodo.org/records/14886553/files/genomad_metadata_v1.9.tsv.gz?download=1"
    # Download and read metadata
    logger.info("Downloading geNomad metadata")
    aria2c_command = f"aria2c -c -d {genomad_dir} -o ./genomad_metadata_v1.9.tsv.gz {metadata_url}"
    subprocess.run(aria2c_command, shell=True)

    metadata_df = pl.read_csv(
        f"{genomad_dir}/genomad_metadata_v1.9.tsv.gz",
        separator="\t",
        null_values=["NA"],
        infer_schema_length=10000,
    )
    # only virus specific markers
    # metadata_df = metadata_df.filter(pl.col("SPECIFICITY_CLASS") == "VV")
    # only RNA viral markers
    metadata_df = metadata_df.filter(
        pl.col("ANNOTATION_DESCRIPTION")
        .str.to_lowercase()
        .str.contains("rna-dependent rna polymerase")
        | pl.col("TAXONOMY").str.contains("Riboviria")
        | pl.col("SOURCE").str.contains("RVMT")
    )
    # Next, filling missing annotation from InterPro.
    # only ones without description
    to_fill = metadata_df.filter(pl.col("ANNOTATION_DESCRIPTION").is_null())
    # if multiple maybe split->explode->groupby->agg->majority vote, something like:
    # nah just using the first if multiple maybe split->explode->first:
    # for now, only using a single accession (but word cloud/majority vote would probably work for multiple accessions)
    # to_fill = to_fill.with_columns(pl.col("ANNOTATION_ACCESSIONS").str.split(";").list.first().alias("ANNOTATION_ACCESSIONS_first"))
    to_fill = to_fill.with_columns(
        pl.col("ANNOTATION_ACCESSIONS")
        .str.split(";")
        .alias("ANNOTATION_ACCESSIONS_struct")
    )
    to_fill = to_fill.explode("ANNOTATION_ACCESSIONS_struct")

    to_fill = to_fill.with_columns(
        pl.when(pl.col("ANNOTATION_ACCESSIONS").str.starts_with("PF"))
        .then(pl.lit("Pfam"))
        .when(pl.col("ANNOTATION_ACCESSIONS").str.starts_with("COG"))
        .then(pl.lit("COG"))
        .when(pl.col("ANNOTATION_ACCESSIONS").str.starts_with("K"))
        .then(pl.lit("KEGG"))
        .otherwise(pl.lit("unknown"))
        .alias("source_db")
    )

    # We (currently) only carte about viral specific markers, so filtering out the rest
    # Not sure Kegg is on interpro.
    to_fill = to_fill.filter(pl.col("source_db").str.contains("Pfam|COG"))

    def query_interpro(entry: str, source_db: str):
        """Fetch the InterPro description for a given entry."""
        # from bs4 import BeautifulSoup
        import requests

        if source_db == "unknown":
            return None
        url = f"https://www.ebi.ac.uk/interpro/api/entry/{source_db}/{entry}"
        # print(url)                     #  debugging

        response = requests.get(url)
        if response.status_code != 200:
            return None

        data = response.json()
        # print(data)                     #  debugging

        # desc = data.get("metadata", {}).get("description")
        desc = data.get("metadata", {}).get("name", {}).get("name", None)
        return desc

    filled_interpro = []
    from rich.progress import track


    tiny_fill = to_fill.select(
        ["ANNOTATION_ACCESSIONS_struct", "source_db"]
    ).unique()

    for row in track(tiny_fill.to_dicts()):
        if row["source_db"] == "unknown":
            filled_interpro.append(None)
        else:
            this_desc = query_interpro(
                row["ANNOTATION_ACCESSIONS_struct"], row["source_db"]
            )
            filled_interpro.append(this_desc)
            print(
                f"{row['ANNOTATION_ACCESSIONS_struct']}\t{this_desc}"
            )  # debugging
            # filled_interpro.append(query_interpro(row["ANNOTATION_ACCESSIONS"], row["source_db"]))

    tiny_fill = tiny_fill.with_columns(
        pl.Series(filled_interpro).alias("interpro")
    )
    to_fill = to_fill.join(
        tiny_fill, on=["ANNOTATION_ACCESSIONS_struct", "source_db"], how="left"
    )
    to_fill = to_fill.with_columns(
        pl.coalesce(pl.col("ANNOTATION_DESCRIPTION"), pl.col("interpro")).alias(
            "ANNOTATION_DESCRIPTION"
        )
    )
    to_fill = to_fill.drop(
        "ANNOTATION_ACCESSIONS_struct", "interpro", "source_db"
    ).unique()
    to_fill = to_fill.filter(pl.col("ANNOTATION_DESCRIPTION").is_not_null())
    # hopefully now we have filled some of the missing descriptions, and we don't have any duplicate MARKERs

    metadata_df = metadata_df.filter(
        ~pl.col("MARKER").is_in(to_fill["MARKER"].implode())
    )
    metadata_df = metadata_df.vstack(to_fill)

    metadata_df.write_csv(
        f"{genomad_dir}/rna_viral_markers_with_annotation.csv"
    )

    # Download MSAs
    logger.info("Downloading geNomad database")
    aria2c_command = (
        f"aria2c -c -d {genomad_dir} -o ./genomad_msa_v1.9.tar.gz {db_url}"
    )
    subprocess.run(aria2c_command, shell=True)

    # Extract RNA viral MSAs
    marker_ids = metadata_df["MARKER"].to_list()

    with tarfile.open(f"{genomad_dir}/genomad_msa_v1.9.tar.gz", "r") as tar:
        for member in tar.getmembers():
            if (
                member.name.removeprefix("genomad_msa_v1.9/").removesuffix(
                    ".faa"
                )
                in marker_ids
            ):
                tar.extract(member, genomad_alignments_dir)
    # need to move all files in genomad/genomad_db/markers/alignments/genomad_msa_v1.9/* to genomad/genomad_db/markers/alignments/
    for file in os.listdir(genomad_alignments_dir + "/genomad_msa_v1.9"):
        shutil.move(
            genomad_alignments_dir + "/genomad_msa_v1.9/" + file,
            genomad_alignments_dir + "/" + file,
        )
    # remove the genomad_msa_v1.9 directory
    shutil.rmtree(genomad_alignments_dir + "/genomad_msa_v1.9")

    output_hmm = os.path.join(
        os.path.join(data_dir, "profiles/hmmdbs"),
        "genomad_rna_viral_markers.hmm",
    )
    hmmdb_from_directory(
        msa_dir=genomad_alignments_dir,
        output=output_hmm,
        msa_pattern="*.faa",
        info_table=f"{profile_dir}/genomad_rna_viral_markers_with_annotation.csv",
        name_col="MARKER",
        accs_col="ANNOTATION_ACCESSIONS",
        desc_col="ANNOTATION_DESCRIPTION",
        gath_col=None,  # no gathering theshold pre-defined for genomad
    )

    mmseqs_profile_db_from_directory(
        msa_dir=genomad_alignments_dir,
        output=os.path.join(
            data_dir, "profiles/mmseqs_dbs/genomad/", "rna_viral_markers"
        ),
        info_table=f"{profile_dir}/genomad_rna_viral_markers_with_annotation.csv",
        msa_pattern="*.faa",
        name_col="MARKER",
        accs_col="ANNOTATION_ACCESSIONS",
        desc_col="ANNOTATION_DESCRIPTION",
    )
    # clean up
    try:
        os.remove(f"{genomad_dir}/genomad_metadata_v1.9.tsv.gz")
        # move the info table to the head folder of profiles (from all dbs)
        shutil.move(
            f"{genomad_dir}/rna_viral_markers_with_annotation.csv",
            os.path.join(data_dir, "profiles/genomad_rna_viral_markers_with_annotation.csv"),
        )
        os.remove(f"{genomad_dir}/genomad_msa_v1.9.tar.gz")
        shutil.rmtree(genomad_db_dir)
    except Exception as e:
        logger.warning(f"Could not remove file: {e}")

    logger.info(f"Created genomad RNA viral HMM database at {output_hmm}")

 
def prepare_rdrp_scan(data_dir, threads, logger: logging.Logger):
    """Download and prepare RdRp profiles from RdRp-scan.

    Args:
        data_dir (str): Base directory for data storage
        threads (int): Number of CPU threads to use
        logger: Logger object for recording progress and errors
    """

    logger.info("Preparing RdRp-scan HMM and MMseqs databases")
    fetch_and_extract(
        "https://github.com/JustineCharon/RdRp-scan/archive/refs/heads/main.zip",
        fetched_to=hmmdb_dir + "/RdRp-scan.zip",
        extract_to=hmmdb_dir + "/RdRp-scan",
    )

    # Use utility function to build HMM database from MSAs
    rdrp_scan_msa_dir = os.path.join(
        hmmdb_dir, "RdRp-scan/RdRp-scan-main/Profile_db_and_alignments"
    )
    rdrp_scan_output = os.path.join(hmmdb_dir, "rdrp_scan.hmm")
    hmmdb_from_directory(
        msa_dir=rdrp_scan_msa_dir,
        output=rdrp_scan_output,
        msa_pattern="*.fasta.CLUSTALO",
    )
    # Also build an MMseqs profile DB from the RdRp-scan MSAs for fast searches
    mmseqs_profile_db_from_directory(
        msa_dir=rdrp_scan_msa_dir,
        output=os.path.join(mmseqs_dbs, "rdrp_scan/rdrp_scan"),
        msa_pattern="*.fasta.CLUSTALO",
        info_table=None,
    )
    # clean up
    shutil.rmtree(hmmdb_dir + "/RdRp-scan")
    os.remove(hmmdb_dir + "/RdRp-scan.zip")

    logger.debug("Finished preparing rdrp-scan databases")


def prepare_pfam_rdrps_rt(data_dir, threads, logger: logging.Logger):
    """Download and prepare RdRp profiles from PFAM.

    Args:
        data_dir (str): Base directory for data storage
        threads (int): Number of CPU threads to use
        logger: Logger object for recording progress and errors
    """

    logger.info("Preparing PFAM RdRps and RTs HMM database")

    # PFAM_A_38 RdRps and RTs
    # fetch Pfam-A.hmm.gz to the hmmdb directory and extract into that directory
    pfam_hmm_url = "https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam38.0/Pfam-A.hmm.gz"
    pfam_msa_url = "https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam38.0/Pfam-A.fasta.gz"
    pfam_gz_path = os.path.join(hmmdb_dir, "Pfam-A.hmm.gz")
    os.makedirs(hmmdb_dir, exist_ok=True)
    fetch_and_extract(
        url=pfam_hmm_url, fetched_to=pfam_gz_path, extract_to=hmmdb_dir
    )

    RdRps_and_RTs = [
        "PF04197.17",
        "PF04196.17",
        "PF22212.1",
        "PF22152.1",
        "PF22260.1",
        "PF00680.25",
        "PF00978.26",
        "PF00998.28",
        "PF02123.21",
        "PF07925.16",
        "PF00078.32",
        "PF07727.19",
        "PF13456.11",
    ]

    # Use hmmfetch to extract the small set of Pfam HMMs we care about
    selected_pfam_output = os.path.join(hmmdb_dir, "pfam_rdrps_and_rts.hmm")
    from rolypoly.utils.bio.alignments import hmm_fetch

    hmm_fetch(accessions=RdRps_and_RTs,
              hmm_db=os.path.join(hmmdb_dir, "Pfam-A.hmm"),
              output=selected_pfam_output,
              strip_after_char=".",
                logger=logger)

    # also prepare mmseqs profile db 
    subprocess.run(
                "mmseqs databases Pfam-A.seed data/profiles/mmseqs_dbs/pfam_a/pfam_a_38_seed tmp",
                shell=True,
                check=True,
    )
    # TODO: Filter the pfam_A mmseqs db to remove "hypothetical protein" + similar entries?

    from rolypoly.utils.bio.alignments import fetchPfamMSA
    # fetch Pfam MSAs for these accessions
    pfam_msa_folder = os.path.join(hmmdb_dir, "pfam_msas")
    os.makedirs(pfam_msa_folder, exist_ok=True)
    for acc in [acc.split(".")[0] for acc in RdRps_and_RTs]:
            fetchPfamMSA(
                acc=acc,
                output_folder=pfam_msa_folder,
                logger=logger,
            )
            logger.info(f"Fetched MSA for Pfam accession {acc}")
        # build mmseqs profile db from these msas
    mmseqs_profile_db_from_directory(
        msa_dir=pfam_msa_folder,
        output=os.path.join(mmseqs_dbs, "pfam_rdrps_and_rts/pfam_rdrps_and_rts"),
        msa_pattern="*.sth",
        info_table=None)
    

    # clean up downloaded gz
    try:
        os.remove(pfam_gz_path)
        os.remove("tmp")

    except Exception:
        pass

    logger.debug("Finished preparing pfam and pfam rdrps+rts sub-database")


def prepare_neordrp_profiles (data_dir, threads, logger: logging.Logger):
    """Prepare NeoRdRp v2.1 RdRp profile HMM database 
    NOTE: this is a USING A PRECOMPUTED HMM, not building from MSA! MIGHT NOT BE COMPATIBLE WITH FUTURE VERSIONS OF HMMER!
    """
    logger.info("Preparing NeoRdRp v2.1 HMM database")
    neordrp_url = "https://zenodo.org/records/10851672/files/NeoRdRp.2.1.hmm.xz?download=1"
    neordrp_path = os.path.join(hmmdb_dir, "NeoRdRp.2.1.hmm.xz")
    fetch_and_extract(
        url=neordrp_url,
        fetched_to=neordrp_path,
        extract_to=hmmdb_dir,
    )
    shutil.move(os.path.join(hmmdb_dir, "NeoRdRp.2.1.hmm"), os.path.join(hmmdb_dir, "neordrp2.1.hmm"))
    os.unlink(neordrp_path)



def prepare_RVMT_profiles(data_dir, threads, logger: logging.Logger):
    """Prepare RVMT RdRp profile + annotation db (NVPC)."""
    logger.info("Preparing RVMT HMM and MMseqs databases")
    rvmt_url = "https://portal.nersc.gov/dna/microbial/prokpubs/Riboviria/RiboV1.4/Alignments/zip.ali.220515.tgz"
    rvmt_path = os.path.join(hmmdb_dir, "zip.ali.220515.tgz")
    fetch_and_extract(
        url=rvmt_url,
        fetched_to=rvmt_path,
        extract_to=os.path.join(hmmdb_dir, "RVMT/"),
    )
    hmmdb_from_directory(
        msa_dir=os.path.join(hmmdb_dir, "RVMT/"),
        output=os.path.join(hmmdb_dir, "rvmt.hmm"),
        msa_pattern="ali*/*.FASTA",
        info_table=None,
    )

    mmseqs_profile_db_from_directory(
        msa_dir=os.path.join(hmmdb_dir, "RVMT/"),
        output=os.path.join(mmseqs_dbs, "RVMT/RVMT"),
        msa_pattern="ali*/*.FASTA",
        info_table=None,
    )

    # now for nvpc
    fetch_and_extract(url="https://portal.nersc.gov/dna/microbial/prokpubs/Riboviria/RiboV1.4/zenodo/v4/RVMT_Zenodo_V4/Domains_Annotations/NVPC/msaFiles.tar.gz",
        fetched_to=os.path.join(hmmdb_dir, "NVPC_msaFiles.tar.gz"),
        extract_to=os.path.join(hmmdb_dir, "RVMT/NVPC/"),
    )

    # nvpc_info = pl.read_csv("https://portal.nersc.gov/dna/microbial/prokpubs/Riboviria/RiboV1.4/zenodo/v4/RVMT_Zenodo_V4/Domains_Annotations/NVPC/NVPC_info.tsv",
    #     separator="\t",
    #     null_values=["NA", ""],
    # ).rename({"New_Name":"Name"}).unique()
    # nvpc_descriptions = pl.read_excel("https://portal.nersc.gov/dna/microbial/prokpubs/Riboviria/RiboV1.4/zenodo/v4/RVMT_Zenodo_V4/Domains_Annotations/misc/NeoCM3_full.xlsx")
    # nvpc_descriptions = nvpc_descriptions.select(["profile_accession","New_Name","Comment"]).rename({"New_Name":"Name","Comment":"Description"}).unique()
    # # add a column with now manyu times each new_name appears
    # # nvpc_descriptions = nvpc_descriptions.filter(pl.col("Description").str.contains_any(["rdrp_fragment","caution","the profile should be split or ditched altogeth"],ascii_case_insensitive=True))
    # nvpc_info = nvpc_info.join(temp_df, on="Name", how="inner")
    # nvpc_descriptions = nvpc_descriptions.join(nvpc_info, on="Name", how="inner")
    # nvpc_descriptions = nvpc_descriptions.filter(pl.col("profile_accession").is_in(nvpc_info["profile_accession"].implode()))
    # #  sort by number of new_name (most duplicates first)
    # nvpc_descriptions.write_csv(os.path.join(hmmdb_dir, "RVMT/NVPC/NVPC_descriptions.csv"),include_header=True)

    info_table=pl.read_csv(os.path.join(profile_dir, "NVPC_descriptions.csv"))
    # remove any msa file that doesn't have a matching profile_accession in the info table
    import glob
    msa_files = glob.glob(os.path.join(hmmdb_dir, "RVMT/NVPC/msaFiles/*.afa"))
    valid_accessions = set(info_table["profile_accession"].to_list())
    for msa_file in msa_files:
        base_name = os.path.basename(msa_file)
        #get the NVPC.NNNn part (first two parts separated by .)
        acc = ".".join(os.path.splitext(base_name)[0].split(".")[:2])
        if acc not in valid_accessions:
            logger.info(f"Removed MSA file without matching accession: {msa_file}")
            os.remove(msa_file)

    hmmdb_from_directory(
        msa_dir=os.path.join(hmmdb_dir, "RVMT/NVPC/"),
        output=os.path.join(hmmdb_dir, "nvpc.hmm"),
        msa_pattern="msaFiles/*.afa",
        info_table=os.path.join(profile_dir, "NVPC_descriptions.csv"),
        accs_col="profile_accession",
        name_col="Name",
        desc_col="Description",
        missing_include=False,
        default_gath="5",
        debug=False,
    )

    mmseqs_profile_db_from_directory(
        msa_dir=os.path.join(hmmdb_dir, "RVMT/NVPC/msaFiles/"),
        output=os.path.join(mmseqs_dbs, "nvpc/nvpc"),
        msa_pattern="*.afa",
        info_table=os.path.join(profile_dir, "NVPC_descriptions.csv"),
        accs_col="profile_accession",
        name_col="Name",
        desc_col="Description",
        # missing_include=False,
    )

    
    # clean up
    shutil.rmtree(os.path.join(hmmdb_dir, "RVMT/"))
    os.remove(os.path.join(hmmdb_dir, "zip.ali.220515.tgz"))
    os.remove(os.path.join(hmmdb_dir, "NVPC_msaFiles.tar.gz"))
    logger.info("Finished preparing RVMT databases")


def prepare_vfam(data_dir, logger: logging.Logger):
    """Prepare VFAM HMM database."""
    # fetch_and_extract(
    #     url="https://fileshare.csb.univie.ac.at/vog/latest/vfam.hmm.tar.gz",
    #     fetched_to=os.path.join(data_dir, "profiles","hmmdbs", "vfam","vfam.tar.gz"),
    #     extract_to=os.path.join(data_dir, "profiles","hmmdbs", "vfam"),
    # )
    os.makedirs(
        os.path.join(data_dir, "profiles", "hmmdbs", "vfam"), exist_ok=True
    )
    vfam_df = pl.read_csv(
        "https://fileshare.lisc.univie.ac.at//vog/latest/vfam.annotations.tsv.gz",
        separator="\t",
    )
    # In [54]: vfam_df.shape
    # Out[54]: (39624, 5)
    # In [56]: vfam_df.collect_schema()
    # Out[56]:
    # Schema([('#GroupName', String),
    #     ('ProteinCount', Int64),
    #     ('SpeciesCount', Int64),
    #     ('FunctionalCategory', String),
    #     ('ConsensusFunctionalDescription', String)])

    vfam_df.write_csv(
        os.path.join(
            data_dir, "profiles", "hmmdbs", "vfam.annotations.tsv.gz"
        )
    )
    version = requests.get(
        "https://fileshare.lisc.univie.ac.at/vog/latest/release.txt"
    ).text.strip()
    logger.info(f"VFAM version: {version}")

    # the raw MSAs
    fetch_and_extract(
        url="https://fileshare.lisc.univie.ac.at//vog/latest/vfam.raw_algs.tar.gz",
        fetched_to=os.path.join(
            data_dir, "profiles", "hmmdbs", "vfam.raw_algs.tar.gz"
        ),
        extract_to=os.path.join(data_dir, "profiles", "hmmdbs", "vfam"),
    )

    import glob
    # remove GroupNames and msa files that only have 2 or less sequences, or that have "hypothetical" in the description
    msa_files = glob.glob(os.path.join(os.path.join(data_dir, "profiles", "hmmdbs", "vfam"), "msa/*.msa"))
    valid_accessions = set(vfam_df.filter(
        (~pl.col("ConsensusFunctionalDescription").str.contains("hypothetical")) & 
        (pl.col("SpeciesCount") > 2)
    )["#GroupName"].to_list())
    for msa_file in msa_files:
        base_name = os.path.basename(msa_file)
        #get the NVPC.NNNn part (first two parts separated by .)
        acc = ".".join(os.path.splitext(base_name)[0].split(".")[:2])
        if acc not in valid_accessions:
            # logger.info(f"Removed MSA file without matching accession: {msa_file}")
            os.remove(msa_file)


    output_hmm = os.path.join(
        os.path.join(data_dir, "profiles/hmmdbs"), "vfam.hmm"
    )
    
    hmmdb_from_directory(
        os.path.join(data_dir, "profiles", "hmmdbs", "vfam", "msa"),
        output_hmm,
        msa_pattern="*.msa",
        info_table=(
            os.path.join(
                data_dir,
                "profiles",
                "vfam.annotations.tsv.gz",
            )
        ),
        name_col="#GroupName",
        accs_col="#GroupName",
        desc_col="ConsensusFunctionalDescription",
        gath_col=None,  # no gathering theshold pre-defined
    )

    # Build the mmseqs database
    mmseqs_profile_db_from_directory(
        os.path.join(data_dir, "profiles", "hmmdbs", "vfam", "msa"),
        output=os.path.join(mmseqs_dbs, "vfam/vfam"),
        msa_pattern="*.msa",
        info_table=(
            os.path.join(
                data_dir,
                "profiles",
                "vfam.annotations.tsv.gz",
            )
        ),
        name_col="#GroupName",
        accs_col="#GroupName",
        desc_col="ConsensusFunctionalDescription",
    )

    # remove the vfam msa directory
    shutil.rmtree(os.path.join(data_dir, "profiles", "hmmdbs", "vfam", "msa"))
    logger.info(
        f"Created VFAM HMM database at {os.path.join(data_dir, 'hmmdbs', 'vfam.hmm')}"
    )

def prepare_ncbi_ribovirus(data_dir, threads, logger: logging.Logger):
    """Download and prepare NCBI ribovirus reference sequences (RefSeq only).

    Downloads complete RefSeq genomes for RNA viruses (Riboviria), processes them
    with entropy masking and compression for efficient searches.

    Args:
        data_dir (str): Base directory for data storage
        threads (int): Number of CPU threads to use
        logger: Logger object for recording progress and errors
    """

    logger.info("Preparing NCBI ribovirus reference sequences")

    ncbi_ribovirus_dir = os.path.join(
        data_dir, "reference_seqs", "ncbi_ribovirus"
    )
    os.makedirs(ncbi_ribovirus_dir, exist_ok=True)
    mmdb_dir = os.path.join(ncbi_ribovirus_dir, "mmseqs")
    os.makedirs(mmdb_dir, exist_ok=True)

    # Define file paths
    raw_fasta_path = os.path.join(
        ncbi_ribovirus_dir, "refseq_ribovirus_genomes.fasta"
    )
    entropy_masked_path = os.path.join(
        ncbi_ribovirus_dir, "refseq_ribovirus_genomes_entropy_masked.fasta"
    )  # noqa (F841)
    compressed_path = os.path.join(
        ncbi_ribovirus_dir, "refseq_ribovirus_genomes_flat.fasta"
    )  # noqa (F841)

    # Riboviria taxid
    taxid = "2559587"

    # Use esearch and efetch to download complete RefSeq ribovirus genomes
    logger.info(f"Downloading RefSeq ribovirus genomes for taxid {taxid}")

    # if from_ena == True:
    #     # Use EBI/ENA REST API instead of NCBI E-utilities
    #     logger.info("Searching EBI/ENA for Riboviria sequences")

    #     # EBI/ENA API search for Riboviria complete genomes
    #     import requests

    #     # Search for Riboviria sequences in ENA
    #     search_url = "https://www.ebi.ac.uk/ena/portal/api/search"
    #     search_params = {
    #         "result": "sequence",
    #         "query": f'tax_tree({taxid}) AND mol_type="genomic RNA" AND base_count>1000',
    #         "fields": "accession,scientific_name,description,mol_type,tax_id,tax_lineage",
    #         "format": "json",
    #         "limit": "100"  # Get all results
    #     }

    #     logger.info("Querying EBI/ENA for sequence metadata")
    #     response = requests.get(search_url, params=search_params)
    #     response.raise_for_status()

    #     sequences_metadata = response.json()
    #     logger.info(f"Found {len(sequences_metadata)} Riboviria sequences")

    #     # Get FASTA sequences using EBI API
    #     logger.info("Downloading sequences from EBI/ENA")
    #     accessions = [seq["accession"] for seq in sequences_metadata[:1000]]  # Limit to avoid overwhelming

    #     with open(raw_fasta_path, "w") as fasta_out:
    #         for i, accession in enumerate(accessions):
    #             if i % 50 == 0:
    #                 logger.info(f"Downloaded {i}/{len(accessions)} sequences")

    #             # Get FASTA from EBI
    #             fasta_url = f"https://www.ebi.ac.uk/ena/browser/api/fasta/{accession}"
    #             fasta_response = requests.get(fasta_url)

    #             if fasta_response.status_code == 200:
    #                 fasta_out.write(fasta_response.text)
    #                 fasta_out.write("\n")
    #             else:
    #                 logger.warning(f"Failed to download {accession}: {fasta_response.status_code}")

    #     logger.info("Downloaded RefSeq ribovirus genomes from EBI/ENA")

    #     # Apply entropy masking first (consistent with RVMT approach)
    #     logger.info("Applying entropy masking")

    # if from_edirect == True:
    # esearch_query = f"txid{taxid}[Organism:exp] AND srcdb_refseq[PROP] AND complete genome[title]"
    #     logger.info("Running esearch | efetch pipeline")
    #     pipeline_cmd = f"~/bin/edirect/esearch -db nuccore -query '{esearch_query}' | ~/bin/edirect/efetch -format fasta > {raw_fasta_path}"

    #     run_command_comp(
    #         base_cmd=pipeline_cmd,
    #         params={},
    #         output_file=raw_fasta_path,
    #         logger=logger,
    #         check_output=True
    #     )

    from_ncbi_ftp = True  # for now, above methods is 1. edirect dependent, 2. ENA API dependent which seems slow/limited (or I'm not filtering prorperly - very likely)
    if from_ncbi_ftp == True:
        # genomes
        fetch_and_extract(
            url="https://ftp.ncbi.nlm.nih.gov/refseq/release/viral/viral.1.1.genomic.fna.gz",
            fetched_to=os.path.join(
                ncbi_ribovirus_dir, "viral.1.1.genomic.fna.gz"
            ),
            extract_to=ncbi_ribovirus_dir,
            rename_extracted=raw_fasta_path,
        )
        # orfs
        fetch_and_extract(
            url="https://ftp.ncbi.nlm.nih.gov/refseq/release/viral/viral.1.protein.faa.gz",
            fetched_to=os.path.join(
                ncbi_ribovirus_dir, "viral.1.protein.faa.gz"
            ),
            extract_to=ncbi_ribovirus_dir,
            rename_extracted=raw_fasta_path.replace(".fasta", "_orfs.faa"),
        )

    logger.info("Downloaded NCBI ribovirus genomes")

    # Create MMseqs2 database
    logger.info("Creating MMseqs2 database")
    run_command_comp(
        base_cmd="mmseqs createdb",
        positional_args_location="start",
        positional_args=[
            raw_fasta_path,
            os.path.join(mmdb_dir, "ncbi_ribovirus_cleaned"),
        ],
        params={"dbtype": "2"},
        logger=logger,
    )

    # Clean up intermediate files
    try:
        os.remove(os.path.join(ncbi_ribovirus_dir, "viral.1.1.genomic.fna.gz"))
        os.remove(os.path.join(ncbi_ribovirus_dir, "viral.1.protein.faa.gz"))
    except FileNotFoundError:
        logger.warning("Some intermediate files not found for cleanup")

    logger.info(f"NCBI ribovirus preparation completed in {ncbi_ribovirus_dir}")


def prepare_rvmt_motifs(data_dir, threads, logger):
    """Prepare RVMT motif sequences for profile-based searches.

    Extracts and processes the RVMT motif sequence library from the pre-downloaded
    tar.gz file, organizing motifs by type (A=mot.1, B=mot.2, C=mot.3) and taxon.
    Creates both HMM and MMseqs profile databases for fast searches.

    Args:
        data_dir (str): Base directory for data storage
        threads (int): Number of CPU threads to use
        logger: Logger object for recording progress and errors

    Note:
        This function assumes motif_sequence_library.tar.gz has been downloaded
        to data_dir/profiles/motif_sequence_library.tar.gz
    """

    logger.info("Preparing RVMT motif sequences")
    motif_dir = os.path.join(data_dir, "profiles/rvmt_motifs")
    motif_alignments_dir = os.path.join(motif_dir, "Sequence_Library")

    os.makedirs(motif_dir, exist_ok=True)
    os.makedirs(motif_alignments_dir, exist_ok=True)

    motif_archive = os.path.join(
        data_dir, "profiles/motif_sequence_library.tar.gz"
    )
    # fetch motif archive
    logger.info(
        "fetching Motif archive from https://portal.nersc.gov/dna/microbial/prokpubs/Riboviria/RiboV1.4/rdrps/motif_sequence_library.tar.gz"
    )
    fetch_and_extract(
        url="https://portal.nersc.gov/dna/microbial/prokpubs/Riboviria/RiboV1.4/rdrps/motif_sequence_library.tar.gz",
        fetched_to=motif_archive,
        extract_to=motif_dir,
        logger=logger,
    )

    # The archive contains Sequence_Library/ with mot.1/, mot.2/, mot.3/ subdirectories
    sequence_library_dir = os.path.join(motif_dir, "Sequence_Library")

    if not os.path.exists(sequence_library_dir):
        logger.error(
            "Expected Sequence_Library directory not found after extraction"
        )
        return False

    # Process each motif type directory
    motif_metadata = {}

    for motif_type_dir in os.listdir(sequence_library_dir):
        if not motif_type_dir.startswith("mot."):
            continue

        motif_type_path = os.path.join(sequence_library_dir, motif_type_dir)
        if not os.path.isdir(motif_type_path):
            continue

        # Map motif directory names to letters (A, B, C, D)
        motif_type_map = {
            "mot.1": "A",
            "mot.2": "B",
            "mot.3": "C",
            "mot.4": "D",
        }
        motif_letter = motif_type_map.get(motif_type_dir, motif_type_dir)

        logger.info(f"Processing motif type {motif_letter} ({motif_type_dir})")

        # Copy alignment files to organized structure
        for afa_file in os.listdir(motif_type_path):
            if afa_file.endswith(".afa"):
                src = os.path.join(motif_type_path, afa_file)
                # Create descriptive filename: motifA_taxon_id.afa
                base_name = afa_file.replace(".afa", "")
                new_name = f"motif{motif_letter}_{base_name}.afa"
                dst = os.path.join(motif_alignments_dir, new_name)
                shutil.copy2(src, dst)

                # Store metadata
                motif_metadata[new_name.replace(".afa", "")] = {
                    "motif_type": motif_letter,
                    "original_name": afa_file,
                    "taxon": base_name,  # this is a placeholder for if eventually the taxon info is incorporated.
                    "file_path": dst,
                }

    # Save metadata
    metadata_file = os.path.join(motif_dir, "motif_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(motif_metadata, f, indent=2)

    logger.info(f"Processed {len(motif_metadata)} motif alignments")

    # Create HMM database
    output_hmm = os.path.join(data_dir, "profiles/hmmdbs", "rvmt_motifs.hmm")
    logger.info(f"Building HMM database: {output_hmm}")

    hmmdb_from_directory(
        msa_dir=motif_alignments_dir,
        output=output_hmm,
        msa_pattern="*.afa",
        info_table=None,  # We'll use our metadata file instead
        name_col=None,
        accs_col=None,
        desc_col=None,
    )

    # Create MMseqs profile database
    mmseqs_output = os.path.join(
        data_dir, "profiles/mmseqs_dbs/rvmt_motifs", "rvmt_motifs"
    )
    os.makedirs(os.path.dirname(mmseqs_output), exist_ok=True)
    logger.info(f"Building MMseqs profile database: {mmseqs_output}")

    mmseqs_profile_db_from_directory(
        msa_dir=motif_alignments_dir,
        output=mmseqs_output,
        info_table=None,
        msa_pattern="*.afa",
        name_col=None,
        accs_col=None,
        desc_col=None,
    )

    # place holder for dimanond db creation

    # Clean up extracted directory
    try:
        # move the metadata file out before removing
        shutil.move(metadata_file, os.path.join(data_dir, "profiles"))

        shutil.rmtree(motif_dir)
        os.remove(motif_archive)
        os.remove(os.path.join(motif_dir, "motif_sequence_library.tar.gz"))

    except Exception as e:
        logger.warning(f"Could not remove motif alignments directory: {e}")

    # logger.info(f"RVMT motif preparation completed. Metadata saved to: {metadata_file}")
    logger.info(f"HMM database: {output_hmm}")
    logger.info(f"MMseqs database: {mmseqs_output}")

    return True


def prepare_contamination_seqs(data_dir, threads, logger):
    """Prepare the masking and contamination sequence sets used in sequence filtering.
    The contamination sequences refers to adapters (from bbtools and Fire lab) and rRNA sequences (SILVA + NCBI).
    The masking refers to creating a compressed set of viral sequences (made from concatenating RVMT and NCBI ribovirus, applies entropy masking and compressesion) that can be used for masking potentail viral sequences in host data.

    Args:
        data_dir (str): Base directory for data storage
        threads (int): Number of CPU threads to use
        logger: Logger object for recording progress and errors
    returns:
        None
    """

    logger.info(
        "Preparing masking sequences by combining RVMT and NCBI ribovirus"
    )

    # Create directories (if not already existing)
    contam_dir = os.path.join(data_dir, "contam")
    rrna_dir = os.path.join(contam_dir, "rrna")
    adapter_dir = os.path.join(contam_dir, "adapters")
    masking_dir = os.path.join(contam_dir, "masking")
    os.makedirs(contam_dir, exist_ok=True)
    os.makedirs(rrna_dir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)
    os.makedirs(masking_dir, exist_ok=True)

    # Masking sequences preparation
    rvmt_fasta_path = os.path.join(
        data_dir, "reference_seqs", "RVMT", "RVMT_cleaned_contigs.fasta"
    )
    ncbi_ribovirus_fasta_path = os.path.join(
        data_dir,
        "reference_seqs",
        "ncbi_ribovirus",
        "refseq_ribovirus_genomes.fasta",
    )

    # Deduplicate directly from multiple files (no concatenation needed)
    deduplicated_fasta = os.path.join(
        masking_dir, "combined_deduplicated.fasta"
    )
    logger.info(
        f"Deduplicating sequences from {len([rvmt_fasta_path, ncbi_ribovirus_fasta_path])} files"
    )

    stats = remove_duplicates(
        input_file=[rvmt_fasta_path, ncbi_ribovirus_fasta_path],
        output_file=deduplicated_fasta,
        by="seq",
        revcomp_as_distinct=False,  # Treat reverse complement as duplicate
        return_stats=True,
        logger=logger,
    )
    #     (rolypoly_tk) ➜  rolypoly git:(main) ✗ time seqkit rmdup -i  -s /clusterfs/jgi/scratch/science/metagen/neri/code/rolypoly/data/reference_seqs/RVMT/RVMT_cleaned_contigs.fasta   /clusterfs/jgi/scratch/science/metagen/neri/code/rolypoly/data/reference_seqs/ncbi_ribovirus/refseq_ribovirus_genomes.fasta > /dev/null
    # [INFO] 5399 duplicated records removed
    # seqkit rmdup -i -s   > /dev/null  14.70s user 0.53s system 75% cpu 20.095 total
    #     #(rolypoly_tk) ➜  rolypoly git:(main) ✗ time seqkit rmdup --quiet /clusterfs/jgi/scratch/science/metagen/neri/code/rolypoly/data/reference_seqs/RVMT/RVMT_cleaned_contigs.fasta   /clusterfs/jgi/scratch/science/metagen/neri/code/rolypoly/data/reference_seqs/ncbi_ribovirus/refseq_ribovirus_genomes.fasta --quiet | seqkit stats
    # file  format  type  num_seqs        sum_len  min_len  avg_len    max_len
    # -     FASTA   DNA    397,135  1,582,230,847      136  3,984.1  2,473,870
    # seqkit rmdup --quiet   --quiet  0.85s user 0.58s system 10% cpu 13.454 total
    # seqkit stats  10.93s user 0.31s system 83% cpu 13.450 total
    # #In [10]: remove_duplicates(
    # ...:         input_file=[rvmt_fasta_path, ncbi_ribovirus_fasta_path],
    # ...:         output_file=deduplicated_fasta,
    # ...:         by="seq",
    # ...:         revcomp_as_distinct=False,  # Treat reverse complement as duplicate
    # ...:         return_stats=True,
    # ...:         logger=logger
    # ...:     )
    # INFO     2025-11-21 12:26:16 - Processing 2 input files                                                                               sequences.py:451
    # INFO     2025-11-21 12:26:25 - Processed 397135 records: 391736 unique, 5399 duplicates removed                                       sequences.py:597
    # Out[10]: {'total_records': 397135, 'unique_records': 391736, 'duplicates_removed': 5399}

    if stats:
        logger.info(
            f"Deduplication stats: {stats['unique_records']} unique sequences from {stats['total_records']} total, {stats['duplicates_removed']} duplicates removed"
        )

    # Apply entropy masking to the deduplicated sequences
    logger.info("Applying entropy masking to combined sequences")
    entropy_masked_path = os.path.join(
        masking_dir, "combined_entropy_masked.fasta"
    )

    bbmask(
        in1=deduplicated_fasta,
        out=entropy_masked_path,
        entropy=0.1,
        entropywindow=30,
        threads=threads,
    )

    # reduce size with kcompress
    logger.info("Compressing sequences with kcompress")
    compressed_path = os.path.join(masking_dir, "combined_compressed.fasta")

    kcompress(
        in1=entropy_masked_path,
        out=compressed_path,
        fuse=500,
        k=31,
        prealloc=True,
        threads=threads,
    )

    #  now complexity masing again just to be sure
    bbmask(
        in1=compressed_path,
        out=entropy_masked_path,
        entropy=0.2,
        entropywindow=25,
        threads=threads,
    )

    # now a similar process for the orfs
    rvmt_fasta_path = os.path.join(
        data_dir, "reference_seqs", "RVMT", "RVMT_cleaned_orfs.faa"
    )
    ncbi_ribovirus_fasta_path = os.path.join(
        data_dir,
        "reference_seqs",
        "ncbi_ribovirus",
        "refseq_ribovirus_genomes_orfs.faa",
    )

    # Deduplicate directly from multiple files (no concatenation needed)
    deduplicated_fasta = os.path.join(
        masking_dir, "combined_deduplicated_orfs.faa"
    )
    
    logger.info(
        f"Deduplicating sequences from {len([rvmt_fasta_path, ncbi_ribovirus_fasta_path])} files"
    )

    stats = remove_duplicates(
        input_file=[rvmt_fasta_path, ncbi_ribovirus_fasta_path],
        output_file=deduplicated_fasta,
        by="seq",
        revcomp_as_distinct=False,  # Treat reverse complement as duplicate
        return_stats=True,
        logger=logger,
    )


    # clean up intermediate files
    try:
        os.remove(deduplicated_fasta)
        os.remove(compressed_path)
        os.remove(rvmt_fasta_path)
    except Exception as e:
        logger.warning(f"Could not remove intermediate files: {e}")

    # Prepare adapter sequences
    logger.info("Fetching adapter sequences")
    fetch_and_extract(
        url="https://raw.githubusercontent.com/bbushnell/BBTools/refs/heads/master/resources/adapters.fa",
        fetched_to=os.path.join(adapter_dir, "bbmap_adapters.fa"),
        rename_extracted=os.path.join(adapter_dir, "bbmap_adapters.fa"),
    )
    fetch_and_extract(
        url="https://raw.githubusercontent.com/FireLabSoftware/CountRabbit/refs/heads/main/illuminatetritis1223wMultiN.fa",
        fetched_to=os.path.join(adapter_dir, "AFire_illuminatetritis1223.fa"),
        rename_extracted=os.path.join(
            adapter_dir, "AFire_illuminatetritis1223.fa"
        ),
    )
    # remove the poly-monomer from Fire lab adapters
    filter_fasta_by_headers(
        fasta_file=os.path.join(adapter_dir, "AFire_illuminatetritis1223.fa"),
        output_file=os.path.join(
            adapter_dir, "AFire_illuminatetritis1223_filtered.fa"
        ),
        headers=["A70", "T70"],
        invert=True,
    )
    shutil.move(
        os.path.join(adapter_dir, "AFire_illuminatetritis1223_filtered.fa"),
        os.path.join(adapter_dir, "AFire_illuminatetritis1223.fa"),
    )

    # ===== rRNA Database Preparation with Metadata =====
    # Download and prepare ribosomal RNA sequences (bacterial, archaeal, eukaryotic)
    # Creates a metadata table with taxonomy lineages and FTP download links for host genomes/transcriptomes
    # This eliminates dependencies on taxonkit, ncbi-datasets CLI, and taxdump files

    logger.info("Preparing rRNA database with metadata")
    rrna_dir = os.path.join(contam_dir, "rrna")
    os.makedirs(rrna_dir, exist_ok=True)

    silva_release = "138.2"

    # Download SILVA rRNA sequences (SSU and LSU)
    logger.info(f"Downloading SILVA {silva_release} rRNA sequences")
    silva_ssu_path = os.path.join(
        rrna_dir, f"SILVA_{silva_release}_SSURef_NR99_tax_silva.fasta"
    )
    silva_lsu_path = os.path.join(
        rrna_dir, f"SILVA_{silva_release}_LSURef_NR99_tax_silva.fasta"
    )

    fetch_and_extract(
        f"https://www.arb-silva.de/fileadmin/silva_databases/release_{silva_release.replace('.', '_')}/Exports/SILVA_{silva_release}_SSURef_NR99_tax_silva.fasta.gz",
        fetched_to=os.path.join(rrna_dir, "tmp_ssu.fasta.gz"),
        extract_to=rrna_dir,
        rename_extracted=silva_ssu_path,
        logger=logger,
    )
    fetch_and_extract(
        f"https://www.arb-silva.de/fileadmin/silva_databases/release_{silva_release.replace('.', '_')}/Exports/SILVA_{silva_release}_LSURef_NR99_tax_silva.fasta.gz",
        fetched_to=os.path.join(rrna_dir, "tmp_lsu.fasta.gz"),
        extract_to=rrna_dir,
        rename_extracted=silva_lsu_path,
        logger=logger,
    )

    # Download SILVA taxonomy mappings (maps accessions to NCBI taxids)
    logger.info("Fetching/making SILVA taxonomy mappings (to NCBI taxids)")

    silva_ssu_taxmap = pl.read_csv(
        "https://www.arb-silva.de/fileadmin/silva_databases/current/Exports/taxonomy/ncbi/taxmap_embl-ebi_ena_ssu_ref_nr99_138.2.txt.gz",
        truncate_ragged_lines=True,
        separator="\t",
        infer_schema_length=123123,
    )
    silva_lsu_taxmap = pl.read_csv(
        "https://www.arb-silva.de/fileadmin/silva_databases/current/Exports/taxonomy/ncbi/taxmap_embl-ebi_ena_lsu_ref_nr99_138.2.txt.gz",
        truncate_ragged_lines=True,
        separator="\t",
        infer_schema_length=123123,
    )
    silva_taxmap = pl.concat([silva_lsu_taxmap, silva_ssu_taxmap])

    # Parse SILVA headers and extract accessions
    logger.info("Parsing SILVA sequences and extracting metadata")

    silva_fasta_df = pl.concat(
        [
            from_fastx_eager(silva_ssu_path).with_columns(
                pl.lit("SSU").alias("rRNA_type")
            ),
            from_fastx_eager(silva_lsu_path).with_columns(
                pl.lit("LSU").alias("rRNA_type")
            ),
        ]
    )
    logger.info(f"total SILVA sequences {silva_fasta_df.height}")

    # Extract accession from header (format: >accession.version rest_of_header)
    silva_fasta_df = silva_fasta_df.with_columns(
        primaryAccession=pl.col("header").str.extract(
            r"^([A-Za-z0-9_]+)(?:\.\d+)*", 1
        ),  # DQ150555.1.2478 -> DQ150555
        accession=pl.col("header").str.extract(
            r"^([A-Za-z0-9_]+(?:\.\d+)?)", 1
        ),  # AY846379 or DQ150555.1
        taxonomy_raw=pl.col("header").str.replace(r"^\S+\s+", ""),
    )
    # silva_fasta_df = silva_fasta_df.with_columns(
    #     pl.col("sequence").str.len_chars().alias("seq_length")
    # )
    # silva_taxmap = silva_taxmap.with_columns(
    #     (pl.col("stop") - pl.col("start")).alias("seq_length")
    # )

    silva_df = silva_fasta_df.join(
        silva_taxmap.select(
            ["primaryAccession", "ncbi_taxonid", "submitted_path"]
        ).unique(),  # seq_length
        on=["primaryAccession"],
        how="inner",
    )
    silva_df.write_parquet(os.path.join(rrna_dir, "silva_rrna_sequences.parquet"))
    # silva_df.height
    # silva_df["ncbi_taxonid"].null_count()

    # Load SILVA taxonomy mappings
    logger.info(
        f"Merged taxonomy for {silva_df.filter(pl.col('ncbi_taxonid').is_not_null()).height} SILVA sequences"
    )

    unique_taxids = (
        silva_df.filter(pl.col("ncbi_taxonid").is_not_null())
        .select("ncbi_taxonid")
        .unique()["ncbi_taxonid"]
        .to_list()
    )
    logger.info(
        f"Total of {len(unique_taxids)} unique NCBI taxids found in SILVA sequences"
    )

    # Generate FTP download URLs for host genomes/transcriptomes
    fetch_and_extract(
        url="https://ftp.ncbi.nlm.nih.gov/genomes/genbank/assembly_summary_genbank.txt",
        fetched_to=os.path.join(rrna_dir, "assembly_summary_genbank.txt.gz"),
        extract=False,
    )
    logger.info("Loading NCBI GenBank assembly summary")
    # genbank_summary = pl.read_csv(os.path.join(rrna_dir, "assembly_summary_genbank.txt.gz",),
    # infer_schema_length=100020, separator="\t", skip_rows=1,
    # null_values=["na","NA","-"],ignore_errors=True,
    # has_header=True)
    # polars failed me, so using line by line iterator
    from gzip import open as gz_open
    with gz_open(
        os.path.join(rrna_dir, "assembly_summary_genbank.txt.gz"), "r"
    ) as f:
        header = None
        records = []
        i = 0
        for line in f:
            if i == 0:
                i += 1
                continue
            line = line.rstrip(b"\n")
            if i == 1:
                header = line.decode()[1:].strip().split("\t")
                i += 1
                continue
            fields = line.decode().strip().split("\t")
            record = dict(zip(header, fields))
            records.append(record)
    genbank_summary = pl.from_records(records).rename({"taxid": "ncbi_taxonid"})
    genbank_summary.collect_schema()
        # Schema([('assembly_accession', String),
    #         ('bioproject', String),
    #         ('biosample', String),
    #         ('wgs_master', String),
    #         ('refseq_category', String),
    #         ('ncbi_taxonid', String),
    #         ('species_taxid', String),
    #         ('organism_name', String),
    #         ('infraspecific_name', String),
    #         ('isolate', String),
    #         ('version_status', String),
    #         ('assembly_level', String),
    #         ('release_type', String),
    #         ('genome_rep', String),
    #         ('seq_rel_date', String),
    #         ('asm_name', String),
    #         ('asm_submitter', String),
    #         ('gbrs_paired_asm', String),
    #         ('paired_asm_comp', String),
    #         ('ftp_path', String),
    #         ('excluded_from_refseq', String),
    #         ('relation_to_type_material', String),
    #         ('asm_not_live_date', String),
    #         ('assembly_type', String),
    #         ('group', String),
    #         ('genome_size', String),
    #         ('genome_size_ungapped', String),
    #         ('gc_percent', String),
    #         ('replicon_count', String),
    #         ('scaffold_count', String),
    #         ('contig_count', String),
    #         ('annotation_provider', String),
    #         ('annotation_name', String),
    #         ('annotation_date', String),
    #         ('total_gene_count', String),
    #         ('protein_coding_gene_count', String),
    #         ('non_coding_gene_count', String),
    #         ('pubmed_id', String)])

    genbank_summary.write_csv(
        os.path.join(rrna_dir, "genbank_assembly_summary.tsv"), separator="\t"
    )
    genbank_summary = pl.read_csv(
        os.path.join(rrna_dir, "genbank_assembly_summary.tsv"),
        infer_schema_length=100020,
        separator="\t",
        null_values=["na", "NA", "-"],
        ignore_errors=True,
        has_header=True,
    )
    # In [91]: genbank_summary.collect_schema()
    # Out[91]: 
    # Schema([('assembly_accession', String),
    #         ('bioproject', String),
    #         ('biosample', String),
    #         ('wgs_master', String),
    #         ('refseq_category', String),
    #         ('ncbi_taxonid', Int64),
    #         ('species_taxid', Int64),
    #         ('organism_name', String),
    #         ('infraspecific_name', String),
    #         ('isolate', String),
    #         ('version_status', String),
    #         ('assembly_level', String),
    #         ('release_type', String),
    #         ('genome_rep', String),
    #         ('seq_rel_date', String),
    #         ('asm_name', String),
    #         ('asm_submitter', String),
    #         ('gbrs_paired_asm', String),
    #         ('paired_asm_comp', String),
    #         ('ftp_path', String),
    #         ('excluded_from_refseq', String),
    #         ('relation_to_type_material', String),
    #         ('asm_not_live_date', String),
    #         ('assembly_type', String),
    #         ('group', String),
    #         ('genome_size', Int64),
    #         ('genome_size_ungapped', Int64),
    #         ('gc_percent', Float64),
    #         ('replicon_count', Int64),
    #         ('scaffold_count', Int64),
    #         ('contig_count', Int64),
    #         ('annotation_provider', String),
    #         ('annotation_name', String),
    #         ('annotation_date', String),
    #         ('total_gene_count', Int64),
    #         ('protein_coding_gene_count', Int64),
    #         ('non_coding_gene_count', Int64),
    #         ('pubmed_id', String)])

    genbank_summary.write_parquet(
        os.path.join(rrna_dir, "genbank_assembly_summary.parquet")
    )
    genbank_summary.write_csv(
        os.path.join(rrna_dir, "genbank_assembly_summary.tsv"), separator="\t"
    )
    
    # next, for every unique ncbi_taxonid, we select the one that has the most protein_coding_gene_count, then refseq_category, then tie breaking with non_coding_gene_count, tie breaking by latest assembly (by seq_rel_date).
    temp_genbank = genbank_summary.sort(
        by=[
            pl.col("protein_coding_gene_count").cast(pl.Int64).reverse(),
            pl.col("refseq_category").reverse(),
            pl.col("non_coding_gene_count").cast(pl.Int64).reverse(),
            pl.col("seq_rel_date").reverse(),
        ]
    ).unique(subset=["ncbi_taxonid"], keep="first")
    logger.info(
        f"Filtered GenBank summary to {temp_genbank.height} unique taxid entries for SILVA sequences"
    )
    temp_genbank = temp_genbank.filter(pl.col("ncbi_taxonid").is_in(unique_taxids)).unique()
    temp_genbank.height
    # only 30482 out ok ~100k?
    fetch_and_extract( url="http://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2accession.gz",
        fetched_to=os.path.join(rrna_dir, "gene2accession.gz"),
        extract=False,
    )
    gene2accession = pl.read_csv(
        os.path.join(rrna_dir, "gene2accession.gz"),
        separator="\t",
        # skip_rows=1,
        # infer_schema_length=100020,
        null_values=["na", "NA", "-"],
        ignore_errors=True,
        has_header=True,
        # n_rows=100
    )
    gene2accession.write_parquet(os.path.join(rrna_dir, "gene2accession.parquet"))
    # gene2accession = pl.read_parquet(os.path.join(rrna_dir, "gene2accession.parquet"))
    # gene2accession.collect_schema()
    # Schema([('#tax_id', Int64),
    #     ('GeneID', Int64),
    #     ('status', String),
    #     ('RNA_nucleotide_accession.version', String),
    #     ('RNA_nucleotide_gi', String),
    #     ('protein_accession.version', String),
    #     ('protein_gi', Int64),
    #     ('genomic_nucleotide_accession.version', String),
    #     ('genomic_nucleotide_gi', Int64),
    #     ('start_position_on_the_genomic_accession', Int64),
    #     ('end_position_on_the_genomic_accession', Int64),
    #     ('orientation', String),
    #     ('assembly', String),
    #     ('mature_peptide_accession.version', String),
    #     ('mature_peptide_gi', String),
    #     ('Symbol', String)])
    gene2accession = gene2accession.rename({"#tax_id": "ncbi_taxonid"})
    test_df = gene2accession.filter(pl.col("ncbi_taxonid").is_in(unique_taxids))
    test_df.height # 148449745
    test_df2 = gene2accession.select(["ncbi_taxonid","assembly"]).unique()
    test_df2.height # 52548

    silva_df = silva_df.with_columns(
        ncbi_taxonid=pl.col("ncbi_taxonid").cast(pl.String)
    )

    silva_df1 = silva_df.join(
        genbank_summary.select(["ncbi_taxonid", "ftp_path"]),
        on=["ncbi_taxonid"],
        how="left",
    )
    silva_df1

    silva_df = silva_df.with_columns(
        genome_ftp_url=pl.when(pl.col("ncbi_taxonid").is_not_null())
        .then(
            pl.format(
                "https://ftp.ncbi.nlm.nih.gov/genomes/all/refseq/taxid_{}/",
                pl.col("ncbi_taxonid"),
            )
        )
        .otherwise(None),
        datasets_api_url=pl.when(pl.col("ncbi_taxonid").is_not_null())
        .then(
            pl.format(
                "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/taxon/{}/download?include_annotation_type=GENOME_FASTA,RNA_FASTA",
                pl.col("ncbi_taxonid"),
            )
        )
        .otherwise(None),
    )

    # Save metadata table
    metadata_output = os.path.join(rrna_dir, "rrna_metadata.tsv")
    silva_df.write_csv(metadata_output, separator="\t")
    logger.info(
        f"Saved rRNA metadata table with {len(silva_df)} entries to {metadata_output}"
    )

    # Merge SILVA sequences and apply entropy masking
    logger.info("Merging and masking SILVA sequences")
    silva_merged = os.path.join(rrna_dir, "SILVA_merged.fasta")
    silva_masked = os.path.join(rrna_dir, "SILVA_merged_masked.fasta")

    # Concatenate SILVA files
    run_command_comp(
        base_cmd="cat",
        positional_args=[silva_ssu_path, silva_lsu_path],
        positional_args_location="end",
        params={},
        output_file=silva_merged,
        logger=logger,
    )

    # Apply entropy masking
    bbduk(
        in1=silva_merged,
        out=silva_masked,
        entropy=0.6,
        entropyk=4,
        entropywindow=24,
        maskentropy=True,
        ziplevel=9,
    )

    logger.info(f"Created masked SILVA rRNA database: {silva_masked}")

    # clean up
    try:
        os.remove(deduplicated_fasta)
        os.remove(compressed_path)
    except Exception as e:
        logger.warning(f"Could not remove intermediate files: {e}")

    logger.info(f"Masking sequences prepared in {masking_dir}")


def prepare_trna_data(data_dir, logger):
    trna_dir = os.path.join(data_dir, "tr", "trna")
    file_url = "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/fasta_files/RF00005.fa.gz"
    trna_seqs = os.path.join(trna_dir, "tRNA_sequences.fasta")
    gz_filename = "RF00005.fa.gz"
    deduplicated_fasta = os.path.join(trna_dir, "tRNA_sequences_deduplicated.fasta")

    fetch_and_extract(
                    url=file_url,
                    fetched_to=os.path.join(trna_dir, gz_filename),
                    extract_to=trna_dir,
                    expected_file=trna_seqs,
                )
    logger.info(f"Downloaded tRNA sequences to {trna_seqs}")
    # remove duplicates
    remove_duplicates(
        input_file=trna_seqs,
        output_file=deduplicated_fasta,
        return_stats=True,
        by = "seq"
    )
    from rolypoly.utils.bio.polars_fastx import fasta_stats
    from rolypoly.utils.bio.sequences import write_fasta_file
    info_table = fasta_stats(deduplicated_fasta)
    info_table = info_table.filter(
    pl.col("length").is_between(60,250),
    pl.col("gc_content") >= 0.01,
    )

    write_fasta_file(
    seqs=info_table["sequence"].to_list(),
    headers=info_table["header"].to_list(),
    output_file=os.path.join(trna_dir, "tRNA_sequences_deduplicated_filtered.fasta"),
    )
    logger.info(f"Wrote filtered tRNA sequences to {os.path.join(trna_dir, 'tRNA_sequences_deduplicated_filtered.fasta')}")


def prepare_plastid_data(data_dir, logger):
    """Prepare plastid sequence data for contamination filtering.

    Downloads NCBI RefSeq plastid sequences, combines them, and removes duplicates.

    Args:
        data_dir (str): Base directory for data storage
        logger: Logger object for recording progress and errors
    returns:
        None
    """
    plastid_dir = os.path.join(data_dir, "reference_seqs", "plastid_refseq")
    os.makedirs(plastid_dir, exist_ok=True)
    
    logger.info("Downloading NCBI RefSeq plastid sequences")
    
    base_url = "https://ftp.ncbi.nlm.nih.gov/refseq/release/plastid/plastid."
    suffix = ".genomic.fna.gz"
    files_to_get = ["1.1", "1.2", "2.1", "2.2", "3.1"]
    
    all_files =[]
    downloaded_files = []
    
    for version in files_to_get:
        file_url = f"{base_url}{version}{suffix}"
        gz_filename = f"plastid.{version}.genomic.fna.gz"
        fasta_filename = f"plastid.{version}.genomic.fna"
        
        logger.info(f"Downloading plastid version {version}")
        
        # Download and extract the file
        try:
            extracted_path = fetch_and_extract(
                url=file_url,
                fetched_to=os.path.join(plastid_dir, gz_filename),
                extract_to=plastid_dir,
                expected_file=fasta_filename,
                logger=logger,
            )
            downloaded_files.append(extracted_path)
            all_files.append(extracted_path)
            all_files.append(os.path.join(plastid_dir, gz_filename))
            logger.info(f"Successfully downloaded and extracted {fasta_filename}")
        except Exception as e:
            logger.error(f"Failed to download plastid version {version}: {e}")
            continue
    
    if not downloaded_files:
        logger.error("No plastid files were successfully downloaded")
        return
    
    # Combine and deduplicate the sequences
    combined_fasta = os.path.join(plastid_dir, "combined_plastid_refseq.fasta")
    logger.info(f"Combining and deduplicating {len(downloaded_files)} plastid files")
    
    stats = remove_duplicates(
        input_file=downloaded_files,
        output_file=combined_fasta,
        by="seq",
        revcomp_as_distinct=False,  # Treat reverse complement as duplicate
        return_stats=True,
        logger=logger,
    )
    
    if stats:
        logger.info(
            f"Plastid deduplication stats: {stats['unique_records']} unique sequences from {stats['total_records']} total, {stats['duplicates_removed']} duplicates removed"
        )
    
    # Clean up individual files
    try:
        for file_path in all_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Removed intermediate file: {os.path.basename(file_path)}")
    except Exception as e:
        logger.warning(f"Could not remove intermediate plastid files: {e}")
    
    logger.info(f"Plastid sequences prepared in {plastid_dir}")


def prepare_mito_data(data_dir, logger):
    """Prepare mito sequence data for contamination filtering.

    Downloads NCBI RefSeq mito sequences, combines them, and removes duplicates.

    Args:
        data_dir (str): Base directory for data storage
        logger: Logger object for recording progress and errors
    returns:
        None
    """
    mito_dir = os.path.join(data_dir, "reference_seqs", "mito_refseq")
    os.makedirs(mito_dir, exist_ok=True)
    
    logger.info("Downloading NCBI RefSeq mito sequences")
    
    base_url = "https://ftp.ncbi.nlm.nih.gov/refseq/release/mitochondrion/mitochondrion."
    suffix = ".genomic.fna.gz"
    files_to_get = ["1.1"] #
    
    all_files =[]
    downloaded_files = []
    
    for version in files_to_get:
        file_url = f"{base_url}{version}{suffix}"
        gz_filename = f"mito.{version}.genomic.fna.gz"
        fasta_filename = f"mito.{version}.genomic.fna"
        
        logger.info(f"Downloading mito version {version}")
        
        # Download and extract the file
        try:
            extracted_path = fetch_and_extract(
                url=file_url,
                fetched_to=os.path.join(mito_dir, gz_filename),
                extract_to=mito_dir,
                expected_file=fasta_filename,
                logger=logger,
            )
            downloaded_files.append(extracted_path)
            all_files.append(extracted_path)
            all_files.append(os.path.join(mito_dir, gz_filename))
            logger.info(f"Successfully downloaded and extracted {fasta_filename}")
        except Exception as e:
            logger.error(f"Failed to download mito version {version}: {e}")
            continue
    
    if not downloaded_files:
        logger.error("No mito files were successfully downloaded")
        return
    
    # Combine and deduplicate the sequences
    combined_fasta = os.path.join(mito_dir, "combined_mito_refseq.fasta")
    logger.info(f"Combining and deduplicating {len(downloaded_files)} mito files")
    
    stats = remove_duplicates(
        input_file=downloaded_files,
        output_file=combined_fasta,
        by="seq",
        revcomp_as_distinct=False,  # Treat reverse complement as duplicate
        return_stats=True,
        logger=logger,
    )
    
    if stats:
        logger.info(
            f"mito deduplication stats: {stats['unique_records']} unique sequences from {stats['total_records']} total, {stats['duplicates_removed']} duplicates removed"
        )
    
    # Clean up individual files to save space (optional)
    try:
        for file_path in all_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Removed intermediate file: {os.path.basename(file_path)}")
    except Exception as e:
        logger.warning(f"Could not remove intermediate mito files: {e}")
    
    logger.info(f"mito sequences prepared in {mito_dir}")



if __name__ == "__main__":
    build_data()


# source ~/.bashrc
# conda activate crispy
# export PATH=$PATH:/clusterfs/jgi/groups/science/homes/uneri/code/mmseqs/bin/

# THREADS=24

# ## Prepare NCBI RNA virus ####
# cd $rolypoly_dir/data/
# mkdir NCBI_ribovirus
# cd NCBI_ribovirus
# taxid="2559587"
# # Perform the search and download the genomes
# alias esearch='~/bin/edirect/esearch'
# esearch -db nuccore -query "txid$taxid[Organism:exp] AND srcdb_refseq[PROP] AND complete genome[title]" | efetch -format fasta > refseq_ribovirus_genomes.fasta
# kcompress.sh in=refseq_ribovirus_genomes.fasta out=refseq_ribovirus_genomes_flat.fasta fuse=2000 k=31  prealloc=true  threads=$THREADS # prefilter=true
# bbmask.sh in=refseq_ribovirus_genomes.fasta out=refseq_ribovirus_genomes_entropy_masked.fasta entropy=0.7  ow=t


# #### Prepare the RVMT mmseqs database ####
# cd $rolypoly_dir/data/
# mkdir RVMT
# mkdir mmdb
# wget https://portal.nersc.gov/dna/microbial/prokpubs/Riboviria/RiboV1.4/RiboV1.6_Contigs.fasta.gz
# extract RiboV1.6_Contigs.fasta.gz
# seqkit grep  -f ./chimeras_RVMT.lst RiboV1.6_Contigs.fasta --invert-match  > tmp_nochimeras.fasta
# mmseqs createdb  tmp_nochimeras.fasta  mmdb/RVMT_mmseqs_db2 --dbtype 2
# RVMTdb=/clusterfs/jgi/scratch/science/metagen/neri/rolypoly/data/RVMT/mmdb/RVMT_mmseqs_db2
# kcompress.sh in=tmp_nochimeras.fasta out=RiboV1.6_Contigs_flat.fasta fuse=2000 k=31  prealloc=true  threads=$THREADS # prefilter=true

# cd ../
# cat RVMT/RiboV1.6_Contigs_flat.fasta NCBI_ribovirus/refseq_ribovirus_genomes_flat.fasta > tmp_target.fas
# bbmask.sh in=tmp_target.fas out=tmp_target_ent_masked.fas entropy=0.7  ow=t
# mv RiboV1.6_Contigs_flat.fasta1 RiboV1.6_Contigs_flat.fasta

# bbmap.sh ref=$input_Fasta in=other_fasta outm=mapped.sam minid=0.9 overwrite=true threads=$THREADS  -Xmx"$MEMORY"
# bbmask.sh in=$input_file out=$output_file entropy=0.2 sam=mapped.sam
# bbduk.sh ref=$input_file sam=mapped.sam k=21 maskmiddle=t in=tmp_target_ent_masked.fas overwrite=true threads=$THREADS  -Xmx"$MEMORY"

# # Test #
# THREADS=4
# MEMORY=40g
# fetched_genomes /clusterfs/jgi/scratch/science/metagen/neri/rolypoly/bench/test_sampled_005_bb_metaTs_spiced_RVMT/temp_dir_sampled_005_bb_metaTs_spiced_RVMT/stats_rRNA_filt_sampled_005_bb_metaTs_spiced_RVMT.txt output.fasta
# input_file=/clusterfs/jgi/scratch/science/metagen/neri/rolypoly/data/output.fasta
# bbduk.sh ref=$input_file sam=mapped.sam k=21 maskmiddle=t in=tmp_target.fas overwrite=true threads=$THREADS  -Xmx"$MEMORY"


# ##### Create rRNA DB #####
# cd $rolypoly/data/
# mkdir rRNA
# cd rRNA
# wget https://www.arb-silva.de/fileadmin/silva_databases/release_138_1/Exports/SILVA_138.1_LSURef_NR99_tax_silva.fasta.gz
# wget https://www.arb-silva.de/fileadmin/silva_databases/release_138_1/Exports/SILVA_138.1_SSURef_NR99_tax_silva.fasta.gz

# gzip SILVA_138.1_SSURef_NR99_tax_silva.fasta.gz
# gzip SILVA_138.1_LSURef_NR99_tax_silva.fasta.gz

# cat *fasta > merged.fas


# bbduk.sh -Xmx1g in=merged.fas out=merged_masked.fa zl=9 entropy=0.6 entropyk=4 entropywindow=24 maskentropy

# # # Define the search term
# # search_term="ribosomal RNA[title] AND srcdb_refseq[PROP] AND 200:7000[SLEN]"
# # # Perform the search and download the sequences
# # esearch -db nuccore -query "$search_term" | efetch -format fasta > "rrna_genes_refseq.fasta"
# bbduk.sh -Xmx1g in=rmdup_rRNA_ncbi.fasta  out=rmdup_rRNA_ncbi_masked.fa zl=9 entropy=0.6 entropyk=4 entropywindow=24 maskentropy

# setup taxonkit
# TODO: CONVERT TO PYTHON
# cd "$DATA_PATH"
# aria2c http://ftp.ncbi.nih.gov/pub/taxonomy/taxdump.tar.gz
# tar -xzvf taxdump.tar.gz
# mv names.dmp nodes.dmp merged.dmp delnodes.dmp "$DATA_PATH/taxdump/"
# rm -rf  taxdump.tar.gz
