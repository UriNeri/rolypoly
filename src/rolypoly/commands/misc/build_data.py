import os
import shutil
import tarfile
import datetime
import requests
import subprocess
import logging
from pathlib import Path as pt
import polars as pl

from rich_click import command, option
from rolypoly.utils.logging.citation_reminder import remind_citations

from rolypoly.utils.logging.loggit import setup_logging, get_version_info
from rolypoly.utils.bio.alignments import hmmdb_from_directory, hmm_from_msa, mmseqs_profile_db_from_directory

from rolypoly.utils.various import (
    extract,
    fetch_and_extract,
)

from rich.console import Console
console = Console()
global tools
tools = []


@command()
@option(
    "--data-dir",
    required=True,
    help="Path to the data directory",
)
@option("--threads", default=4, help="Number of threads to use")
@option(
    "--log-file",
    default="./prepare_external_data_logfile.txt",
    help="Path to the log file",
)
def build_data(data_dir, threads, log_file):
    """Build external data required for RolyPoly. This is an internal scratch script, wrapped in a click command for convenience (and logging), but exposed to users.
        1. Build geNomad RNA viral HMMs
        2. Build protein HMMs RdRp-scan, RVMT, Neordrp_v2.1, tsa_2018 and PFAM_A_37 
        3. Download and prepare rRNA databases SILVA_138.1_SSURef_NR99_tax_silva.fasta and SILVA_138.1_LSURef_NR99_tax_silva.fasta
        4. Download Rfam data.
    """

    logger = setup_logging(log_file)
    logger.info(f"Starting data preparation to : {data_dir}")

    contam_dir = os.path.join(data_dir, "contam")
    os.makedirs(contam_dir, exist_ok=True)

    rrna_dir = os.path.join(contam_dir, "rRNA")
    os.makedirs(rrna_dir, exist_ok=True)

    adapter_dir = os.path.join(contam_dir, "adapters")
    os.makedirs(adapter_dir, exist_ok=True)

    masking_dir = os.path.join(contam_dir, "masking")
    os.makedirs(masking_dir, exist_ok=True)

    taxonomy_dir = os.path.join(data_dir, "taxdump")
    os.makedirs(taxonomy_dir, exist_ok=True)

    reference_seqs = os.path.join(data_dir, "reference_seqs")
    os.makedirs(reference_seqs, exist_ok=True)

    rvmt_dir = os.path.join(reference_seqs, "RVMT")
    os.makedirs(rvmt_dir, exist_ok=True)

    ncbi_ribovirus_dir = os.path.join(reference_seqs, "RVMT")
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

    # RdRp-scan
    prepare_rdrp_scan(data_dir, theards, logger)

    # RVMT
    rvmt_url = "https://portal.nersc.gov/dna/microbial/prokpubs/Riboviria/RiboV1.4/Alignments/zip.ali.220515.tgz"
    rvmt_path = os.path.join(hmmdb_dir, "zip.ali.220515.tgz")
    fetch_and_extract(
        url=rvmt_url, fetched_to=rvmt_path, extract_to=os.path.join(hmmdb_dir, "RVMT/")
    )
    
    # Use utility function to build HMMs from each subdirectory
    os.makedirs(os.path.join(hmmdb_dir, "RVMT_HMMs"), exist_ok=True)
    current_dir = os.getcwd()  # Save current directory
    
    for ali_folder in os.listdir(os.path.join(hmmdb_dir, "RVMT")):
        ali_folder_path = os.path.join(hmmdb_dir, "RVMT", ali_folder)
        if os.path.isdir(ali_folder_path):
            for alignm_file in os.listdir(ali_folder_path):
                if alignm_file.endswith(".FASTA"):
                    ali_name = "RVMT_" + os.path.splitext(alignm_file)[0].replace("FASTA", "")
                    msa_file_path = os.path.join(ali_folder_path, alignm_file)
                    output_hmm_path = os.path.join(hmmdb_dir, "RVMT_HMMs", f"{ali_name}.hmm")
                    hmm_from_msa(
                        msa_file=msa_file_path,
                        output=output_hmm_path,
                        alphabet="amino",
                        name=ali_name
                    )
    
    # Concatenate all HMMs into a single file
    subprocess.run(f"cat {hmmdb_dir}/RVMT_HMMs/*.hmm > {hmmdb_dir}/RVMT.hmm", shell=True)
    os.chdir(current_dir)  # Restore original directory

    # Build an MMseqs profile DB from the RVMT MSAs (if present)
    try:
        rvmt_msa_dir = os.path.join(hmmdb_dir, "RVMT")
        mmseqs_profile_db_from_directory(
            msa_dir=rvmt_msa_dir,
            output=os.path.join(mmseqs_dbs, "RVMT_mmseqs_db"),
            msa_pattern="*.FASTA",
            info_table=None,
        )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to build mmseqs DB for RVMT MSAs: {e}")

    # PFAM_A_37 RdRps and RTs
    fetch_and_extract(
        url="https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam37.0/Pfam-A.hmm.gz",
        fetched_to="tmp.hmm.gz",
        extract_to="Pfam-A.hmm",
    )
    subprocess.run(
        "echo PF04197.17,PF04196.17,PF22212.1,PF22152.1,PF22260.1,PF00680.25,PF00978.26,PF00998.28,PF02123.21,PF07925.16,PF00078.32,PF07727.19,PF13456.11 >tmp.lst",
        shell=True,
    )
    subprocess.run("sed 's|,|\n|g' tmp.lst -i", shell=True)
    subprocess.run("hmmfetch -f Pfam-A.hmm tmp.lst > rt_rdrp_pfamA37.hmm", shell=True)

    subprocess.run(
        "cat NCBI_ribovirus/proteins/datasets_efetch_refseq_ribovirus_proteins_rmdup.faa RVMT/RVMT_allorfs_filtered_no_chimeras.faa | seqkit rmdup | seqkit seq -w0 > prots_for_masking.faa",
        shell=True,
    )
    logger.info("Finished data preparation")


def prepare_rvmt_mmseqs(data_dir, threads, log_file):
    """Prepare RVMT database for MMseqs2 searches.

    Processes the RVMT (RNA Virus MetaTranscriptomes) database alignments
    and creates formatted databases for MMseqs2 searches.

    Args:
        data_dir (str): Base directory for data storage
        threads (int): Number of CPU threads to use
        log_file (str): Path to write log messages

    Note:
        This function assumes RVMT alignments have been downloaded and
        extracted to the appropriate location in data_dir.
    """

    console.print("Preparing RVMT mmseqs database")
    rvmt_dir = os.path.join(data_dir, "RVMT")
    mmdb_dir = os.path.join(data_dir, "mmdb")
    os.makedirs(rvmt_dir, exist_ok=True)
    os.makedirs(mmdb_dir, exist_ok=True)

    os.chdir(rvmt_dir)

    fetch_and_extract(
        "https://portal.nersc.gov/dna/microbial/prokpubs/Riboviria/RiboV1.4/RiboV1.6_Contigs.fasta.gz",
        fetched_to="tmp.fasta.gz",
        extract_to="RiboV1.6_Contigs.fasta.gz",
    )

    seqkit_command = "seqkit grep --invert-match -f ./chimeras_RVMT.lst RiboV1.6_Contigs.fasta  > tmp_nochimeras.fasta"
    subprocess.run(seqkit_command, shell=True)

    mmseqs_command = (
        "mmseqs createdb tmp_nochimeras.fasta mmdb/RVMT_mmseqs_db2 --dbtype 2"
    )
    subprocess.run(mmseqs_command, shell=True)

    kcompress_command = f"kcompress.sh in=tmp_nochimeras.fasta out=RiboV1.6_Contigs_flat.fasta fuse=2000 k=31 prealloc=true threads={threads}"
    subprocess.run(kcompress_command, shell=True)

    os.chdir(data_dir)


def prepare_rrna_db(data_dir, log_file):
    """Download and prepare ribosomal RNA databases.

    Downloads and processes reference databases for identifying ribosomal RNA
    sequences, including bacterial, archaeal, and eukaryotic rRNAs.

    Args:
        data_dir (str): Base directory for data storage
        log_file (str): Path to write log messages

    Note:
        Creates formatted databases suitable for sequence similarity searches
        and taxonomic classification.
    """

    console.print("Preparing rRNA database")
    rrna_dir = os.path.join(data_dir, "rRNA")
    os.makedirs(rrna_dir, exist_ok=True)
    os.chdir(rrna_dir)

    fetch_and_extract(
        "https://www.arb-silva.de/fileadmin/silva_databases/release_138_2/Exports/SILVA_138.2_SSURef_NR99_tax_silva.fasta.gz",
        fetched_to="tmp.fasta.gz",
        extract_to="SILVA_138.2_SSURef_NR99_tax_silva.fasta",
    )
    fetch_and_extract(
        "https://www.arb-silva.de/fileadmin/silva_databases/release_138_2/Exports/SILVA_138.2_LSURef_NR99_tax_silva.fasta.gz",
        fetched_to="tmp.fasta.gz",
        extract_to="SILVA_138.2_LSURef_NR99_tax_silva.fasta",
    )

    subprocess.run("cat SILVA*fasta > merged.fas", shell=True)

    bbduk_command = "bbduk.sh -Xmx1g in=merged.fas out=SILVA_138_merged_masked.fa zl=9 entropy=0.6 entropyk=4 entropywindow=24 maskentropy"
    subprocess.run(bbduk_command, shell=True)

    # now for NCBI refseq.
    fetch_and_extract(
        "https://ftp.ncbi.nlm.nih.gov/blast/db/16S_ribosomal_RNA.tar.gz",
        fetched_to="tmp.fasta.gz",
        extract_to="NCBI_16S_ribosomal_RNA.fasta",
    )    
    fetch_and_extract(
        "https://ftp.ncbi.nlm.nih.gov/blast/db/18S_fungal_sequences.tar.gz",
        fetched_to="tmp.fasta.gz",
        extract_to="NCBI_18S_ribosomal_RNA.fasta",
    )    
    fetch_and_extract(
        "https://ftp.ncbi.nlm.nih.gov/blast/db/28S_fungal_sequences.tar.gz",
        fetched_to="tmp.fasta.gz",
        extract_to="28S_fungal_sequences.fasta",
    )    
    fetch_and_extract(
        "https://ftp.ncbi.nlm.nih.gov/blast/db/16S_ribosomal_RNA.tar.gz",
        fetched_to="tmp.fasta.gz",
        extract_to="NCBI_16S_ribosomal_RNA.fasta",
    )    
    
#     16S_ribosomal_RNA-nucl-metadata.json       2025-08-26 05:36  468   
# 16S_ribosomal_RNA.tar.gz                   2025-08-26 05:36   64M  
# 16S_ribosomal_RNA.tar.gz.md5               2025-08-26 05:36   59   
# 18S_fungal_sequences-nucl-metadata.json    2025-08-28 05:36  489   
# 18S_fungal_sequences.tar.gz                2025-08-28 05:36   58M  
# 18S_fungal_sequences.tar.gz.md5            2025-08-28 05:36   62   
# 28S_fungal_sequences-nucl-metadata.json    2025-08-28 05:36  491   
# 28S_fungal_sequences.tar.gz                2025-08-28 05:36   60M  
# 28S_fungal_sequences.tar.gz.md5  




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
    rfam_cm_path = data_dir / "Rfam.cm.gz"
    rfam_extract_path = data_dir / "Rfam.cm"
    subprocess.run("cmpress Rfam.cm", shell=True)

    logger.info("Downloading Rfam database    ")
    try:
        fetch_and_extract(
            rfam_url, fetched_to=str(rfam_cm_path), extract_to=str(rfam_extract_path)
        )
        logger.info("Rfam database downloaded and extracted successfully.")
        rfam_cm_path.unlink()  # Remove the .gz file after extraction
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading Rfam database: {e}")
    except Exception as e:
        logger.error(f"Error processing Rfam database: {e}")


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
        f_out.write(f"RolyPoly version: {version}\n")
        f_out.write(f"Date: {datetime.datetime.now()}\n")
        f_out.write(f"Data dir: {data_dir}\n")
        f_out.write(
            "for more details see: https://pages.jgi.doe.gov/rolypoly/docs/\n"
        )
        f_out.write("Changes in this version: \n")
        f_out.write(" - Removed eukaryotic RdRp Pfam (see d1a0f1b3e2452253a4d47e20b81ac71652ccb944) \n")
        f_out.write("Software / DBs used in the creation of this data: \n")
        tools.append("RolyPoly")
        tools.append("seqkit")
        tools.append("bbmap")
        tools.append("mmseqs2")
        tools.append("mmseqs")
        tools.append("hmmer")
        tools.append("pyhmmer")
        tools.append("datasets")
        tools.append("eutils")
        tools.append("silva")
        tools.append("Rfam")
        tools.append("rvmt")
        tools.append("rdrp-scan")
        tools.append("neordrp_v2.1")
        tools.append("tsa_2018")
        tools.append("pfam_a_37")
        tools.append("refseq")
        f_out.write(remind_citations(tools, return_as_text=True) or "")

    tar_command = f"tar --use-compress-program='pigz -p 8 --best' -cf rpdb.tar.gz {data_dir}"  # threads

    subprocess.run(tar_command, shell=True)

    # # On NERSC
    # scp uneri@xfer.jgi.lbl.gov:/REDACTED_HPC_PATH/projects/data2/data.tar.gz /REDACTED_NERSC_PATH/prokpubs/www/rolypoly/data/
    # chmod +777 -R /REDACTED_NERSC_PATH/prokpubs/www/rolypoly/data/

    # upload_command = f"gsutil cp {data_dir}.tar.gz gs://rolypoly-data/"
    # subprocess.run(upload_command, shell=True)


def prepare_genomad_rna_viral_markers(data_dir, threads, logger: logging.Logger):
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
    genomad_mmseqs_dir = os.path.join(genomad_markers_dir, "mmseqs_dbs")
    os.makedirs(genomad_dir, exist_ok=True)
    os.makedirs(genomad_db_dir, exist_ok=True)
    os.makedirs(genomad_markers_dir, exist_ok=True)
    os.makedirs(genomad_alignments_dir, exist_ok=True)

    # Download metadata and database
    genomad_data = "https://zenodo.org/api/records/14886553/files-archive" # noqa
    db_url = (
        "https://zenodo.org/records/14886553/files/genomad_msa_v1.9.tar.gz?download=1"
    )
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
    metadata_df = metadata_df.filter(pl.col("ANNOTATION_DESCRIPTION").str.to_lowercase().str.contains("rna-dependent rna polymerase") |
                                    pl.col("TAXONOMY").str.contains("Riboviria") | 
                                    pl.col("SOURCE").str.contains("RVMT")
                                    )
    # Next, filling missing annotation from InterPro.
    # only ones without description
    to_fill = metadata_df.filter(
        pl.col("ANNOTATION_DESCRIPTION").is_null()
    )
    # if multiple maybe split->explode->groupby->agg->majority vote, something like:
    # nah just using the first if multiple maybe split->explode->first:
    # for now, only using a single accession (but word cloud/majority vote would probably work for multiple accessions)
    # to_fill = to_fill.with_columns(pl.col("ANNOTATION_ACCESSIONS").str.split(";").list.first().alias("ANNOTATION_ACCESSIONS_first"))
    to_fill = to_fill.with_columns(pl.col("ANNOTATION_ACCESSIONS").str.split(";").alias("ANNOTATION_ACCESSIONS_struct"))
    to_fill = to_fill.explode("ANNOTATION_ACCESSIONS_struct")

    to_fill = to_fill.with_columns(
        pl.when(pl.col("ANNOTATION_ACCESSIONS").str.starts_with("PF")).then(pl.lit("Pfam"))
        .when(pl.col("ANNOTATION_ACCESSIONS").str.starts_with("COG")).then(pl.lit("COG"))
        .when(pl.col("ANNOTATION_ACCESSIONS").str.starts_with("K")).then(pl.lit("KEGG"))
        .otherwise(pl.lit("unknown")).alias("source_db"))

    # We (currently) only carte about viral specific markers, so filtering out the rest
    # Not sure Kegg is on interpro.
    to_fill = to_fill.filter(pl.col("source_db").str.contains("Pfam|COG"))
    
    def query_interpro(entry: str, source_db: str) :
        """Fetch the InterPro description for a given entry.
        """
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
        desc = data.get("metadata", {}).get("name",{}).get("name",None)
        return desc

    filled_interpro = []
    from tqdm import tqdm
    tiny_fill = to_fill.select(["ANNOTATION_ACCESSIONS_struct","source_db"]).unique()
    
    for row in tqdm(tiny_fill.to_dicts()):
        if row["source_db"] == "unknown":
            filled_interpro.append(None)
        else:
            this_desc =  query_interpro(row["ANNOTATION_ACCESSIONS_struct"], row["source_db"])
            filled_interpro.append(this_desc)
            print(f"{row['ANNOTATION_ACCESSIONS_struct']}\t{this_desc}")  # debugging
            # filled_interpro.append(query_interpro(row["ANNOTATION_ACCESSIONS"], row["source_db"]))

    tiny_fill = tiny_fill.with_columns(pl.Series(filled_interpro).alias("interpro"))
    to_fill = to_fill.join(tiny_fill, on = ["ANNOTATION_ACCESSIONS_struct","source_db"], how="left")
    to_fill = to_fill.with_columns(
    pl.coalesce(
                pl.col("ANNOTATION_DESCRIPTION"),
                pl.col("interpro")).alias("ANNOTATION_DESCRIPTION"))
    to_fill = to_fill.drop("ANNOTATION_ACCESSIONS_struct","interpro","source_db").unique()
    to_fill = to_fill.filter(pl.col("ANNOTATION_DESCRIPTION").is_not_null()) 
    # hopefully now we have filled some of the missing descriptions, and we don't have any duplicate MARKERs

    metadata_df = metadata_df.filter(~pl.col("MARKER").is_in(to_fill["MARKER"].implode()))
    metadata_df = metadata_df.vstack(to_fill)

    metadata_df.write_csv(
            f"{genomad_dir}/rna_viral_markers_with_annotation.csv"
    )
    

    # Download MSAs
    logger.info("Downloading geNomad database")
    aria2c_command = f"aria2c -c -d {genomad_dir} -o ./genomad_msa_v1.9.tar.gz {db_url}"
    subprocess.run(aria2c_command, shell=True)

    # Extract RNA viral MSAs
    marker_ids = metadata_df["MARKER"].to_list()

    with tarfile.open(f"{genomad_dir}/genomad_msa_v1.9.tar.gz", "r") as tar:
        for member in tar.getmembers():
            if (
                member.name.removeprefix("genomad_msa_v1.9/").removesuffix(".faa")
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
        os.path.join(data_dir, "profiles/hmmdbs"), "genomad_rna_viral_markers.hmm"
    )
    hmmdb_from_directory(
        msa_dir=genomad_alignments_dir,
        output=output_hmm,
        msa_pattern="*.faa",
        info_table=f"{genomad_dir}/rna_viral_markers_with_annotation.csv",
        name_col="MARKER",
        accs_col="ANNOTATION_ACCESSIONS",
        desc_col="ANNOTATION_DESCRIPTION",
        gath_col=None # no gathering theshold pre-defined for genomad
    )

    mmseqs_profile_db_from_directory(
        msa_dir =genomad_alignments_dir,
        output = os.path.join(data_dir, "profiles/mmseqs_dbs/genomad/", "genomad_rna_viral_markers"),
        info_table=f"{genomad_dir}/rna_viral_markers_with_annotation.csv",
        msa_pattern="*.faa",
        name_col="MARKER",
        accs_col="ANNOTATION_ACCESSIONS",
        desc_col="ANNOTATION_DESCRIPTION",
    )
    # clean up
    try: 
        os.remove(f"{genomad_dir}/genomad_metadata_v1.9.tsv.gz")
    except Exception as e:
        logger.warning(f"Could not remove file: {e}")
    

    logger.info(f"Created RNA viral HMM database at {output_hmm}")

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
    rdrp_scan_msa_dir = os.path.join(hmmdb_dir, "RdRp-scan/RdRp-scan-main/Profile_db_and_alignments")
    rdrp_scan_output = os.path.join(hmmdb_dir, "RdRp-scan.hmm")
    hmmdb_from_directory(
        msa_dir=rdrp_scan_msa_dir,
        output=rdrp_scan_output,
        msa_pattern="*.fasta.CLUSTALO"
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
    

    logger.debug("Finished preparing rdrp-scan databases")

    # Create directories
    genomad_dir = os.path.join(data_dir, "profiles/genomad")
    genomad_db_dir = os.path.join(genomad_dir, "genomad_db")
    genomad_markers_dir = os.path.join(genomad_db_dir, "markers")
    genomad_alignments_dir = os.path.join(genomad_markers_dir, "alignments")
    genomad_mmseqs_dir = os.path.join(genomad_markers_dir, "mmseqs_dbs")
    os.makedirs(genomad_dir, exist_ok=True)
    os.makedirs(genomad_db_dir, exist_ok=True)
    os.makedirs(genomad_markers_dir, exist_ok=True)
    os.makedirs(genomad_alignments_dir, exist_ok=True)

    # Download metadata and database
    genomad_data = "https://zenodo.org/api/records/14886553/files-archive" # noqa
    db_url = (
        "https://zenodo.org/records/14886553/files/genomad_msa_v1.9.tar.gz?download=1"
    )
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
    metadata_df = metadata_df.filter(pl.col("ANNOTATION_DESCRIPTION").str.to_lowercase().str.contains("rna-dependent rna polymerase") |
                                    pl.col("TAXONOMY").str.contains("Riboviria") | 
                                    pl.col("SOURCE").str.contains("RVMT")
                                    )
    # Next, filling missing annotation from InterPro.
    # only ones without description
    to_fill = metadata_df.filter(
        pl.col("ANNOTATION_DESCRIPTION").is_null()
    )
    # if multiple maybe split->explode->groupby->agg->majority vote, something like:
    # nah just using the first if multiple maybe split->explode->first:
    # for now, only using a single accession (but word cloud/majority vote would probably work for multiple accessions)
    # to_fill = to_fill.with_columns(pl.col("ANNOTATION_ACCESSIONS").str.split(";").list.first().alias("ANNOTATION_ACCESSIONS_first"))
    to_fill = to_fill.with_columns(pl.col("ANNOTATION_ACCESSIONS").str.split(";").alias("ANNOTATION_ACCESSIONS_struct"))
    to_fill = to_fill.explode("ANNOTATION_ACCESSIONS_struct")

    to_fill = to_fill.with_columns(
        pl.when(pl.col("ANNOTATION_ACCESSIONS").str.starts_with("PF")).then(pl.lit("Pfam"))
        .when(pl.col("ANNOTATION_ACCESSIONS").str.starts_with("COG")).then(pl.lit("COG"))
        .when(pl.col("ANNOTATION_ACCESSIONS").str.starts_with("K")).then(pl.lit("KEGG"))
        .otherwise(pl.lit("unknown")).alias("source_db"))

    # We (currently) only carte about viral specific markers, so filtering out the rest
    # Not sure Kegg is on interpro.
    to_fill = to_fill.filter(pl.col("source_db").str.contains("Pfam|COG"))
    
    def query_interpro(entry: str, source_db: str) :
        """Fetch the InterPro description for a given entry.
        """
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
        desc = data.get("metadata", {}).get("name",{}).get("name",None)
        return desc

    filled_interpro = []
    from tqdm import tqdm
    tiny_fill = to_fill.select(["ANNOTATION_ACCESSIONS_struct","source_db"]).unique()
    
    for row in tqdm(tiny_fill.to_dicts()):
        if row["source_db"] == "unknown":
            filled_interpro.append(None)
        else:
            this_desc =  query_interpro(row["ANNOTATION_ACCESSIONS_struct"], row["source_db"])
            filled_interpro.append(this_desc)
            print(f"{row['ANNOTATION_ACCESSIONS_struct']}\t{this_desc}")  # debugging
            # filled_interpro.append(query_interpro(row["ANNOTATION_ACCESSIONS"], row["source_db"]))

    tiny_fill = tiny_fill.with_columns(pl.Series(filled_interpro).alias("interpro"))
    to_fill = to_fill.join(tiny_fill, on = ["ANNOTATION_ACCESSIONS_struct","source_db"], how="left")
    to_fill = to_fill.with_columns(
    pl.coalesce(
                pl.col("ANNOTATION_DESCRIPTION"),
                pl.col("interpro")).alias("ANNOTATION_DESCRIPTION"))
    to_fill = to_fill.drop("ANNOTATION_ACCESSIONS_struct","interpro","source_db").unique()
    to_fill = to_fill.filter(pl.col("ANNOTATION_DESCRIPTION").is_not_null()) 
    # hopefully now we have filled some of the missing descriptions, and we don't have any duplicate MARKERs

    metadata_df = metadata_df.filter(~pl.col("MARKER").is_in(to_fill["MARKER"].implode()))
    metadata_df = metadata_df.vstack(to_fill)

    metadata_df.write_csv(
            f"{genomad_dir}/rna_viral_markers_with_annotation.csv"
    )
    

    # Download MSAs
    logger.info("Downloading geNomad database")
    aria2c_command = f"aria2c -c -d {genomad_dir} -o ./genomad_msa_v1.9.tar.gz {db_url}"
    subprocess.run(aria2c_command, shell=True)

    # Extract RNA viral MSAs
    marker_ids = metadata_df["MARKER"].to_list()

    with tarfile.open(f"{genomad_dir}/genomad_msa_v1.9.tar.gz", "r") as tar:
        for member in tar.getmembers():
            if (
                member.name.removeprefix("genomad_msa_v1.9/").removesuffix(".faa")
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
        os.path.join(data_dir, "profiles/hmmdbs"), "genomad_rna_viral_markers.hmm"
    )
    hmmdb_from_directory(
        genomad_alignments_dir,
        output_hmm,
        msa_pattern="*.faa",
        info_table=f"{genomad_dir}/rna_viral_markers_with_annotation.csv",
        name_col="MARKER",
        accs_col="ANNOTATION_ACCESSIONS",
        desc_col="ANNOTATION_DESCRIPTION",
        gath_col=None # no gathering theshold pre-defined for genomad
    )

    mmseqs_profile_db_from_directory(
        msa_dir =genomad_alignments_dir,
        output = os.path.join(data_dir, "profiles/mmseqs_dbs/genomad/", "genomad_rna_viral_markers"),
        info_table=f"{genomad_dir}/rna_viral_markers_with_annotation.csv",
        msa_pattern="*.faa",
        name_col="MARKER",
        accs_col="ANNOTATION_ACCESSIONS",
        desc_col="ANNOTATION_DESCRIPTION",
    )

    logger.info(f"Created RNA viral HMM database at {output_hmm}")


def prepare_vfam(data_dir, logger: logging.Logger):
    """Prepare VFAM HMM database."""
    # fetch_and_extract(
    #     url="https://fileshare.csb.univie.ac.at/vog/latest/vfam.hmm.tar.gz",
    #     fetched_to=os.path.join(data_dir, "profiles","hmmdbs", "vfam","vfam.tar.gz"),
    #     extract_to=os.path.join(data_dir, "profiles","hmmdbs", "vfam"),
    # )
    os.makedirs(os.path.join(data_dir, "profiles","hmmdbs", "vfam"), exist_ok=True)
    vfam_df = pl.read_csv("https://fileshare.lisc.univie.ac.at//vog/latest/vfam.annotations.tsv.gz", separator="\t")
    # In [54]: vfam_df.shape
    # Out[54]: (39624, 5)
    # In [56]: vfam_df.collect_schema()
    # Out[56]: 
    # Schema([('#GroupName', String),
    #     ('ProteinCount', Int64),
    #     ('SpeciesCount', Int64),
    #     ('FunctionalCategory', String),
    #     ('ConsensusFunctionalDescription', String)])

    vfam_df.write_csv(os.path.join(data_dir, "profiles","hmmdbs", "vfam", "vfam.annotations.tsv.gz"))
    version = requests.get("https://fileshare.lisc.univie.ac.at/vog/latest/release.txt").text.strip()
    logger.info(f"VFAM version: {version}")
    
    # the raw MSAs
    fetch_and_extract(
        url="https://fileshare.lisc.univie.ac.at//vog/latest/vfam.raw_algs.tar.gz",
        fetched_to=os.path.join(data_dir, "profiles","hmmdbs", "vfam","vfam.raw_algs.tar.gz"),
        extract_to=os.path.join(data_dir, "profiles","hmmdbs", "vfam"),
    )

    output_hmm = os.path.join(
            os.path.join(data_dir, "profiles/hmmdbs"), "vfam.hmm"
        )
    hmmdb_from_directory(
            os.path.join(data_dir, "profiles","hmmdbs", "vfam","msa"),
            output_hmm,
            msa_pattern="*.msa",
            info_table=(os.path.join(data_dir, "profiles","hmmdbs", "vfam", "vfam.annotations.tsv.gz")),
            name_col="#GroupName",
            accs_col="#GroupName",
            desc_col="ConsensusFunctionalDescription",
            gath_col=None # no gathering theshold pre-defined 
        )
    
    # Build the mmseqs database
    mmseqs_commands = [f"mmseqs createdb tmp_nochimeras.fasta mmdb/RVMT_mmseqs_db2 --dbtype 2",
        f"mmseqs createdb tmp_nochimeras.fasta mmdb/RVMT_mmseqs_db2 --dbtype 2"    ]
    subprocess.run(mmseqs_commands, shell=True)



    # remove the vfam msa directory
    shutil.rmtree(os.path.join(data_dir, "profiles","hmmdbs", "vfam","msa"))
    logger.info(f"Created VFAM HMM database at {os.path.join(data_dir, 'hmmdbs', 'vfam.hmm')}")

# setup taxonkit
# TODO: CONVERT TO PYTHON
# cd "$DATA_PATH"
# aria2c http://ftp.ncbi.nih.gov/pub/taxonomy/taxdump.tar.gz
# tar -xzvf taxdump.tar.gz
# mv names.dmp nodes.dmp merged.dmp delnodes.dmp "$DATA_PATH/taxdump/"
# rm -rf  taxdump.tar.gz


if __name__ == "__main__":
    build_data()


# source ~/.bashrc
# conda activate crispy
# export PATH=$PATH:<HOME_PATH>/code/mmseqs/bin/

# THREADS=24

# ## Prepare NCBI RNA virus ####
# cd $rolypoly_dir/data/
# mkdir NCBI_ribovirus
# cd NCBI_ribovirus
# taxid="2559587"
# # Perform the search and download the genomes
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
# RVMTdb=/REDACTED_HPC_PATH/rolypoly/data/RVMT/mmdb/RVMT_mmseqs_db2
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
# fetched_genomes /REDACTED_HPC_PATH/rolypoly/bench/test_sampled_005_bb_metaTs_spiced_RVMT/temp_dir_sampled_005_bb_metaTs_spiced_RVMT/stats_rRNA_filt_sampled_005_bb_metaTs_spiced_RVMT.txt output.fasta
# input_file=/REDACTED_HPC_PATH/rolypoly/data/output.fasta
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

# ##### Create AMR DBs #####
# mkdir dbs
# mkdir dbs/NCBI_pathogen_AMR
# cd dbs/NCBI_pathogen_AMR
# wget https://ftp.ncbi.nlm.nih.gov/pathogen/Antimicrobial_resistance/Data/2024-05-02.2/ReferenceGeneCatalog.txt
# awk -v RS='\t' '/refseq_nucleotide_accession/{print NR; exit}' ReferenceGeneCatalog.txt
# # 11
# # awk -F'\t' -v ORS=" " '{print $11}' ReferenceGeneCatalog.txt |sed 's|genbank_nucleotide_accession||g' > genbank_nucleotide_accessions.lst
# awk -F'\t'  '{print $11}' ReferenceGeneCatalog.txt |sed 's|genbank_nucleotide_accession||g' > genbank_nucleotide_accessions.lst

# datasets download gene accession $(cat genbank_nucleotide_accessions.lst)   --filename AMR_genes.zip
# echo "CXAL01000043.1 CXAL01000043.1 CXAL01000043.1 CXAL01000043.1 CXAL01000043.1 CXAL01000043.1 CXAL01000043.1 CXAL01000043.1 CXAL01000043.1 CXAL01000043.1 CP001050.1 CP001050.1 CP001050.1 CP001050.1 CP001050.1 CP030821.1 CP030821.1 CP030821.1 CP030821.1 CP030821.1 CP030821.1 BX571856.1 BX571856.1 BX571856.1 BX571856.1 BX571856.1 BX571856.1 " > small_file.txt
# datasets download gene accession  $(cat small_file.txt)  --filename AMR_genes.zip
# datasets download gene accession NG_048523
# datasets download gene accession CXAL01000043.1 CXAL01000043.1
#   # Read each taxon name from the file and fetch the corresponding genome data (zip from ncbi)
#   while IFS= read -r line;
#   do
#       echo "Processing $line    "
#       datasets download gene accession "${line}"  --filename "${line}"_fetched_genomes.zip
#   done < genbank_nucleotide_accessions.lst

# # cd2mec
# # cd dbs
# # cd nt
# # aws s3 cp --no-sign-request s3://ncbi-blast-databases/2024-06-01-01-05-03/ ./ --recursive --exclude "*" --include "nt.*"
