import os
from pathlib import Path as pt

from rich.console import Console
from rich_click import command, option

from rolypoly.utils.various import (
    extract,
    fetch_and_extract,
    find_most_recent_folder,
    move_contents_to_parent,
)

console = Console()
global tools
tools = []


@command()
@option(
    "--try-hard",
    default=False,
    help="""Instead of downloading the data from a public FTP (TBD also zenodo), try to recreate all the databases and files from scratch (this will include some downloading though    ) \n
        In theory, this might get your more updated reference sequences than those I fetched on 07/08/2024. """,
)
@option(
    "--ROLYPOLY_DATA",
    required=False,
    help="If you do not want to download the the data to same location as the rolypoly code, specify an alternative path. TODO: remind user to provide such alt path in other scripts? envirometnal variable maybe",
)
@option("--threads", default=4, help="Number of threads to use")
@option(
    "--log-file",
    default=f"./prepare_external_data_logfile.txt",
    help="Path to the log file",
)
def prepare_external_data(try_hard, rolypoly_data, threads, log_file):
    """Download or build external data required for RolyPoly.

    This command either downloads pre-built databases and reference data from
    a public repository, or builds them from scratch using the latest source data.

    Args:
        try_hard (bool): If True, build databases from scratch instead of downloading
            pre-built versions. May result in more up-to-date references.
        rolypoly_data (str, optional): Alternative directory to store data. If None,
            uses the default RolyPoly data directory.
        threads (int, optional): Number of CPU threads to use.
        log_file (str, optional): Path to write log messages. Defaults to
            "./prepare_external_data_logfile.txt".

    Note:
        When try_hard=True, this will:
        1. Build RdRp-scan HMM profiles from MSAs
        2. Process RVMT alignments and create HMMs
        3. Download and prepare rRNA databases
        4. Download and process Rfam data

    """
    import json
    import subprocess
    from importlib import resources

    import requests

    from rolypoly.utils.loggit import setup_logging

    if rolypoly_data == None:
        ROLYPOLY_DATA = pt(resources.files("rolypoly")) / "data"
    else:
        ROLYPOLY_DATA = pt(os.path.abspath(rolypoly_data))

    config_path = pt(resources.files("rolypoly")) / "rpconfig.json"

    if config_path.exists():
        with config_path.open("r") as f:
            config = json.load(f)
    else:
        config = {"ROLYPOLY_DATA": ""}

    config["ROLYPOLY_DATA"] = str(ROLYPOLY_DATA)
    os.environ["ROLYPOLY_DATA"] = str(ROLYPOLY_DATA)
    print(os.environ["ROLYPOLY_DATA"])
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    logger = setup_logging(log_file)
    logger.info(f"Starting data preparation to : {ROLYPOLY_DATA}")

    ROLYPOLY_DATA.mkdir(exist_ok=True)

    if try_hard == False:
        response = requests.get(
            "https://portal.nersc.gov/dna/microbial/prokpubs/rolypoly/data/data.tar.gz",
            stream=True,
        )
        with open(f"{ROLYPOLY_DATA}/data.tar.gz", "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        extract(
            archive_path=f"{ROLYPOLY_DATA}/data.tar.gz", extract_to=f"{ROLYPOLY_DATA}"
        )
        most_recent_folder = find_most_recent_folder(f"{ROLYPOLY_DATA}")
        move_contents_to_parent(most_recent_folder)
        os.remove(f"{ROLYPOLY_DATA}/data.tar.gz")
        logger.info(f"Finished fetching and extracting data to : {ROLYPOLY_DATA}")
        return 0

    # If try_hard is True, prepare databases from scratch
    hmmdb_dir = os.path.join(ROLYPOLY_DATA, "hmmdbs")
    os.makedirs(hmmdb_dir, exist_ok=True)

    # Add geNomad RNA viral HMMs preparation
    prepare_genomad_rna_viral_hmms(ROLYPOLY_DATA, threads, logger)

    # RdRp-scan
    fetch_and_extract(
        "https://github.com/JustineCharon/RdRp-scan/archive/refs/heads/main.zip",
        fetched_to=hmmdb_dir + "/RdRp-scan.zip",
        extract_to=hmmdb_dir + "/RdRp-scan",
    )
    subprocess.run("mkdir " + hmmdb_dir + "/RdRp-scan_HMMs", shell=True)
    import pyhmmer

    alphabet = pyhmmer.easel.Alphabet.amino()
    for alignm_file in os.listdir(
        hmmdb_dir + "/RdRp-scan/RdRp-scan-main/Profile_db_and_alignments"
    ):
        if alignm_file.endswith(".fasta.CLUSTALO"):
            with pyhmmer.easel.MSAFile(
                hmmdb_dir
                + "/RdRp-scan/RdRp-scan-main/Profile_db_and_alignments/"
                + alignm_file,
                digital=True,
                alphabet=alphabet,
            ) as msa_file:
                msa = msa_file.read()
                ali_name = "RdRp-scan_" + alignm_file.replace(".fasta.CLUSTALO", "")
                msa.name = bytes(ali_name.encode())
                builder = pyhmmer.plan7.Builder(alphabet)
                background = pyhmmer.plan7.Background(alphabet)
                hmm, _, _ = builder.build_msa(msa, background)
                with open(
                    hmmdb_dir + f"/RdRp-scan_HMMs/{ali_name}.hmm", "wb"
                ) as output_file:
                    hmm.write(output_file)

    subprocess.run(
        f"cat  {hmmdb_dir}/RdRp-scan_HMMs/*.hmm > {hmmdb_dir}/RdRp-scan.hmm ",
        shell=True,
    )

    # RVMT
    rvmt_url = "https://portal.nersc.gov/dna/microbial/prokpubs/Riboviria/RiboV1.4/Alignments/zip.ali.220515.tgz"
    rvmt_path = os.path.join(hmmdb_dir, "zip.ali.220515.tgz")
    fetch_and_extract(
        url=rvmt_url, fetched_to=rvmt_path, extract_to=os.path.join(hmmdb_dir, "RVMT/")
    )
    os.makedirs(os.path.join(hmmdb_dir, "RVMT_HMMs"), exist_ok=True)
    for ali_folder in os.listdir(os.path.join(hmmdb_dir, "RVMT")):
        ali_folder_path = os.path.join(hmmdb_dir, "RVMT", ali_folder)
        if os.path.isdir(ali_folder_path):
            os.chdir(ali_folder_path)
            for alignm_file in os.listdir("."):
                if alignm_file.endswith(".FASTA"):
                    ali_name = os.path.splitext(alignm_file)[0]
                    with pyhmmer.easel.MSAFile(
                        alignm_file, digital=True, alphabet=alphabet
                    ) as msa_file:
                        msa = msa_file.read()
                        ali_name = "RVMT_" + ali_name.replace("FASTA", "")
                        msa.name = bytes(ali_name.encode())
                        builder = pyhmmer.plan7.Builder(alphabet)
                        background = pyhmmer.plan7.Background(alphabet)
                        hmm, _, _ = builder.build_msa(msa, background)
                        with open(f"../RVMT_HMMs/{ali_name}.hmm", "wb") as output_file:
                            hmm.write(output_file)
    os.chdir(os.path.join(hmmdb_dir, "RVMT_HMMs"))
    subprocess.run(f"cat ./*hmm > ../RVMT.hmm", shell=True)
    os.chdir("../../")

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


def prepare_rvmt_mmseqs(ROLYPOLY_DATA, threads, log_file):
    """Prepare RVMT database for MMseqs2 searches.

    Processes the RVMT (RNA Virus MetaTranscriptomes) database alignments
    and creates formatted databases for MMseqs2 searches.

    Args:
        ROLYPOLY_DATA (str): Base directory for data storage
        threads (int): Number of CPU threads to use
        log_file (str): Path to write log messages

    Note:
        This function assumes RVMT alignments have been downloaded and
        extracted to the appropriate location in ROLYPOLY_DATA.
    """
    import subprocess

    console.print("Preparing RVMT mmseqs database")
    rvmt_dir = os.path.join(ROLYPOLY_DATA, "RVMT")
    mmdb_dir = os.path.join(ROLYPOLY_DATA, "mmdb")
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

    os.chdir(ROLYPOLY_DATA)


def prepare_rrna_db(ROLYPOLY_DATA, log_file):
    """Download and prepare ribosomal RNA databases.

    Downloads and processes reference databases for identifying ribosomal RNA
    sequences, including bacterial, archaeal, and eukaryotic rRNAs.

    Args:
        ROLYPOLY_DATA (str): Base directory for data storage
        log_file (str): Path to write log messages

    Note:
        Creates formatted databases suitable for sequence similarity searches
        and taxonomic classification.
    """
    import subprocess

    console.print("Preparing rRNA database")
    rrna_dir = os.path.join(ROLYPOLY_DATA, "rRNA")
    os.makedirs(rrna_dir, exist_ok=True)
    os.chdir(rrna_dir)

    fetch_and_extract(
        "https://www.arb-silva.de/fileadmin/silva_databases/release_138_1/Exports/SILVA_138.1_SSURef_NR99_tax_silva.fasta.gz",
        fetched_to="tmp.fasta.gz",
        extract_to="SILVA_138.1_SSURef_NR99_tax_silva.fasta",
    )
    fetch_and_extract(
        "https://www.arb-silva.de/fileadmin/silva_databases/release_138_1/Exports/SILVA_138.1_LSURef_NR99_tax_silva.fasta.gz",
        fetched_to="tmp.fasta.gz",
        extract_to="SILVA_138.1_LSURef_NR99_tax_silva.fasta",
    )

    subprocess.run("cat SILVA*fasta > merged.fas", shell=True)

    bbduk_command = "bbduk.sh -Xmx1g in=merged.fas out=SILVA_138_merged_masked.fa zl=9 entropy=0.6 entropyk=4 entropywindow=24 maskentropy"
    subprocess.run(bbduk_command, shell=True)


# From Antonio https://github.com/apcamargo/diversify_pfam/blob/main/scripts/generate_hmms.py
def create_hmm_from_msa(msa, alphabet, set_ga):
    """Create a Hidden Markov Model from a multiple sequence alignment.

    Uses pyhmmer to build an HMM profile from an input MSA, with options
    for setting gathering thresholds.

    Args:
        msa (pyhmmer.easel.MSA): Input multiple sequence alignment
        alphabet (pyhmmer.easel.Alphabet): Sequence alphabet (amino/nucleic)
        set_ga (bool): Whether to set gathering thresholds

    Returns:
        pyhmmer.plan7.HMM: Built HMM profile

    Note:
        The function handles both protein and nucleotide alignments based
        on the provided alphabet.
    """
    import pyhmmer

    builder = pyhmmer.plan7.Builder(alphabet)
    background = pyhmmer.plan7.Background(alphabet)
    hmm, _, _ = builder.build_msa(msa, background)
    hmm.command_line = None
    if set_ga:
        hmm.cutoffs.gathering = set_ga, set_ga
    return hmm


def generate_hmms(input_msas, input_format, set_ga):
    """Generate multiple HMM profiles from a collection of MSAs.

    Batch processes multiple sequence alignments to create corresponding
    HMM profiles using pyhmmer.

    Args:
        input_msas (list): List of MSA file paths
        input_format (str): Format of input MSAs (e.g., "stockholm", "fasta")
        set_ga (bool): Whether to set gathering thresholds

    Returns:
        list: List of generated HMM profiles

    Example:
             hmms = generate_hmms(["msa1.sto", "msa2.sto"], "stockholm", True)
    """
    import pyhmmer

    alphabet = pyhmmer.easel.Alphabet.amino()
    hmm_list = []
    for p in input_msas:
        with pyhmmer.easel.MSAFile(
            p, digital=True, alphabet=alphabet, format=input_format
        ) as fi:
            msa = fi.read()
            msa.name = p.stem.encode("utf-8")  # type: ignore
            msa.accession = p.stem.encode("utf-8")  # type: ignore
            hmm = create_hmm_from_msa(msa, alphabet, set_ga)
            hmm_list.append(hmm)
    return hmm_list


def write_hmms(hmms, output_hmm, write_ascii):
    """Write HMM profiles to a file.

    Saves one or more HMM profiles to a single output file in either
    binary or ASCII format.

    Args:
        hmms (list): List of HMM profiles to write
        output_hmm (str): Path to output HMM file
        write_ascii (bool): If True, write in ASCII format; otherwise binary
    """
    binary = not write_ascii
    with open(output_hmm, "wb") as fo:
        for hmm in hmms:
            hmm.write(fo, binary=binary)


def download_and_extract_rfam(ROLYPOLY_DATA, logger):
    """Download and process Rfam database files.

    Retrieves Rfam database files and processes them for use in RNA
    family identification and annotation.

    Args:
        ROLYPOLY_DATA (str): Base directory for data storage
        logger: Logger object for recording progress and errors

    Note:
        Downloads both the sequence database and covariance models,
        and processes them for use with Infernal.
    """
    import subprocess

    import requests

    rfam_url = "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.cm.gz"
    rfam_cm_path = ROLYPOLY_DATA / "Rfam.cm.gz"
    rfam_extract_path = ROLYPOLY_DATA / "Rfam.cm"
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


def tar_everything_and_upload_to_NERSC(ROLYPOLY_DATA, version=""):
    """Package and upload prepared data to NERSC.

    Creates a tarball of all prepared databases and reference data,
    then uploads it to NERSC for distribution.

    Args:
        ROLYPOLY_DATA (str): Directory containing data to package
        version (str, optional): Version string to append to archive name.

    Note:
        Requires appropriate NERSC credentials and permissions to upload.
    """
    import datetime
    import subprocess
    from pathlib import Path
    from rolypoly.utils.citation_reminder import remind_citations

    if version == "":
        from rolypoly.utils.loggit import get_version_info

        version = get_version_info()
    with open(Path(ROLYPOLY_DATA) / "README.md", "w") as f_out:
        f_out.write(f"RolyPoly version: {version}\n")
        f_out.write(f"Date: {datetime.datetime.now()}\n")
        f_out.write(f"Data dir: {ROLYPOLY_DATA}\n")
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
        f_out.write(remind_citations(tools, return_as_text=True))

    tar_command = f"tar --use-compress-program='pigz -p 8 --best' -cf rpdb.tar.gz {ROLYPOLY_DATA}"  # threads

    subprocess.run(tar_command, shell=True)

    # # On NERSC
    # scp uneri@xfer.jgi.lbl.gov:/clusterfs/jgi/scratch/science/metagen/neri/projects/data2/data.tar.gz /global/dna/projectdirs/microbial/prokpubs/www/rolypoly/data/
    # chmod +777 -R /global/dna/projectdirs/microbial/prokpubs/www/rolypoly/data/

    # upload_command = f"gsutil cp {ROLYPOLY_DATA}.tar.gz gs://rolypoly-data/"
    # subprocess.run(upload_command, shell=True)


def prepare_genomad_rna_viral_hmms(ROLYPOLY_DATA, threads, logger=None):
    """Download and prepare RNA viral HMMs from geNomad markers.

    Downloads the geNomad database, analyzes the marker metadata to identify
    RNA viral specific markers, and creates an HMM database from their alignments.

    Args:
        ROLYPOLY_DATA (str): Base directory for data storage
        threads (int): Number of CPU threads to use
        logger: Logger object for recording progress and errors
    """
    import os
    import shutil
    import subprocess
    import tarfile

    import polars as pl

    from rolypoly.utils.fax import hmmdb_from_directory

    logger.info("Starting geNomad RNA viral HMM preparation")

    # Create directories
    genomad_dir = os.path.join(ROLYPOLY_DATA, "genomad")
    genomad_db_dir = os.path.join(genomad_dir, "genomad_db")
    genomad_markers_dir = os.path.join(genomad_db_dir, "markers")
    genomad_alignments_dir = os.path.join(genomad_markers_dir, "alignments")
    os.makedirs(genomad_dir, exist_ok=True)
    os.makedirs(genomad_db_dir, exist_ok=True)
    os.makedirs(genomad_markers_dir, exist_ok=True)
    os.makedirs(genomad_alignments_dir, exist_ok=True)
    # Download metadata and database
    genomad_data = "https://zenodo.org/api/records/14886553/files-archive"
    db_url = (
        "https://zenodo.org/records/14886553/files/genomad_msa_v1.9.tar.gz?download=1"
    )
    metadata_url = "https://zenodo.org/records/14886553/files/genomad_metadata_v1.9.tsv.gz?download=1"
    try:
        # Download and read metadata
        logger.info("Downloading geNomad metadata")
        aria2c_command = f"aria2c -c -o genomad_metadata_v1.9.tsv.gz {metadata_url}"
        subprocess.run(aria2c_command, shell=True)
        extract(archive_path="./genomad_metadata_v1.9.tsv.gz", extract_to="./genomad")
        metadata_df = pl.read_csv(
            "./genomad/genomad_metadata_v1.9.tsv",
            separator="\t",
            null_values=["NA"],
            infer_schema_length=10000,
        )
        # Filter for RNA viral specific markers
        rna_viral_markers = metadata_df.filter(
            (pl.col("VIRUS_HALLMARK") == 1)
            & (pl.col("TAXONOMY").str.contains("Riboviria;Orthornavira"))
            & (pl.col("SPECIFICITY_CLASS") == "VV")
        )
        rna_viral_markers.write_csv("./genomad/rna_viral_markers.csv")
        null_ANNOTATION_DESCRIPTION = rna_viral_markers.filter(
            pl.col("ANNOTATION_DESCRIPTION").is_null()
        )
        null_ANNOTATION_DESCRIPTION.write_csv(
            "./genomad/null_annotation_description.csv"
        )
        notation_added = pl.read_csv("./genomad/notation_added.csv")
        # merge the two dataframes, colhece the notation_added dataframe into rna_viral_markers
        rna_viral_markers = (
            rna_viral_markers.join(
                notation_added["ANNOTATION_DESCRIPTION", "MARKER"],
                on="MARKER",
                how="left",
            )
            .with_columns(
                pl.coalesce(
                    [
                        pl.col("ANNOTATION_DESCRIPTION"),
                        pl.col("ANNOTATION_DESCRIPTION_right"),
                    ]
                ).alias("ANNOTATION_DESCRIPTION")
            )
            .drop("ANNOTATION_DESCRIPTION_right")
        )
        rna_viral_markers.write_csv("./genomad/rna_viral_markers_with_notation.csv")
        # Download database
        logger.info("Downloading geNomad database")
        aria2c_command = f"aria2c -c -o genomad_msa_v1.9.tar.gz {db_url}"
        subprocess.run(aria2c_command, shell=True)

        # Extract RNA viral MSAs
        marker_ids = rna_viral_markers["MARKER"].to_list()

        with tarfile.open("genomad/genomad_msa_v1.9.tar", "r") as tar:
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
            os.path.join(ROLYPOLY_DATA, "hmmdbs"), "genomad_rna_viral_markers.hmm"
        )
        hmmdb_from_directory(
            genomad_alignments_dir,
            output_hmm,
            msa_pattern="*.faa",
            info_table="./genomad/rna_viral_markers_with_notation.csv",
            name_col="MARKER",
            accs_col="ANNOTATION_ACCESSIONS",
            desc_col="ANNOTATION_DESCRIPTION",
        )

        logger.info(f"Created RNA viral HMM database at {output_hmm}")

    except Exception as e:
        logger.error(f"Error preparing geNomad RNA viral HMMs: {e}")
        raise


# setup taxonkit
# TODO: CONVERT TO PYTHON
# cd "$DATA_PATH"
# aria2c http://ftp.ncbi.nih.gov/pub/taxonomy/taxdump.tar.gz
# tar -xzvf taxdump.tar.gz
# mv names.dmp nodes.dmp merged.dmp delnodes.dmp "$DATA_PATH/taxdump/"
# rm -rf  taxdump.tar.gz


if __name__ == "__main__":
    prepare_external_data()


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
