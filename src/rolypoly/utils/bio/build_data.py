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
from rolypoly.utils.bio.pyhmm_utils import hmmdb_from_directory, hmm_from_msa

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

    data_dir = pt(os.path.abspath(data_dir))
    data_dir.mkdir(exist_ok=True)

    hmmdb_dir = os.path.join(data_dir, "hmmdbs")
    os.makedirs(hmmdb_dir, exist_ok=True)

    # Add geNomad RNA viral HMMs preparation
    prepare_genomad_rna_viral_hmms(data_dir, threads, logger)

    # RdRp-scan
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
    # scp uneri@xfer.jgi.lbl.gov:/clusterfs/jgi/scratch/science/metagen/neri/projects/data2/data.tar.gz /global/dna/projectdirs/microbial/prokpubs/www/rolypoly/data/
    # chmod +777 -R /global/dna/projectdirs/microbial/prokpubs/www/rolypoly/data/

    # upload_command = f"gsutil cp {data_dir}.tar.gz gs://rolypoly-data/"
    # subprocess.run(upload_command, shell=True)


def prepare_genomad_rna_viral_hmms(data_dir, threads, logger: logging.Logger):
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
    genomad_dir = os.path.join(data_dir, "genomad")
    genomad_db_dir = os.path.join(genomad_dir, "genomad_db")
    genomad_markers_dir = os.path.join(genomad_db_dir, "markers")
    genomad_alignments_dir = os.path.join(genomad_markers_dir, "alignments")
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
            os.path.join(data_dir, "hmmdbs"), "genomad_rna_viral_markers.hmm"
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

def prepare_vfam_hmm(data_dir, logger: logging.Logger):
    """Prepare VFAM HMM database."""
    fetch_and_extract(
        "https://fileshare.csb.univie.ac.at/vog/vog230/vfam.hmm.tar.gz",
        extract_to=os.path.join(data_dir, "hmmdbs", "vfam"),
    )
    # move all files in vfam/hmm to vfam/
    for file in os.listdir(os.path.join(data_dir, "hmmdbs", "vfam", "hmm")):
        shutil.move(os.path.join(data_dir, "hmmdbs", "vfam", "hmm", file), os.path.join(data_dir, "hmmdbs", "vfam", file))
    # concat all the hmms and remove subdirectories
    with open(os.path.join(data_dir, "hmmdbs", "vfam.hmm"), "w") as f_out:
        for file in os.listdir(os.path.join(data_dir, "hmmdbs", "vfam")):
            if file.endswith(".hmm"):
                with open(os.path.join(data_dir, "hmmdbs", "vfam", file), "r") as f_in:
                    f_out.write(f_in.read())
    # remove the vfam directory
    shutil.rmtree(os.path.join(data_dir, "hmmdbs", "vfam"))
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
