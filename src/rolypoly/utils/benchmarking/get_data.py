import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote

import requests
from requests.exceptions import RequestException

from rolypoly.utils.fax import download_genome


def download_file(url, filename):
    response = requests.get(url, stream=True)
    with open(filename, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)


def run_command(command):
    subprocess.run(command, shell=True, check=True)


##################################################################################################
# Written by Uri Neri.
# Last modified 17.07.2024 ---- WIP
# Contact:
# Description: scratch script using mason and public data to generate input files for benchmarking and fine-tunning.

##################################################################################################
##### Set enviroment #####

# THREADS=6 # $SLURM_CPUS_PER_TASK
THREADS = 6
# export THREADS=$THREADS

# MEMORY=15g # "$SLURM_MEM_PER_NODE" # Might need to add "g" suffix.
MEMORY = "15g"
# export MEMORY=$MEMORY
# MEMORY_nsuffix=$(echo $MEMORY | sed 's|g||g')
MEMORY_BYTES = int(float(MEMORY[:-1]) * 1e9)

# rolypoly_dir=/clusterfs/jgi/scratch/science/metagen/neri/rolypoly/
rolypoly_dir = Path("/clusterfs/jgi/scratch/science/metagen/neri/rolypoly/")
# export rolypoly_dir=$rolypoly_dir
# source "$rolypoly_dir"/rolypoly/utils/load_bins.sh
# source "$rolypoly_dir"/rolypoly/utils/bash_functions.sh
# datadir="$rolypoly_dir"/data/
datadir = rolypoly_dir / "data"

##################################################################################################
##### Main #####
# mkdir /clusterfs/jgi/scratch/science/metagen/neri/rolypoly/bench/gen_data
# cd /clusterfs/jgi/scratch/science/metagen/neri/rolypoly/bench/gen_data
# mkdir dsRNA_data mock_data synt_data VANA_data
output_dir = Path("/clusterfs/jgi/scratch/science/metagen/neri/rolypoly/bench/gen_data")
output_dir.mkdir(parents=True, exist_ok=True)
os.chdir(output_dir)

for subdir in ["dsRNA_data", "mock_data", "synt_data", "VANA_data"]:
    (output_dir / subdir).mkdir(exist_ok=True)

# output_dir="$(pwd)" #o
# logfile=$(pwd)/scratch_logfile.txt
logfile = output_dir / "scratch_logfile.txt"

## Get public data ##
# Source 1 - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10688312/
# A. https://doi.org/10.57745/42WNRJ Trimmed sequencing reads for all viral communities analyzed by dsRNA or VANA approaches.
# B. https://doi.org/10.57745/T4UYPC. Normalized 10M reads dsRNA or VANA data sets generated using the 60-viruses community; community composition and the complete or near-complete reference genomic sequences.
# wget https://entrepot.recherche.data.gouv.fr/api/access/datafile/:persistentId\?persistentId\=doi:10.57745/GQI86B -O cacarrot_populations_info.tab
download_file(
    "https://entrepot.recherche.data.gouv.fr/api/access/datafile/:persistentId?persistentId=doi:10.57745/GQI86B",
    "cacarrot_populations_info.tab",
)
# wget https://entrepot.recherche.data.gouv.fr/api/access/datafile/:persistentId?persistentId=doi:10.57745/ZC1OGQ  -O Schonegger2023_carrot_populations_datasets.zip -b -o /dev/null
download_file(
    "https://entrepot.recherche.data.gouv.fr/api/access/datafile/:persistentId?persistentId=doi:10.57745/ZC1OGQ",
    "Schonegger2023_carrot_populations_datasets.zip",
)
# wget https://entrepot.recherche.data.gouv.fr/api/access/datafile/:persistentId?persistentId=doi:10.57745/YVOMW6 -O 10M_reads_dsRNA_dataset_60_viruses_community.fas -b -o /dev/null
download_file(
    "https://entrepot.recherche.data.gouv.fr/api/access/datafile/:persistentId?persistentId=doi:10.57745/YVOMW6",
    "10M_reads_dsRNA_dataset_60_viruses_community.fas",
)
# wget https://entrepot.recherche.data.gouv.fr/api/access/datafile/:persistentId?persistentId=doi:10.57745/1DUXEE -O 10M_reads_VANA_dataset_60_viruses_community.fas -b -o /dev/null
download_file(
    "https://entrepot.recherche.data.gouv.fr/api/access/datafile/:persistentId?persistentId=doi:10.57745/1DUXEE",
    "10M_reads_VANA_dataset_60_viruses_community.fas",
)
# wget https://entrepot.recherche.data.gouv.fr/api/access/datafile/:persistentId?persistentId=doi:10.57745/ICDLUQ -O 	Reference_genomes.fas -b -o /dev/null
download_file(
    "https://entrepot.recherche.data.gouv.fr/api/access/datafile/:persistentId?persistentId=doi:10.57745/ICDLUQ",
    "Reference_genomes.fas",
)

# Source 2 - https://github.com/cbg-ethz/5-virus-mix
# SRR961514 - MiSeq 2x250bp Illumina run of the 5 HIV-1 virus mix https://www.ncbi.nlm.nih.gov/sra/?term=SRR961514
# mkdir 5_hiv_mix
# cd 5_hiv_mix
hiv_mix_dir = output_dir / "5_hiv_mix"
hiv_mix_dir.mkdir(exist_ok=True)
os.chdir(hiv_mix_dir)

# wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR961/SRR961514/SRR961514_2.fastq.gz
run_command(
    "wget ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR961/SRR961514/SRR961514_2.fastq.gz"
)
# wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR961/SRR961514/SRR961514_1.fastq.gz
run_command(
    "wget ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR961/SRR961514/SRR961514_1.fastq.gz"
)
shutil.move("SRR961514_1.fastq.gz", "inputs/reads/hiv_R1.fq.gz")
shutil.move("SRR961514_2.fastq.gz", "inputs/reads/hiv_R2.fq.gz")
# git clone https://github.com/cbg-ethz/5-virus-mix.git
# subprocess.run(["git", "clone", "https://github.com/cbg-ethz/5-virus-mix.git"])
# rm 5-virus-mix/.git -r
# subprocess.run(["rm", "-r", "5-virus-mix/.git"])
# reformat.sh in1=./SRR961514_1.fastq.gz in2=./SRR961514_2.fastq.gz out=SRR961514_interleaved.fq.gz
run_command(
    f"reformat.sh in1=./SRR961514_1.fastq.gz in2=./SRR961514_2.fastq.gz out=SRR961514_interleaved.fq.gz"
)

# cd ../
os.chdir(output_dir)

# Source nn https://gitlab.com/ilvo/VIROMOCKchallenge/-/tree/master/Dataset_creation

# Source 3 - https://www.ncbi.nlm.nih.gov/pubmed/34970230
# SRR14871112
# mkdir lyssa
# cd lyssa
lyssa_dir = output_dir / "lyssa"
lyssa_dir.mkdir(exist_ok=True)
os.chdir(lyssa_dir)

# wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR148/012/SRR14871112/SRR14871112_2.fastq.gz
download_file(
    "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR148/012/SRR14871112/SRR14871112_2.fastq.gz",
    "SRR14871112_2.fastq.gz",
)
# wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR148/012/SRR14871112/SRR14871112_1.fastq.gz
download_file(
    "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR148/012/SRR14871112/SRR14871112_1.fastq.gz",
    "SRR14871112_1.fastq.gz",
)

# wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR148/711/SRR14871112/SRR14871112_2.fastq.gz
# wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR148/711/SRR14871112/SRR14871112_1.fastq.gz
# rename.sh fixsra=t in1=SRR14871112_1.fastq.gz in2=SRR14871112_2.fastq.gz out1=SRR14871112_1.fq.gz out2=SRR14871112_2.fq.gz
run_command(
    "rename.sh fixsra=t in1=SRR14871112_1.fastq.gz in2=SRR14871112_2.fastq.gz out1=SRR14871112_1.fq.gz out2=SRR14871112_2.fq.gz"
)
# reformat.sh in1=./SRR14871112_1.fastq.gz in2=./SRR14871112_2.fastq.gz out=SRR15886080_interleaved.fq.gz=
run_command(
    "reformat.sh in1=./SRR14871112_1.fastq.gz in2=./SRR14871112_2.fastq.gz out=SRR15886080_interleaved.fq.gz"
)

# host genome - https://www.ncbi.nlm.nih.gov/nuccore/NC_000913.3
download_genome(taxid="9606")
shutil.unpack_archive("9606_fetched_genomes.zip", extract_dir="inputs/hosts")
shutil.move(
    "inputs/hosts/ncbi  GCF_000001405.40_GRCh38.p14_genomic.fna",
    "inputs/hosts/human_genome.fasta",
)
run_command(
    "seqkit rmdup inputs/hosts/ncbi_dataset/data/*/*fna -w0 > inputs/hosts/human_genome_rmdup.fasta"
)
shutil.rmtree("inputs/hosts/ncbi_dataset")
os.unlink("9606_fetched_genomes.zip")
os.unlink("inputs/hosts/md5sum.txt")
os.unlink("inputs/hosts/README.md")


# mism_prob=0.000
# mason_simulator --illumina-prob-mismatch 0 --illumina-read-length 150 --illumina-prob-insert 0 --illumina-prob-deletion 0 \
# --input-reference  Reference_genomes.fasta \
# --fragment-min-size 1 \
# --fragment-max-size  500 \
# --out-alignment  test_mason_simulator.sam \
# --out test_mason_simulator_R1.fastq.gz \
# --out-right test_mason_simulator_R2.fastq.gz \
# --num-threads $THREADS \
# --num-fragments  200000 \
# --very-verbose
mason_cmd = f"""
mason_simulator --illumina-prob-mismatch 0 --illumina-read-length 150 --illumina-prob-insert 0 --illumina-prob-deletion 0 \
--input-reference Reference_genomes.fasta \
--fragment-min-size 1 \
--fragment-max-size 500 \
--out-alignment test_mason_simulator.sam \
--out test_mason_simulator_R1.fastq.gz \
--out-right test_mason_simulator_R2.fastq.gz \
--num-threads {THREADS} \
--num-fragments 200000 \
--very-verbose
"""
run_command(mason_cmd)

# reformat.sh in=test_mason_simulator_R#.fastq.gz out=test_mason_simulator_interleaved.fq.gz
run_command(
    "reformat.sh in=test_mason_simulator_R#.fastq.gz out=test_mason_simulator_interleaved.fq.gz"
)

# coverage = (read count * read length ) / total genome size.
# (cov * gl)/rl = rc
# (10 * 5000 ) / 150 = 333.3** (*5 for 15kbp)
# --illumina-prob-mismatch-scale #  --illumina-right-template-fastq --illumina-left-template-fastq --fragment-mean-size 150 \

# randomreads.sh coverage=10 prefix=three_sim ref=cated_3sims.fasta paired=t illuminanames=t minq=28 maxq=35 maxinsert=500 mininsert=-100 interleaved=t  out=simreasd3sim.fq.gz
run_command(
    "randomreads.sh coverage=10 prefix=three_sim ref=cated_3sims.fasta paired=t illuminanames=t minq=28 maxq=35 maxinsert=500 mininsert=-100 interleaved=t out=simreasd3sim.fq.gz"
)
# randomreads.sh coverage=15 prefix=three_sim ref=ncbi_dataset/data/GCF_000005845.2/GCF_000005845.2_ASM584v2_genomic.fna paired=t illuminanames=t minq=28 maxq=35 maxinsert=500 mininsert=-100 interleaved=t  out=colisims.fq.gz ow=t
run_command(
    "randomreads.sh coverage=15 prefix=three_sim ref=ncbi_dataset/data/GCF_000005845.2/GCF_000005845.2_ASM584v2_genomic.fna paired=t illuminanames=t minq=28 maxq=35 maxinsert=500 mininsert=-100 interleaved=t out=colisims.fq.gz ow=t"
)

# cat *fq.gz > cated_fastq.gz
run_command("cat *fq.gz > cated_fastq.gz")

# Download Dataset_9.fastq.gz.zip from Dryad - I can't understand how to download it using the API so just copy pasting the url from the browser.
run_command(
    "wget 'https://dryad-assetstore-merritt-west.s3.us-west-2.amazonaws.com/ark%3A/13030/m5tr2r1q%7C1%7Cproducer/Dataset_9.fastq.gz.zip?response-content-disposition=attachment%3B filename%3DDataset_9.fastq.gz.zip&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA2KERHV5E3OITXZXC%2F20240919%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20240919T180623Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=0d6053b9495a1e005c000775da37a592359c502f6bf408ad411658b63ea576a5'"
)
# run_command(" mv Dataset_9.fastq.gz.zip\?response-content-disposition=attachment\;\ filename=Dataset_9.fastq.gz.zip\&X-Amz-Algorithm=AWS4-HMAC-SHA256\&X-Amz-Credential=AKIA2KERHV5E3OITXZXC%2F20240919%2Fus-west-2%2Fs3%2Faws4_request\&X-Amz-Date=20240919T180623 Dataset_9.fastq.gz.zip")
run_command("unzip Dataset_9.fastq.gz.zip")
run_command("mv Dataset_9_R1.fastq.gz ./inputs/reads/plant_R1.fq.gz")
run_command("mv Dataset_9_R2.fastq.gz ./inputs/reads/plant_R2.fq.gz")
# # Download Dataset_9.fastq.gz.zip from Dryad
# dryad_doi = "10.5061/dryad.0zpc866z8"
# encoded_doi = quote(dryad_doi)
# dryad_api_url = f"https://datadryad.org/api/v2/datasets/{encoded_doi}"

# try:
#     # Get the dataset metadata
#     response = requests.get(dryad_api_url)
#     response.raise_for_status()  # Raises an HTTPError for bad responses

#     dataset_metadata = response.json()

#     # Find the correct file in the dataset
#     target_file = next((file for file in dataset_metadata['files'] if file['path'].endswith('Dataset_9.fastq.gz.zip')), None)

#     if target_file:
#         download_url = f"https://datadryad.org/api/v2/files/{target_file['id']}/download"
#         print(f"Downloading from: {download_url}")
#         download_file(download_url, "Dataset_9.fastq.gz.zip")
#     else:
#         print("Dataset_9.fastq.gz.zip not found in the Dryad dataset.")
# except RequestException as e:
#     print(f"An error occurred while accessing the Dryad API: {e}")
#     print(f"Response content: {e.response.text if hasattr(e, 'response') else 'No response content'}")


# source 4 - SRR11097768 https://www.nature.com/articles/s41564-020-0755-4 concentrated viral; RNA seq of river estuary.
# SRR11097768
download_file(
    "https://sra-pub-run-odp.s3.amazonaws.com/sra/SRR11097768/SRR11097768",
    "SRR11097768.fastq.gz",
)
shutil.move(
    "SRR11097768.fastq.gz",
    "inputs/reads/meta/viral_RNA_metatranscriptome_river_estuary_interleaved.fq.gz",
)

# source 5 - SRR14039684 https://www.nature.com/articles/s41564-022-01180-2 Total RNA metatranscriptome from soil.
download_file(
    "https://sra-pub-run-odp.s3.amazonaws.com/sra/SRR14039684/SRR14039684",
    "SRR14039684.fastq.gz",
)
shutil.move(
    "SRR14039684.fastq.gz",
    "inputs/reads/meta/total_RNA_metatranscriptome_soil_interleaved.fq.gz",
)


# source 6 - all orthorna viruses from ncbi
download_genome(taxid="2732396")
shutil.unpack_archive("2732396_fetched_genomes.zip", extract_dir="inputs/contigs")
