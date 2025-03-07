#!/bin/bash
## specify an email address
#SBATCH --mail-user=uneri@lbl.gov
## specify when to send the email when job is (a)borted, (b)egins, or (e)nds
#SBATCH --mail-type=FAIL
## specify allocation - we want jgi_shared since we don't want to use the whole node for nothing
#SBATCH -A grp-org-sc-metagen
#SBATCH -q jgi_normal
## specify number of nodes
#SBATCH -N 1
#######SBATCH --exclusive
## specify number of procs
#SBATCH -c 24
## specify ram
#SBATCH --mem=400G 
## specify runtime
#SBATCH -t 12:00:00
## specify job name
#SBATCH -J SE_IMGPR
## specify output and error file
#SBATCH -o /REDACTED_HPC_PATH/rolypoly/slurmsout/Slurmout-%A_%a.out

source ~/.bashrc
conda activate crispy   
export PATH=$PATH:<HOME_PATH>/code/mmseqs/bin/

THREADS=24

## Prepare NCBI RNA virus ####
cd $rolypoly_dir/data/
mkdir NCBI_ribovirus
cd NCBI_ribovirus
taxid="2559587"
# Perform the search and download the genomes
esearch -db nuccore -query "txid$taxid[Organism:exp] AND srcdb_refseq[PROP] AND complete genome[title]" | efetch -format fasta > refseq_ribovirus_genomes.fasta
kcompress.sh in=refseq_ribovirus_genomes.fasta out=refseq_ribovirus_genomes_flat.fasta fuse=2000 k=31  prealloc=true  threads=$THREADS # prefilter=true


#### Prepare the RVMT mmseqs database ####
cd $rolypoly_dir/data/
mkdir RVMT
mkdir mmdb
wget https://portal.nersc.gov/dna/microbial/prokpubs/Riboviria/RiboV1.4/RiboV1.6_Contigs.fasta.gz
extract RiboV1.6_Contigs.fasta.gz
seqkit grep  -f ./chimeras_RVMT.lst RiboV1.6_Contigs.fasta --invert-match  > tmp_nochimeras.fasta
mmseqs createdb  tmp_nochimeras.fasta  mmdb/RVMT_mmseqs_db2 --dbtype 2
RVMTdb=/REDACTED_HPC_PATH/rolypoly/data/RVMT/mmdb/RVMT_mmseqs_db2
kcompress.sh in=tmp_nochimeras.fasta out=RiboV1.6_Contigs_flat.fasta fuse=2000 k=31  prealloc=true  threads=$THREADS # prefilter=true

cd ../
cat RVMT/RiboV1.6_Contigs_flat.fasta NCBI_ribovirus/refseq_ribovirus_genomes_flat.fasta > tmp_target.fas
bbmask.sh in=tmp_target.fas out=tmp_target_ent_masked.fas entropy=0.7  ow=t
mv RiboV1.6_Contigs_flat.fasta1 RiboV1.6_Contigs_flat.fasta

bbmap.sh ref=$input_Fasta in=other_fasta outm=mapped.sam minid=0.9 overwrite=true threads=$THREADS  -Xmx"$MEMORY"
bbmask.sh in=$input_file out=$output_file entropy=0.2 sam=mapped.sam
bbduk.sh ref=$input_file sam=mapped.sam k=21 maskmiddle=t in=tmp_target_ent_masked.fas overwrite=true threads=$THREADS  -Xmx"$MEMORY"

# Test #
THREADS=4
MEMORY=40g
fetched_genomes /REDACTED_HPC_PATH/rolypoly/bench/test_sampled_005_bb_metaTs_spiced_RVMT/tmp_dir_sampled_005_bb_metaTs_spiced_RVMT/stats_rRNA_filt_sampled_005_bb_metaTs_spiced_RVMT.txt output.fasta
input_file=/REDACTED_HPC_PATH/rolypoly/data/output.fasta
bbduk.sh ref=$input_file sam=mapped.sam k=21 maskmiddle=t in=tmp_target.fas overwrite=true threads=$THREADS  -Xmx"$MEMORY"


##### Create rRNA DB #####
cd $rolypoly/data/
mkdir rRNA
cd rRNA
wget https://www.arb-silva.de/fileadmin/silva_databases/release_138_1/Exports/SILVA_138.1_LSURef_NR99_tax_silva.fasta.gz
wget https://www.arb-silva.de/fileadmin/silva_databases/release_138_1/Exports/SILVA_138.1_SSURef_NR99_tax_silva.fasta.gz    

gzip SILVA_138.1_SSURef_NR99_tax_silva.fasta.gz
gzip SILVA_138.1_LSURef_NR99_tax_silva.fasta.gz

cat *fasta > merged.fas


bbduk.sh -Xmx1g in=merged.fas out=merged_masked.fa zl=9 entropy=0.6 entropyk=4 entropywindow=24 maskentropy

# # Define the search term
# search_term="ribosomal RNA[title] AND srcdb_refseq[PROP] AND 200:7000[SLEN]"
# # Perform the search and download the sequences
# esearch -db nuccore -query "$search_term" | efetch -format fasta > "rrna_genes_refseq.fasta"
bbduk.sh -Xmx1g in=rmdup_rRNA_ncbi.fasta  out=rmdup_rRNA_ncbi_masked.fa zl=9 entropy=0.6 entropyk=4 entropywindow=24 maskentropy

##### Create AMR DBs #####
mkdir dbs
mkdir dbs/NCBI_pathogen_AMR
cd dbs/NCBI_pathogen_AMR
wget https://ftp.ncbi.nlm.nih.gov/pathogen/Antimicrobial_resistance/Data/2024-05-02.2/ReferenceGeneCatalog.txt
awk -v RS='\t' '/refseq_nucleotide_accession/{print NR; exit}' ReferenceGeneCatalog.txt
# 11
# awk -F'\t' -v ORS=" " '{print $11}' ReferenceGeneCatalog.txt |sed 's|genbank_nucleotide_accession||g' > genbank_nucleotide_accessions.lst
awk -F'\t'  '{print $11}' ReferenceGeneCatalog.txt |sed 's|genbank_nucleotide_accession||g' > genbank_nucleotide_accessions.lst

datasets download gene accession $(cat genbank_nucleotide_accessions.lst)   --filename AMR_genes.zip
echo "CXAL01000043.1 CXAL01000043.1 CXAL01000043.1 CXAL01000043.1 CXAL01000043.1 CXAL01000043.1 CXAL01000043.1 CXAL01000043.1 CXAL01000043.1 CXAL01000043.1 CP001050.1 CP001050.1 CP001050.1 CP001050.1 CP001050.1 CP030821.1 CP030821.1 CP030821.1 CP030821.1 CP030821.1 CP030821.1 BX571856.1 BX571856.1 BX571856.1 BX571856.1 BX571856.1 BX571856.1 " > small_file.txt
datasets download gene accession  $(cat small_file.txt)  --filename AMR_genes.zip
datasets download gene accession NG_048523
datasets download gene accession CXAL01000043.1 CXAL01000043.1
  # Read each taxon name from the file and fetch the corresponding genome data (zip from ncbi)
  while IFS= read -r line;
  do
      echo "Processing $line    "
      datasets download gene accession "${line}"  --filename "${line}"_fetched_genomes.zip 
  done < genbank_nucleotide_accessions.lst

# cd2mec
# cd dbs
# cd nt
# aws s3 cp --no-sign-request s3://ncbi-blast-databases/2024-06-01-01-05-03/ ./ --recursive --exclude "*" --include "nt.*"





