#!/usr/bin/bash

# Written by Uri Neri
# Last modified 13.06.2024 ---- WIP
# Contact: 
# Description: Get and install needed binaries and 3rd party code
# I assume these could work as standalones without requiring even more dependencies - but I might be wrong.

# rolypoly_dir=/REDACTED_HPC_PATH/rolypoly/
depen_dir=$1
mkdir $depen_dir
current_path=$(pwd)
cd $depen_dir

## Downloads!
# seqkit 
wget https://github.com/shenwei356/seqkit/releases/download/v2.8.2/seqkit_linux_amd64.tar.gz ; tar xvfz seqkit_linux_amd64.tar.gz

# bbmap.sh 
wget https://sourceforge.net/projects/bbmap/files/BBMap_39.06.tar.gz ; tar xvfz BBMap_39.06.tar.gz

# NCBI's datasets 
wget https://ftp.ncbi.nlm.nih.gov/pub/datasets/command-line/v2/linux-amd64/datasets
wget https://ftp.ncbi.nlm.nih.gov/pub/datasets/command-line/v2/linux-amd64/dataformat

# SPAdes
wget https://github.com/ablab/spades/releases/download/v4.0.0/SPAdes-4.0.0-Linux.tar.gz ; tar xvfz SPAdes-4.0.0-Linux.tar.gz

# MEGAHIT
wget https://github.com/voutcn/megahit/releases/download/v1.2.9/MEGAHIT-1.2.9-Linux-x86_64-static.tar.gz ; tar zvxf MEGAHIT-1.2.9-Linux-x86_64-static.tar.gz

# FastQC
wget https://www.bioinformatics.babraham.ac.uk/projects/fastqc/fastqc_v0.12.1.zip ; unzip  fastqc_v0.12.1.zip 

# MMseqs2 
wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz; tar xvfz mmseqs-linux-avx2.tar.gz

# plass and pinguin
wget https://mmseqs.com/plass/plass-linux-avx2.tar.gz; tar xvfz plass-linux-avx2.tar.gz 

# aws CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" ; unzip awscliv2.zip

#NCBI's blast+
wget https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-2.15.0+-x64-linux.tar.gz ;  tar xvfz ncbi-blast-2.15.0+-x64-linux.tar.gz

# rush
wget https://github.com/shenwei356/rush/releases/download/v0.5.4/rush_linux_amd64.tar.gz ; tar xvfz rush_linux_amd64.tar.gz

# Diamond
wget https://github.com/bbuchfink/diamond/releases/download/v2.1.9/diamond-linux64.tar.gz; tar xvfz diamond-linux64.tar.gz

## TBD / needs compiling\glibc? ##
# # hstlib  -for bzip2?
wget https://github.com/samtools/htslib/releases/download/1.20/htslib-1.20.tar.bz2 ;  tar -xvjf htslib-1.20.tar.bz2
cd   htslib-1.20
./configure
make
cd ..


# GNU parallel 
# wget https://ftpmirror.gnu.org/parallel/parallel-latest.tar.bz2; tar -xvjf parallel-latest.tar.bz2
# rm parallel-latest.tar.bz2
# cd parallel*
# ./configure
# make
# cd ..   

# # rust-parallel 
# wget https://github.com/aaronriekenberg/rust-parallel/releases/download/v1.18.1/rust-parallel-x86_64-unknown-linux-gnu.tar.gz  ;  tar xvfz rust-parallel-x86_64-unknown-linux-gnu.tar.gz


### Delete compressed archives ###
rm ./*gz
rm ./*bz2
rm ./*zip
rm ./*tar.gz

### Make the files executable  ###
chmod +x  datasets
chmod +x  datasets
cd $current_path+

## Python deps are handled by     pip
# pyhmmer and pyrodigal-gv
# pip install pyhmmer
# pip install pyrodigal-gv
