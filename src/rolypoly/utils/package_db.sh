#!/bin/bash

# Set source and destination directories
SOURCE_DIR="./data"
DEST_DIR="../data2/data"

# Create destination directory structure
mkdir -p "$DEST_DIR"
mkdir -p "$DEST_DIR/hmmdbs"
mkdir -p "$DEST_DIR/NCBI_ribovirus"
mkdir -p "$DEST_DIR/RVMT"
mkdir -p "$DEST_DIR/rRNA"
mkdir -p "$DEST_DIR/contam"
mkdir -p "$DEST_DIR/Rfam"
mkdir -p "$DEST_DIR/masking"
mkdir -p "$DEST_DIR/taxdump"
# mkdir -p "$DEST_DIR/citations"

# Copy HMM databases
cp "$SOURCE_DIR/hmmdbs/neordrp/zenodo/"*.hmm "$DEST_DIR/hmmdbs/"
cp "$SOURCE_DIR/hmmdbs/RdRp-scan.hmm" "$DEST_DIR/hmmdbs/"
cp "$SOURCE_DIR/hmmdbs/RVMT.hmm" "$DEST_DIR/hmmdbs/"
cp "$SOURCE_DIR/hmmdbs/TSA_Olendraite.hmm" "$DEST_DIR/hmmdbs/"
cp "$SOURCE_DIR/hmmdbs/rt_rdrp_pfamA37.hmm" "$DEST_DIR/hmmdbs/"
cp "$SOURCE_DIR/hmmdbs/neordrp/" "$DEST_DIR/hmmdbs/neordrp/" -r
cp "$SOURCE_DIR"/hmmdbs/genomad* "$DEST_DIR/hmmdbs/" 

# Copy NCBI RNA virus files
cp "$SOURCE_DIR/NCBI_ribovirus/refseq_ribovirus_genomes_flat.fasta" "$DEST_DIR/NCBI_ribovirus/"
cp "$SOURCE_DIR/NCBI_ribovirus/refseq_ribovirus_genomes_entropy_masked.fasta" "$DEST_DIR/NCBI_ribovirus/"
cp "$SOURCE_DIR/NCBI_ribovirus/proteins/datasets_efetch_refseq_ribovirus_proteins_rmdup.faa" "$DEST_DIR/NCBI_ribovirus/proteins.faa"

# Copy RVMT files
cp "$SOURCE_DIR/RVMT/RiboV1.6_Contigs_flat.fasta" "$DEST_DIR/RVMT/"
cp "$SOURCE_DIR/RVMT/RVMT_allorfs_filtered_no_chimeras.faa" "$DEST_DIR/RVMT/"

# Copy rRNA database files
cp "$SOURCE_DIR"/rRNA/* "$DEST_DIR/rRNA/" -r 

# Copy Rfam database
cp "$SOURCE_DIR"/Rfam/Rfam.cm* "$DEST_DIR/Rfam"

# Copy contamination files
cp "$SOURCE_DIR/contam" "$DEST_DIR/contam/" -r

# Copy masking files
cp "$SOURCE_DIR/masking" "$DEST_DIR/masking/" -r

# Copy taxdump
cp "$SOURCE_DIR/taxdump" "$DEST_DIR/taxdump/" -r

# Create a tar.gz archive
cd "../data2"
tar -czf data.tar.gz data/

# On NERSC
scp uneri@xfer.jgi.lbl.gov:/REDACTED_HPC_PATH/projects/data2/data.tar.gz /REDACTED_NERSC_PATH/prokpubs/www/rolypoly/data/
chmod +777 -R /REDACTED_NERSC_PATH/prokpubs/www/rolypoly/data/
