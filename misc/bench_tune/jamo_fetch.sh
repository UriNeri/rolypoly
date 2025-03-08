#!/usr/bin/env bash


source ~/.bashrc
conda activate crispy   
cd /clusterfs/jgi/scratch/science/metagen/neri/ynp/
mkdir raw_reads
mkdir YNP_DNA
alias jamo='apptainer run docker://doejgi/jamo-dori jamo'

apptainer run docker://doejgi/jamo-dori jamo info
apptainer run docker://doejgi/jamo-dori jamo info -s dori all id 5567e4830d87852b48481cfd

apptainer run docker://doejgi/jamo-dori jamo info all spid 1061603 


# Show all filtered fastqs for sequencing project id 1109457
jamo info filtered -s dori spid $(cat /clusterfs/jgi/scratch/science/metagen/neri/ynp/metaTs_ITS_spIDs.lst) > YNP_RNA_info_fastqs_08072024.tsv
jamo fetch filtered -s dori spid $(cat /clusterfs/jgi/scratch/science/metagen/neri/ynp/metaTs_ITS_spIDs.lst)
# awk '$3 == "RESTORED" {print $2}' ./YNP_RNA_info_fastqs.tsv  | xargs -I %  cp % ./raw_reads/

# # cut -f1 YNP_RNA_info_fastqs_08072024.tsv -d' ' | sort -u > spids.lst
# # while IFS= read -r line
# # do
# # mkdir spids/$line
# # done < spids.lst

jamo info img -s dori spid 1255018
# jamo fetch  -s dori  id 5e5d4ce3de209cf12aa1dea2
jamo info  -s dori  id 5e5d4ce2de209cf12aa1de8c
jamo info  -s dori  id 5e5d4ce1de209cf12aa1de7e
jamo info  -s dori  id 5e273e1f7776dfea0bd9b8f9 5e273e1f7776dfea0bd9b8c7 5e273e1f7776dfea0bd9b8d5 5e273e1f7776dfea0bd9b8e0 5e273e1f7776dfea0bd9b8ce


# Fetch DNA assemblies for read substraction/filteration
jamo info assembled.fna -s dori spid $(cat /clusterfs/jgi/scratch/science/metagen/neri/ynp/metaGs_GS_ITS_spIDs.lst) > YNP_DNA/YNP_DNA_info_assembelies.tsv
jamo fetch assembled.fna -s dori spid $(cat /clusterfs/jgi/scratch/science/metagen/neri/ynp/metaGs_GS_ITS_spIDs.lst)

awk '$3 == "RESTORED" {print $2}' ./YNP_DNA/YNP_DNA_info_assembelies.tsv 
# awk '$3 == "RESTORED" {print $2}' YNP_DNA_info_assembelies.tsv  | xargs -I %  cp % ./YNP_DNA/


##### 
#!/bin/bash

# Specify the file name
# file=../metaTs_ITS_spIDs.lst

# Read the file line by line
# while IFS= read -r line
# do
#     # Do something with each line
#     jamo info all -s dori spid $line > spid_"$line".info
# done < "$file"