#!/bin/bash
##################################################################################################
# Written by Uri Neri, Brian Bushnell.
# Last modified 13.06.2024 ---- WIP 
# Description: Mask an input fasta file for sequences that could be RNA viral, or that
# could be mistaken for them, based on minimal identity and shared kmers (see bbmask.sh for more info).
echo ${0}
echo ${@}

usage() {
echo ' 
Written by Uri Neri, Brian Bushnell, Simon Roux.
Description: Mask an input fasta file for sequences that could be RNA viral, or that
could be mistaken for them, based on minimal identity and shared kmers (see bbmask.sh for more info).
Input is the full path to a single fasta file, and the output is a masked fasta file.
Optionally, the masked file could be compressed - flatten - using kcompress.sh so it only includes unique kmers.
Mapping to the kcompressed file can reduce required memory downstream, but the process itself can require a lot of memory so    

Example Usage:
bash mask_dna.sh -t 1 -M 10 -i blabla.fasta -o output_file.fasta -f true  -S path/to/rolypoly/
Arguments:
#	Desc (suggestion) [default]
-t	Threads (all) [1]
-M	MEMORY in gb (more) [6]  
-o  output file name. Note, that some tmp files could be written to that directory.
-f	Attempt to kcompress.sh the masked file.
-i	Input fasta file
-S  Path to rolypoly project directory - the target files to mask against are in there. ([/REDACTED_HPC_PATH/rolypoly/])


Dependencies for this script (the script will attempt to load them from the rolypoly bin directory):
bbmap.sh
see download_dependencies.sh for more information.
'
exit 
}

if [[ $# -eq 0 ]] ; then
usage
exit
fi


##################################################################################################
##### Set defaults: #####
THREADS=1 #t
MEMORY=6 #M
kompres=true #f
rolypoly_dir=/REDACTED_HPC_PATH/rolypoly/ #S
# output_file= #o
# input='*.fasta' #i

##### Parse args: #####
while getopts t:i:o:f:M:S: flag
do
case "${flag}" in
t) THREADS=${OPTARG};;
i) input=${OPTARG};;
o) output_file=${OPTARG};;
f) kompres=${OPTARG} ;;
M) MEMORY=${OPTARG};;
S) rolypoly_dir=${OPTARG};;
*) usage
esac
done

# Mandatory arguments
if [ ! "$input" ]|| [ ! "$output_file" ] ; then
usage
echo "arguments -i and -o must be provided"
fi

##### Set enviroment #####
datadir="$rolypoly_dir"/data/
export rolypoly_dir=$rolypoly_dir
source "$rolypoly_dir"/rolypoly/utils/load_bins.sh
source "$rolypoly_dir"/rolypoly/utils/bash_functions.sh

export MEMORY=$MEMORY
export THREADS=$THREADS

# RVMT_flat="$datadir"/RVMT/RiboV1.6_Contigs_flat.fasta # RVMT contigs shreded to unique kmers 
# RefSeq_ribovirus_flat="$datadir"/NCBI_ribovirus/refseq_ribovirus_genomes_entropy_masked.fasta
# cat $RefSeq_ribovirus_flat $RVMT_flat > tmp_target.fas # Keeping it as a cat instead of a path to a pre-merged file, as I want to add the option for the user to supply a fasta file to (also/instead) use for masking.

##### Check dependencies #####
Dependency_List=(bbmap.sh)
check_dependencies "${Dependency_List[@]}"

##################################################################################################

##### Main #####
input_file=$(echo $(readlink -f "$input")  ) # Get abs path.
# file_name=$(basename $(echo $input_file) .fasta)
bbmap.sh ref=$input_file in="$datadir"/RVMT_NCBI_Ribo_Japan_for_masking.fasta outm=mapped.sam minid=0.7 overwrite=true threads=$THREADS  -Xmx"$MEMORY"
bbmask.sh in=$input_file out=$output_file entropy=0.2 sam=mapped.sam
rm mapped.sam 
# rm tmp_target.fas

##### Based on arg -f#####
if [ "$kompres" != "False" ];
then
    kcompress.sh in=$output_file out="$output_file"_flat.fa fuse=2000 k=31  prealloc=true overwrite=true threads=$THREADS  -Xmx"$MEMORY" # prefilter=true
    mv "$output_file"_flat.fa $output_file
fi

rm ref -r
