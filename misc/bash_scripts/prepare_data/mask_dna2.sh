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
Written by Uri Neri, Brian Bushnell.
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

RVMT_flat="$datadir"/RVMT/RiboV1.6_Contigs_flat.fasta # RVMT contigs shreded to unique kmers 
RefSeq_ribovirus_flat="$datadir"/NCBI_ribovirus/refseq_ribovirus_genomes_flat.fasta
cat $RefSeq_ribovirus_flat $RVMT_flat > tmp_target.fas


##### Check dependencies #####
Dependency_List=(bbmap.sh)
check_dependencies "${Dependency_List[@]}"

##################################################################################################

##### Main #####
input_file=$(echo $(readlink -f "$input")  ) # Get abs path.
# file_name=$(basename $(echo $input_file) .fasta)
bbmap.sh ref=$input_file in=tmp_target.fas outm=mapped.sam minid=0.7 overwrite=true threads=$THREADS  -Xmx"$MEMORY"
bbmask.sh in=$input_file out=$output_file entropy=0.2 sam=mapped.sam
# rm mapped.sam tmp_target.fas
{
kmercountexact.sh in=tmp_target_ent_masked.fas out=tmp_target_ent_masked_31kmers.fa mincount=1 k=31 threads=$THREADS  -Xmx"$MEMORY"
bbduk.sh in=output.fasta ref=tmp_target_ent_masked_31kmers.fa out=masked_output.fasta ktrim=N k=31 mm=f threads=$THREADS  -Xmx"$MEMORY"

tmp_target_ent_masked.fas 
mkdir RVDB
mmseqs createdb tmp_target_nochimeras.fasta RVDB/rvdb --dbtype 2
mkdir dnadb/
mmseqs createdb output.fasta dnadb/dnadb --dbtype 2

dnaomedb=dnadb/dnadb

mkdir ./dnaome_vs_RVDB

 querydb=RVDB/rvdb
 targetdb=$dnaomedb
 resultsdb=./dnaome_vs_RVDB/results


 echo resultdb is $resultsdb 
 echo targetdb is $targetdb
 echo querydb is $querydb 

 mmseqs search $querydb $targetdb $resultsdb ./tmp/  --threads $THREADS --search-type 3  --cov-mode 2 -c 0.3 --min-seq-id 0.75  --min-aln-len 90  -e 0.001 --max-seqs 350  --max-accept 350 --split-memory-limit 34G -a  --force-reuse 
 mmseqs convertalis $querydb $targetdb $resultsdb ./RVDB_vs_DNAdb.tsv --search-type 3 --format-output "qheader,theader,evalue,pident,qstart,qend,qlen,tstart,tend,tlen,alnlen,bits,mismatch,qcov,tcov"
 echo 
 sed -i '1s/^/qheader	theader	evalue	pident	qstart	qend	qlen	tstart	tend	tlen	alnlen	bits	mismatch	qcov	tcov\n/' ./RVDB_vs_DNAdb.tsv 


mkdir ./dnaome_vs_RVDB_map
resultsdb=./dnaome_vs_RVDB_map/results
mmseqs createindex $querydb ./tmp/   --search-type 3 # https://github.com/soedinglab/MMseqs2/issues/507 ## WTF mmseqs    
mmseqs createindex $targetdb ./tmp/   --search-type 3 # https://github.com/soedinglab/MMseqs2/issues/507 

 mmseqs map  $querydb $targetdb $resultsdb ./tmp/  --threads $THREADS --max-seqs 1 -a # --split-memory-limit 34G   # --force-reuse 
 mmseqs convertalis $querydb $targetdb $resultsdb ./RVDB_vs_DNAdb_map.tsv --search-type 3 --format-output "qheader,theader,evalue,pident,qstart,qend,qlen,tstart,tend,tlen,alnlen,bits,mismatch,qcov,tcov"

}
bbduk.sh in=./tmp_target.fas out=rRNA_filt_tmp_target.fas outm=tmp_target_fas_chimeras_bbduk_rRNA.fas ref=$rrna_fas1,$rrna_fas2 k=31 mincovfraction=0.5 hdist=0 stats=stats_rRNA_filt_tmp_target.txt  threads=$THREADS  -Xmx"$MEMORY" # speed=15 # rskip=7 
