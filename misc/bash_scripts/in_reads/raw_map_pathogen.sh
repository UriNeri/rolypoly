##################################################################################################
# Written by Uri Neri, Brian Bushnell.
# Last modified 13.06.2024 ---- WIP
# Contact: 
# Description: Processing RNA-seq (transcriptome, RNA virome, metatranscriptomes) Illumina raw reads with the 
# sole goal of identification of RNA viruses. 

echo ${@}

usage() {
echo ' 
Written by Uri Neri, Brian Bushnell.
Description: Map input fastq file to a collection of known viral pathogens. 
Input is the full path to a single fastq file with paired end reads.
In the output directory, you will see several files (with the same basename as the input):
1. SAM file with the mapping results.
2. Stats file (see bbmap documentation for details).
3. fastq.gz file with the matched reads.
4. A rough assembly (attempt) with contigs generated from the reads in 3. using tadpole.sh extension and assembly modules (see tadpole.sh -h for more).

Example Usage:
bash raw_map_pathogen.sh -t 1 -M 10 -i blabla.fastq.gz -r False -o output_dir/ -g ./logog.txt -D custom_pathogen.fasta -S /REDACTED_HPC_PATH/rolypoly/
Arguments:
#	Desc (suggestion) [default]
-t	Threads (all) [1]
-M	MEMORY in gb (more) [6]  
-S  path to the rolypoly PARENT project directory ([/REDACTED_HPC_PATH/rolypoly/])
-o  output location. Note, this will be used as the working directory, and to this the final files will be written to. ([pwd])
-r  Remove tmps? ([False]) --- if this is anything other than "False" (capital F), they will be kept.  
-g	Abs path to logfile ([pwd/logfile.txt]). Note that most of the info will go to the slurmout, this is more for documenting the parameters used in this run.
-i	Input fastq.gz file
-D	option to use a custom fasta file for the raw read mapping.

Dependencies for this script (the script will attempt to load them from the rolypoly bin directory):
awk, seqkit, bbmap.sh, rush, bgzip
see download_dependencies.sh for more information.
'
}


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
rolypoly_dir=/REDACTED_HPC_PATH/rolypoly/
output_dir=$(pwd) #o
rm_tmp="False" #r
logfile=$(pwd)/"$(date "+%F+%T")"_"logfile.txt"
Overwritebbduks=true
# input='*.fq.gz' #i
# known_DNA_present=""  #D #Commeneted out as these are mandtory and the user should supply them    
speed=0

##### Parse args: #####
while getopts t:i:o:r:g:M:D:S:s: flag
do
case "${flag}" in
t) THREADS=${OPTARG};;
i) input=${OPTARG};;
o) output_dir=${OPTARG};;
r) rm_tmp=${OPTARG} ;;
g) logfile=${OPTARG} ;;
M) MEMORY=${OPTARG};;
D) known_DNA_present=${OPTARG};;
S) rolypoly_dir=${OPTARG};;
s) speed=${OPTARG};;
*) usage
esac
done

# Mandatory arguments
if [ ! "$input" ]|| [ ! "$known_DNA_present" ] ; then
usage
echo "arguments -i and -D must be provided"
fi

##### Set enviroment #####
datadir="$rolypoly_dir"/data/
export rolypoly_dir=$rolypoly_dir
source "$rolypoly_dir"/rolypoly/utils/load_bins.sh
source "$rolypoly_dir"/rolypoly/utils/bash_functions.sh

logfile=$(echo $(readlink -f "$logfile")  ) # Get abs path.

export MEMORY=$MEMORY
export THREADS=$THREADS

rrna_fas1="$datadir"/rRNA/merged_masked.fa # SILVA_138
rrna_fas2="$datadir"/rRNA/rmdup_rRNA_ncbi_masked.fa # NCBI RefSeq rRNA
RVMT_flat="$datadir"/RVMT/RiboV1.6_Contigs_flat.fasta # RVMT contigs shreded to unique kmers 
RVMT_contigs="$datadir"/RVMT/RiboV1.6_Contigs.fasta # RVMT contigs (raw)


##### Check dependencies #####
Dependency_List=(awk rush seqkit bbmap.sh datasets)
check_dependencies "${Dependency_List[@]}"

##################################################################################################

##### Ready or not message #####
logit $logfile "script: $0" 
logit $logfile "Params: echo ${@}" 
logit $logfile "Launch location: $PWD "
logit $logfile "Time and date: $(date)"
logit $logfile "Submitter name: $USER"
logit $logfile "HOSTNAME: $HOSTNAME"

##### Main #####
output_dir=$(echo $(readlink -f "$output_dir")  ) # Get abs path.
fastq_file=$(echo $(readlink -f "$input")  ) # Get abs path.
file_name=$(basename $(echo $fastq_file) .fastq.gz)
known_DNA_present=$(echo $(readlink -f "$known_DNA_present")  ) # Get abs path.

mkdir $output_dir
cd $output_dir
mkdir tmp_dir_"$file_name"
work_dir=tmp_dir_"$file_name"

cd "$work_dir"

# --- Preprocessing ---
logit $logfile "Started preprocessnig for:  $fastq_file"


