##################################################################################################
# Written by Uri Neri, Brian Bushnell.
# Last modified 13.06.2024 ---- WIP
# Contact: 
# Description: Processing RNA-seq (transcriptome, RNA virome, metatranscriptomes) Illumina raw reads with the 
# sole goal of identification of RNA viruses. 

echo ${0}
echo ${@}

usage() {
echo ' 
Written by Uri Neri, Brian Bushnell, Simon Roux.
Description: Processing RNA-seq (transcriptome, RNA virome, metatranscriptomes) Illumina raw reads with the 
sole goal of identification of RNA viruses.
Input is the full path to a single fastq file with paired end reads, and the output is two fq.gz files with the same basename, 
in the given output path with the prefixes: qtrimmed_<original file name> and merged_<original file name>
If the program fails and runs out of memory, try setting -s true, which will set bbduk to use speed=15 - 
this will cut memory use by 87% and still match most reads (see bbduk docs for more info).


Example Usage:
bash filter_reads.sh -t 1 -M 10 -i blabla.fastq.gz -r False -o output_dir/ -g ./logog.txt -D host_dna.fasta -S /REDACTED_HPC_PATH/rolypoly/
Arguments:
#	Desc (suggestion) [default]
-t	Threads (all) [1]
-M	MEMORY in gb (more) [6]  
-S  path to the rolypoly PARENT project directory ([/REDACTED_HPC_PATH/rolypoly/])
-o  output location. Note, this will be used as the working directory, and to this the final files will be written to. ([pwd])
-r  Remove tmps? ([False]) --- if this is anything other than "False" (capital F), they will be kept.  
-g	Abs path to logfile ([pwd/logfile.txt]) note that most of the info will go to the slurmout, this is more for documenting the parameters used in this run.
-i	Input fastq.gz file
-D	fasta file of known DNA entities to be used for filtering likely host reads
-s  Set`s bbduk.sh speed to a vlue, for the filtering by mapping to the input DNA sequences. ([0])


Dependencies for this script (the script will attempt to load them from the rolypoly bin directory):
awk, seqkit, bbmap.sh, rush, bgzip\bzip
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

#Decontamination by mapping
##NOTE: if sample is RNA-seq (metaT, bulk, polyA etc) gunned for RNA viruses, consider removing reads perfectly mapping to DNA.
#Decontamination by mapping (deduped SILVA rRNA)
logit $logfile "Starting filtering by matches to rRNAs (SILVA and NCBI's RefSeq)"
bbduk.sh in=$fastq_file out=rRNA_filt_"$file_name".fq.gz ref=$rrna_fas1,$rrna_fas2 k=31 mincovfraction=0.5 hdist=0 stats=stats_rRNA_filt_"$file_name".txt overwrite=$Overwritebbduks threads=$THREADS  -Xmx"$MEMORY" # speed=15 # rskip=7 
check_file_exist_isempty rRNA_filt_"$file_name".fq.gz


## Decontamination by mapping - fetch DNA assemblies (from NCBI) based on the top rRNA matches, to be used for further filtering.
mkdir fetched_dna_genomes
cd fetched_dna_genomes
fetch_genomes ../stats_rRNA_filt_"$file_name".txt gbs_50m.fasta
mask_dna.sh -t $THREADS -M $MEMORY -i gbs_50m.fasta -o masked_gbs_50m_"$file_name".fasta -f False # Mask the fetched genomes for low complexity region, and regions that could be similar to RNA viruses. see https://jgi.doe.gov/data-and-tools/software-tools/bbtools/bb-tools-user-guide/bbmask-guide/
mv masked_gbs_50m_"$file_name".fasta ../masked_gbs_50m_"$file_name".fasta
cd ..
rm fetched_dna_genomes -r
logit $logfile "Starting filtering by matches to rRNA-present fetched DNA genomes"
bbduk.sh in=rRNA_filt_"$file_name".fq.gz out=rRNA_cgbs_filt_"$file_name".fq.gz ref=masked_gbs_50m_"$file_name".fasta k=31 mincovfraction=0.7 hdist=0 speed=$speed stats=stats_rRNA_cgbs_filt_"$file_name".txt threads=$THREADS  -Xmx"$MEMORY" # speed=15 # rskip=7 
check_file_exist_isempty rRNA_cgbs_filt_"$file_name".fq.gz

#Decontamination by mapping (DNA - ideally a user supplied !preclustered! representative set)
logit $logfile "Starting filtering by matches to $known_DNA_present"
mask_dna.sh -t $THREADS -M $MEMORY -i $known_DNA_present -o masked_known_DNA_present.fasta -f False  # We first mask the known_DNA_present set for sequences that might resemble RNA viruses (TBD more work to mask potentially non-retroviral EVEs).
# Mapping and filtering
bbduk.sh in=rRNA_cgbs_filt_"$file_name".fq.gz out=DNA_filt_"$file_name".fq.gz ref=masked_known_DNA_present.fasta k=31 mincovfraction=0.6 hdist=0 speed=$speed stats=stats_known_DNA_filt_"$file_name".txt threads=$THREADS  -Xmx"$MEMORY" #rskip=7 speed=15
check_file_exist_isempty DNA_filt_"$file_name".fq.gz

# Remove  duplicates (not !optical!) **** N.B! if these are from NCBI's SRA, i.e. headers are weird, so this is only dedupe, not by flowcell/optical. See https://github.com/BioInfoTools/BBMap/issues/15
logit $logfile "Starting removing non-optical duplicates"
clumpify.sh in=DNA_filt_"$file_name".fq.gz out="$file_name"_clumped.fq.gz dedupe  -Xmx"$MEMORY" threads=$THREADS overwrite=t # stats=stats_clumpify_"$file_name".txt
check_file_exist_isempty "$file_name"_clumped.fq.gz # End of DNA filtering

# Remove low-quality regions
logit $logfile "Starting filtering by tile (only makes sense if the input is a single illumina library - do not do for synthetic reads or combined libraries)"
filterbytile.sh nullifybrokenquality=t in="$file_name"_clumped.fq.gz out=filtered_by_tile_"$file_name".fq.gz  -Xmx"$MEMORY" threads=$THREADS overwrite=t #stats="$sample"_stats_filterbytile.txt
check_file_exist_isempty filtered_by_tile_"$file_name".fq.gz

# Trim adapters.  Optionally, reads with Ns can be discarded by adding "maxns=0" and reads with really low average quality can be discarded with "maq=8".
logit $logfile "Starting trimming adapters "
bbduk.sh   in=filtered_by_tile_"$file_name".fq.gz out=trimmed_"$file_name".fq.gz ktrim=r k=23 mink=11 hdist=1 tbo tpe minlen=45 ref=adapters ftm=5 maq=6 maxns=1 ordered -Xmx"$MEMORY" threads=$THREADS overwrite=t stats="$file_name"_adapters_stats_bbduk.txt
check_file_exist_isempty trimmed_"$file_name".fq.gz

# Remove synthetic artifacts and spike-ins by kmer-matching.
logit $logfile "Starting filtering synthetic artifacts "
bbduk.sh  nullifybrokenquality in=trimmed_"$file_name".fq.gz out=filtered_"$file_name".fq.gz k=31 ref=artifacts,phix ordered cardinality  -Xmx"$MEMORY" threads=$THREADS overwrite=t stats="$file_name"_synthethic_stats_bbduk.txt
check_file_exist_isempty filtered_"$file_name".fq.gz

# Entropy filtering (check parameters!!!)
logit $logfile "Starting Entropy filtering "
bbduk.sh in=filtered_"$file_name".fq.gz out=filtered_entrophy_"$file_name".fq.gz entropy=0.001 entropywindow=30   -Xmx"$MEMORY" threads=$THREADS  overwrite=t
check_file_exist_isempty filtered_entrophy_"$file_name".fq.gz

#Error-correct phase 1  
logit $logfile "Starting Error-correct phase 1 (clumpify ecco)"
bbmerge.sh in=filtered_entrophy_"$file_name".fq.gz out=ecco_"$file_name".fq.gz ecco mix ordered ihist="$file_name"_ihist_merge1.txt   -Xmx"$MEMORY" threads=$THREADS  overwrite=t stats="$file_name"_stats_bbmerge1.txt
check_file_exist_isempty ecco_"$file_name".fq.gz

#Error-correct phase 2
logit $logfile "Starting Error-correct phase 2 (clumpify ecc)"
clumpify.sh in=ecco_"$file_name".fq.gz out=eccc_"$file_name".fq.gz ecc passes=1 reorder nullifybrokenquality  -Xmx"$MEMORY" threads=$THREADS  overwrite=t
check_file_exist_isempty eccc_"$file_name".fq.gz

#Merge
# This phase handles overlapping reads,
# and also nonoverlapping reads, if there is sufficient coverage and sufficiently short inter-read gaps
# For very large datasets, "prefilter=1" or "prefilter=2" can be added to conserve MEMORY.
logit $logfile "Starting bbmerge-auto"
bbmerge-auto.sh in=eccc_"$file_name".fq.gz out=merged_"$file_name".fq.gz outu=unmerged_"$file_name".fq.gz k=93 extend2=80 rem ordered ihist="$file_name"_ihist_merge.txt  -Xmx"$MEMORY" threads=$THREADS overwrite=t stats="$file_name"_stats_bbmerge2.txt
check_file_exist_isempty unmerged_"$file_name".fq.gz
check_file_exist_isempty merged_"$file_name".fq.gz


# #Quality-trim the unmerged reads.
logit $logfile "Starting Quality-trim unmerged"
bbduk.sh  in=unmerged_"$file_name".fq.gz out=qtrimmed_"$file_name".fq.gz qtrim=rl trimq=5 minlen=50 ordered   -Xmx"$MEMORY" threads=$THREADS  overwrite=t stats="$file_name"_stats_bbduk2.txt
check_file_exist_isempty qtrimmed_"$file_name".fq.gz  

logit $logfile "qtrimmed_"$file_name".fq exists. Proceeding to remove intermediate .fq.gz files     based on your -r flag"
##### Based on arg -f, remove intermediate files/dirs. #####
if [ "$rm_tmp" != "False" ];
then
mv merged_"$file_name".fq.gz merged_"$file_name".fq.gz1
mv qtrimmed_"$file_name".fq.gz qtrimmed_"$file_name".fq.gz1  
rm ./*.fq.gz
mv merged_"$file_name".fq.gz1 merged_"$file_name".fq.gz
mv qtrimmed_"$file_name".fq.gz1 qtrimmed_"$file_name".fq.gz 
rm masked_gbs_50m_*
rm masked_known_DNA_present.fasta
fi

# #Fastqc post processing Illumina reads.
logit $logfile "Generating FastQC report for reads post bbduk etc"

mkdir ./FastQC_post_trim_reads/
fastqc -t $THREADS qtrimmed_"$file_name".fq.gz merged_"$file_name".fq.gz -o ./FastQC_post_trim_reads/
logit $logfile "Running multiQC report "

multiqc  ./ --outdir "$file_name"_multiqc

cd ..
mv $work_dir/* ./ 
rm $work_dir -r
logit $logfile "Finished running filter_reads.sh"

