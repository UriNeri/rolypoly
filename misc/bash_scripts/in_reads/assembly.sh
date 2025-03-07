##################################################################################################
# Written by Uri Neri, Brian Bushnell.
# Last modified 12.07.2024 ---- WIP
# Contact: 
# Description: Assembler wraper - takes in (presumably filtered) reads, and passes them to an assembler(s) of choice.

echo ${@}

usage() {
echo ' 
Written by Uri Neri, Brian Bushnell, Simon Roux.
Description: Assembler wraper - takes in (presumably filtered) reads, and passes them to an assembler(s) of choice.
Assemblers are supplied with the -A flag. Multiple assemblers can be supplied via a comma seprated list (spades,megahit, xxx - TODO: more options?)

Example Usage:
bash assembly.sh -t 1 -M 10g -i ./spids/121231231 -r False -o output_dir/ -g ./logog.txt -A spades,megahit -S /REDACTED_HPC_PATH/rolypoly/
Arguments:
#	Desc (suggestion) [default]
-t	Threads (all) [1]
-M	MEMORY in gb (more) [6]  
-S    path to the rolypoly PARENT project directory ([/REDACTED_HPC_PATH/rolypoly/])
-o    output location. Note, this will be used as the working directory, and to this the final files will be written to. ({-i} + _assembly or something)
-r    Remove tmps? ([False]) --- if this is anything other than "False" (capital F), they will be kept.  
-g	Abs path to logfile ([pwd/logfile.txt]) note that most of the info will go to the slurmout, this is more for documenting the parameters used in this run.
-i    Input path to a unique Spid folder (Mandatory flag).
-A	Assembler ([spades]). Multiple assemblers can be supplied via a comma seprated list (spades,megahit,pengiun)


Dependencies for this script (the script will attempt to load them from the rolypoly bin directory):
awk, seqkit, bbmap.sh, The assembler you choose (-A).
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
output_dir="$(pwd)"_RP_assembly #o
rm_tmp="False" #r
logfile=$(pwd)/"$(date "+%F+%T")"_"logfile.txt"
assembler=spades
# input_reads ='*.fq.gz' #i
# merged=none

#### Parse args: #####
while getopts t:i:o:r:g:M:A:S:s:m: flag
do
case "${flag}" in
t) THREADS=${OPTARG};;
i) sample_dir=${OPTARG};;
o) output_dir=${OPTARG};;
r) rm_tmp=${OPTARG} ;;
g) logfile=${OPTARG} ;;
M) MEMORY=${OPTARG};;
A) assembler=${OPTARG};;
S) rolypoly_dir=${OPTARG};;
m) merged=${OPTARG};;
*) usage
esac
done


# THREADS=6
# sample_dir=/REDACTED_HPC_PATH/ynp/raw_reads/spids/1042213/
# output_dir=/REDACTED_HPC_PATH/ynp/raw_reads/spids/1042213/Assembly
# rm_tmp=False
# logfile=/REDACTED_HPC_PATH/ynp/raw_reads/spids/1061592/assembly_test.log
# MEMORY=20g
# assembler=spades
# rolypoly_dir=/REDACTED_HPC_PATH/rolypoly/
# cd /REDACTED_HPC_PATH/ynp//raw_reads/spids/1042213/


# Mandatory arguments
if [ ! "$sample_dir" ] ; then
usage
echo "argument -i must be provided"
exit
fi

##### Set enviroment #####
datadir="$rolypoly_dir"/data/
export rolypoly_dir=$rolypoly_dir
source "$rolypoly_dir"/rolypoly/utils/load_bins.sh
source "$rolypoly_dir"/rolypoly/utils/bash_functions.sh

logfile=$(echo $(readlink -f "$logfile")) # Get abs path.

export MEMORY=$MEMORY
MEMORY_bytes=$(numfmt --from si  $(echo $MEMORY | sed  's|g|G|g' ))
MEMORY_nsuffix=$(echo $MEMORY | sed 's|g||g')

export THREADS=$THREADS


##### Check dependencies #####
Dependency_List=(awk bbmap.sh spades.py megahit spades.py)
check_dependencies "${Dependency_List[@]}"

##################################################################################################
##### Ready or not message #####
logit $logfile "script: $0" 
logit $logfile "Params: ${@}" 
logit $logfile "Launch location: $PWD "
logit $logfile "Time and date: $(date)"
logit $logfile "Submitter name: $USER"
logit $logfile "HOSTNAME: $HOSTNAME"

##### Main #####
output_dir=$(echo $(readlink -f "$output_dir")  ) # Get abs path.
file_name=$(basename $(echo $sample_dir) .fastq.gz)
file_name=$(basename $(echo $file_name) .fq.gz) # just in case.
file_name=$(echo $file_name | sed 's|qtrimmed_||g'  ) # just in case.

mkdir $output_dir # try in case the it isn't there
cd $output_dir

# --- Assembly ---
logit $logfile "Started assembly for:  $file_name"
run_spades=$( echo $assembler  | grep -i spades -c -m1 )
run_megahit=$( echo $assembler  | grep -i  megahit -c -m1 )
run_penguin=$( echo $assembler  | grep -i  penguin -c -m1 )

    all_qtrimmed=$(find "$sample_dir" -name 'qtrimmed*.fq.gz')
    all_merged=$(find "$sample_dir" -name 'merged*.fq.gz')
    n_libraries=$(ls ${all_merged} | wc -l) ; #
    logit $logfile "found  $n_libraries libraries, files:
    $all_qtrimmed, $all_merged"

    qtrimmed=$(echo $all_qtrimmed | sed 's| |,|g') # comma seperated
    merged=$(echo $all_merged | sed 's| |,|g') # comma seperated

if [ "$run_spades" == 1 ];
then
    logit $logfile "Started SPAdes assembly for:  $file_name"
    # mode=rnaviral # meta
    mode=meta # meta
    # Initialize a variable to hold the spades command
    spades_cmd="spades.py --$mode -o '$output_dir'/spades_'$mode'_output --threads $THREADS --only-assembler  -k 21,33,45,57,63,69,71,83,95,103,107,111,119 --phred-offset 33  -m $MEMORY_nsuffix "

    if [ "$n_libraries" -gt 9 ]|| [ "$n_libraries" -eq 1 ] || [ "$mode" == "meta" ]; then
        logit $logfile "Found $n_libraries libraries, (>9 || ==1), so Running SPAdes directly or on a concatenation of the reads (max supported number of libraries is 9)"
        cat $all_merged > all_merged.fq.gz
        cat $all_qtrimmed > all_qtrimmed.fq.gz
        spades_cmd+=" --pe-12 1 all_qtrimmed.fq.gz --pe-m 1 all_merged.fq.gz"
        eval $spades_cmd

    # fi

    # if [ "$n_libraries" -lt 9 ]; then
    else
        logit $logfile "Running SPAdes via multiple supplied libraries"
        # Initialize a library counter
        lib_num=1
        # Loop through each subdirectory within the base directory
        for library_dir in "$sample_dir"/*; do
        # Check if it is a directory
        if [ -d "$library_dir" ]; then
            logit $logfile "looking in directory: $library_dir"
                    # Find qtrimmed and merged files
                    nqtrimmed=$(find "$library_dir" -name 'qtrimmed*.fq.gz')
                    nmerged=$(find "$library_dir" -name 'merged*.fq.gz')
                    
                    # Check if both files are found
                    if [ -n "$nqtrimmed" ] && [ -n "$nmerged" ]; then
                        logit $logfile "    Found qtrimmed file: $nqtrimmed"
                        logit $logfile "    Found merged file: $nmerged"
                        
                        # Add the files to the spades command with appropriate library numbers
                        spades_cmd+=" --pe-12 $lib_num $nqtrimmed --pe-m $lib_num $nmerged"
                        
                        # Increment the library counter
                        lib_num=$((lib_num + 1))
                    else
                        logit $logfile "    Required files not found in: $library_dir"
                    fi
                fi
            done

        # Run the spades command if it contains the required parameters
        if [ "$lib_num" -gt 1 ]; then
            logit $logfile "Running SPAdes command: $spades_cmd"
            eval $spades_cmd
        else
            logit $logfile "  No valid qtrimmed and merged file pairs found in: $SAMPLE_DIR"
        fi
    fi
    contigs4eval="$output_dir"/spades_"$mode"_output/scaffolds.fasta
    logit $logfile "Finished SPAdes assembly for:  $file_name"
fi

if [ "$run_megahit" == 1 ];
then
    mode=custom
    logit $logfile "Started Megahit assembly for:  $file_name"
    megahit --k-min 21 --k-max 147 --k-step 8  --min-contig-len 30 --12 $qtrimmed --read $merged  --out-dir "$output_dir"/megahit_"$mode"_out   --num-cpu-threads $THREADS  --memory $MEMORY_bytes  # --k-list 21,29,39,47,59,79,99,119,141  --merge-level 10,0.7
    final_k=$(ls -1 "$output_dir"/megahit_"$mode"_out/intermediate_contigs/*\.final.contigs.fa | sed 's/.*_contigs\///g' | sort -n -k1.2 | tail -1 | cut -d'k' -f2 | cut -d'.' -f1 )
    megahit_toolkit contig2fastg $final_k  "$output_dir"/megahit_"$mode"_out/final.contigs.fa  > "$output_dir"/megahit_"$mode"_out/final_megahit_assembly_k"$final_k".fastg
    contigs4eval+=" $output_dir"/megahit_"$mode"_out/final.contigs.fa
    logit $logfile "Finished Megahit assembly for:  $file_name"
fi

if [ "$run_penguin" == 1 ];
then
    mkdir tmp
    logit $logfile "Started penguin assembly for:  $file_name"
    penguin guided_nuclassemble  $all_qtrimmed $all_merged "$output_dir"/penguin_Fguided_1_nuclassemble_c0.fasta ./tmp/   --min-contig-len 150  --contig-output-mode 0 --num-iterations  aa:1,nucl:10  --min-seq-id nucl:0.990,aa:1 --min-aln-len nucl:23,aa:150 --clust-min-seq-id 0.98 --clust-min-cov 0.99 --threads $THREADS  #--split-memory-limit $MEMORY
    seqkit rmdup  -n "$output_dir"/penguin_Fguided_1_nuclassemble_c0.fasta | seqkit seq -m 150 -w 0 > "$output_dir"/penguin_Fguided_m150_nuclassemble_out.fasta # mmseqs and Co. weirdness

    # penguin guided_nuclassemble  $all_qtrimmed $all_merged penguin_guided_1_nuclassemble_c1_"$file_name".fasta ./tmp/   --min-contig-len 30  --contig-output-mode 0 --num-iterations  aa:1,nucl:10  --min-seq-id nucl:0.990,aa:1 --min-aln-len nucl:21,aa:150 --clust-min-seq-id 1 --clust-min-cov 1 --threads $THREADS  #--split-memory-limit $MEMORY
    # penguin nuclassemble  $all_qtrimmed $all_merged "$output_dir"/penguin_nuclassemble_out.fasta ./tmp/ --min-aln-len 21 --min-seq-id 0.97 --num-iterations 12  --min-contig-len 30  --contig-output-mode 0 --threads $THREADS --sort-results 1 #--split-memory-limit $MEMORY
    # mmseqs easy-linclust "$output_dir"/penguin_nuclassemble_out.fasta "$output_dir"/penguin_nuclassemble_out_clstr tmp --min-seq-id 1 -c 1 --threads $THREADS 
    # seqkit rmdup  -n penguin_nuclassemble_out_clstr_rep_seq.fasta | seqkit seq -m 30 > penguin_nuclassemble_out.fasta # mmseqs and Co. weirdness
    # rm penguin_nuclassemble_out_clstr_cluster.tsv penguin_nuclassemble_out_clstr_all_seqs.fasta penguin_nuclassemble_out_clstr_rep_seq.fasta
    # Doesn't make much sense for OLC I think    
    # megahit_toolkit contig2fastg 21 penguin_nuclassemble_out.fasta > penguin_nuclassemble_out.fastg
    # spades-gbuilder penguin_nuclassemble_out.fasta  penguin_nuclassemble_out_gubilder.gfa --gfa -k 21 -t  $THREADS
    contigs4eval+=" $output_dir"/penguin_Fguided_m150_nuclassemble_out.fasta
fi


# --- Evaluation ---
contigs4eval_comma=$(echo $contigs4eval |sed 's| |,|g' )
seqkit fx2tab -s  $(echo $contigs4eval)  | sed 's| |_|g' |awk  '{print ">"$3"_"$1"\n"$2}' >> "$output_dir"/merged_contigs.fasta
mmseqs easy-linclust "$output_dir"/merged_contigs.fasta "$output_dir"/merged_contigs_clst tmp --min-seq-id 1 -c 1 --threads $THREADS  --cov-mode 1

logit $logfile "Started assembly evaluation on:  $contigs4eval"
statswrapper.sh in=$(echo $contigs4eval_comma,merged_contigs_clst_rep_seq.fasta) format=3 out="$file_name"_assembly_stats.tsv ow=t 

#Calculate the coverage distribution, and capture reads that did not make it into the assembly
bbwrap.sh ref=merged_contigs_clst_rep_seq.fasta in=$qtrimmed,$merged out=bbwrap_output.sam threads=$THREADS  nodisk covhist="$file_name"_assebmly_covhist.txt covstats="$file_name"_assebmly_covstats.txt outm="$file_name"_assebmly_bbw_assembled.fq.gz outu="$file_name"_assebmly_bbw_unassembled.fq.gz maxindel=200 minid=90 untrim ambig=best append=t

#Search for reads via MMseqs2 which allows multiple HSPs on different target seqs (easily)
mkdir contigs_mmdb raw_reads_mmdb searchmmdb tmp 
mmseqs createdb  $contigs4eval  contigs_mmdb/mmdb 
mmseqs createdb $all_qtrimmed $all_merged raw_reads_mmdb/rdb 
mmseqs search contigs_mmdb/mmdb  raw_reads_mmdb/rdb  searchmmdb/res ./tmp/  --min-seq-id 0.7 --search-type 3 --threads $THREADS -a # --format-mode 1  
mmseqs convertalis contigs_mmdb/mmdb  raw_reads_mmdb/rdb  searchmmdb/res "$file_name"_assebmly_mm_out.tab --format-mode 4 --format-output qheader,theader,qlen,tlen,qstart,qend,tstart,tend,alnlen,mismatch,qcov,tcov,bits,evalue,gapopen,pident,nident
mmseqs convertalis contigs_mmdb/mmdb  raw_reads_mmdb/rdb  searchmmdb/res "$file_name"_assebmly_mm_out.sam --format-mode 1  --search-type 3

rm tmp searchmmdb -rf
rm $(find . -name "tmp*" ) -r
bgzip  -@$THREADS *_assebmly_covstats.txt
bgzip  -@$THREADS *_assebmly_mm_out.sam
bgzip  -@$THREADS *_assebmly_mm_out.tab

logit $logfile "Finished assembly evaluation on:  $contigs4eval"
