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
#SBATCH --mem=40G 
## specify runtime
#SBATCH -t 24:00:00
## specify job name
#SBATCH -J filter_search_rdrp
## specify output and error file
#SBATCH -o /REDACTED_HPC_PATH/ynp/slurmsout/search/Slurmout-%A_%a.out
## specify that we never run more than XX jobs at a time (using "%", e.g. --array=0-15%4)
#SBATCH --array=1-1%1

# SasdBATCH --array=1-153%1
# 278,284,285,286,287,288,289,290,291,295,296,302,303,304,305,306,307,308,311,312,320,321,322,324,325,326,327,328,330,331,332,333%1


source ~/.bashrc
conda activate <HOME_PATH>/miniconda3/envs/rolypoly
# echo $SLURM_MEM_PER_NODE

# job_id_in_array=$SLURM_ARRAY_TASK_ID
# job_id_in_array=245
echo $SLURM_ARRAY_TASK_ID
THREADS=$SLURM_CPUS_PER_TASK
export THREADS=$THREADS


MEMORY=40g # "$SLURM_MEM_PER_NODE" # Might need to add "g" suffix.
export MEMORY=$MEMORY
MEMORY_nsuffix=$(echo $MEMORY | sed 's|g||g')
MEMORY_bytes=$(numfmt --from si  $(echo $MEMORY | sed  's|g|G|g' ))

rolypoly_dir=/REDACTED_HPC_PATH/rolypoly/
export rolypoly_dir=$rolypoly_dir
source "$rolypoly_dir"/rolypoly/utils/load_bins.sh
source "$rolypoly_dir"/rolypoly/utils/bash_functions.sh

spid=$(awk -v varrr="$SLURM_ARRAY_TASK_ID" 'NR==varrr' /REDACTED_HPC_PATH/ynp/meta/spids.lst)
cd /REDACTED_HPC_PATH/ynp/raw_reads/spids
echo $spid
if [ -d "$spid" ]; then
echo $spid found in /REDACTED_HPC_PATH/ynp/raw_reads/spids/
cd $spid
spid_path=$(realpath ./)
output_dir="$spid"_assemblies
logfile="$spid"_assembly_filter_log.txt

input="$spid_path"/"$spid"_assemblies/merged_contigs_clst_rep_seq.fasta
rolypoly filter_assembly -i  $input -o "$spid_path"/"$spid"_assemblies/filtered_merged_assembly.fasta --threads $THREADS --log-file $logfile --host /REDACTED_HPC_PATH/ynp/YNP_DNA/host_db/dnammdb --filter2 "qcov >= 0.9 & pident > 0.9" --mmseqs-args "--min-seq-id 0.5" --memory $MEMORY
# input="$spid_path"/"$spid"_assemblies/filtered_merged_assembly.fasta
# python /REDACTED_HPC_PATH/rolypoly/rolypoly/pyrolypoly/search_viruses.py --input  $input --threads $THREADS --db RVMT --output "$spid_path"/"$spid"_assemblies/rp_sv  -S $rolypoly_dir 
# python /REDACTED_HPC_PATH/rolypoly/rolypoly/pyrolypoly/marker_searchpy --input-file $input --threads $THREADS   --output-file "$spid_path"/"$spid"_assemblies/rp_rs  -S $rolypoly_dir  #--db RVMT

else
echo $spid not found in /REDACTED_HPC_PATH/ynp/raw_reads/spids/
fi


# /REDACTED_HPC_PATH/rolypoly/bin/cath-resolver --input-format hmmer_domtblout --html-output-to-file test.html --quiet --html-exclude-rejected-hits test_rp_rs # --worst-permissible-score 19
