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
#SBATCH -c 30
## specify ram
#SBATCH --mem=160G 
## specify runtime
#SBATCH -t 48:00:00
## specify job name
#SBATCH -J spid_assembly
## specify output and error file
#SBATCH -o /REDACTED_HPC_PATH/ynp/slurmsout/assemblies/Slurmout-%A_%a.out
## specify that we never run more than XX jobs at a time (using "%", e.g. --array=0-15%4)
#SBATCH --array=1-154%3


source ~/.bashrc
conda activate rolypoly   
# echo $SLURM_MEM_PER_NODE


THREADS=30 # $SLURM_CPUS_PER_TASK
export THREADS=$THREADS

MEMORY=160g # "$SLURM_MEM_PER_NODE" # Might need to add "g" suffix.
export MEMORY=$MEMORY
MEMORY_nsuffix=$(echo $MEMORY | sed 's|g||g')
MEMORY_bytes=$(numfmt --from si  $(echo $MEMORY | sed  's|g|G|g' ))

rolypoly_dir=/REDACTED_HPC_PATH/rolypoly/
export rolypoly_dir=$rolypoly_dir

cd /REDACTED_HPC_PATH/ynp/
cd raw_reads

# cat YNP_RNA_info_fastqs.tsv | cut -f1 -d' ' | sort -u -r > spids.lst
spid=$(awk -v varrr="$SLURM_ARRAY_TASK_ID" 'NR==varrr' /REDACTED_HPC_PATH/ynp/meta/spids.lst)
echo $spid
if [ -d "spids/$spid" ]; then
echo $spid found in /REDACTED_HPC_PATH/ynp/raw_reads/spids/
cd spids/$spid
spid_path=$(realpath ./)
assembler=spades,megahit,penguin
output_dir="$spid"_assemblies
logfile="$spid"_assembly_log.txt
rm $output_dir -r
rm $logfile

bash /REDACTED_HPC_PATH/rolypoly/rolypoly/in_reads/assembly.sh -i $spid_path -r False -o "$output_dir" -g $logfile -S $rolypoly_dir -A $assembler -t $THREADS -M $MEMORY
else
echo $spid not found in /REDACTED_HPC_PATH/ynp/raw_reads/spids/
fi
