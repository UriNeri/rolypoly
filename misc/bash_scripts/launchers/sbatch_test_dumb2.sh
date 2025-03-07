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
#SBATCH --mem=100G 
## specify runtime
#SBATCH -t 12:00:00
## specify job name
#SBATCH -J SE_IMGPR
## specify output and error file
#SBATCH -o /REDACTED_HPC_PATH/spasm/Slurmout-%A_%a.out


source ~/.bashrc
conda activate crispy   
cd /REDACTED_HPC_PATH/

# ~/code/MEGAHIT-1.2.9-Linux-x86_64-static/bin/megahit -r ../spacers.fasta  --min-count 1 -t 24 -o test_dmb3 --k-list 21,23,25,27,29,31,33,35,37,39,59,79,99,119,141 --merge-level 20,0.85 --prune-depth 1 --min-contig-len 20
~/code/MEGAHIT-1.2.9-Linux-x86_64-static/bin/megahit -r ./test_dmb3/final.contigs.fa  --min-count 1 -t 24 -o test_dmb4 --k-list 21,23,25,27,29,31,33,35,37,39,59,79,99,119,141 --merge-level 20,0.80 --prune-depth 1 --min-contig-len 20
# ~/code/SPAdes-4.0.0-Linux/bin/spades.py -s ./test_dmb3/final.contigs.fa --sc -k 31,33,35,37,53,61,75,91 -o test_spades4 --only-assembler --threads 24 --memory 100 --cov-cutoff off

# ~/code/SPAdes-4.0.0-Linux/bin/spades.py  -k 31,33,35,37,53,61,75,91 -o test_spades4 --threads 24 --memory 100 --cov-cutoff off --restart-from k75 --careful
