source ~/.bashrc
conda activate crispy   

THREADS=10

alias jamo='apptainer run docker://doejgi/jamo-dori jamo'


awk -v RS='\t' '/ITS SP ID/{print NR; exit}' /clusterfs/jgi/scratch/science/metagen/neri/rolypoly/rolypoly/metaTs_for_rolypoly_tunning.tsv
awk -F'\t' '{print $23}' /clusterfs/jgi/scratch/science/metagen/neri/rolypoly/rolypoly/metaTs_for_rolypoly_tunning.tsv | tail |sort -u -r   > spids.tsv



cd /clusterfs/jgi/scratch/science/metagen/neri/rolypoly/bench/
wget https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/025/685/GCF_000025685.1_ASM2568v1/GCF_000025685.1_ASM2568v1_genomic.fna.gz
extract  GCF_000025685.1_ASM2568v1_genomic.fna.gz

cd /clusterfs/jgi/scratch/science/metagen/neri/rolypoly/bench/metaTs

jamo  fetch assembled.fna -s dori spid  $(cat spids.tsv)  > jamo_info.tsv
awk -F' ' '$2=="RESTORED" {print $1}' ./jamo_info.tsv  | xargs -I %  cp % ./

cp /clusterfs/jgi/scratch/dsi/aa/dm_archive/img/submissions/65383/65383.assembled.fna ./
cp /clusterfs/jgi/scratch/dsi/aa/dm_archive/img/submissions/61747/61747.assembled.fna ./
cp /clusterfs/jgi/scratch/dsi/aa/dm_archive/img/submissions/183094/183094.assembled.fna ./
cp /clusterfs/jgi/scratch/dsi/aa/dm_archive/img/submissions/76857/76857.assembled.fna ./

jamo  info img -s dori spid  1036156 1036157 1036156 1057031 1068976 1089242 1176361 1335836 1407040 > jamo_info.tsv
jamo  fetch filtered -s dori spid  1036156 1036157 1036156 1057031 1068976 1089242 1176361 1335836 1407040 > filtered_jamo_info.tsv
jamo  info filtered -s dori spid  1036156 1036157 1036156 1057031 1068976 1089242 1176361 1335836 1407040 > filtered_jamo_info.tsv

grep "contigs.fna" -F  jamo_info.tsv >> files2get.lst
grep "assembled.fna" -F  jamo_info.tsv >> files2get.lst
awk -F' ' '$3=="RESTORED" {print $2}' ./files2get.lst  | xargs -I %  cp % ./
awk -F' ' ' {print $2}' ./files2get.lst  | xargs -I %  cp % ./


# The closest thing I could find to a TPA of human gut/blood etc related metatranscriptome. Noted as a "benchmarking" so this should be taken with caution.
# wget ftp://ftp.ebi.ac.uk/pub/databases/ena/tsa/public/hbd/HBDA01.fasta.gz # https://www.ebi.ac.uk/ena/browser/view/HBDA01000000?show=xrefs
# extract HBDA01.fasta.gz
shred.sh in=metaTs_spiced_RVMT.fasta out=metaTs_spiced_RVMT.fastq.gz qfake=33

# cat *.fna *fasta > combined_metaTs.fas
cat *fastq.gz > cated.fq.gz
cat cated.fq.gz shreded_rvmt.fq.gz > metaTs_spiced_RVMT.fastq.gz

# cat /clusterfs/jgi/scratch/science/metagen/neri/rolypoly/data/RVMT/RiboV1.6_Contigs.fasta combined_metaTs.fas > metaTs_spiced_RVMT.fasta



THREADS=6
MEMORY=10g
export THREADS=$THREADS

rolypoly_dir=/clusterfs/jgi/scratch/science/metagen/neri/rolypoly/
export rolypoly_dir=$rolypoly_dir
datadir="$rolypoly_dir"/data/

source "$rolypoly_dir"/rolypoly/utils/bash_functions.sh  

known_DNA_present=/clusterfs/jgi/scratch/science/metagen/neri/rolypoly/bench/GCF_000025685.1_ASM2568v1_genomic.fna

cd /clusterfs/jgi/scratch/science/metagen/neri/rolypoly/bench

fastq_file=/clusterfs/jgi/scratch/science/metagen/neri/rolypoly/bench/metaTs/metaTs_spiced_RVMT.fastq.gz # Big 
fastq_file=/clusterfs/jgi/scratch/science/metagen/neri/rolypoly/bench/sampled_0.05_metaTs_spiced_RVMT.fastq.gz # Small - Sampled 0.05% (proportaion) of the reads.


file_name=$(basename $(echo $fastq_file) .fastq.gz)

bash /clusterfs/jgi/scratch/science/metagen/neri/rolypoly/rolypoly/filter_reads.sh -t $THREADS -M $MEMORY -i $fastq_file -D $known_DNA_present -r true -o "test_"$file_name -g "$file_name"_log.txt -S /clusterfs/jgi/scratch/science/metagen/neri/rolypoly/





echo $SLURM_MEM_PER_NODE
THREADS=24 # $SLURM_CPUS_PER_TASK
MEMORY=115g # "$SLURM_MEM_PER_NODE" # Might need to add "g" suffix.
known_DNA_present=/clusterfs/jgi/scratch/science/metagen/neri/rolypoly/bench/GCF_000025685.1_ASM2568v1_genomic.fna


export THREADS=$THREADS

rolypoly_dir=/clusterfs/jgi/scratch/science/metagen/neri/rolypoly/
export rolypoly_dir=$rolypoly_dir
# source "$rolypoly_dir"/rolypoly/utils/bash_functions.sh  


cd /clusterfs/jgi/scratch/science/metagen/neri/rolypoly/bench

# fastq_file=/clusterfs/jgi/scratch/science/metagen/neri/rolypoly/bench/metaTs/metaTs_spiced_RVMT.fastq.gz # Big 
fastq_file=/clusterfs/jgi/scratch/science/metagen/neri/rolypoly/bench/metaTs/sampled_005_bb_metaTs_spiced_RVMT.fastq.gz # Small - Sampled 0.05% (proportaion) of the reads.

file_name=$(basename $(echo $fastq_file) .fastq.gz)

bash /clusterfs/jgi/scratch/science/metagen/neri/rolypoly/rolypoly/in_reads/filter_reads.sh -t $THREADS -M $MEMORY -i $fastq_file -D $known_DNA_present -r true -o "test_"$file_name -g "$file_name"_log.txt -S $rolypoly_dir

