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
#SBATCH --mem=1250G 
## specify runtime
#SBATCH -t 48:00:00
## specify job name
#SBATCH -J co_assembly
## specify output and error file
#SBATCH -o /REDACTED_HPC_PATH/ynp/slurmsout/assemblies/Slurmout-%A_%a.out
## specify that we never run more than XX jobs at a time (using "%", e.g. --array=0-15%4)
# SasdBATCH --array=1-154%3


source ~/.bashrc
conda activate rolypoly   
# echo $SLURM_MEM_PER_NODE

THREADS=30 # $SLURM_CPUS_PER_TASK
export THREADS=$THREADS

MEMORY=1250g # "$SLURM_MEM_PER_NODE" # Might need to add "g" suffix.
export MEMORY=$MEMORY
MEMORY_nsuffix=$(echo $MEMORY | sed 's|g||g')
MEMORY_bytes=$(numfmt --from si  $(echo $MEMORY | sed  's|g|G|g' ))

rolypoly_dir=/REDACTED_HPC_PATH/rolypoly/
export rolypoly_dir=$rolypoly_dir
source "$rolypoly_dir"/rolypoly/utils/load_bins.sh
source "$rolypoly_dir"/rolypoly/utils/bash_functions.sh

mkdir /REDACTED_HPC_PATH/ynp/combined_assembly/
output_dir=/REDACTED_HPC_PATH/ynp/combined_assembly/
qtrim12=/REDACTED_HPC_PATH/ynp/raw_reads/all_qtrimmed.fq.gz
merged=/REDACTED_HPC_PATH/ynp/raw_reads/all_merged.fq.gz
logfile=/REDACTED_HPC_PATH/ynp/combined_assembly/coassembly_log.txt

## megahit
    mode=custom
    # logit $logfile "Started Megahit assembly for:  $qtrim12 and $merged"
    # megahit --continue --k-min 21 --k-max 147 --k-step 8  --min-contig-len 150 --12 $qtrim12 --read $merged  --out-dir "$output_dir"/megahit_"$mode"_out   --num-cpu-threads $THREADS  --memory $MEMORY_bytes  # --k-list 21,29,39,47,59,79,99,119,141  --merge-level 10,0.7
    # final_k=$(ls -1 "$output_dir"/megahit_"$mode"_out/intermediate_contigs/*\.final.contigs.fa | sed 's/.*_contigs\///g' | sort -n -k1.2 | tail -1 | cut -d'k' -f2 | cut -d'.' -f1 )
    # megahit_toolkit contig2fastg $final_k  "$output_dir"/megahit_"$mode"_out/final.contigs.fa  > "$output_dir"/megahit_"$mode"_out/final_megahit_assembly_k"$final_k".fastg
    contigs4eval="$output_dir"/megahit_"$mode"_out/final.contigs.fa
    # logit $logfile "Finished Megahit assembly for: $qtrim12 and $merged"

## SPAdes
    mode=meta #metaviral # rnaviral
    # spades.py --pe-12 1 $qtrim12  --pe-m 1 $merged --$mode -o "$output_dir"/spades_"$mode"_out --threads $THREADS --only-assembler  -k 21,33,45,57,63,69,71,83,95,103,107,111,119 --phred-offset 33  -m $MEMORY_nsuffix 
    # spades.py  -o "$output_dir"/spades_"$mode"_out --continue # --threads $THREADS  --restart-from last # --only-assembler  -k 21,33,45,57,63,69,71,83,95,103,107,111,119 --phred-offset 33  -m $MEMORY_nsuffix 
    
    contigs4eval+=" $output_dir"/spades_"$mode"_out/scaffolds.fasta
    logit $logfile "Finished SPAdes assembly for:  $qtrim12 and $merged"

## penguin
    mkdir tmp
    logit $logfile "Started penguin assembly for:  $qtrim12 and $merged"
    penguin guided_nuclassemble  $qtrim12 $merged "$output_dir"/penguin_Fguided_1_nuclassemble_c0.fasta ./tmp/   --min-contig-len 150  --contig-output-mode 0 --num-iterations  aa:1,nucl:10  --min-seq-id nucl:0.990,aa:1 --min-aln-len nucl:23,aa:150 --clust-min-seq-id 0.98 --clust-min-cov 0.99 --threads $THREADS  #--split-memory-limit $MEMORY
    seqkit rmdup  -n "$output_dir"/penguin_Fguided_1_nuclassemble_c0.fasta | seqkit seq -m 150 -w 0 > "$output_dir"/penguin_Fguided_m150_nuclassemble_out.fasta # mmseqs and Co. weirdness

    # penguin nuclassemble $qtrim12 $merged "$output_dir"/penguin_nuclassemble_out.fasta ./tmp/ --min-aln-len 23 --min-seq-id 0.97 --num-iterations 12  --min-contig-len 30  --contig-output-mode 0 --threads $THREADS  #--split-memory-limit $MEMORY

    # mmseqs easy-linclust "$output_dir"/penguin_nuclassemble_out.fasta "$output_dir"/penguin_nuclassemble_out_clstr tmp --min-seq-id 0.99 -c 0.99 --cov-mode 5 --threads $THREADS 
    # seqkit rmdup  -n "$output_dir"/penguin_nuclassemble_out_clstr_rep_seq.fasta | seqkit seq -m 150 -w 0 > "$output_dir"/penguin_nuclassemble_out.fasta # mmseqs and Co. weirdness
    # rm penguin_nuclassemble_out_clstr_cluster.tsv penguin_nuclassemble_out_clstr_all_seqs.fasta penguin_nuclassemble_out_clstr_rep_seq.fasta
    # Doesn't make much sense for OLC I think    
    # megahit_toolkit contig2fastg 21 penguin_nuclassemble_out.fasta > penguin_nuclassemble_out.fastg
    # spades-gbuilder penguin_nuclassemble_out.fasta  penguin_nuclassemble_out_gubilder.gfa --gfa -k 21 -t  $THREADS
    contigs4eval+=" $output_dir"/penguin_Fguided_m150_nuclassemble_out.fasta
    logit $logfile "Finished penguin assembly for:  $qtrim12 and $merged"



# --- Evaluation ---
cd "$output_dir"
contigs4eval_comma=$(echo $contigs4eval |sed 's| |,|g' )

seqkit fx2tab -s  $(echo $contigs4eval)  | sed 's| |_|g' |awk  '{print ">"$3"_"$1"\n"$2}' |seqkit seq -m150 | seqkit rmdup -s -D "$output_dir"/dup_cross_assembly.lst  --ignore-case  -w 0  --threads $THREADS  >> "$output_dir"/merged_contigs.fasta
mmseqs easy-linclust $(echo $contigs4eval) "$output_dir"/merged_linclust_covmode5_c99_id_99 "$output_dir"/tmp/ --min-seq-id 0.99 -c 0.99 --threads 5 --cov-mode 5

logit $logfile "Started evaluation on:  $(echo $contigs4eval)"
statswrapper.sh in=$(echo $contigs4eval_comma) format=3 out="$output_dir"/coassemblies_stats.tsv

# #Calculate the coverage distribution, and capture reads that did not make it into the assembly
# bbwrap.sh ref="$output_dir"/merged_contigs.fasta in=$qtrim12,$merged out=bbwrap_output_merged_contigs.sam threads=$THREADS  nodisk covhist="$file_name"_assebmly_covhist.txt covstats="$file_name"_assebmly_covstats.txt outm="$file_name"_assebmly_bbw_assembled.fq.gz outu="$file_name"_assebmly_bbw_unassembled.fq.gz maxindel=200 minid=90 untrim ambig=best

# #Search for reads via MMseqs2 which allows multiple HSPs on different target seqs (easily)
# mkdir contigs_mmdb raw_reads_mmdb searchmmdb tmp 
# mmseqs createdb  $contigs4eval  contigs_mmdb/mmdb 
# mmseqs createdb $qtrim12 $merged raw_reads_mmdb/rdb 
# mmseqs search contigs_mmdb/mmdb  raw_reads_mmdb/rdb  searchmmdb/res ./tmp/  --min-seq-id 0.7 --search-type 3 --threads $THREADS -a # --format-mode 1  
# mmseqs convertalis contigs_mmdb/mmdb  raw_reads_mmdb/rdb  searchmmdb/res "$file_name"_assebmly_mm_out.tab --format-mode 4 --format-output qheader,theader,qlen,tlen,qstart,qend,tstart,tend,alnlen,mismatch,qcov,tcov,bits,evalue,gapopen,pident,nident
# mmseqs convertalis contigs_mmdb/mmdb  raw_reads_mmdb/rdb  searchmmdb/res "$file_name"_assebmly_mm_out.sam --format-mode 1  --search-type 3

# rm tmp searchmmdb -rf
# bgzip  -@$THREADS *_assebmly_covstats.txt
# bgzip  -@$THREADS *_assebmly_mm_out.sam
# bgzip  -@$THREADS *_assebmly_mm_out.tab

# logit $logfile "Finished assembly evaluation on:  $contigs4eval"



# mmseqs easy-search $qtrim12 $merged $(echo $contigs4eval) /REDACTED_HPC_PATH/ynp/raw_reads/mm_out_eleven.sam ./tmp/ --min-aln-len 100 --min-seq-id 0.85 --format-mode 1  --search-type 3 --threads $THREADS 



# # # spades.py  -o combined_assembly/spades_out --pe-12 1 $qtrim12  --pe-m 1 $merged --threads $THREADS  -m $MEMORY_nsuffix  --only-assembler --phred-offset 33 --cov-cutoff off 
# spades.py  -o combined_assembly/spades_out --restart-from last --threads $THREADS -m $MEMORY_nsuffix
# # # spades.py  -o combined_assembly/spades_out_rnaviral --pe-12 1 $qtrim12  --pe-m 1 $merged --threads $THREADS  -m $MEMORY_nsuffix  --only-assembler --phred-offset 33 --cov-cutoff off --rnaviral

# # ## Penguin 
# #     penguin nuclassemble  $qtrim12 $merged penguin_nuclassemble_c1_"$file_name".fasta ./tmp/   --min-contig-len 30  --contig-output-mode 1 --threads $THREADS --sort-results 1 #--split-memory-limit $MEMORY



# #  contgs4eval=/REDACTED_HPC_PATH/ynp/raw_reads/Eleven.fa
# #  mmseqs easy-search $qtrim12 $merged $contgs4eval /REDACTED_HPC_PATH/ynp/raw_reads/mm_out_eleven.sam ./tmp/ --format-mode 1  --search-type 3 --threads $THREADS --force-reuse true
# # bbwrap.sh ref=$contgs4eval in=$qtrim12,$merged out=bbwrap_output_selected.sam threads=$THREADS  nodisk covhist=covhist.txt covstats=covstats.txt outm=bbwrap_output_selected_mapped.sam outu=unassembled.fq.gz maxindel=200 minid=70  untrim ambiguous=best secondary=t maxsites=15 sssr=0.1

# # testtt=temp.fasta.split/temp.part_001.fasta,temp.fasta.split/temp.part_002.fasta,temp.fasta.split/temp.part_002.fasta
# # bbwrap.sh ref=$testtt in=$qtrim12,$merged out=bbwrap_output_selected.sam
# # bbmapskimmer.sh ref=$contgs4eval in=$qtrim12 out=bbmapskimmer_output_selected.sam threads=$THREADS k=8 nodisk covhist=covhist.txt covstats=covstats.txt outm=bbmapskimmer_output_selected_mapped.sam outu=unassembled.fq.gz maxindel=40 minid=70  untrim ambiguous=best secondary=t  maxsites=1 sssr=0.9
# # bbmapskimmer.sh ref=$contgs4eval in=$merged out=bbmapskimmer_output_selected.sam threads=$THREADS k=8 nodisk covhist=covhist.txt covstats=covstats.txt outm=bbmapskimmer_output_selected_mapped.sam outu=unassembled.fq.gz maxindel=40 minid=70  untrim ambiguous=best secondary=t maxsites=11 sssr=0.9
# # # in1=read1.fq,singleton.fq in2=read2.fq,null
# # # 
# # contgs4eval=temp.fasta  





# # cd /REDACTED_HPC_PATH/ynp/
# # cd raw_reads

# # # blalba=$(awk -v varrr="$job_id_in_array" 'NR==varrr' YNP_RNA_info_fastqs.tsv)
# # # fastq_file=$(echo  $blalba | cut -f2 -d' ')
# # # file_name=$(basename $(echo $fastq_file) .fastq.gz)

# # # spid=$(echo  $blalba | cut -f1 -d' ')
# # # mkdir $spid
# # # cd $spid
# # # cd $file_name
# # # qtrim12=qtrimmed_"$file_name".fq.gz
# # # merged=merged_"$file_name".fq.gz
# # # assembler=spades,megahit
# # # outdir="$file_name"_assemblies
# # # logfile="$file_name"_log.txt

# # # bash /REDACTED_HPC_PATH/rolypoly/rolypoly/in_reads/assembly.sh -m $merged -i $fastq_file -r False -o "$outdir" -g $logfile -S $rolypoly_dir -A $assembler -t $THREADS -M $MEMORY


# # mkdir combined_assembly
# # qtrim12=all_qtrimmed.fq.gz
# # merged=all_merged.fq.gz
# # assembler=spades,megahit
# # outdir="$file_name"_assemblies
# # logfile="$file_name"_log.txt
# # ## megahit
# #     # mode=meta-large
# #     # megahit --12 $qtrim12 --read $merged  --out-dir combined_assembly/megahit_"$mode"_out   --min-contig-len 30 --presets $mode --num-cpu-threads $THREADS  --memory $MEMORY_bytes  --continue # --k-list 21,29,39,47,59,79,99,119,141  --merge-level 10,0.7
# #     # final_k=$(ls -1 combined_assembly/megahit_"$mode"_out/intermediate_contigs/*\.final.contigs.fa | sed 's/.*_contigs\///g' | sort -n -k1.2 | tail -1 | cut -d'k' -f2 | cut -d'.' -f1 )
# #     # megahit_toolkit contig2fastg $final_k  combined_assembly/megahit_"$mode"_out/final.contigs.fa  > combined_assembly/megahit_"$mode"_out/final_megahit_assembly_k"$final_k".fastg

# # ## SPAdes
# # # spades.py  -o combined_assembly/spades_out --pe-12 1 $qtrim12  --pe-m 1 $merged --threads $THREADS  -m $MEMORY_nsuffix  --only-assembler --phred-offset 33 --cov-cutoff off 
# # spades.py  -o combined_assembly/spades_out --restart-from last --threads $THREADS -m $MEMORY_nsuffix
# # # spades.py  -o combined_assembly/spades_out_rnaviral --pe-12 1 $qtrim12  --pe-m 1 $merged --threads $THREADS  -m $MEMORY_nsuffix  --only-assembler --phred-offset 33 --cov-cutoff off --rnaviral

# # ## Penguin 
# #     penguin nuclassemble  $qtrim12 $merged penguin_nuclassemble_c1_"$file_name".fasta ./tmp/   --min-contig-len 30  --contig-output-mode 1 --threads $THREADS --sort-results 1 #--split-memory-limit $MEMORY



# # #  contgs4eval=/REDACTED_HPC_PATH/ynp/raw_reads/Eleven.fa
# # #  mmseqs easy-search $qtrim12 $merged $contgs4eval /REDACTED_HPC_PATH/ynp/raw_reads/mm_out_eleven.sam ./tmp/ --format-mode 1  --search-type 3 --threads $THREADS --force-reuse true
# # # bbwrap.sh ref=$contgs4eval in=$qtrim12,$merged out=bbwrap_output_selected.sam threads=$THREADS  nodisk covhist=covhist.txt covstats=covstats.txt outm=bbwrap_output_selected_mapped.sam outu=unassembled.fq.gz maxindel=200 minid=70  untrim ambiguous=best secondary=t maxsites=15 sssr=0.1

# # # testtt=temp.fasta.split/temp.part_001.fasta,temp.fasta.split/temp.part_002.fasta,temp.fasta.split/temp.part_002.fasta
# # # bbwrap.sh ref=$testtt in=$qtrim12,$merged out=bbwrap_output_selected.sam
# # # bbmapskimmer.sh ref=$contgs4eval in=$qtrim12 out=bbmapskimmer_output_selected.sam threads=$THREADS k=8 nodisk covhist=covhist.txt covstats=covstats.txt outm=bbmapskimmer_output_selected_mapped.sam outu=unassembled.fq.gz maxindel=40 minid=70  untrim ambiguous=best secondary=t  maxsites=1 sssr=0.9
# # # bbmapskimmer.sh ref=$contgs4eval in=$merged out=bbmapskimmer_output_selected.sam threads=$THREADS k=8 nodisk covhist=covhist.txt covstats=covstats.txt outm=bbmapskimmer_output_selected_mapped.sam outu=unassembled.fq.gz maxindel=40 minid=70  untrim ambiguous=best secondary=t maxsites=11 sssr=0.9
# # # # in1=read1.fq,singleton.fq in2=read2.fq,null
# # # # 
# # # contgs4eval=temp.fasta  


    # megahit --k-min 21 --k-max 147 --k-step 8  --min-contig-len 300 --12 $qtrim12 --read $merged  --out-dir "$output_dir"/megahit_"$mode"_out  --num-cpu-threads $THREADS  --memory $MEMORY_bytes  # --k-list 21,29,39,47,59,79,99,119,141  --merge-level 10,0.7
