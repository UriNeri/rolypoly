#!/bin/bash
# Written by Uri Neri
# Last modified 13.06.2024 ---- WIP
# Contact: 
# Description: To simplify workflows.


# Check if a file exists
check_file_exists() {
  local file_path="$1"
  if [ ! -e "$file_path" ]; then
    echo "File not found: $file_path Tüdelü!"
    exit 1
  fi
}


# Check if file size is 0 bytes
check_file_size() {
    local file="$1"

    # Get the file size using stat
    file_size=$(stat --format="%s" "$file")

    # Check if the file size is 0
    if [ "$file_size" -eq 0 ]; then
        echo "File '$file' is empty"
        # exit 1
    else
        echo "File '$file' size is $file_size"
    fi
}

# Check if file size is 0 and return true or false
is_file_empty() {
  local file="$1"

  # Check if the file exists
  if [ ! -e "$file" ]; then
      echo "File '$file' does not exist."
      return 1
  fi

  # Get the file size using stat
  file_size=$(stat --format="%s" "$file")

  # Check if the file size is 0
  if [ "$file_size" -eq 0 ]; then
      return 0  # true (file is empty)
  fi
  if [ "$file_size" -eq 28 ]; then
      return 0  # true (file is empty - 28bytes is around what an empty .gz file should weigh).
  else
      return 1  # false (file is not empty)
  fi
      return 1  # false (file is not empty)

}

# Check if a given file (e.g. a fastq.gz) exist and if so that it is not empty. Needs the THREADS and file_name variables to be defined.
check_file_exist_isempty() {
    local file_to_check="$1"
    # Check if the file exists    
    check_file_exists "$file_to_check"

    # Check if the file is empty and process accordingly
    if is_file_empty "$file_to_check"; then
        echo "File $file_to_check exists, but is empty."
        echo "This might mean all reads were filtered,
        so we'll try to do a hasty wrap-up and exit without proceeding to downstream steps."
        mkdir ./FastQC_post_trim_reads/
        fastqc -t "$THREADS" *.fq.gz -o ./FastQC_post_trim_reads/
        echo "Running multiQC report $THREADS"
        multiqc ./ --outdir "${file_name}_multiqc"
        echo "Tata, Exiting!!!"
        exit 1
    else
        echo "File '$file_to_check' size is" $(stat --format="%s" "$file_to_check") bytes "(i.e. not empty) Continuing    "
    fi
}


# Downloads genomes from NCBI using the NCBI nucleotide accession. input file is should have a list of taxons (one per row).
# Requires `rush` and `datasets`.
fetch_genomes() {
  # input_file=/REDACTED_HPC_PATH/rolypoly/bench/test_sampled_005_bb_metaTs_spiced_RVMT/tmp_dir_sampled_005_bb_metaTs_spiced_RVMT/stats_rRNA_filt_sampled_005_bb_metaTs_spiced_RVMT.txt
  # output_file=test.fasta
  local input_file="$1"
  local output_file="$2"
  awk 'NR > 4 && $3 > 5000 { if (!seen[$1]++) print $0 }' $input_file  | cut -f2,3,4,5 -d' ' | awk -F'\t'  '{print $1}'| head -n 100 | rev | cut -f1 -d";" | rev | sort -u > tmp_gbs_50m.lst ## this counts as one line somehow.
  sed -i '/meta/d' tmp_gbs_50m.lst # Otherwise it woudl download metagenomes too - which might have been good except as far as NCBI taxonomy goes, metatranscriptomes (the dwelling of RNA viruses) are also "metagenome"
  sed -i '/uncultured/d' tmp_gbs_50m.lst # Unfortunately there are too much noise there.
  sed -i '/^unidentified$/d' tmp_gbs_50m.lst # Unfortunately there are too much noise there.
  sed -i '/^synthetic construct$/d' tmp_gbs_50m.lst # Unfortunately there are too much noise there.

  
  taxonkit name2taxid tmp_gbs_50m.lst | cut -f2 > tmp_gbs_50m_taxids.lst

  # Read each taxon name from the file and fetch the corresponding genome data (zip from ncbi)
  # while IFS= read -r line;
  # do
  #     echo "Processing $line    "
  #     datasets download genome taxon "${line}" --include genome  --filename "${line}"_fetched_genomes.zip --reference --assembly-version latest  --exclude-atypical --assembly-source RefSeq --no-progressbar
  # done < tmp_gbs_50m.lst
sed 's| |_|g' tmp_gbs_50m.lst > blabla
paste blabla tmp_gbs_50m.lst > blabla2
# cat blabla2 | rush  -d"\\t+" 'echo Trying to fetch "{2}"  \n | datasets download genome taxon "{2}" --include genome  --filename "{1}"_fetched_genomes.zip --reference --assembly-version latest  --exclude-atypical --assembly-source RefSeq --no-progressbar '
# cat tmp_gbs_50m_taxids.lst | rush  -d"\\t+" 'echo Trying to fetch "{2}"  \n | datasets download genome taxon "{2}" --include genome  --filename "{1}"_fetched_genomes.zip --reference --assembly-version latest  --exclude-atypical --assembly-source RefSeq --no-progressbar '
cat tmp_gbs_50m_taxids.lst | rush  -d"\\t+" 'echo Trying to fetch "{2}"  \n | datasets download genome taxon "{2}" --include rna,genome  --filename "{1}"_fetched_genomes.zip --reference --assembly-version latest  --exclude-atypical --assembly-source RefSeq --no-progressbar '

  # parallel  unzip {} -d {.} ::: ls ./*.zip #-j"$THREADS"
  # rust-parallel  unzip {} -d {.} ::: ls ./*.zip #-j"$THREADS"
  ls *.zip | rush -d'@' 'unzip "{}" -d {.}'
# Find all .fna files and sort them
find . -type f -name "*.fna" | sort > all_files.txt

# Use awk to select rna.fna files over other .fna files
awk '
{
    # Extract the directory path
    match($0, /(.+\/GCF_[^\/]+)\//, arr)
    dir = arr[1]

    # Prioritize rna.fna if found
    if ($0 ~ /rna\.fna$/) {
        chosen[dir] = $0
    } else if (!(dir in chosen)) {
        chosen[dir] = $0
    }
}

END {
    # Print the chosen files in the correct order
    for (dir in chosen) {
        print chosen[dir]
    }
}
' all_files.txt > selected_files.txt

# Concatenate the selected files into the output file
cat $(cat selected_files.txt) | seqkit rmdup -s  > "$output_file"
echo "Concatenation complete. Output saved to $output_file"

# Clean up temporary files
  rm all_files.txt selected_files.txt
  rm *.zip  
  rm *fetched_genomes -r
  rm blabla*
  echo "Download complete. Output saved to $output_file."
}



# One archive decompress function to rule them all and avoid wasting time going to the same stackoverflow pages to figure out.
extract () {
   if [ -f $1 ] ; then
       case $1 in
        *.tar.bz2)      tar xvjf $1 ;;
        *.tar.gz)       tar xvzf $1 ;;
        *.tar.xz)       tar Jxvf $1 ;;
        *.bz2)          bunzip2 $1 ;;
        *.rar)          unrar x $1 ;;
        *.gz)           gunzip $1 ;;
        *.tar)          tar xvf $1 ;;
        *.tbz2)         tar xvjf $1 ;;
        *.tgz)          tar xvzf $1 ;;
        *.zip)          unzip $1 ;;
        *.Z)            uncompress $1 ;;
        *.7z)           7z x $1 ;;
        *)              echo "don't know how to extract '$1'    " ;;
       esac
   else
       echo "'$1' is not a valid file!"
   fi
}

# Check dependencies - tries to run commands and based on the return value of the command breaks or proceeds.
check_dependencies() {
    local dependencies=("$@")
    for dpnc in "${dependencies[@]}"; do
        $dpnc -V &>/dev/null # ← puts the command whose exit code you want to check here &>/dev/null
        if [ $? -eq 127 ]; then
            echo "$dpnc Not Found! Exiting!!!"
            exit 1
        else
            echo "$dpnc Found!"
        fi
    done
}

# Prints (echo) something (first arg) and also saves it to a log file (second arg) 
logit () {
   echo "$(date +"%Y-%m-%d %T") $2" | tee -a $1
}
