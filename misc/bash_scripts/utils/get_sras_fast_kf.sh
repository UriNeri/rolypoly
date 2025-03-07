#!/bin/bash

### mamba activate <HOME_PATH>/miniconda3/envs/kingfisher
# Default values
OUTPUT_DIR="fetch_sras_outdir"
THREADS=10

# Function to display usage
usage() {
    echo "Usage: $0 -f <sra_list_file> [-o <output_dir>] [-t <threads>]"
    exit 1
}

# Parse command line arguments
while getopts ":f:o:t:" opt; do
    case ${opt} in
        f)
            SRA_LIST_FILE=$OPTARG
            ;;
        o)
            OUTPUT_DIR=$OPTARG
            ;;
        t)
            THREADS=$OPTARG
            ;;
        \?)
            usage
            ;;
    esac
done

# Check if SRA list file is provided
if [ -z "$SRA_LIST_FILE" ]; then
    usage
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Read SRA IDs from the file and download each one
# while IFS= read -r SRA_ID; do
#     SRA_DIR="$OUTPUT_DIR/$SRA_ID"
#     mkdir -p "$SRA_DIR"
#     kingfisher get -r "$SRA_ID" -m aws-http -t "$THREADS" -o "$SRA_DIR"
# done < "$SRA_LIST_FILE"

# mkdir -p "$SRA_DIR"
kingfisher get --run-identifiers-list $SRA_LIST_FILE  -m aws-http -t "$THREADS" --output-directory "$OUTPUT_DIR"


echo "Download completed."