# Fetch Sra

<!-- Auto-generated draft from CLI metadata for `rolypoly fetch-sra`. -->
<!-- Expand this page with command-specific context, examples, and citations. -->

## Summary

Download SRA run FASTQ files and optional XML metadata from *ENA*

## Description

Takes either a single SRA run ID (e.g., SRR12345678) or a file containing multiple run IDs (one per line).
    Downloads FASTQ files and optionally XML metadata reports to the specified output directory.

    Example usage:


    # Download single run:
    rolypoly fetch-sra -i SRR12345678 -o output_dir

    # Download multiple runs with metadata:
    rolypoly fetch-sra -i run_ids.txt -o output_dir --report

    * Note: The fastq headers may vary for the same SRA run/expriemtn based on the source and fetching method (s3, ftp, ena...)

## Usage

```bash
rolypoly fetch-sra [OPTIONS]
```

## Options

- `-i`, `--input`: SRA run ID or file containing run IDs (one per line) (type: `TEXT`; required; default: `Sentinel.UNSET`)
- `-o`, `--output-dir`: Directory to save downloaded files (type: `DIRECTORY`; default: `./`)
- `--report`: Download XML report for each run (type: `BOOLEAN`; default: `False`)




