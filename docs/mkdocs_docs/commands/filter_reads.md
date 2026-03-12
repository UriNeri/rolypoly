# Filter Reads

<!-- Auto-generated draft from CLI metadata for `rolypoly filter-reads`. -->
<!-- Expand this page with command-specific context, examples, and citations. -->

## Summary

Process RNA-seq Illumina reads through the read-cleaning pipeline.

## Description

The workflow combines host/contaminant removal, optional fetched-reference
filtering, adapter/quality trimming, and optional error correction based on
configured steps and speed presets.

Input can be a single file, paired files, or a directory of FASTQ files.
Use `--skip-steps` and `--override-parameters` to tailor the workflow.

## Usage

```bash
rolypoly filter-reads [OPTIONS]
```

## Options

- `-t`, `--threads`: Number of threads to use. Example: -t 4 (type: `INTEGER`; default: `1`)
- `-M`, `-mem`, `--memory`: Memory. Example: -M 8gb (type: `TEXT`; default: `10gb`)
- `-o`, `-out`, `--output`: Output directory. Example: -o output (type: `PATH`; default: `/clusterfs/jgi/scratch/science/metagen/neri/code/rolypoly`)
- `--keep-tmp`: Keep temporary files (type: `BOOLEAN`; default: `False`)
- `-g`, `--log-file`: Path to log_file. Example: -g logfile.log, if not provided, a log file will be created in the current directory. (type: `PATH`; default: `/clusterfs/jgi/scratch/science/metagen/neri/code/rolypoly/rolypoly.log`)
- `-i`, `-in`, `--input`: Input raw reads file(s) or directory containing them. For paired-end reads, you can provide an interleaved file or the R1 and R2 files separated by comma. Example: -i sample_R1.fastq.gz,sample_R2.fastq.gz If --input is a directory, all fastq files in the directory will be used - paired end files of the same base name would be assumed as from the same sample, otherwise a fastq is assumed interleaved. All interleaved and R1/R2 files would be concatenated into a single file before processing, and certain processing steps would be skipped as they assume a single sequencing library (error_correct_1, error_correct_2). (type: `TEXT`; default: `Sentinel.UNSET`)
- `-D`, `--known-dna`: Fasta file of known DNA entities. Example: -D known_dna.fasta (type: `PATH`; default: `Sentinel.UNSET`)
- `-s`, `--speed`: Set bbduk.sh speed value (0-15, where 0 uses all kmers and 15 skips most). Example: -s 5 (type: `INTEGER`; default: `0`)
- `-se`, `--skip-existing`: Skip steps if output files already exist (type: `BOOLEAN`; default: `False`)
- `-ss`, `--skip-steps`: Comma-separated list of steps to skip. Example: --skip-steps filter_by_tile,entropy_filter (type: `TEXT`; default: ``)
- `-op`, `-override-params`, `--override-parameters`: JSON-like string of parameters to override. Example: --override-parameters '{"decontaminate_rrna": {"k": 29}, "filter_dna_genomes": {"mincovfraction": 0.8}}' (type: `TEXT`)
- `--config-file`: Path to configuration file. Example: --config-file my_config.json (type: `PATH`; default: `Sentinel.UNSET`)
- `-to`, `-timeout`, `--step-timeout`: Timeout for every step in the workflow in seconds. if you think the some processes are hanging (not terminated correctly) this would help debug that. Example: --timeout 600 (type: `INTEGER`; default: `3600`)
- `-n`, `-name`, `--file-name`: Base name of the output files. Example: -file-name my_filtered_reads. If not set, default would be "rp_filtered_reads" unless the --input has a structure like somethingsomething_R1.fastq.gz,somethingsomething_R2.fastq.gz or somethingsomething.fastq.gz in which case it would be somethingsomething (type: `TEXT`; default: `Sentinel.UNSET`)
- `-ow`, `--overwrite`: Do not overwrite the output directory if it already exists (type: `BOOLEAN`; default: `False`)
- `-z`, `--zip-reports`: Zip the reports into a single file (type: `BOOLEAN`; default: `False`)




