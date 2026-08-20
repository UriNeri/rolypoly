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

- `-o`, `-out`, `--output`: Output directory. Example: -o output (type: `PATH`; default: `/home/neri/Documents/GitHub/rps/rolypoly`)
- `-i`, `-in`, `--input`: Input raw reads file(s) or directory containing them. For paired-end reads, you can provide an interleaved file or the R1 and R2 files separated by comma. Example: -i sample_R1.fastq.gz,sample_R2.fastq.gz If --input is a directory, all fastq files in the directory will be used - paired end files of the same base name would be assumed as from the same sample, otherwise a fastq is assumed interleaved. All interleaved and R1/R2 files would be concatenated into a single file before processing, and certain processing steps would be skipped as they assume a single sequencing library (error_correct_1, error_correct_2). (type: `TEXT`; default: `Sentinel.UNSET`)
- `-D`, `--known-dna`: Fasta file of known DNA entities. Example: -D known_dna.fasta (type: `PATH`; default: `Sentinel.UNSET`)
- `-s`, `--speed`: Set bbduk.sh speed value (0-15, where 0 uses all kmers and 15 skips most). Example: -s 5 (type: `INTEGER`; default: `0`)
- `-se`, `--skip-existing`: Skip steps if output files already exist (type: `BOOLEAN`; default: `False`)
- `-ss`, `--skip-steps`: Comma-separated list of steps to skip. Example: --skip-steps filter_by_tile,entropy_filter (type: `TEXT`; default: ``)
- `--preset`: Apply a named read-filtering preset (overrides individual step parameters unless those are given explicitly via --override-parameters). 'rna_virus_metat': RNA virus metatranscriptome: rRNA removal (mincovfraction=0.6), known-DNA + identified-DNA filtering, adapter trim, lenient quality trim (trimq=5 minlen=25); no polyA trimming 'total_rna_ribodepleted': Total RNA ribo-depleted: stricter rRNA removal (mincovfraction=0.6), known-DNA + identified-DNA filtering, adapter trim, lenient quality trim (trimq=5 minlen=20); no polyA trimming 'poly_a_selected': Poly-A selected mRNA: polyA tail trimming enabled (trimpolya=18), stricter quality trim (trimq=12 minlen=20); rRNA and DNA filtering still applied 'fast': Fast: skips overlap error correction (error_correct_1/2) and identified-DNA filtering; all other steps run at default parameters 'strict': Strict: aggressive quality trim (trimq=20 minlen=20), two-pass deduplication; all filtering steps enabled 'all_virus_metat': All-virus metatranscriptome: relaxed rRNA removal (mincovfraction=0.5), skips identified-DNA filter, moderate quality trim (trimq=10 minlen=20); known-DNA filtering still applied 'all_virus_metag': All-virus metagenomics: skips rRNA and identified-DNA filtering entirely, moderate quality trim (trimq=10 minlen=20); known-DNA (host) filtering still applied if --known-dna is provided (type: `CHOICE`)
- `--disable-auto`: Disable automatic trim/minlen tuning from detected read stats. (type: `BOOLEAN`; default: `False`)
- `--trim-polya`, `--poly-selection`: Enable optional terminal polyA/polyT tail trimming after adapter trimming. Uses the trim_polya_tails preset and can be customized with --override-parameters. (type: `BOOLEAN`; default: `False`)
- `--adapters`: Optional adapter FASTA to use instead of built-in (or discovered via bbmerge) adapters. (type: `PATH`)
- `--artifacts`: Optional synthetic-artifact FASTA to use. Turns on --remove-synthetic-artifacts. (type: `PATH`)
- `--remove-synthetic-artifacts`: Enable the synthetic-artifact removal step using the built-in artifacts reference unless --artifacts is provided. (type: `BOOLEAN`; default: `False`)
- `-op`, `-override-params`, `--override-parameters`: JSON-like string of parameters to override. Example: --override-parameters '{"decontaminate_rrna": {"k": 29}, "trim_polya_tails": {"trimpolya": 28, "minlen": 30}}' (type: `TEXT`)
- `--config-file`: Path to configuration file. Example: --config-file my_config.json (type: `PATH`; default: `Sentinel.UNSET`)
- `-to`, `-timeout`, `--step-timeout`: Timeout for every step in the workflow in seconds. if you think the some processes are hanging (not terminated correctly) this would help debug that. Example: --timeout 600 (type: `INTEGER`; default: `3600`)
- `-n`, `-name`, `--file-name`: Base name of the output files. Example: -file-name my_filtered_reads. If not set, default would be "rp_filtered_reads" unless the --input has a structure like somethingsomething_R1.fastq.gz,somethingsomething_R2.fastq.gz or somethingsomething.fastq.gz in which case it would be somethingsomething (type: `TEXT`; default: `Sentinel.UNSET`)
- `-ow`, `--overwrite`: Do not overwrite the output directory if it already exists (type: `BOOLEAN`; default: `False`)
- `-z`, `--zip-reports`: Zip the reports into a single file (type: `BOOLEAN`; default: `False`)
- `-t`, `--threads`: Number of worker threads. (type: `INTEGER RANGE`; default: `1`)
- `-M`, `-mem`, `--memory`: Memory limit, for example 8g. (type: `MEMORY`; default: `8g`)
- `-k`, `--keep-tmp`: Keep temporary files. (type: `BOOLEAN`; default: `False`)
- `-tmp`, `--temp-dir`: Temporary working directory. (type: `DIRECTORY`)
- `-g`, `--log-file`: Path to the log file. (type: `FILE`; default: `rolypoly.log`)




