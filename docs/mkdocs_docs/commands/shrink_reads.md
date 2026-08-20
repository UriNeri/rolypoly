# Shrink Reads

<!-- Auto-generated draft from CLI metadata for `rolypoly shrink-reads`. -->
<!-- Expand this page with command-specific context, examples, and citations. -->

## Summary

Subset FASTQ reads by count or fraction for lightweight test datasets.

## Description

Supports deterministic head-style subsampling (`first_n`) and random
sampling (`random`) for single-end, interleaved, and paired-end layouts.

This command is intended for quick dry runs and resource-reduced tests,
not as a full read-normalization strategy.

## Usage

```bash
rolypoly shrink-reads [OPTIONS]
```

## Options

- `-i`, `-in`, `--input`: Input raw reads file(s) or directory containing them. For paired-end reads, you can provide an interleaved file or the R1 and R2 files separated by comma. If a directory is provided, one output per input identified file/pair will be created. (type: `TEXT`; default: `Sentinel.UNSET`)
- `-st`, `--subset-type`: how to sample reads from input. (type: `CHOICE`; default: `top_reads`)
- `-sz`, `--sample-size`: For top_reads/random, at most this many reads (or proportion if <1). For bbnorm, this is the target k-mer depth. (type: `FLOAT`; default: `1000`)
- `--bbnorm-min-depth`: Minimum depth threshold for bbnorm normalization (min in bbnorm.sh). (type: `INTEGER`; default: `2`)
- `-t`, `--threads`: Number of worker threads. (type: `INTEGER RANGE`; default: `1`)
- `-g`, `--log-file`: Path to the log file. (type: `FILE`; default: `rolypoly.log`)




