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

- `-i`, `-in`, `--input`: Input raw reads file(s) or directory containing them. For paired-end reads, you can provide an interleaved file or the R1 and R2 files separated by comma. (type: `TEXT`; default: `Sentinel.UNSET`)
- `-st`, `--subset-type`: how to sample reads from input. (type: `CHOICE`; default: `top_reads`)
- `-sz`, `--sample-size`: Will only return (at most) this much reads (if <1, will be interpreted as a proportion of total reads, else as the exact number of reads to get) (type: `FLOAT`; default: `1000`)
- `-g`, `--log-file`: Path to save loggging message to. defaults to current folder. (type: `PATH`; default: `/clusterfs/jgi/scratch/science/metagen/neri/code/rolypoly/rolypoly.log`)
- `-ll`, `--log-level`: Log level. Options: debug, info, warning, error, critical (type: `CHOICE`; default: `info`)




