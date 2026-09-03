# Map

<!-- Auto-generated draft from CLI metadata for `rolypoly map`. -->
<!-- Expand this page with command-specific context, examples, and citations. -->

## Summary

Map reads to a reference FASTA with one or more aligners.

## Description

Input libraries are auto-detected from a file, comma-separated file list, or
directory supplied with --input / --input-dir, or declared explicitly via
--paired-end / --single-end / --merged / --long-read. Each input file is
mapped independently; outputs land in per-mapper sub-directories under
--output.

Pair filtering is deliberately optional. In fragmented assemblies, a real
biological fragment may span an unassembled gap and place its mates on two
different contigs; --concordant and --proper will exclude such evidence.

## Usage

```bash
rolypoly map [OPTIONS]
```

## Options

- `-i`, `--input`: Input FASTQ file, comma-separated FASTQ files, or directory to auto-detect libraries from (type: `TEXT`)
- `-id`, `--input-dir`: Input directory to scan for FASTQ files (alias for --input) (type: `DIRECTORY`)
- `-r`, `--reference`: Reference FASTA to map against (type: `TEXT`; required; default: `Sentinel.UNSET`)
- `-o`, `--output`: Output directory (type: `DIRECTORY`; default: `RP_mapping_output`)
- `--paired-end`: Explicit paired-end library: <lib_num> <R1> <R2> (type: `TEXT`; default: ``)
- `--single-end`: Explicit single-end library: <lib_num> <fastq> (type: `TEXT`; default: ``)
- `--merged`: Explicit merged-read library: <lib_num> <fastq> (type: `TEXT`; default: ``)
- `--long-read`: Long-read FASTQ file(s) (type: `TEXT`; default: ``)
- `-m`, `--mapper`: Mapper (read aligner) choice. Use multiple -m flags to run multiple mappers. (type: `CHOICE`; default: `bbmap`)
- `-ow`, `--overwrite`: Overwrite existing output directory (type: `BOOLEAN`; default: `False`)
- `--concordant`, `--concord`: Keep only paired alignments whose mates map to the same reference in inward-facing FR orientation. Does not enforce insert size. (type: `BOOLEAN`; default: `False`)
- `--proper`: Keep only alignments marked proper-pair (SAM flag 0x2) by the mapper. Supported by bbmap and bwa-mem2. (type: `BOOLEAN`; default: `False`)
- `-z`, `--compressed`: Compress final SAM output from every mapper with pigz. (type: `BOOLEAN`; default: `False`)
- `--bwa-mem2-all`, `--no-bwa-mem2-all`: bwa-mem2: pass -a, which reports all alignments for single-end or unpaired paired-end reads. (type: `BOOLEAN`; default: `False`)
- `--bwa-mem2-extra-flags`: Additional flags to pass verbatim to bwa-mem2 mem (e.g. '-Y -q'). (type: `TEXT`; default: ``)
- `-t`, `--threads`: Number of worker threads. (type: `INTEGER RANGE`; default: `1`)
- `-M`, `--memory`: Memory limit, for example 8g. (type: `MEMORY`; default: `8g`)
- `-k`, `--keep-tmp`: Keep temporary files. (type: `BOOLEAN`; default: `False`)
- `-tmp`, `--temp-dir`: Temporary working directory. (type: `DIRECTORY`)
- `-g`, `--log-file`: Path to the log file. (type: `FILE`; default: `rolypoly.log`)
