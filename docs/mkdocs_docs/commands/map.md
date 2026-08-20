# Map

<!-- Auto-generated draft from CLI metadata for `rolypoly map`. -->
<!-- Expand this page with command-specific context, examples, and citations. -->

## Summary

Map reads to a reference FASTA with one or more aligners.

## Description

Input libraries are auto-detected from --input / --input-dir (same detection
logic as assemble) or declared explicitly via --paired-end / --single-end /
--merged / --long-read. Each input file is mapped independently; outputs land
in per-mapper sub-directories under --output.

## Usage

```bash
rolypoly map [OPTIONS]
```

## Options

- `-i`, `--input`: Input FASTQ file or directory to auto-detect libraries from (type: `TEXT`)
- `-id`, `--input-dir`: Input directory to scan for FASTQ files (alias for --input) (type: `DIRECTORY`)
- `-r`, `--reference`: Reference FASTA to map against (type: `TEXT`; required; default: `Sentinel.UNSET`)
- `-o`, `--output`: Output directory (type: `DIRECTORY`; default: `RP_mapping_output`)
- `--paired-end`: Explicit paired-end library: <lib_num> <R1> <R2> (type: `TEXT`; default: ``)
- `--single-end`: Explicit single-end library: <lib_num> <fastq> (type: `TEXT`; default: ``)
- `--merged`: Explicit merged-read library: <lib_num> <fastq> (type: `TEXT`; default: ``)
- `--long-read`: Long-read FASTQ file(s) (type: `TEXT`; default: ``)
- `-m`, `--mapper`: Mapper (read aligner) choice. Use multiple -m flags to run multiple mappers. (type: `CHOICE`; default: `bbmap`)
- `-ow`, `--overwrite`: Overwrite existing output directory (type: `BOOLEAN`; default: `False`)
- `--bwa-mem2-all`: bwa-mem2: report all valid alignments (-a flag). Recommended for quasispecies/strain disentangling. (type: `BOOLEAN`; default: `True`)
- `--bwa-mem2-extra-flags`: Additional flags to pass verbatim to bwa-mem2 mem (e.g. '-Y -q'). (type: `TEXT`; default: ``)
- `-t`, `--threads`: Number of worker threads. (type: `INTEGER RANGE`; default: `1`)
- `-M`, `--memory`: Memory limit, for example 8g. (type: `MEMORY`; default: `8g`)
- `-k`, `--keep-tmp`: Keep temporary files. (type: `BOOLEAN`; default: `False`)
- `-tmp`, `--temp-dir`: Temporary working directory. (type: `DIRECTORY`)
- `-g`, `--log-file`: Path to the log file. (type: `FILE`; default: `rolypoly.log`)




