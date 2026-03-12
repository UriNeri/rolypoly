# Assemble

> Auto-generated draft from CLI metadata for `rolypoly assemble`.
> Expand this page with command-specific context, examples, and citations.

## Summary

Assemble reads/contigs with one or more backends and optional post-processing.

## Description

Inputs can be provided explicitly (`--paired-end`, `--single-end`,
    `--merged`, `--long-read`, `--raw-fasta`) and/or discovered from
    `--input-dir`.

    Selected assembler outputs are normalized and optionally post-processed
    (for example `rmdup` or `linclust`) before writing final contigs and run
    metadata to the output directory.

## Usage

```bash
rolypoly assemble [OPTIONS]
```

## Options

- `-t`, `--threads`: Threads (type: `INTEGER`; default: `1`)
- `-M`, `--memory`: RAM limit (more is betterer, see the docs for more info) (type: `TEXT`; default: `6gb`)
- `-o`, `--output`: Output path (folder will be created if it doesn't exist) (type: `DIRECTORY`; default: `RP_assembly_output`)
- `-k`, `--keep-tmp`: Keep temporary files (type: `BOOLEAN`; default: `False`)
- `-g`, `--log-file`: Path to a logfile, should exist and be writable (permission wise) (type: `TEXT`; default: `/clusterfs/jgi/scratch/science/metagen/neri/code/rolypoly/assemble_logfile.txt`)
- `-id`, `--input-dir`: Input directory to scan for fastq files (type: `DIRECTORY`)
- `--paired-end`: Library number and paired FASTQ files: <lib_num> <R1> <R2> (type: `TEXT`; default: ``)
- `--single-end`: Library number and single-end FASTQ: <lib_num> <fastq> (type: `TEXT`; default: ``)
- `--merged`: Library number and merged FASTQ: <lib_num> <fastq> (type: `TEXT`; default: ``)
- `--long-read`: path to long read FASTQ: <fastq> Note: long read files are not currently supported by all assemblers/configurations: SPAdes: supported in hybrid assembly mode (--nanopore or --pacbio). PacBio input needs to be prefiltered (i.e. the circular consensus sequences), see spades manual for more details. MEGAHIT: not supported Penguin: TODO: check if supported. I think it should be as the inputs can include a long list of fasta (type: `TEXT`; default: ``)
- `--raw-fasta`: Raw FASTA file(s) to include, note that not all assemblers support this: SPAdes: supported via the --trusted-contigs flag (see spades manual for more details) MEGAHIT: not supported Penguin: TODO: check if supported. I think it should be as the inputs can include a long list of fasta (type: `FILE`; default: ``)
- `-A`, `--assembler`: Assembler choice. For multiple, use multiple -A flags or give a comma-separated list. SPAdes: iterative de bruijn graph assembler - relatively slow and memory heavy, but potentially more accurate. MEGAHIT: multiple kmer based de bruijn graph assembler - Fast and memory light, but potentially less accurate. Penguin: mmseqs2 based, more similar to an overlap-layout-consensus method - while it claims to identify many more sequences, many of them are likely false positives. Note1 : Penguin offers a amino-acid (translation) guided assembly mode, but RolyPoly bypasses it. Note2 : SPAdes is the default assembler for RolyPoly. (type: `CHOICE`; default: `spades, megahit`)
- `-op`, `--override-parameters`: JSON-like string of parameters to override. Example: --override-parameters '{"spades": {"k": "21,33,55"}, "megahit": {"k-min": 31}}' (type: `TEXT`; default: `{}`)
- `-ss`, `--skip-steps`: Comma-separated list of steps to skip. Example: --skip-steps post_processing,rename_seqs (type: `CHOICE`; default: ``)
- `-ow`, `--overwrite`: Do not overwrite the output directory if it already exists (type: `BOOLEAN`; default: `False`)
- `-p`, `--post-processing`: Method for merging or clustering the assembler output(s), options: - linclust: use MMseqs2 linclust to cluster the assembler output at 99% identity and 99% coverage using coverage-mode 1. These parameters mean that most subsequences that are wholly contained within a larger sequence will dropped (use with caution, as a chimeras from one assembler may be merged with a chimera from another assembler may 'engulf' a non-chimeric sequence from the other assembler) - rmdup: use seqkit rmdup to remove identical sequences (same sequence, same length, or its' reverse complement) - none: do not perform any post assembly processing (type: `CHOICE`; default: `rmdup`)




