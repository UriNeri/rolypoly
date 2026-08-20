# Assemble

<!-- Auto-generated draft from CLI metadata for `rolypoly assemble`. -->
<!-- Expand this page with command-specific context, examples, and citations. -->

## Summary

Assemble reads/contigs with one or more backends and optional dereplication.

## Description

Inputs can be provided explicitly (`--paired-end`, `--single-end`,
`--merged`, `--long-read`, `--raw-fasta`) and/or discovered from
`--input-dir`.

Selected assembler outputs are normalized and optionally dereplicated
before writing final contigs and run metadata to the output directory.

## Usage

```bash
rolypoly assemble [OPTIONS]
```

## Options

- `-o`, `--output`: Output path (folder will be created if it doesn't exist) (type: `DIRECTORY`; default: `RP_assembly_output`)
- `-id`, `--input-dir`: Input directory to scan for fastq files (type: `DIRECTORY`)
- `--paired-end`: Library number and paired FASTQ files: <lib_num> <R1> <R2> (type: `TEXT`; default: ``)
- `--single-end`: Library number and single-end FASTQ: <lib_num> <fastq> (type: `TEXT`; default: ``)
- `--merged`: Library number and merged FASTQ: <lib_num> <fastq> (type: `TEXT`; default: ``)
- `--long-read`: path to long read FASTQ: <fastq> Note: long read files are not currently supported by all assemblers/configurations: SPAdes: supported in hybrid assembly mode (--nanopore or --pacbio). PacBio input needs to be prefiltered (i.e. the circular consensus sequences), see spades manual for more details. MEGAHIT: not supported Penguin: TODO: check if supported. I think it should be as the inputs can include a long list of fasta (type: `TEXT`; default: ``)
- `--raw-fasta`: Raw FASTA file(s) to include, note that not all assemblers support this: SPAdes: supported via the --trusted-contigs flag (see spades manual for more details) MEGAHIT: not supported Penguin: TODO: check if supported. I think it should be as the inputs can include a long list of fasta (type: `FILE`; default: ``)
- `-A`, `--assembler`: Assembler choice. For multiple, use multiple -A flags or give a comma-separated list. SPAdes: iterative de bruijn graph assembler - relatively slow and memory heavy, but potentially more accurate. MEGAHIT: multiple kmer based de bruijn graph assembler - Fast and memory light, but potentially less accurate. Penguin: mmseqs2 based, more similar to an overlap-layout-consensus method - while it claims to identify many more sequences, many of them are likely false positives. Note1 : Penguin offers a amino-acid (translation) guided assembly mode, but RolyPoly bypasses it. Note2 : SPAdes is the default assembler for RolyPoly. (type: `CHOICE`; default: `spades, megahit`)
- `--spades-mode`: SPAdes mode for the 'spades' assembler. (type: `CHOICE`; default: `meta`)
- `--preset`: Apply a named assembly preset (overrides --assembler and --dereplicate unless those flags are given explicitly on the command line). 'rna_virus': RNA virus-focused: rnaviralSPAdes + MEGAHIT, broad k-mer range. Removes duplicate contigs (rmdup). Recommended for viral metatranscriptomes. 'metatranscriptome': Metatranscriptome: rnaSPAdes + MEGAHIT, broad k-mer range. Suited for poly-A selected or mixed transcriptome libraries. 'fast': Fast: MEGAHIT only, narrow k-mer range and larger step. Trades an unknown amount of sensitivity for an unknown amount of speed; suitable for quick previews or roll --mini runs. 'complete': Complete: metaSPAdes + rnaviralSPAdes + MEGAHIT with thorough k-mer ranges. Different assemblers may produce better results - the onus of choice is on the user. This will increase the runtime and memory usage significantly 'metag': Metagenomics: metaSPAdes (meta mode) only, broad k-mer range. Suited for DNA-based or mixed metagenomic libraries. (type: `CHOICE`)
- `-op`, `--override-parameters`: JSON-like string of parameters to override. Example: --override-parameters '{"spades": {"k": "21,33,55"}, "megahit": {"k-min": 31}}' (type: `TEXT`; default: `{}`)
- `-ss`, `--skip-steps`: Comma-separated list of steps to skip. Example: --skip-steps dereplicate,rename_seqs (type: `CHOICE`; default: ``)
- `-ow`, `--overwrite`: Do not overwrite the output directory if it already exists (type: `BOOLEAN`; default: `False`)
- `--dereplicate`, `--no-rmdup`: Dereplicate assembler output by default. Disable with --no-rmdup. (type: `BOOLEAN`; default: `True`)

    - dereplicate: remove identical sequences (same sequence, same length, or its' reverse complement)
    - no-rmdup: do not perform assembler-output dereplication

- `-t`, `--threads`: Number of worker threads. (type: `INTEGER RANGE`; default: `1`)
- `-M`, `--memory`: Memory limit, for example 8g. (type: `MEMORY`; default: `8g`)
- `-k`, `--keep-tmp`: Keep temporary files. (type: `BOOLEAN`; default: `False`)
- `-tmp`, `--temp-dir`: Temporary working directory. (type: `DIRECTORY`)
- `-g`, `--log-file`: Path to the log file. (type: `FILE`; default: `rolypoly.log`)
