# Nucleic Search

<!-- Auto-generated draft from CLI metadata for `rolypoly nucleic-search`. -->
<!-- Expand this page with command-specific context, examples, and citations. -->

## Summary

Search nucleotide reads or contigs against virus reference databases.

## Description

Input can be one FASTA/FASTQ file, a comma-separated list, a directory, or
an existing MMseqs2 database. Sequence inputs are combined into one MMseqs2
query database. Records are searched independently; paired-read
concordance is not evaluated here. Use ``rolypoly map`` for pair-aware read
mapping.

For a custom nucleotide reference distributed as FASTA (including
gzip-compressed FASTA), use `--db other --db-path reference.fasta.gz`; RolyPoly
will create the temporary MMseqs2 target database. This can also serve as a
workaround when an older installation points to an obsolete built-in database
path.

## Usage

```bash
rolypoly nucleic-search [OPTIONS]
```

## Options

- `-o`, `--output`: output file location - set suffix to .tab, .sam or html (type: `TEXT`; default: `/home/neri/Documents/GitHub/rps/rolypoly_RP_mapping`)
- `--db`, `--database`: Select the database to search against. 'all' retains its historical meaning: the two RNA-virus databases (RVMT and NCBI_Ribovirus). (type: `CHOICE`; default: `all`)
- `--db-path`: Path to the user-supplied source (required if --db is 'other'). Either a fasta or a path to formatted MMseqs2 virus database (type: `TEXT`; default: ``)
- `-i`, `--input`: Input FASTA/FASTQ file, comma-separated sequence files, directory of sequence files, or one preformatted MMseqs2 database prefix (type: `TEXT`; required; default: `Sentinel.UNSET`)
- `-mo`, `--matched-output`: Output path for matched virus contigs. set to 'no' to skip writing matched contigs (type: `TEXT`; default: `Sentinel.UNSET`)
- `-e`, `--mmseqs-evalue`: E-value threshold for MMseqs2 search) (type: `FLOAT`; default: `0.1`)
- `-id`, `--mmseqs-identity`: minimum Identity threshold for MMseqs2 search) (type: `FLOAT`; default: `0.7`)
- `-al`, `--mmseqs-min-aln-len`: Minimum alignment length for MMseqs2 search) (type: `INTEGER`; default: `95`)
- `-t`, `--threads`: Number of worker threads. (type: `INTEGER RANGE`; default: `1`)
- `-M`, `--memory`: Memory limit, for example 8g. (type: `MEMORY`; default: `8g`)
- `-k`, `--keep-tmp`: Keep temporary files. (type: `BOOLEAN`; default: `False`)
- `-tmp`, `--temp-dir`: Temporary working directory. (type: `DIRECTORY`)
- `-g`, `--log-file`: Path to the log file. (type: `FILE`; default: `rolypoly.log`)
