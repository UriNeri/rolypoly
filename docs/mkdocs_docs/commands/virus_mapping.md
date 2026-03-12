# Virus Mapping

> Auto-generated draft from CLI metadata for `rolypoly virus-mapping`.
> Expand this page with command-specific context, examples, and citations.

## Summary

Search nucleotide reads/contigs against virus reference databases.

## Description

Input can be FASTA/FASTQ or an existing MMseqs2 database. The command
    converts sequence inputs to MMseqs2 format as needed and runs searches
    against built-in viral databases (`RVMT`, `NCBI_Ribovirus`, or `all`) or a
    user-supplied target via `--db other --db-path`.

## Usage

```bash
rolypoly virus-mapping [OPTIONS]
```

## Options

- `-t`, `--threads`: Threads (all) [1] (type: `INTEGER`; default: `1`)
- `-M`, `--memory`: MEMORY in gb (more) [6] (type: `TEXT`; default: `6g`)
- `-o`, `--output`: output file location - set suffix to .tab, .sam or html [default: .tab] (type: `TEXT`; default: `/clusterfs/jgi/scratch/science/metagen/neri/code/rolypoly_RP_mapping`)
- `--keep-tmp`: Keep temporary files (type: `BOOLEAN`; default: `False`)
- `--db`: Select the database to search against. (type: `CHOICE`; default: `all`)
- `--db-path`: Path to the user-supplied source (required if --db is 'other'). Either a fasta or a path to formatted MMseqs2 virus database (type: `TEXT`; default: ``)
- `-g`, `--log-file`: Abs path to logfile (type: `TEXT`; default: `/clusterfs/jgi/scratch/science/metagen/neri/code/rolypoly/search_viruses_logfile.txt`)
- `-i`, `--input`: Input path to nucl fasta file OR preformatted mmseqs db (type: `TEXT`; required; default: `Sentinel.UNSET`)




