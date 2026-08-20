# Filter Contigs

<!-- Auto-generated draft from CLI metadata for `rolypoly filter-contigs`. -->
<!-- Expand this page with command-specific context, examples, and citations. -->

## Summary

Filter contigs against user-supplied host/contamination references.

## Description

Depending on `--mode`, the command applies nucleotide filtering, protein
filtering, or both, using two-stage rule sets (`filter1_*` and
`filter2_*`) to retain likely non-host contigs.

Host references can be masked first (default) unless `--dont-mask` is set.

## Usage

```bash
rolypoly filter-contigs [OPTIONS]
```

## Options

- `-i`, `--input`: Input path to fasta file (type: `PATH`; required; default: `Sentinel.UNSET`)
- `-d`, `--known-dna`, `--host`: Path to the user-supplied host/contamination fasta (type: `PATH`; required; default: `Sentinel.UNSET`)
- `-o`, `--output`: Output file location. (type: `TEXT`; default: `/home/neri/Documents/GitHub/rps/rolypoly/filtered_contigs.fasta`)
- `-m`, `--mode`: Filtering mode: nucleotide, amino acid, or both (nuc / aa / both) (type: `CHOICE`; default: `both`)
- `-Fm1`, `--filter1_nuc`: First set of rules for nucleic filtering by aligned stats (type: `TEXT`; default: `alnlen >= 120 & pident>=75`)
- `-Fm2`, `--filter2_nuc`: Second set of rules for nucleic match filtering (type: `TEXT`; default: `qcov >= 0.95 & pident>=95`)
- `-Fd1`, `--filter1_aa`: First set of rules for amino (protein) match filtering (type: `TEXT`; default: `length >= 80 & pident>=75`)
- `-Fd2`, `--filter2_aa`: Second set of rules for protein match filtering (type: `TEXT`; default: `qcovhsp >= 95 & pident>=80`)
- `--dont-mask`: If set, host fasta won't be masked for potential RNA virus-like seqs (type: `BOOLEAN`; default: `False`)
- `--mmseqs-args`: Additional arguments for MMseqs2 (type: `TEXT`; default: `--min-seq-id 0.5 --min-aln-len 80`)
- `--diamond-args`: Additional arguments for Diamond (type: `TEXT`; default: `--id 50 --min-orf 50`)
- `-ow`, `--overwrite`: Do not overwrite the output directory if it already exists (type: `BOOLEAN`; default: `False`)
- `-t`, `--threads`: Number of worker threads. (type: `INTEGER RANGE`; default: `1`)
- `-M`, `--memory`: Memory limit, for example 8g. (type: `MEMORY`; default: `8g`)
- `-k`, `--keep-tmp`: Keep temporary files. (type: `BOOLEAN`; default: `False`)
- `-tmp`, `--temp-dir`: Temporary working directory. (type: `DIRECTORY`)
- `-g`, `--log-file`: Path to the log file. (type: `FILE`; default: `rolypoly.log`)




