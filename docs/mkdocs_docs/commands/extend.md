# Extend

> Auto-generated draft from CLI metadata for `rolypoly extend`.
> Expand this page with command-specific context, examples, and citations.

## Summary

Extend contigs by ANI-guided overlap pileup.

## Description

Workflow:
    1) Read input seqs (FASTA/FASTQ).
    2) Cluster input by ANI using pyskani.
    3) Within each multi-contig cluster, attempt overlap pileup extension
       to produce a longer representative contig.
    4) Write extended contigs (FASTA) and cluster membership table.

    Use this command to get more complete genomes from fragmented assemblies,
    for example when combining data from multiple experiments or samples.
    Supports multiple input files, e.g. one file with contigs from sample A and
    another file with contigs/reads-derived contigs from sample B.

## Usage

```bash
rolypoly extend [OPTIONS]
```

## Options

- `-i`, `--input`: Input contig FASTA/FASTQ file (pass multiple times to merge inputs) (type: `FILE`; required; default: `Sentinel.UNSET`)
- `-o`, `--output`: Output path for extended contigs FASTA (type: `FILE`; default: `extended_contigs.fasta`)
- `--clusters-output`: Output path for ANI cluster membership table (default: <output_stem>.clusters.tsv) (type: `FILE`)
- `--ani-min-identity`: Minimum ANI identity (0-1) for contigs to be placed in the same cluster (type: `FLOAT RANGE`; default: `0.95`)
- `--ani-min-af`: Minimum aligned fraction (0-1) for ANI clustering (type: `FLOAT RANGE`; default: `0.8`)
- `--pileup-min-overlap`: Minimum overlap length for pileup extension (type: `INTEGER RANGE`; default: `50`)
- `--pileup-min-identity`: Minimum identity (0-1) within overlap windows when merging cluster members (type: `FLOAT RANGE`; default: `0.98`)
- `--pileup-min-overlap-fraction-shorter`: Minimum overlap fraction relative to the shorter sequence for merge acceptance (type: `FLOAT RANGE`; default: `0.6`)
- `--include-pileup-path-in-header`, `--no-include-pileup-path-in-header`: Append pileup path to FASTA headers for extended representatives (type: `BOOLEAN`; default: `False`)
- `--pileup-bidirectional`, `--pileup-single-direction`: Allow chain growth on both sides (bidirectional) or constrain to one direction (type: `BOOLEAN`; default: `False`)
- `--pileup-aligner`: Pairwise aligner backend for overlap checks (type: `CHOICE`; default: `parasail`)
- `--pileup-parasail-algorithm`: Parasail algorithm: ov (semiglobal overlap) or sw (local Smith-Waterman) (type: `CHOICE`; default: `ov`)
- `--pileup-parasail-gap-open`: Parasail gap-open penalty (type: `INTEGER RANGE`; default: `3`)
- `--pileup-parasail-gap-extend`: Parasail gap-extend penalty (type: `INTEGER RANGE`; default: `1`)
- `--pileup-parasail-match`: Parasail nucleotide match score for ov mode matrix (type: `INTEGER RANGE`; default: `5`)
- `--pileup-parasail-mismatch`: Parasail nucleotide mismatch score for ov mode matrix (type: `INTEGER RANGE`; default: `0`)
- `--pileup-repeat-precheck`, `--no-pileup-repeat-precheck`: Pre-screen contigs for strong terminal direct/inverted repeat signatures before overlap candidate evaluation (type: `BOOLEAN`; default: `True`)
- `--pileup-repeat-check-span`: Terminal span (bp) to compare when detecting repeat-risk contigs (type: `INTEGER RANGE`; default: `40`)
- `--pileup-repeat-max-terminal-identity`: If terminal direct or inverted identity >= this threshold, contig is flagged as repeat-risk (type: `FLOAT RANGE`; default: `0.95`)
- `--pileup-repeat-risk-policy`: How repeat-risk flags are applied to candidate pair filtering (type: `CHOICE`; default: `target-only`)
- `--pileup-repeat-dotplot-min-track-len`: Minimum self dotplot track span to flag internal direct/inverted repeats (type: `INTEGER RANGE`; default: `80`)
- `--output-format`: Tabular output format for the cluster membership table (type: `CHOICE`; default: `tsv`)
- `--log-file`: Optional log file path (type: `FILE`)
- `-t`, `--threads`: Number of worker processes for parallel pileup alignment (type: `INTEGER RANGE`; default: `4`)




