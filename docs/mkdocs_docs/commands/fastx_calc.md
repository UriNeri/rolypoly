# Fastx Calc

<!-- Auto-generated draft from CLI metadata for `rolypoly fastx-calc`. -->
<!-- Expand this page with command-specific context, examples, and citations. -->

## Summary

Calculate per-sequence metrics (length, GC content, hash, etc.).

## Description

This command computes metrics for each sequence in a FASTA/FASTQ file.
For aggregate statistics across all sequences, use the 'fastx-stats' command instead.

Note:
    - No support yet for reverse complement (not in circular or hash).

## Usage

```bash
rolypoly fastx-calc [OPTIONS]
```

## Options

- `-i`, `--input`: Input file (fasta, fa, fna, faa) (type: `PATH`; required; default: `Sentinel.UNSET`)
- `-o`, `--output`: Output path (use 'stdout' to print to console) (type: `PATH`; default: `rp_sequence_calc.tsv`)
- `--format`: Output format for per-sequence annotations (type: `CHOICE`; default: `tsv`)
- `-f`, `--fields`: Fields to annotate for each sequence. Available: length - sequence length gc_content - percentage of GC nucleotides n_count - total number of Ns hash - md5 hash of the sequence (type: `CHOICE`; default: `length, gc_content, n_count`)
- `-c`, `--circular`: Treat sequences as circular (rotate to minimal lexicographical form before hashing) (type: `BOOLEAN`; default: `False`)




