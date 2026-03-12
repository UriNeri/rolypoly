# Fastx Stats

> Auto-generated draft from CLI metadata for `rolypoly fastx-stats`.
> Expand this page with command-specific context, examples, and citations.

## Summary

Calculate aggregate statistics for sequences (min, max, mean, median, etc.).

## Description

This command computes dataset-level summaries across sequences in one or
more FASTA/FASTQ inputs, including count, length, and composition fields.

For per-sequence annotations instead of aggregates, use `fastx-calc`.

## Usage

```bash
rolypoly fastx-stats [OPTIONS]
```

## Options

- `-i`, `--input`: Input file (fasta, fa, fna, faa) (type: `PATH`; required; default: `Sentinel.UNSET`)
- `-o`, `--output`: Output path (use 'stdout' to print to console) (type: `PATH`; default: `stdout`)
- `--format`: Output format for aggregate statistics (type: `CHOICE`; default: `tsv`)
- `-f`, `--fields`: Fields to calculate statistics for (type: `CHOICE`; default: `length, gc_content, n_count`)
- `-c`, `--circular`: Treat sequences as circular (rotate to minimal lexicographical form before analysis) (type: `BOOLEAN`; default: `False`)




