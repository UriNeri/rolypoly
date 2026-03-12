# Rename Seqs

> Auto-generated draft from CLI metadata for `rolypoly rename-seqs`.
> Expand this page with command-specific context, examples, and citations.

## Summary

Rename sequences in a FASTA file with consistent IDs (supports numbering or hashing, appending attributes like GC and length).

## Description

This tool renames sequences in a FASTA file using either sequential numbers
    or hashes, and generates a lookup table mapping old IDs to new IDs.
    Optionally includes sequence statistics (length, GC content).

## Usage

```bash
rolypoly rename-seqs [OPTIONS]
```

## Options

- `-i`, `--input`: Input FASTA file (type: `TEXT`; required; default: `Sentinel.UNSET`)
- `-o`, `--output`: Output FASTA file (type: `TEXT`; required; default: `Sentinel.UNSET`)
- `-m`, `--mapping`: Output mapping file (TSV) (type: `TEXT`; required; default: `Sentinel.UNSET`)
- `-p`, `--prefix`: Prefix for new sequence IDs (type: `TEXT`; default: `CID`)
- `--hash`, `--no-hash`: Use hash instead of a padded running number for IDs (type: `BOOLEAN`; default: `False`)
- `--stats`, `--no-stats`: Include sequence statistics in mapping file (length, GC content) (type: `BOOLEAN`; default: `True`)




