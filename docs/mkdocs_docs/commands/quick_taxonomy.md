# Quick Taxonomy

> Auto-generated draft from CLI metadata for `rolypoly quick-taxonomy`.
> Expand this page with command-specific context, examples, and citations.

## Summary

Assign lightweight taxonomy labels to viral sequences.

## Description

The command combines optional marker-search evidence with geNomad RNA viral
HMM matches and writes assignments as text, TSV, or JSON.

Use `--min_score` to control assignment strictness and `--summarize` to
include aggregate summary metrics.

## Usage

```bash
rolypoly quick-taxonomy [OPTIONS]
```

## Options

- `-i`, `--input`: Input file or directory (type: `TEXT`; required; default: `Sentinel.UNSET`)
- `-o`, `--output`: Output directory (type: `TEXT`; default: `output`)
- `-t`, `--threads`: Number of threads (type: `INTEGER`; default: `1`)
- `--log-file`: Path to log file (type: `TEXT`; default: `command.log`)
- `--marker_results`: marker_results (type: `TEXT`)
- `--format`: format (type: `CHOICE`; default: `text`)
- `--min_score`: min_score (type: `FLOAT`; default: `50.0`)
- `--summarize`: summarize (type: `BOOLEAN`; default: `True`)




