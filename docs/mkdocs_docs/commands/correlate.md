# Correlate

> Auto-generated draft from CLI metadata for `rolypoly correlate`.
> Expand this page with command-specific context, examples, and citations.

## Summary

Find co-occurring contigs across samples

## Description

Group contigs by cross-sample association patterns.

    Supports correlation-based edges, co-occurrence-based edges, or both,
    with prevalence and shared-sample thresholds to reduce spurious links.

    Writes pairwise edge tables and connected-component style groups using the
    selected output prefix.

## Usage

```bash
rolypoly correlate [OPTIONS]
```

## Options

- `-i`, `--input`: Input table (contig IDs x sample IDs) with presence/absence or abundance values (type: `PATH`; required; default: `Sentinel.UNSET`)
- `-o`, `--output-prefix`: Output file prefix (type: `TEXT`; default: `/clusterfs/jgi/scratch/science/metagen/neri/code/rolypoly/correlate`)
- `-m`, `--mode`: Analysis mode (type: `CHOICE`; default: `both`)
- `--method`: Correlation method used in correlation mode (type: `CHOICE`; default: `spearman`)
- `--table-type`: Input value type (type: `CHOICE`; default: `auto`)
- `--min-prevalence`: Minimum fraction of samples where contig must be present (type: `FLOAT`; default: `0.1`)
- `--min-correlation`: Minimum correlation threshold for keeping contig pairs (type: `FLOAT`; default: `0.5`)
- `--min-shared-samples`: Minimum number of shared-present samples for permissive co-occurrence (type: `INTEGER`; default: `1`)
- `--presence-threshold`: Values greater than this threshold count as present (type: `FLOAT`; default: `0.0`)
- `--separator`: Input delimiter (type: `CHOICE`; default: `auto`)
- `-t`, `--threads`: Number of threads (type: `INTEGER`; default: `1`)
- `-g`, `--log-file`: Path to log file (type: `TEXT`; default: `/clusterfs/jgi/scratch/science/metagen/neri/code/rolypoly/correlate_logfile.txt`)




