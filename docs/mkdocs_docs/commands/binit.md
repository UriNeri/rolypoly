# Binit

> Auto-generated draft from CLI metadata for `rolypoly binit`.
> Expand this page with command-specific context, examples, and citations.

## Summary

Run integrated segment binning workflow

## Description

Run an integrated segment-binning workflow over RdRp and CP contigs.

	The workflow chains cluster, correlate, extend, and termini steps, then
	joins their outputs to report candidate segment pairs with supporting
	evidence from abundance/correlation and termini/motif consistency.

## Usage

```bash
rolypoly binit [OPTIONS]
```

## Options

- `--rdrp-fasta`: Input FASTA containing RdRp candidate contigs (type: `FILE`; required; default: `Sentinel.UNSET`)
- `--cp-fasta`: Input FASTA containing CP candidate contigs (type: `FILE`; required; default: `Sentinel.UNSET`)
- `-o`, `--outdir`: Output directory for all intermediate and final workflow artifacts (type: `DIRECTORY`; required; default: `Sentinel.UNSET`)
- `-t`, `--threads`: Threads used by cluster/extend/termini stages (type: `INTEGER`; default: `8`)
- `--ani-min-identity`: ANI identity threshold (0-1) used for cluster and extend (type: `FLOAT RANGE`; default: `0.9`)
- `--ani-min-af`: ANI aligned fraction threshold (0-1) used for cluster and extend (type: `FLOAT RANGE`; default: `0.8`)
- `--min-correlation`: Minimum correlation for the correlate stage (type: `FLOAT`; default: `0.01`)
- `--min-prevalence`: Minimum sample prevalence used in the correlate stage (type: `FLOAT RANGE`; default: `0.05`)
- `--min-shared-samples`: Minimum shared sample count used in the correlate stage (type: `INTEGER RANGE`; default: `1`)
- `--termini-length`: Termini length or range passed to rolypoly termini (type: `TEXT`; default: `4-1145`)
- `--termini-distance`: Maximum Hamming mismatch distance for termini seed grouping (type: `INTEGER RANGE`; default: `0`)
- `--reuse-extend`, `--no-reuse-extend`: Reuse existing extend outputs if present instead of rerunning extend (type: `BOOLEAN`; default: `False`)
- `--write-single-rdrp-strict`, `--no-write-single-rdrp-strict`: Write strict candidate pairs that pass the complementarity check (type: `BOOLEAN`; default: `False`)
- `--log-file`: Optional workflow-level log file path (type: `FILE`)




