# Annotate Rna

<!-- Auto-generated draft from CLI metadata for `rolypoly annotate-rna`. -->
<!-- Expand this page with command-specific context, examples, and citations. -->

## Summary

Annotate RNA structural features and motifs on viral nucleotide inputs.

## Description

The default pipeline runs secondary-structure prediction, ribozyme/CM
search, and tRNA detection, while optional modules can add IRES and motif
analyses depending on selected tools and skip settings.

Use `--skip-steps` to disable modules and `--override-parameters` to pass
tool-specific tuning values for individual stages.

## Usage

```bash
rolypoly annotate-rna [OPTIONS]
```

## Options

- `-i`, `--input`: Input nucleotide sequence file (fasta, fna, fa, or faa) (type: `PATH`; required; default: `Sentinel.UNSET`)
- `-o`, `--output-dir`: Output directory path (type: `TEXT`; default: `./annotate_RNA_output`)
- `-op`, `--override_parameters`, `--override-parameters`: JSON-like string of parameters to override. Example: --override-parameters '{"RNAfold": {"temperature": 37}, "cmscan": {"E": 1e-5}}' (type: `TEXT`; default: `{}`)
- `--skip-steps`: Comma-separated list of steps to skip. Example: --skip-steps RNAfold,cmsearch (type: `TEXT`; default: ``)
- `--secondary-structure-tool`: Tool for secondary structure prediction. LinearFold is faster but less configurable. (type: `CHOICE`; default: `LinearFold`)
- `--ires-tool`: Tool for IRES identification (type: `CHOICE`; default: `IRESfinder`)
- `--trna-tool`: Tool for tRNA identification (type: `CHOICE`; default: `tRNAscan-SE`)
- `--rnamotif-tool`: Tool for RNAmotif identification (PSSM search). Not fully supported yet. (type: `CHOICE`; default: `lightmotif`)
- `--cm-db`: Database for cmscan (type: `CHOICE`; default: `Rfam`)
- `--custom-cm-db`: Path to a custom cm database in nhmmer/cm format (mandatory to use with --cm-db custom) (type: `TEXT`; default: ``)
- `--output-format`: Output format for the combined results (type: `CHOICE`; default: `tsv`)
- `--motif-db`: Database to use for RNA motif scanning - RolyPoly, jaspar_core, or a path to a folder containg a pwm/msa files (type: `TEXT`; default: `RolyPoly`)
- `-rm`, `--resolve-mode`: How to deal with overlapping RNA element hits in the same sequence. (type: `CHOICE`; default: `simple`)

    - merge: all overlapping hits are merged into one range

    - one_per_range: one hit per range is reported

    - one_per_query: one hit per query sequence is reported

    - split: each overlapping element is split into a new row

    - drop_contained: hits that are contained within other hits are dropped

    - none: no resolution of overlapping hits is performed

    - simple: heuristic-based approach using drop_contained

- `-mo`, `--min-overlap-positions`: Minimal number of overlapping positions between two intersecting ranges before they are considered as overlapping (used in some resolve_mode(s)). (type: `INTEGER`; default: `10`)
- `-t`, `--threads`: Number of worker threads. (type: `INTEGER RANGE`; default: `1`)
- `-M`, `--memory`: Memory limit, for example 8g. (type: `MEMORY`; default: `8g`)
- `-g`, `--log-file`: Path to the log file. (type: `FILE`; default: `rolypoly.log`)
