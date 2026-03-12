# Annotate

<!-- Auto-generated draft from CLI metadata for `rolypoly annotate`. -->
<!-- Expand this page with command-specific context, examples, and citations. -->

## Summary

Run combined RNA + protein annotation on nucleotide viral sequences.

## Description

This command orchestrates `annotate-rna` and `annotate-prot` into a single
    workflow and writes results into `rna_annotation/` and
    `protein_annotation/` subdirectories under the selected output path.

    Use `--skip-steps` to disable specific stages and `--override-parameters`
    to forward JSON overrides to sub-tools.

## Usage

```bash
rolypoly annotate [OPTIONS]
```

## Options

- `-i`, `--input`: Input nucleotide sequence file (fasta, fna, fa, or faa) (type: `PATH`; required; default: `Sentinel.UNSET`)
- `-o`, `--output`: Output file location. (type: `TEXT`; default: `rolypoly_annotation`)
- `-t`, `--threads`: Number of threads (type: `INTEGER`; default: `1`)
- `-g`, `--log-file`: Path to log file (type: `TEXT`; default: `/clusterfs/jgi/scratch/science/metagen/neri/code/rolypoly/annotate_logfile.txt`)
- `-M`, `--memory`: Memory in GB. Example: -M 8gb (type: `TEXT`; default: `6gb`)
- `--override-parameters`: JSON-like string of parameters to override. Example: --override-parameters '{"RNAfold": {"temperature": 37}, "ORFfinder": {"minimum_length": 150}}' (type: `TEXT`; default: `{}`)
- `--skip-steps`: Comma-separated list of steps to skip. Example: --skip-steps RNA_annotation,protein_annotation or --skip-steps IRESfinder,RNAMotif or --skip-steps ORFfinder,hmmsearch (type: `TEXT`; default: ``)
- `--secondary-structure-tool`: Tool for secondary structure prediction (type: `CHOICE`; default: `LinearFold`)
- `--ires-tool`: Tool for IRES identification (type: `CHOICE`; default: `IRESfinder`)
- `--trna-tool`: Tool for tRNA identification (type: `CHOICE`; default: `tRNAscan-SE`)
- `--rnamotif-tool`: Tool for RNA sequence motif identification (type: `CHOICE`; default: `lightmotif`)
- `--gene-prediction-tool`: Tool for gene prediction (type: `CHOICE`; default: `pyrodigal`)
- `--domain-db`: Database for domain detection (NOTE: currently packaged with rolypoly data: Pfam, genomad, RVMT) (type: `CHOICE`; default: `Pfam`)
- `--custom-domain-db`: Path to a custom domain database in HMM format (for use with --domain-db custom) (type: `TEXT`; default: ``)
- `--min-orf-length`: Minimum ORF length for gene prediction (type: `INTEGER`; default: `30`)
- `--search-tool`: Tool/command for protein domain detection. hmmer commands are via pyhmmer bindings. (type: `CHOICE`; default: `hmmsearch`)




