# Roll

<!-- Auto-generated draft from CLI metadata for `rolypoly roll`. -->
<!-- Expand this page with command-specific context, examples, and citations. -->

## Summary

End-to-end pipeline for RNA virus discovery from raw sequencing data.

## Description

This pipeline performs a complete analysis workflow including:
    1. Read filtering and quality control
    2. De novo assembly
    3. Contig filtering
    4. Marker gene search (default: RdRps)
    5. Genome annotation
    6. Virus characteristics prediction

## Usage

```bash
rolypoly roll [OPTIONS]
```

## Options

- `-i`, `--input`: Input path to raw RNA-seq data (fastq/gz file or directory with fastq/gz files) (type: `TEXT`; required; default: `Sentinel.UNSET`)
- `-o`, `--output-dir`: Output directory (type: `TEXT`; default: `/clusterfs/jgi/scratch/science/metagen/neri/code/rolypoly_rp_e2e`)
- `-t`, `--threads`: Number of threads (type: `INTEGER`; default: `1`)
- `-M`, `--memory`: Memory allocation (type: `TEXT`; default: `6g`)
- `-D`, `--host`: Path to the user-supplied host/contamination fasta /// Fasta file of known DNA entities expected in the sample (type: `TEXT`; default: `Sentinel.UNSET`)
- `--keep-tmp`: Keep temporary files (type: `BOOLEAN`; default: `False`)
- `--log-file`: Path to log file (type: `TEXT`; default: `/clusterfs/jgi/scratch/science/metagen/neri/code/rolypoly/rolypoly_pipeline.log`)
- `-ll`, `--log-level`: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (type: `TEXT`; default: `INFO`)
- `--skip-existing`: Skip commands if output files already exist (type: `BOOLEAN`; default: `False`)
- `-A`, `--assembler`: Assembler choice (spades,megahit,penguin). For multiple, give a comma-separated list (type: `TEXT`; default: `spades,megahit`)
- `-d`, `--post-processing`: Method for merging or clustering the assembler output(s), options: - linclust: use MMseqs2 linclust to cluster the assembler output at 99% identity and 99% coverage using coverage-mode 1. These parameters mean that most subsequences that are wholly contained within a larger sequence will dropped (use with caution, as a chimeras from one assembler may be merged with a chimera from another assembler may 'engulf' a non-chimeric sequence from the other assembler) - rmdup: use seqkit rmdup to remove identical sequences (same sequence, same length, or its' reverse complement) - none: do not perform any post assembly processing (type: `TEXT`; default: `none`)
- `-Fm1`, `--filter1_nuc`: First set of rules for nucleic filtering by aligned stats (type: `TEXT`; default: `alnlen >= 120 & pident>=75`)
- `-Fm2`, `--filter2_nuc`: Second set of rules for nucleic match filtering (type: `TEXT`; default: `qcov >= 0.95 & pident>=95`)
- `-Fd1`, `--filter1_aa`: First set of rules for amino (protein) match filtering (type: `TEXT`; default: `length >= 80 & pident>=75`)
- `-Fd2`, `--filter2_aa`: Second set of rules for protein match filtering (type: `TEXT`; default: `qcovhsp >= 95 & pident>=80`)
- `--dont-mask`: If set, host fasta won't be masked for potential RNA virus-like seqs (type: `BOOLEAN`; default: `False`)
- `--mmseqs-args`: Additional arguments to pass to MMseqs2 search command (type: `TEXT`; default: `Sentinel.UNSET`)
- `--diamond-args`: Additional arguments to pass to Diamond search command (type: `TEXT`; default: `--id 50 --min-orf 50`)
- `--db`: Database to use for marker gene search (type: `TEXT`; default: `all`)




