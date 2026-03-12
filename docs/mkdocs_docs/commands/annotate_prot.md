# Annotate Prot

> Auto-generated draft from CLI metadata for `rolypoly annotate-prot`.
> Expand this page with command-specific context, examples, and citations.

## Summary

Identify coding sequences (ORFs) from fasta, and predicts their translated seqs putative function via homology search.

## Description

Currently supported tools and databases:

    * Translations: ORFfinder, pyrodigal, six-frame

    * Search engines:

    - (py)hmmsearch: Pfam, NVPC, RVMT, genomad, vfam

    - mmseqs2: NVPC, RVMT, genomad, vfam

    - diamond: Uniref50 (viral subset)

    * custom: user supplied database. Needs to be in tool appropriate format, or a directory of aligned fasta files (for hmmsearch)

## Usage

```bash
rolypoly annotate-prot [OPTIONS]
```

## Options

- `-i`, `--input`: Fasta file or input directory containing rolypoly's virus identification results (type: `TEXT`; required; default: `Sentinel.UNSET`)
- `-o`, `--output-dir`: Output directory path (type: `TEXT`; default: `./annotate_prot_output`)
- `-t`, `--threads`: Number of threads (type: `INTEGER`; default: `1`)
- `-g`, `--log-file`: Path to log file (type: `TEXT`; default: `./annotate_prot_logfile.txt`)
- `-op`, `--override-parameters`, `--override-params`: JSON-like string of parameters to override. Example: --override-parameters '{"ORFfinder": {"minimum_length": 150}, "hmmsearch": {"E": 1e-3}}' (type: `TEXT`; default: `{}`)
- `-ss`, `--skip-steps`: Comma-separated list of steps to skip. Example: --skip-steps ORFfinder,hmmsearch (type: `TEXT`; default: ``)
- `-gp`, `--gene-prediction-tool`: Tool for gene prediction. * pyrodigal-rv: might work well for some viruses, but it's not as well tested for RNA viruses. Includes internal genetic code assignment. * ORFfinder: The default ORFfinder settings may have some false positives, but it's fast and easy to use. * six-frame: includes all 6 reading frames, so all possible ORFs are predicted - prediction is quick but will include many false positives, and the input for the domain search will be larger. (type: `CHOICE`; default: `ORFfinder`)
- `-st`, `--search-tool`: Tool/command for protein domain detection. Only one tool can be used at a time. (type: `CHOICE`; default: `hmmsearch`)
- `-d`, `--domain-db`: comma-separated list of database(s) for domain detection. * Pfam: Pfam-A (only hmmsearch) * RVMT: RVMT RdRp profiles * NVPC: RVMT's New Viral Profile Clusters, filtered to remove "hypothetical" proteins * genomad: genomad virus-specific markers - note these can be good for identification but not ideal for annotation. * vfam: VFam profiles fetched on December 2025, filtered to remove "hypothetical" proteins * uniref50: UniRef50 viral subset (for diamond only) * custom: custom (path to a custom database in HMM format or a directory of MSA/hmms files) * all: all (all databases) (type: `TEXT`; default: `Pfam,NVPC`)
- `-ml`, `--min-orf-length`: Minimum ORF length for gene prediction (type: `INTEGER`; default: `30`)
- `-gc`, `--genetic-code`: Genetic code (a.k.a. translation table) NOT REALLY USED CURRENTLY (type: `INTEGER`; default: `11`)
- `-e`, `--evalue`: E-value for search result filtering. Note, this is for inital filteringg only, you are encouraged to filter the results further using e.g. profile coverage and scores. (type: `FLOAT`; default: `0.1`)
- `--db-create-mode`: How to handle custom database directories: auto=guess, mmseqs=build mmseqs profile DB, hmm=build concatenated HMM (type: `CHOICE`; default: `auto`)
- `--output-format`: Output format for the combined results (type: `CHOICE`; default: `tsv`)
- `-rm`, `--resolve-mode`: How to deal with overlapping domain hits in the same query sequence. - merge: all overlapping hits are merged into one range - one_per_range: one hit per range (ali_from-ali_to) is reported - one_per_query: one hit per query sequence is reported - split: each overlapping domain is split into a new row - drop_contained: hits that are contained within (i.e. enveloped by) other hits are dropped - none: no resolution of overlapping hits is performed - simple: heuristic-based approach - chains drop_contained with adaptive overlap detection for polyproteins (type: `CHOICE`; default: `simple`)
- `-mo`, `--min-overlap-positions`: Minimal number of overlapping positions between two intersecting ranges before they are considered as overlapping (used in some resolve_mode(s)). With 'simple' mode, this is adaptively adjusted for polyprotein detection. (type: `INTEGER`; default: `10`)
- `--alignment-strings`, `--no-alignment-strings`: Include alignment identity strings in hmmsearch outputs (applies to modomtblout format). (type: `BOOLEAN`; default: `True`)




