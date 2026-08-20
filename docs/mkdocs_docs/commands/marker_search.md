# Marker Search

<!-- Auto-generated draft from CLI metadata for `rolypoly marker-search`. -->
<!-- Expand this page with command-specific context, examples, and citations. -->

## Summary

RNA virus marker protein search using HMMER or MMseqs2 profile databases.

## Description

Most pre-made DBs are based on RdRp domain (except for geNomad).
Input can be nucleotide contigs or amino acid seqs.
If nucleotide, by default all contigs will be translated to six end-to-end frames (with stops replaced by `X`), or into ORFs called by pyrodigal (meta) or callgenes.sh

Pre-compiled options are:

• NeoRdRp2.1

    GitHub: https://github.com/shoichisakaguchi/NeoRdRp  | Paper: https://doi.org/10.1264/jsme2.ME22001

• RVMT

    GitHub: https://github.com/UriNeri/RVMT  | Zenodo: https://zenodo.org/record/7368133  |  Paper: https://doi.org/10.1016/j.cell.2022.08.023

• RdRp-Scan

    GitHub: https://github.com/JustineCharon/RdRp-scan  |  Paper: https://doi.org/10.1093/ve/veac082

        ⤷ (which IIRC incorporated PALMdb, GitHub: https://github.com/rcedgar/palmdb, Paper: https://doi.org/10.7717/peerj.14055

• Pfam_RTs_RdRp

    RdRp and RT profiles from Pfam 38.2 --- PF04197.18,PF04196.18,PF22212.2,PF22152.2,PF22260.2,PF00680.26,PF00978.27,PF00998.29,PF02123.22,PF07925.16,PF00078.33,PF07727.20,PF13456.13
    Data: https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam38.2/ | Paper https://doi.org/10.1093/nar/gkaa913
• geNomad

    RNA virus marker genes from geNomad v1.9 --- https://zenodo.org/records/14886553
For a custom path, use an HMM/MSA source with hmmsearch or an MMseqs database prefix with mmseqs2.
Please cite accordingly based on the DBs you select.

## Usage

```bash
rolypoly marker-search [OPTIONS]
```

## Options

- `-i`, `--input`: Input fasta file. Preferably nucleotide contigs, but you can provide amino acid input too (the script would skip 6 frame translation) (type: `TEXT`; required; default: `Sentinel.UNSET`)
- `-o`, `--output`: Path to output directory. Note - if multiple DBs are used and the resolve-mode is `none`, multiple outputs are made (DB name appended as suffix). (type: `TEXT`; default: `/home/neri/Documents/GitHub/rps/rolypoly/marker_search_out`)
- `-rm`, `--resolve-mode`: How to deal with regions in your query that match multiple profiles? (type: `CHOICE`; default: `simple`)

    - merge: all overlapping hits are merged into one range

    - one_per_range: one hit per range (ali_from-ali_to) is reported

    - one_per_query: one hit per query sequence is reported

    - split: each overlapping domain is split into a new row

    - drop_contained: hits that are contained within (i.e. enveloped by) other hits are dropped.

    - none: no resolution of overlapping hits is performed. NOTE - EXPECT A POTENTIALLY LARGE OUTPUT

    - simple: heuristic/personal observation based - chains drop_contained output with split mode.

- `-mo`, `--min-overlap-positions`: Minimal number of overlapping positions between two intersecting ranges before they are considered as overlapping (used in some resolve_mode(s) (type: `INTEGER`; default: `10`)
- `--repeat-filter`, `--no-repeat-filter`: Filter hits where the same profile region repeatedly matches distinct parts of one query. (type: `BOOLEAN`; default: `True`)
- `-ie`, `--inc-evalue`: Maximal e-value for including a domain match in the results (type: `FLOAT`; default: `0.001`)
- `-s`, `--score`: Minimal score for including a domain match in the results (type: `INTEGER`; default: `20`)
- `-mla`, `--min-ali-len`: Minimal alignment length for including a domain match in the results (type: `INTEGER`; default: `15`)
- `-am`, `--aa-method`: Method to translate nucleotide sequences into amino acids. Options: six frame translation using seqkit, pyrodigal-rv uses pyrodigal-meta with additional genetic codes, bbmap callgenes.sh (quick but less accurate for metagenomic data) (type: `CHOICE`; default: `six_frame`)
- `-db`, `--database`: comma separated list of databases to search against (or `all`), or path to a custom database. options: NeoRdRp_v2.1, RdRp-scan, RVMT, Pfam_RTs_RdRp, genomad, all. Availability depends on the selected backend. With hmmsearch, a custom path may be an HMM, an MSA, or a directory of either. With mmseqs2, provide an MMseqs database prefix. (type: `TEXT`; default: `RVMT,genomad`)
- `-st`, `--search-tool`: Profile-search backend. MMseqs2 uses the corresponding prebuilt MMseqs profile databases. (type: `CHOICE`; default: `hmmsearch`)
- `-ow`, `--overwrite`: Do not overwrite the output directory if it already exists (type: `BOOLEAN`; default: `False`)
- `--write-matched-regions`, `--no-write-matched-regions`: Write matched query regions to FASTA (enabled by default; disable with --no-write-matched-regions) (type: `BOOLEAN`; default: `True`)
- `-mro`, `--matched-regions-output`: Output FASTA path for matched regions (default: <output>/marker_search_matched_regions.faa) (type: `TEXT`)
- `--include-aligned-region`, `--no-include-aligned-region`: Include aligned query region sequence in marker_search_results.tsv (enabled by default) (type: `BOOLEAN`; default: `True`)
- `--include-alignment-string`, `--no-include-alignment-string`: Include alignment identity string in marker_search_results.tsv (disabled by default) (type: `BOOLEAN`; default: `False`)
- `--write-matched-input-seqs`, `--no-write-matched-input-seqs`: Write full original input sequences (contigs for nucleotide input, whole proteins for AA input) that had at least one marker hit to FASTA (disabled by default) (type: `BOOLEAN`; default: `False`)
- `-miso`, `--matched-input-seqs-output`: Output FASTA path for matched input sequences (default: <output>/marker_search_matched_input_seqs.fna|faa) (type: `TEXT`)
- `-t`, `--threads`: Number of worker threads. (type: `INTEGER RANGE`; default: `1`)
- `-m`, `-M`, `--memory`: Memory limit, for example 8g. (type: `MEMORY`; default: `8g`)
- `-k`, `--keep-tmp`: Keep temporary files. (type: `BOOLEAN`; default: `False`)
- `-td`, `-tempdir`, `-tmp`, `--temp-dir`: Temporary working directory. (type: `DIRECTORY`)
- `-g`, `--log-file`: Path to the log file. (type: `FILE`; default: `rolypoly.log`)
