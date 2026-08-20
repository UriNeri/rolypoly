# Report

<!-- Auto-generated draft from CLI metadata for `rolypoly report`. -->
<!-- Expand this page with command-specific context, examples, and citations. -->

## Summary

Render an interactive per-contig genome-map report from RolyPoly outputs.

## Description

Tabs appear only when data is present: Table / Genome maps (protein domains +
RNA + nucleic tracks), Nucleic hits, Run stats, and any --extra-tab layers.
Toggle all-hits vs best-only and pick the best-by criterion in the toolbar;
protein and RNA hits are resolved separately, sourcing rolypoly's
consolidate_hits. RNA discrete features are classified (rRNA / tRNA / IRES /
ribozyme / riboswitch / frameshift / UTR / CRE / motif). The "source" criterion
applies a precedence order (RVMT > NVPC > Pfam > genomad > VFAM by default)
without excluding lower-priority sources.

## Usage

```bash
rolypoly report [OPTIONS]
```

## Options

- `-i`, `--input`: A protein/marker hit table (combined_annotations.tsv, or any TSV/CSV/Parquet), OR a roll/annotate output directory (RNA, nucleic-search and run-stats are then discovered automatically). (type: `TEXT`; required; default: `Sentinel.UNSET`)
- `-o`, `--output`: Output HTML file. (type: `TEXT`; required; default: `Sentinel.UNSET`)
- `-r`, `--rna`: Optional annotate-rna table (ignored in directory mode, where it is discovered). (type: `TEXT`)
- `-nu`, `--nucleic`: Optional nucleic-search table(s) (repeatable; ignored in directory mode). (type: `TEXT`; default: `Sentinel.UNSET`)
- `-x`, `--extra-tab`: Add a generic table tab as 'Label=path.tsv' (repeatable), e.g. for predicted taxonomy or host prediction. (type: `TEXT`; default: `Sentinel.UNSET`)
- `-tx`, `--taxonomy`: Optional mmtax TSV; adds a taxonomy table and composition chart. (type: `TEXT`)
- `--rrna-mapping`: Path to rrna_to_genome_mapping.parquet to enrich the rRNA stats with reference organism names (default: $ROLYPOLY_DATA/contam/rrna/...). (type: `TEXT`)
- `-T`, `--title`: Title shown in the report header. (type: `TEXT`; default: `RolyPoly — Genome / marker maps`)
- `-ms`, `--min-score`: Drop protein hits with bit score below this value. (type: `FLOAT`)
- `-me`, `--max-evalue`: Drop protein hits with E-value above this value. (type: `FLOAT`)
- `-b`, `--best-only`, `-a`, `--all-hits`: Initial view mode (toggleable in the viewer). (type: `BOOLEAN`; default: `False`)
- `-bb`, `--best-by`: Initial 'best' criterion: score | evalue | longest | source. (type: `CHOICE`; default: `score`)
- `-n`, `--min-overlap`: Min overlapping positions to collapse hits during best-hit resolution (1 = also collapse partial/nested overlaps). (type: `INTEGER`; default: `1`)
- `-sp`, `--source-priority`: Comma-separated precedence order for the 'source' criterion (default: rvmt,nvpc,pfam,genomad,vfam). Lower-priority sources still win any locus no higher-priority hit overlaps. (type: `TEXT`)
- `-st`, `--start-tab`: Which tab to open on load (table | maps | nucleic | stats | <extra id>); falls back to the first available tab. (type: `TEXT`; default: `table`)
- `-rb`, `--rna-bins`: Number of windows for the RNA base-pairing-density strip. (type: `INTEGER`; default: `150`)
- `--no-stats`: Do not collect reads/assembly run statistics (directory mode). (type: `BOOLEAN`; default: `False`)
- `--col-query`: Override the ORF/query id column (default: auto-detect the schema). (type: `TEXT`)
- `--col-profile`: Override the profile/marker name column (default: auto-detect). (type: `TEXT`)
- `--col-source`: Override the source/database column that drives colour (default: auto-detect). (type: `TEXT`)
- `--col-aligned`: Override the aligned-region / consensus column shown on hover. '' disables it. Default: auto-detect (identity_str for hmmsearch). (type: `TEXT`)
- `-lf`, `-g`, `--log-file`: Path to the log file. (type: `FILE`; default: `rolypoly.log`)




