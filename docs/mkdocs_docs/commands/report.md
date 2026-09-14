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

<!-- BEGIN GENERATED CLI OPTIONS -->
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
<!-- END GENERATED CLI OPTIONS -->





## Reading the genome map

The nucleotide axis runs along the contig. Signed reading-frame rows (`rf+1`
through `rf-3`) contain predicted ORF arrows and protein-hit blocks. Negative
frames are measured from the reverse-complement sequence. Arrows indicate
translation direction; block colours identify source databases. Overlapping
features occupy separate lanes. Six-frame hits have no synthetic gene arrows,
so evidence outside predicted ORFs remains visible. Hover over hits for query
IDs, coordinates and search details.

## Filters and supporting evidence

**Best only** uses the selected overlap-resolution criterion; it is not a
confidence threshold. **All hits** still groups covered marker evidence under
annotations marked **+ marker**. Expand **Supporting evidence** below the hit
table to inspect the original marker hits. Grouping requires matching source,
profile, strand and reading phase, and containment of both nucleotide and
profile intervals. Hits that add coverage remain visible. Original scores and
E-values are preserved, not merged or recomputed.

Expand **Protein sources** to filter databases. RNA, nucleic and RdRp-motif
checkboxes control those map tracks. The map's **max E (10^)** field takes an
exponent: `-5` means `1e-5`; `0` disables that filter.

## Sequences and exports

**Alignment details** reveals additional hit-table columns. Partial reference
coverage alone does not establish ORF incompleteness. Predicted-ORF counts use
prediction metadata; a dash means the count is unavailable.

**Show hit** displays the matched amino-acid sequence and allows FASTA export.
Available marker-hit sequences are embedded. Full protein/contig sequences can
be loaded through **Sequence files**; choose a local FASTA if the browser blocks
loading referenced files. Keep the HTML beside its output files to preserve
relative links.

**Export shown hits TSV** exports the displayed primary hits. Grouped supporting
hits remain accessible in their expandable section and the original tables.
**Original files** links to pipeline outputs. The HTML includes an offline
**Help** tab; its documentation links open the online manual.

## Caveats

### Nucleic hits end at 10 kbp length

Nucleic-hit tracks can end at approximately 10,000 bases, even on longer
contigs. This is a symptom of MMseqs2 internally splitting long sequences in
its nucleotide-search workflow, not evidence of a biological boundary or of
absent homology beyond that point. The upstream [search workflow sets a
10,000-base maximum sequence length for nucleotide searches](https://github.com/soedinglab/MMseqs2/blob/master/src/workflow/Search.cpp).
The exact behaviour depends on the MMseqs2 version and search settings. Check
the original hit table and search logs before interpreting a sharp endpoint.

### Taxonomy charts count contigs, not abundance

The pie chart shows the distribution of rows in the taxonomy table (normally
one per classified contig). It is **not an estimate of absolute or relative
abundance**, read depth, biomass, virus particles, or complete viral genomes.
For direct protein input without a contig map, rows instead represent protein
queries, so the chart is not even a contig distribution in that case.

In roll, the represented set depends on the sensitivity and quality of the
markers used to identify candidate contigs, which are primarily RdRp-focused.
Segments without the marker, accessory contigs, and other undetected sequences
may never enter this set and are therefore absent from the denominator. There
is no automatic reconstruction or counting of every segment of a segmented
virus. Multiple represented contigs can also come from one virus. The chart
must not be interpreted as the composition of the whole viral community.

Taxonomic labels themselves may be overly specific: see [mmtax caveats](mmtax.md#caveats)
for the defaults and why sequence-similarity votes do not establish taxonomic
membership under the appropriate biological demarcation criteria.

## Known bugs

### Split nucleotide hits are not reconstructed as one continuous match

The report currently displays the intervals returned by the nucleotide search;
it does not reconstruct a continuous alignment across MMseqs2 internal sequence
splits. The resulting approximately 10-kbp endpoints can misleadingly resemble
biological boundaries. This remains an unresolved search/report integration
limitation; the chart itself does not establish that the remaining sequence
lacks a match. See the [10-kbp caveat](#nucleic-hits-end-at-10-kbp-length).
