# Annotate Prot

<!-- Auto-generated draft from CLI metadata for `rolypoly annotate-prot`. -->
<!-- Expand this page with command-specific context, examples, and citations. -->

## Summary

Identify coding sequences (ORFs) from fasta, and predicts their translated seqs putative function via homology search.

## Description

Currently supported tools and databases:

- Translations: ORFfinder, pyrodigal, six-frame

- Search engines:

    - (py)hmmsearch: Pfam, NVPC, RVMT, genomad, vfam

    - mmseqs2: NVPC, RVMT, genomad, vfam

    - diamond: Uniref50 (viral subset)

- custom: user supplied database. Needs to be in tool appropriate format, or a directory of aligned fasta files (for hmmsearch)

## Usage

```bash
rolypoly annotate-prot [OPTIONS]
```

<!-- BEGIN GENERATED CLI OPTIONS -->
## Options

- `-i`, `--input`: Fasta file or input directory containing rolypoly's virus identification results (type: `TEXT`; required; default: `Sentinel.UNSET`)
- `-o`, `--output-dir`: Output directory path (type: `TEXT`; default: `./annotate_prot_output`)
- `--reuse-translations-from`: Reuse a verified translation bundle from annotation or marker-search output. (type: `DIRECTORY`)
- `-op`, `--override-parameters`, `--override-params`: JSON-like string of parameters to override. Example: --override-parameters '{"ORFfinder": {"minimum_length": 150}, "hmmsearch": {"E": 1e-3}}' (type: `TEXT`; default: `{}`)
- `-ss`, `--skip-steps`: Comma-separated steps to skip: predict_orfs, prepare_translation_metadata, search_protein_domains, resolve_domain_overlaps, combine_results. Example: --skip-steps resolve_domain_overlaps (type: `TEXT`; default: ``)
- `-gp`, `--gene-prediction-tool`: Tool for gene prediction. (type: `CHOICE`; default: `pyrodigal`)

    - pyrodigal-rv: might work well for some viruses, but it's not as well tested for RNA viruses. Includes internal genetic code assignment.

    - ORFfinder: The default ORFfinder settings may have some false positives, but it's fast and easy to use.

    - six-frame: includes all 6 reading frames, so all possible ORFs are predicted - prediction is quick but will include many false positives, and the input for the domain search will be larger.

- `-st`, `--search-tool`: Tool/command for protein domain detection. Only one tool can be used at a time. (type: `CHOICE`; default: `hmmsearch`)
- `-d`, `--domain-db`: comma-separated list of database(s) for domain detection. (type: `TEXT`; default: `Pfam,NVPC`)

    - Pfam: Pfam-A (only hmmsearch)

    - RVMT: RVMT RdRp profiles

    - NVPC: RVMT's New Viral Profile Clusters, filtered to remove "hypothetical" proteins

    - genomad: genomad virus-specific markers - note these can be good for identification but not ideal for annotation.

    - vfam: VFam profiles from VOGDB release 236, filtered to remove low-information profiles

    - uniref50: UniRef50 viral subset (for diamond only)

    - custom: custom (path to a custom database in HMM format or a directory of MSA/hmms files)

    - all: all (all databases)

- `-ml`, `--min-orf-length`: Minimum ORF length for gene prediction (type: `INTEGER`; default: `30`)
- `-gc`, `--genetic-code`: Genetic code (a.k.a. translation table) NOT REALLY USED CURRENTLY (type: `INTEGER`; default: `11`)
- `-e`, `--evalue`: E-value for search result filtering. Note, this is for inital filteringg only, you are encouraged to filter the results further using e.g. profile coverage and scores. (type: `FLOAT`; default: `0.001`)
- `--db-create-mode`: How to handle custom database directories: auto=guess, mmseqs=build mmseqs profile DB, hmm=build concatenated HMM (type: `CHOICE`; default: `auto`)
- `--output-format`: Output format for the combined results (type: `CHOICE`; default: `tsv`)
- `-rm`, `--resolve-mode`: How to deal with overlapping domain hits in the same query sequence. (type: `CHOICE`; default: `simple`)

    - merge: all overlapping hits are merged into one range

    - one_per_range: one hit per range (ali_from-ali_to) is reported

    - one_per_query: one hit per query sequence is reported

    - split: each overlapping domain is split into a new row

    - drop_contained: hits that are contained within (i.e. enveloped by) other hits are dropped

    - none: no resolution of overlapping hits is performed

    - simple: heuristic-based approach - chains drop_contained with adaptive overlap detection for polyproteins

- `-mo`, `--min-overlap-positions`: Minimal number of overlapping positions between two intersecting ranges before they are considered as overlapping (used in some resolve_mode(s)). With 'simple' mode, this is adaptively adjusted for polyprotein detection. (type: `INTEGER`; default: `10`)
- `--alignment-strings`, `--no-alignment-strings`: Include alignment identity strings in hmmsearch outputs (applies to modomtblout format). (type: `BOOLEAN`; default: `True`)
- `-t`, `--threads`: Number of worker threads. (type: `INTEGER RANGE`; default: `1`)
- `-M`, `--memory`: Memory limit, for example 8g. (type: `MEMORY`; default: `8g`)
- `-g`, `--log-file`: Path to the log file. (type: `FILE`; default: `rolypoly.log`)
<!-- END GENERATED CLI OPTIONS -->

## Coordinate provenance

Before search, protein annotation and marker search normalize translated queries
into `predicted_orfs.faa` and `predicted_orfs.gff`. IDs are consistent across both
files: `<contig>_orf_<ordinal>` for predicted ORFs, `<contig>_frame_p1` through
`<contig>_frame_m3` for full-frame translations, and `<input-id>_protein_1` for
provided proteins. Unsafe characters in parent IDs are percent-encoded. ORF
ordinals follow genomic position within each contig; they are not biological
identifiers shared across different predictors.

The complete native FASTA and predictor GFF (when present) are retained under
`tool_outputs/`, including when temporary search files are cleaned. The sidecar
`translation_metadata.tsv` preserves complete original headers, original IDs,
source-contig headers, prediction attributes, and original GFF records. It joins
into the final hit table after search/overlap resolution. New search results join
strictly on normalized IDs; matching old tool IDs requires explicit migration
and rejects ambiguous mappings.

All added amino-acid and nucleotide intervals are **1-based and inclusive**.
Nucleotide bounds always ascend; `strand` is separately stored as `1` or `-1`.
The original search coordinate columns retain their existing meaning.

| Fields | Meaning |
| --- | --- |
| `search_query_id`, `translation_id` | Search-reported ID and normalized translated FASTA ID. These match in new runs; migrated older searches may retain tool-specific search IDs. |
| `original_translation_id`, `original_header`, `original_source_header` | Native identifiers and complete FASTA headers, including text after the first token. |
| `prediction_attributes`, `original_gff_record` | Parsed prediction attributes and source GFF fields stored as JSON. |
| `translation_label`, `orf_ordinal` | Exact canonical ID for display and per-contig ordinal; frames have IDs such as `CID_4_frame_m2` and a null ORF ordinal. |
| `source_seq_id`, `contig_length` | Parent nucleotide sequence and its full length. |
| `translation_method`, `translation_length_aa` | How the query was produced and its protein length. |
| `translation_nt_start`, `translation_nt_end` | Nucleotide interval represented by the translated query, including only codons present in that query. |
| `strand`, `frame_id` | Genomic orientation and signed reading frame (`+1..+3` or `-1..-3`), measured from the corresponding contig end. |
| `orf_nt_start`, `orf_nt_end` | Caller-reported ORF bounds, potentially including an omitted terminal stop; null for full-frame translations. |
| `aa_start`, `aa_end` | Query alignment span in amino acids (HMMER alignment bounds, rather than its envelope). |
| `nt_start`, `nt_end` | Corresponding genomic hit span, including the complete first and last codons. |
| `coordinate_system` | `1-based-inclusive`. |

Pyrodigal coordinates come from its companion GFF when available, with a FASTA
header fallback. ORFfinder's protein FASTA IDs use zero-based oriented endpoints;
these are explicitly converted. Six-frame queries use the frame suffix and
original contig length. Unknown mappings, ambiguous IDs, unsupported split CDS
records, and out-of-bounds hits are rejected rather than assigned invented
coordinates.

Protein-only input retains its input protein ID in `source_seq_id` and has null
genomic coordinates, contig length, strand, and frame. Its report shows an unplaced protein
hit table, and nucleotide GFF export requires a genomic mapping. Six-frame tracks
are labelled as translated frames, not predicted ORFs.

GFF export uses the parent contig, projected nucleotide bounds, and `+`/`-` strand.
Protein-domain features have phase `.`: GFF CDS phase is not the signed reading
frame. Metadata attribute keys and values are percent-escaped so descriptions
containing tabs, newlines, semicolons, commas, or equals signs do not corrupt GFF.

The shared arithmetic lives in `utils/bio/interval_ops.py`; translation metadata
is built in `translation.py`; table enrichment and GFF serialization live in
`polars_fastx.py`. DIAMOND `blastp`, MMseqs2 protein-search output, and HMMER report
positions within translated proteins. Their subject coordinates do not determine
the parent nucleotide strand. These adapters do not interpret nucleotide-search
or BLASTX output.

## Reusing a previous translation

`--reuse-translations-from PATH` accepts a normalized translation bundle from an
existing marker-search or protein-annotation output directory. It reuses **all**
translations for the requested contigs, not just queries that previously had
marker hits. Search engines and reference databases may differ.

```bash
rolypoly annotate-prot --input matched_contigs.fasta --output-dir annotation \
  --gene-prediction-tool six-frame --search-tool diamond --domain-db uniref50 \
  --reuse-translations-from marker_search_results
```

The bundle's `translation_manifest.json` records input IDs, full headers and
sequence hashes, effective prediction parameters, predictor versions, schema
version, and output-file hashes. Reuse requires exact matches, but permits a
subset of the original contigs. Changed headers, sequences, settings, versions,
or bundle files are rejected before copying. The destination must differ from
the source. BBMap reuse is not enabled without a verified version fingerprint.
Older outputs without a manifest must first be regenerated. Reuse is explicit;
`roll --skip-existing` alone does not establish compatibility.

Reports show exact, whitespace-free ORF/frame IDs alongside amino-acid lengths and in the
per-contig hit table. Hovering the table ID reveals the full normalized ID; the
translation mapping is linked for provenance. The reference-span column reports
which part of the target protein/profile aligned. Unaligned reference ends are
not, by themselves, evidence of an incomplete ORF, so they are not drawn as broken
ORF boundaries.

### Reusing marker searches in roll

`roll` passes its marker-search output directory to protein annotation. For
HMM searches, annotation can reuse saved **pre-filtering, pre-resolution** hits
from `search_cache/` when the database contents, pyhmmer version, translation
settings, input headers/sequences, translated IDs/sequences, search thresholds,
and required alignment fields match. The annotation stage still applies its
own overlap resolution and metadata enrichment. A `.reuse.json` sidecar records
the source of each reused search. Missing or incompatible caches cause a normal
search, with the reason recorded in the log; older outputs without a cache are
not assumed reusable.

HMMER infers its statistical search spaces (`Z` and `domZ`) from the protein
population. Removing proteins can change both E-values and domain inclusion.
Therefore ordinary HMM searches require identical protein populations, even
when the nucleotide inputs form a subset. The shared validator allows protein
subsets only when both search spaces were explicitly fixed identically; the
current command paths do not set these overrides. MMseqs2 marker and annotation
searches currently use different output/filtering contracts and are rerun.

The default roll stages also use different translations (six-frame markers,
pyrodigal annotation) and thresholds, so they do not qualify automatically.
Naming both databases RVMT or geNomad is insufficient: their contents and the
other conditions must match. This check avoids changing search sensitivity or
statistical meaning merely to enable reuse.

### Marker evidence in genome maps

Reports also discover `marker_search_results.tsv` and display its hits alongside protein annotation in shared reading-frame rows. These retain the
original query IDs, scores/E-values, database and translation method. The hit
tables and exports identify the stage; marker hits do not compete with
annotation hits during best-hit selection. Original query IDs and stage provenance remain available in the hit table.

Explicit nucleotide coordinates are preferred. Older six-frame marker tables
can be mapped using their original frame IDs and known parent contig lengths.
Hits without a reliable genomic mapping appear in a separate table rather than
being positioned speculatively. Original marker tables are linked for provenance.

The report groups fully covered marker hits as supporting evidence by default.
Grouping requires the same source/profile, strand and nucleotide reading phase,
and containment in both nucleotide and profile coordinates. This is a display
grouping, not a merged alignment or a claim that scores from different searches
are comparable. The underlying evidence and original scores remain intact.
Expand **Supporting evidence** below the hit table to inspect grouped hits; marker
hits that extend coverage remain visible. Available marker-matched sequences
are embedded for direct viewing and FASTA export. Full-sequence file controls
and protein-source filters are collapsible, and **Alignment details** reveals
the additional hit-table columns.

Genome maps group ORFs and protein hits by signed reading frame (`rf+1` through
`rf-3`). Only real ORFs receive gene arrows; six-frame translations contribute
hit intervals without synthetic gene blocks. Overlapping features receive
separate lanes within their frame. There are no marker/frame visibility toggles.
