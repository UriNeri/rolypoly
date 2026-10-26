# Annotate Rna

<!-- Auto-generated draft from CLI metadata for `rolypoly annotate-rna`. -->
<!-- Expand this page with command-specific context, examples, and citations. -->

## Summary

Annotate RNA structural features and motifs on viral nucleotide inputs.

## Description

The default pipeline runs secondary-structure prediction, RNA-family/CM
search, and tRNA detection. IRES detection, RNA motif searches, and
RNAsselem searches are not currently enabled.

Use `--skip-steps` to disable modules and `--override-parameters` to pass
tool-specific tuning values for individual stages.

## RNA family types and exports

Rfam matches are no longer universally labelled `ribozyme`. The runtime loads
`$ROLYPOLY_DATA/profiles/rfam_family.tsv.gz`, built alongside the models by the
`rfam` step in `rolypoly-db`. That step pins both products to Rfam 15.1.
Classification uses the accession and curated Rfam TP categories, not broad
substring searches of model names or descriptions.

Exported `type` values are Sequence Ontology (SO) sequence features:

| Curated evidence | Exported type |
|---|---|
| Known LSU / SSU families | `cytosolic_LSU_rRNA` / `cytosolic_SSU_rRNA` (mitochondrial families use `mt_LSU_rRNA` / `mt_SSU_rRNA`) |
| 5.8S family RF00002 | `cytosolic_5_8S_rRNA` |
| Other curated rRNA / tRNA | `rRNA` / `tRNA` |
| Curated ribozyme / riboswitch | `ribozyme` / `riboswitch` |
| Curated IRES | `internal_ribosome_entry_site` |
| Curated frameshift elements / thermoregulators | `cis_regulatory_frameshift_element` / `RNA_thermometer` |
| Other cis-regulatory elements | `regulatory_region` |
| Curated snoRNA subtypes | `C_D_box_snoRNA`, `H_ACA_box_snoRNA`, `scaRNA` (otherwise `snoRNA`) |
| Curated snRNA, miRNA, lncRNA, antisense RNA | `snRNA`, `miRNA`, `lncRNA`, `antisense_RNA` |
| Other RNA genes / introns | `ncRNA` / `intron` |
| Unclassified models, including custom models without metadata | `nucleotide_match` |

These terms and their `is_a` ancestry were validated against SO 2026-08-07.
`rna_class` is a separate presentation category (for example `frameshift`),
and `so_accession` identifies the exported SO type. Display labels are not
substituted into GFF3 column 3.

The original Infernal `ribozymes.out` remains unchanged search evidence.
`ribozymes_annotations.tsv` contains all parsed, typed hits before overlap
resolution; `ribozymes_resolved.out` contains typed, overlap-resolved hits.
Combined TSV/CSV/GFF3 exports retain `rfam_id`, model name, description,
E-value, model coordinates, original oriented sequence coordinates
(`seq_from_raw`, `seq_to_raw`), and other Infernal evidence. Curated metadata
adds `rfam_type`, `rfam_description`, `rfam_release`, and `metadata_source`.
Reports use these same types and categories rather than re-inferring biology.

The family table does not supply GO annotations; none are invented.
An optional `Ontology_term` column in a supplied metadata table is preserved
as an attribute, never used as a feature type.

## Caveats

### Older bundles and custom models

If the family sidecar is absent, a warning is logged. Exact, curated identities
for common rRNA, tRNA, hammerhead/RNase P, and cobalamin families remain supported;
other models use `nucleotide_match`. An old resolved table without accessions
can use only exact known model names. Refresh the Rfam database bundle for broad
family coverage. New models and metadata should always come from the same release.

A custom CM database may provide a sibling `<custom-cm-db>.families.tsv.gz`
with the same headered schema (`rfam_id`, `profile_name`, `rfam_type`,
`rfam_description`, `rfam_release`, `metadata_source`). Malformed metadata is
an error, not a reason to silently use fallbacks.

This changes annotation semantics only. Search thresholds, overlap-selection
policy, early rRNA filtering, and flag-only behavior are unchanged. It does not
remove rRNA-matched contigs.

## Usage

```bash
rolypoly annotate-rna [OPTIONS]
```

<!-- BEGIN GENERATED CLI OPTIONS -->
## Options

- `-i`, `--input`: Input nucleotide sequence file (fasta, fna, fa, or faa) (type: `PATH`; required; default: `Sentinel.UNSET`)
- `-o`, `--output-dir`: Output directory path (type: `TEXT`; default: `./annotate_RNA_output`)
- `-op`, `--override_parameters`, `--override-parameters`: JSON-like string of parameters to override. Example: --override-parameters '{"RNAfold": {"temperature": 37}, "cmsearch": {"cut_ga": false, "E": 1e-5}}' (type: `TEXT`; default: `{}`)
- `--skip-steps`: Comma-separated steps to skip: predict_secondary_structure, search_ribozymes, predict_trnas, resolve_rna_element_overlaps. Example: --skip-steps predict_secondary_structure,search_ribozymes (type: `TEXT`; default: ``)
- `--secondary-structure-tool`: Tool for secondary structure prediction. LinearFold is faster but less configurable. (type: `CHOICE`; default: `LinearFold`)
- `--ires-tool`: Reserved IRES tool setting; IRES detection is not currently enabled (type: `CHOICE`; default: `IRESfinder`)
- `--trna-tool`: Tool for tRNA identification (type: `CHOICE`; default: `tRNAscan-SE`)
- `--rnamotif-tool`: Reserved RNA motif tool setting; RNA motif searches are not currently enabled. (type: `CHOICE`; default: `lightmotif`)
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
<!-- END GENERATED CLI OPTIONS -->
