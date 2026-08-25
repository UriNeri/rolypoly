# MMTAX

## Summary

Assign ICTV taxonomy to virus contigs from protein matches made by MMseqs2 or
DIAMOND.

Both search backends feed the same post-search classifier:

1. Keep matches within `--top` percent of each protein's best bitscore.
2. Collapse duplicate hits assigned to the same taxid to their best score.
3. Walk down the ICTV ranks using weighted votes, first for each protein and
   then across the proteins on each contig.

Each selected rank must be a child of the previously selected rank, so the
result is always one lineage that exists in the taxonomy. A match labelled only
at a shallow rank contributes there but is neutral in deeper-rank votes. Thus a
large weight assigned only to `root` cannot erase lower-weight family evidence,
while conflicting family-resolved matches still compete normally. This rule is
generic and does not privilege any particular virus lineage.

The vote can be weighted by bitscore (default) or negative log E-value. This
makes backend comparisons less confounded by different built-in ORF and LCA
implementations.

## Usage

With the built-in database:

```bash
rolypoly mmtax \
  --input contigs.fasta \
  --output taxonomy.tsv \
  --backend mmseqs \
  --threads 8
```

Nucleotide contigs are passed through the existing pyrodigal-rv predictor. To
reuse proteins from an earlier RolyPoly step, supply the FASTA and an explicit
two-column `protein<TAB>contig` map:

```bash
rolypoly mmtax \
  --input contigs.fasta \
  --proteins predicted_orfs.faa \
  --protein-map protein_to_contig.tsv \
  --output taxonomy.tsv
```

`--infer-protein-map` is available for RolyPoly/pyrodigal-style headers ending
in an ORF number, but an explicit map is safer for arbitrary protein FASTA
files.

For direct protein input, use `--query-type protein`. Without a map, each
protein is classified as its own output query. Supplying `--protein-map` instead
groups those proteins back onto contigs before weighted assignment.

For a custom database, `--taxdump` is mandatory:

```bash
rolypoly mmtax \
  --input contigs.fasta \
  --database /path/to/ictv_nr_db/ictv_nr_db \
  --taxdump /path/to/ictv_taxdump \
  --backend mmseqs \
  --output taxonomy.tsv
```

A custom MMseqs2 database needs the `_mapping` and `_taxonomy` sidecars from
`mmseqs createtaxdb`. A custom DIAMOND database must be built with `--taxonmap`
and `--taxonnodes`. A plain FASTA file is not a taxonomy database.

## Shared search controls

`--sensitivity` accepts a number from 1 through 8 or the corresponding names:
`faster`, `fast`, `mid-sensitive`, `normal`, `sensitive`, `more-sensitive`,
`very-sensitive`, and `ultra-sensitive`. `normal` is the default and maps to
MMseqs2 sensitivity 4 or DIAMOND's unmodified default preset.

`--min-aln-len` is passed directly to MMseqs2. For DIAMOND it is applied to the
tabular hits after search and before taxonomy assignment. RolyPoly applies
`--top` once, as the same post-search bitscore-window filter for both backends,
before calculating the rank-aware assignment.

`--tax-lineage 1` writes names, `--tax-lineage 2` writes taxids (the default),
and `--tax-lineage 0` omits the compact lineage. Named ICTV-rank columns are
always retained for reporting.

## Output

The headered TSV contains the assigned `query`, `taxid`, `rank`, `taxon_name`,
overall `support`, `informative_fraction`, counts of assigned proteins and
retained hits, `lineage`, `backend`, and `method`. Each ICTV rank also has a name
and conditional support column, for example `family` and `family_support`.
`support` describes agreement among matches that resolve to the assigned rank;
`informative_fraction` reports how much compatible retained-hit weight actually
resolved that deeply. A high-support call with a low informative fraction should
therefore be treated as tentative.

The best individual retained alignment is reported separately as
`best_match_target`, its taxid, name and rank, and its bitscore, E-value,
identity, and alignment length. This is evidence, not necessarily the final
assignment: for example, the best-scoring hit may be labelled only at `root`
while other rank-resolved hits support a family.

Breadth columns make that distinction more auditable:

- `proteins_assigned / total_proteins` and `protein_hit_fraction` report
  how many predicted genes on the contig had retained matches.
- `aligned_residues / total_protein_residues` and
  `residue_alignment_fraction` report the union of covered query-amino-acid
  positions, so overlapping database matches are not counted repeatedly.
- `projected_aligned_nt / genome_length` and
  `projected_alignment_genome_fraction` report
  the union of those amino-acid intervals projected through single-CDS GFF
  coordinates. They remain empty for protein-only mappings or proteins whose
  genomic coordinates cannot be projected safely.

RolyPoly's HTML report discovers `mmtax.tsv` in an output directory and adds a
taxonomy table plus a family-composition chart. The `roll` command runs taxonomy
by default; use `--skip-steps taxonomy` to omit it.

`rvmt` and `nvpc` are reserved database names. They will remain disabled until
their profiles have been enriched with compatible ICTV taxids.
