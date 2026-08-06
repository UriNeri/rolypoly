# MMTAX

## Summary

Assign ICTV taxonomy to virus contigs from protein matches made by MMseqs2 or
DIAMOND.

Both search backends feed the same post-search classifier:

1. Keep matches within `--top` percent of each protein's best bitscore.
2. Assign each protein to the LCA of its retained matches.
3. Assign each contig by a weighted majority vote across its proteins.

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
before calculating an LCA.

`--tax-lineage 1` writes names, `--tax-lineage 2` writes taxids (the default),
and `--tax-lineage 0` omits the compact lineage. Named ICTV-rank columns are
always retained for reporting.

## Output

The headered TSV contains the assigned `query`, `taxid`, `rank`, `taxon_name`,
overall `support`, counts of assigned proteins and retained hits, `lineage`,
`backend`, and `method`. Each ICTV rank also has a name and support column, for
example `family` and `family_support`.

RolyPoly's HTML report discovers `mmtax.tsv` in an output directory and adds a
taxonomy table plus a family-composition chart. The `roll` command runs taxonomy
by default; use `--skip-steps taxonomy` to omit it.

`rvmt` and `nvpc` are reserved database names. They will remain disabled until
their profiles have been enriched with compatible ICTV taxids.
