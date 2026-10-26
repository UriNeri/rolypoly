# MMTAX

## Summary

Assign ICTV taxonomy to virus contigs from protein matches made by MMseqs2 or
DIAMOND. DIAMOND is the default for standalone `mmtax` and every `roll` preset,
including fast/mini. Select `mmtax --backend mmseqs` or
`roll --taxonomy-backend mmseqs` to retain the previous backend. There is no
automatic fallback: the selected backend needs its own taxonomy-enabled database.

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
  --backend diamond \
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

## Memory behavior

`--memory` is **not a hard process RAM cap** for either taxonomy backend.
The default is `8g`; RolyPoly interprets `g`/`GB` as binary GiB.

- **MMseqs2:** RolyPoly reserves the larger of 5% of the requested budget or
  1 GiB, then passes the remainder as `--split-memory-limit`, in bytes rounded
  down. For example, `30g` passes 28.5 GiB and `8g` passes 7 GiB. Budgets of
  1 GiB or less are rejected. This leaves headroom but cannot guarantee that
  total process memory stays within the original budget.
- **DIAMOND:** `--memory` is logged but is not passed as a memory limit.
  RolyPoly retains DIAMOND's default block and index settings rather than
  guessing a hard limit from them. For normal sensitivity these are
  `--block-size 2.0` and `--index-chunks 4`.

DIAMOND's [official memory and performance documentation](https://github.com/bbuchfink/diamond/wiki/3.-Command-line-options#memory--performance-options)
describes `--block-size` in billions of sequence letters, with approximate
memory usage of six times that value in GB. Smaller blocks reduce memory and
temporary storage at a possible runtime cost, but can also change search
results. `--index-chunks` controls partitioning of the seed index; using one
chunk instead of the default four increases memory use. Very/ultra-sensitive
presets have different defaults (block size 0.4, one index chunk). These are
tuning controls, not enforceable RAM limits.

### Local baseline (2026-10-14)

Two bounded `diamond blastp` searches used DIAMOND 2.2.8, normal sensitivity,
default block/index/algorithm settings, and the local full `ncbi_virus.dmnd`:
3,589,746 proteins, 2,684,726,080 residues, and 3,011,057,792 bytes on disk.
Queries were the first 100 or 1,000 proteins from the local
`refseq_ribovirus_genomes_orfs_cleaned.faa.gz`. Both selected the query-indexed
algorithm automatically. `/usr/bin/time -v` measured each search:

| Query proteins | Query residues | Threads | Peak RSS | Wall time |
| --- | --- | --- | --- | --- |
| 100 | 58,236 | 1 | 3.86 GiB | 32.48 s |
| 1,000 | 437,737 | 10 | 4.50 GiB | 7.93 s |

Search options matched the current adapter: `--evalue 1e-5 --id 10
--min-orf 20 --min-query-len 20 --header simple`, with output columns
`qseqid sseqid pident length bitscore evalue staxids qstart qend qlen`.
The second run benefited from a warm filesystem cache; the wall times are
not a scaling comparison. Measurements cover the external protein search,
not protein prediction, Polars/taxopy classification, or the complete pipeline.

These small-query baselines fit below the default 8 GiB budget, so no automatic
block-size reduction has been introduced. They do **not** establish a memory
guarantee for larger inputs, larger/custom references, other sensitivity
presets, or concurrent jobs, nor an assignment-quality comparison with MMseqs2.
When a hard cap is required, use an OS/container/job-scheduler memory limit
and allow for failure if the search cannot fit. On a smaller-memory machine,
benchmark separately before running the full pipeline.

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

## Caveats

### Defaults can produce overly specific assignments

Current defaults in the command implementation are:

| Control | Default | Meaning |
| --- | --- | --- |
| `--backend` | `diamond` | Protein similarity search backend |
| `--sensitivity` | `normal` | MMseqs2 sensitivity 4; DIAMOND default preset |
| `--evalue` | `1e-5` | Maximum hit E-value |
| `--identity` | `0.1` | Minimum aligned amino-acid identity: 10% |
| `--min-aln-len` | `30` | Minimum alignment length in amino acids |
| `--top` | `10` | Retain hits within 10% of the best protein bitscore |
| `--weight` | `bitscore` | Weight for taxonomic voting |
| `--majority` | `0.5` | Minimum weighted support used in rank-aware voting |

These permissive search thresholds and voting rules can assign a divergent or
fragmentary contig too deeply. Support measures agreement among retained,
rank-informative reference matches; it is not a calibrated probability of
membership. A high support value, especially with a low `informative_fraction`,
does not establish that an assignment satisfies the biology of that taxon.

### Similarity assignment is not taxonomic demarcation

Conceptually, mmtax belongs to the similarity-search-to-taxonomy family of
methods often described as BLAST-to-LCA. Its current implementation is more
specifically a lineage-consistent, rank-aware weighted vote at protein and
contig levels, rather than a strict LCA of every retained hit.

It does **not** apply actual rank- and taxon-specific demarcation criteria, nor
calibrated approximations to them. There is no universal RdRp amino-acid identity
cutoff for family, genus or species membership. An aligned-region identity is
also not interchangeable with identity across a complete RdRp, polyprotein or
genome. The relevant comparison and criteria vary across virus groups and
ranks; see the [ICTV explanation of taxonomic classification](https://ictv.global/about/taxonomy).

For example, a contig may receive a family label because its best represented
matches all belong to that family, even when its RdRp similarity falls below
the range or criterion appropriate for membership. Conserved polymerase
homology can support a broader evolutionary relationship without establishing
membership of that family. Genome organization and accessory-protein content
may provide additional conflicting evidence. Their absence from a partial
contig, however, is not proof that the complete virus lacks those genes.

Treat assignments as hypotheses to review using the relevant ICTV report,
alignment extent, phylogenetic placement, genome completeness and gene content.
Raising generic identity or voting thresholds alone does not implement the
appropriate demarcation criteria. See also the [scientific background](../background.md#similarity-based-taxonomy-and-demarcation)
and [report-chart caveats](report.md#caveats).

## Known bugs

The limitations above concern the classifier's interpretation and calibration;
they are not a claim that it implements formal taxon demarcation incorrectly.
Visual/search integration issues are tracked under [report known bugs](report.md#known-bugs).
