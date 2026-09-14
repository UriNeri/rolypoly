# Filter Contigs

<!-- Auto-generated draft from CLI metadata for `rolypoly filter-contigs`. -->
<!-- Expand this page with command-specific context, examples, and citations. -->

## Summary

Filter contigs against host/contamination references, with optional rRNA screening.

## Description

Depending on `--mode`, the command applies nucleotide filtering, protein
filtering, or both, using two-stage rule sets (`filter1_*` and
`filter2_*`) to retain likely non-host contigs.

Host references can be masked first (default) unless `--dont-mask` is set.

## Usage

```bash
rolypoly filter-contigs [OPTIONS]
```

<!-- BEGIN GENERATED CLI OPTIONS -->
## Options

- `-i`, `--input`: Input path to fasta file (type: `PATH`; required; default: `Sentinel.UNSET`)
- `-d`, `--known-dna`, `--host`: Path to the user-supplied host/contamination fasta (type: `PATH`; default: `Sentinel.UNSET`)
- `-o`, `--output`: Output file location. (type: `TEXT`; default: `/home/neri/Documents/Github/rolypoly/filtered_contigs.fasta`)
- `-m`, `--mode`: Filtering mode: nucleotide, amino acid, or both (nuc / aa / both) (type: `CHOICE`; default: `both`)
- `-Fm1`, `--filter1_nuc`: First set of rules for nucleic filtering by aligned stats (type: `TEXT`; default: `alnlen >= 120 & pident>=75`)
- `-Fm2`, `--filter2_nuc`: Second set of rules for nucleic match filtering (type: `TEXT`; default: `qcov >= 0.95 & pident>=95`)
- `-Fd1`, `--filter1_aa`: First set of rules for amino (protein) match filtering (type: `TEXT`; default: `length >= 80 & pident>=75`)
- `-Fd2`, `--filter2_aa`: Second set of rules for protein match filtering (type: `TEXT`; default: `qcovhsp >= 95 & pident>=80`)
- `--dont-mask`: If set, host fasta won't be masked for potential RNA virus-like seqs (type: `BOOLEAN`; default: `False`)
- `--mmseqs-args`: Additional arguments for MMseqs2 (type: `TEXT`; default: `--min-seq-id 0.5 --min-aln-len 80`)
- `--diamond-args`: Additional arguments for Diamond (type: `TEXT`; default: `--id 50 --min-orf 50`)
- `--flag-only`: Retain matching contigs and record warning intervals instead of discarding them. (type: `BOOLEAN`; default: `False`)
- `--rrna`: Opt in to an rRNA-only cmscan after host filtering (also works without --host). (type: `BOOLEAN`; default: `False`)
- `--rrna-db`: Override the bundled rrna/rrna.cm database; requires --rrna. (type: `FILE`)
- `--rrna-min-fraction`: Minimum contig fraction covered by accepted rRNA hits for removal; local hits are flagged. Ignored for removal with --flag-only. (type: `FLOAT RANGE`; default: `0.8`)
- `-ow`, `--overwrite`: Do not overwrite the output directory if it already exists (type: `BOOLEAN`; default: `False`)
- `-t`, `--threads`: Number of worker threads. (type: `INTEGER RANGE`; default: `1`)
- `-M`, `--memory`: Memory limit, for example 8g. (type: `MEMORY`; default: `8g`)
- `-k`, `--keep-tmp`: Keep temporary files. (type: `BOOLEAN`; default: `False`)
- `-tmp`, `--temp-dir`: Temporary working directory. (type: `DIRECTORY`)
- `-g`, `--log-file`: Path to the log file. (type: `FILE`; default: `rolypoly.log`)
<!-- END GENERATED CLI OPTIONS -->





## Retain matches as warnings

By default, contigs matching either configured host-rejection rule are removed.
Use `--flag-only` to retain them and record the accepted nucleotide/protein
match intervals for inspection instead:

```bash
rolypoly filter-contigs --input assembly.fasta --host host.fasta \
  --output screened.fasta --flag-only
```

Only hits passing `filter1_*` or `filter2_*` become warnings. Protein host
matches come from DIAMOND blastx, whose query coordinates are already nucleotide
coordinates; reverse orientation is retained. Original FASTA headers are
preserved. In `both` mode, flag-only allows both searches to examine every
contig; normal removal mode searches surviving nucleotide-filtered contigs in
the protein step.

## Optional early rRNA screening

`--rrna` enables an additional cmscan against `$ROLYPOLY_DATA/rrna/rrna.cm`.
It is **off by default** and runs after host filtering. A host reference is not
required for rRNA-only screening. The scan uses each model's Rfam gathering
threshold (`--cut_ga`), not a single generic score threshold.

```bash
# Remove predominantly rRNA contigs; retain localized rRNA hits as warnings.
rolypoly filter-contigs --input assembly.fasta --output screened.fasta --rrna

# Retain all matches from both host filtering and rRNA screening.
rolypoly filter-contigs --input assembly.fasta --host host.fasta \
  --output screened.fasta --rrna --flag-only
```

The default `--rrna-min-fraction 0.8` removes a contig only when the union of
accepted rRNA intervals covers at least 80% of its length. Overlapping models
are not double-counted. Smaller rRNA regions are flagged and retained; with
`--flag-only`, even predominantly rRNA contigs are retained. This configurable
coverage policy is a screening heuristic, not a biological chimera criterion.
`--rrna-db` selects an alternative CM file with gathering thresholds.

## Evidence and reports

Alongside `screened.fasta`, the command writes:

- `screened.fasta.filter_hits.tsv`: accepted match coordinates (1-based,
  inclusive), strand, source/model, score, E-value, rule and flagged/removed action.
- `screened.fasta.filter_run.json`: screening settings, including reference paths.
- `screened.fasta.rrna.tblout`: raw Infernal results when a nonempty rRNA scan runs.

Reports discover `*.filter_hits.tsv` beneath their input directory. Retained
matches appear as red RNA/QC features and RNA-feature table entries. Matching
rRNA features already covered by an annotate-rna interval are consolidated with
that feature, retaining filter provenance; intervals adding coverage remain
separate. Removed contigs are not reintroduced. Keep sidecars with the outputs
when generating a report.

Red indicates a region to review, not proof of contamination or chimerism.
Co-occurring viral-marker and rRNA evidence is described as overlapping or
occurring elsewhere on the contig; the latter is labelled a possible chimera
for review. A cmscan failure aborts screening rather than being treated as a
negative result.
