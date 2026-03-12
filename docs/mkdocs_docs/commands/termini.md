# Termini

<!-- Auto-generated draft from CLI metadata for `rolypoly termini`. -->
<!-- Expand this page with command-specific context, examples, and citations. -->

!!! warning "🚧 Under Construction"
   This command is under active development and behavior or outputs may change.

## Summary

Group contigs that share termini of length *n* (or a range).

## Description

Workflow:
    1) Optional ANI pre-pass (on by default) clusters highly similar contigs and keeps
       the longest representative per cluster. For overlap pileup extension use the
       extend command (under the assembly group).
    2) First pass groups contigs by the minimum-window motif seed (exact when --distance 0,
       Hamming-tolerant otherwise).
    3) Motifs are then extended up to --length max while all members share the same added bases.
    4) Optional second pass (on by default) collapses groups when one motif is contained in another
       by clipped containment (--clip-mode), up to --max-clipped bases.

    Outputs:
    - Assignments table at --output.
    - Group summary table at --groups-output (or <output>.groups.<ext> by default).
    - Group motifs FASTA at --motifs-fasta (or <output>.motifs.fasta by default).

    Group-output columns:
    - found_in: orientation-aware labels for member placement (e.g., fwd_on_5_end, rev_on_3_end).
    - source_group_ids: first-pass group IDs represented in the final row.
    - clip_contains_source_ids: source groups whose motifs are clipped-contained by this group.
    - clip_contained_by_source_ids: source groups that clipped-contain this group.

## Usage

```bash
rolypoly termini [OPTIONS]
```

## Options

- `-i`, `--input`: Input contig FASTA/FASTQ file (type: `FILE`; required; default: `Sentinel.UNSET`)
- `-n`, `--length`: Terminus length or range (e.g., 30 or 25-40) (type: `TEXT`; default: `40`)
- `-d`, `--distance`: Maximum Hamming mismatches allowed in the first-pass grouping seed (type: `INTEGER RANGE`; default: `0`)
- `--max-clipped`: Second-pass collapse: maximum total clipped bases allowed when one motif is contained in another (type: `INTEGER RANGE`; default: `4`)
- `--max-clipped-collapse`, `--no-max-clipped-collapse`: Enable/disable second-pass collapse of groups linked by clipped motif containment (type: `BOOLEAN`; default: `True`)
- `--clip-mode`: Containment mode for second pass: one-edge clipping only, or clipping distributed across both edges (type: `CHOICE`; default: `both`)
- `--strand`: Strand orientation(s) used when building termini signatures (type: `CHOICE`; default: `both`)
- `--ani-prefilter`, `--no-ani-prefilter`: Before termini grouping, collapse highly similar contigs so each ANI cluster contributes one representative (longest). For overlap-based extension, use the extend command. (type: `BOOLEAN`; default: `True`)
- `--ani-min-identity`: Minimum ANI identity (0-1) for contigs to be considered in the same prefilter cluster (type: `FLOAT RANGE`; default: `0.95`)
- `--ani-min-af`: Minimum aligned fraction (min(query_fraction, reference_fraction), 0-1) for ANI prefilter clustering (type: `FLOAT RANGE`; default: `0.8`)
- `-o`, `--output`: Output path for per-contig termini assignments (type: `FILE`; default: `termini_assignments.tsv`)
- `--groups-output`: Optional output path for grouped motif summary (default: <output>.groups.<ext>) (type: `FILE`)
- `--motifs-fasta`: Output path for group motif FASTA entries (defaults to <output>.motifs.fasta) (type: `FILE`)
- `--output-format`: Tabular output format for assignments and groups tables (type: `CHOICE`; default: `tsv`)
- `--log-file`: Optional log file path (type: `FILE`)
- `-t`, `--threads`: Number of threads to use for parallel processing IF applicable (type: `INTEGER RANGE`; default: `4`)




