# Termini Analysis

!!! warning "🚧 Experimental"
    This command is implemented and usable, but still under active development.

The `termini` command will analyze the edge regions of assembled contigs, possibly assisted by read mapping to potentially identify or determine if the contig appears truncated (and thus the genome is likely not complete). Subsequent commands will be able to use this information to improve genome binning (i.e. potentially by identifying other contigs with shared termini).

## Usage

```bash
rolypoly termini -i contigs.fasta -n 40 -o termini_assignments.tsv
```

## Common options

- `-i, --input`: Input contig FASTA/FASTQ
- `-n, --length`: Terminus length or range (example: `40` or `25-40`)
- `-d, --distance`: Allowed Hamming mismatches in first-pass seed grouping
- `--ani-prefilter/--no-ani-prefilter`: ANI-based representative prefilter toggle
- `--ani-min-identity`, `--ani-min-af`: ANI prefilter thresholds
- `-o, --output`: Per-contig assignments output
- `--groups-output`: Optional grouped motif summary table output
- `--motifs-fasta`: Optional output FASTA for group motifs
