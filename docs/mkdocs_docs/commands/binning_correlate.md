# Correlation Analysis

!!! warning "🚧 Experimental"
    This command is implemented and usable, but still under active development.

The `correlate` command will analyze the distribution and abundance patterns of viral sequences across multiple samples to identify potential relationships (i.e. potential genome fragments / segments) and improve/guide genome binning.

## Usage

```bash
rolypoly correlate -i abundance_table.tsv -o correlate
```

## Common options

- `-i, --input`: Contig x sample table (presence/absence or abundance)
- `-o, --output-prefix`: Output prefix
- `-m, --mode`: `correlation`, `cooccurrence`, or `both`
- `--method`: Correlation method (`spearman`, `pearson`)
- `--table-type`: `auto`, `presence-absence`, or `abundance`
- `--min-correlation`: Threshold for reporting pairs
