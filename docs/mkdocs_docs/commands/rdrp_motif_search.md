# Rdrp Motif Search

<!-- Auto-generated draft from CLI metadata for `rolypoly rdrp-motif-search`. -->
<!-- Expand this page with command-specific context, examples, and citations. -->

## Summary

Search for RdRp motifs (A, B, C, D) in nucleotide or amino acid sequences.

## Description

This command searches input sequences for RNA-dependent RNA polymerase (RdRp)
    motif patterns using pre-built profile databases from the RVMT project.

    The output is a table with one row per input sequence, showing motif locations,
    scores, and conformations found.

## Usage

```bash
rolypoly rdrp-motif-search [OPTIONS]
```

## Options

- `-i`, `--input`: Input FASTA file with nucleotide or amino acid sequences (type: `TEXT`; required; default: `Sentinel.UNSET`)
- `-o`, `--output`: Output directory path (type: `TEXT`; default: `./rdrp_motif_search_output`)
- `-t`, `--threads`: Number of threads for processing (type: `INTEGER`; default: `1`)
- `-g`, `--log-file`: Path to log file (type: `TEXT`; default: `./rdrp_motif_search_logfile.txt`)
- `-M`, `--memory`: Memory limit in GB. Example: -M 8gb (type: `TEXT`; default: `4gb`)
- `-e`, `--evalue`: E-value threshold for motif searches (type: `FLOAT`; default: `0.01`)
- `--min-score`: Minimum score threshold for motif matches (type: `FLOAT`)
- `--max-distance`: Maximum distance between motifs in amino acids (type: `INTEGER`; default: `200`)
- `--search-tool`: Search tool to use (currently only hmmsearch supported) (type: `CHOICE`; default: `hmmsearch`)
- `--aa-method`: Method for amino acid translation from nucleotides (type: `CHOICE`; default: `six_frame`)
- `--min-orf-length`: Minimum ORF length for gene prediction (type: `INTEGER`; default: `30`)
- `--motif-filter`: Filter results by specific motif type (type: `CHOICE`)
- `--no-include-alignment`: Disable including aligned region sequences in output (alignment included by default) (type: `BOOLEAN`; default: `False`)
- `--data-dir`: Path to rolypoly data directory (if not in default location) (type: `TEXT`)
- `--output-format`: Output file format (tsv or parquet) (type: `CHOICE`; default: `tsv`)
- `--output-structure`: Output structure: nested (motif_details as JSON) or flat (separate columns per motif) (type: `CHOICE`; default: `nested`)
- `-k`, `--keep-tmp`: Keep temporary files (type: `BOOLEAN`; default: `False`)
- `-ow`, `--overwrite`: Overwrite output directory if it exists (type: `BOOLEAN`; default: `False`)
- `-ll`, `--log-level`: Logging level (type: `CHOICE`; default: `INFO`)
- `-cf`, `--config-file`: JSON config file with parameters (overrides command line) (type: `TEXT`)

## Additional Notes

```text
EXAMPLES:

  # Basic search with default flat TSV output and alignment
  rolypoly rdrp-motif-search -i sequences.fasta -o results_dir

  # Nested structure for programmatic analysis
  rolypoly rdrp-motif-search -i sequences.fasta -o results_dir --output-structure nested

  # Parquet output with structured data for analysis
  rolypoly rdrp-motif-search -i sequences.fasta -o results_dir --output-format parquet

  # Disable alignment to reduce output size
  rolypoly rdrp-motif-search -i sequences.fasta -o results_dir --no-include-alignment

  # High sensitivity search with custom parameters
  rolypoly rdrp-motif-search -i sequences.fasta -o results_dir -e 0.1 --max-distance 300

OUTPUT FORMATS:

  flat + tsv: separate columns (motif_a_start, motif_b_start, etc.) - DEFAULT
  nested + tsv: motif_details column as JSON string
  flat + parquet: separate columns with native data types
  nested + parquet: motif_details as structured data types
```


