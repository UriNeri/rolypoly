# Testing inputs

Small fixtures for CLI tests are grouped by type:

- `contigs/` : nucleotide FASTA fixtures for annotation and sequence commands
- `reads/` : FASTQ fixtures for read-processing command tests
- `proteins/` : amino-acid FASTA fixtures for `annotate-prot` amino-input tests

Prefer adding new tiny deterministic fixtures here and referencing them from `src/tests/cli_scenarios.json`.
Legacy files in `testing_folder/` are kept for backward compatibility with older scripts.
