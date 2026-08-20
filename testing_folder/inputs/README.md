# Testing inputs

Small fixtures for CLI tests are grouped by type:

- `contigs/` : nucleotide FASTA fixtures for annotation and sequence commands
- `reads/` : FASTQ fixtures for read-processing command tests
- `proteins/` : amino-acid FASTA fixtures for `annotate-prot` amino-input tests

Prefer adding new tiny deterministic fixtures here and referencing them from
`src/tests/cli_scenarios.json`. Do not commit generated indexes or command
outputs alongside their source fixtures.

`reads/some_soil_shrinked.fq` is a bounded 100-read subset of the historical
interleaved soil-read fixture. It is retained only for command smoke tests and
is not a representative dataset or biological validation.

The named segmented-virus FASTA files under `contigs/` are small reference
examples for manual termini/extension experiments; the automated scenario uses
the synthetic `segmented_shared_termini.fasta` fixture.

Files under `partiti_usecase/` support the documented exploratory notebook and
are research-example inputs rather than general test fixtures.
