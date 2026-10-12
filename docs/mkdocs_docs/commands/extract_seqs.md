# Extract sequences

A CLI wrapper around the existing streaming `filter_fasta_by_headers` utility.
The shared helper's matching and output behavior are unchanged.

## Usage

```bash
# Select exact complete headers and print FASTA to stdout.
rolypoly extract-seqs -i sequences.fasta -H headers.txt

# Repeat inline selections; gzip the file output automatically.
rolypoly extract-seqs -i first.fa,second.fa \
  -s 'contig_1 original description' -s contig_2 -o selected.fasta.gz

# Exclude every complete header containing the literal substring 'host'.
rolypoly extract-seqs -i sequences/ --match substring -s host --invert \
  -o retained.fasta
```

## Selection and streaming

`-i/--input` accepts FASTA/FASTQ files, comma-separated inputs, or non-recursive
directories. Supported suffixes: `.fa`, `.fasta`, `.fna`, `.faa`, `.fas`, `.fq`,
`.fastq`, and their `.gz` forms. Directory files are sorted; explicit inputs
retain their order. Duplicate paths are removed.

Supply a plain UTF-8 `-H/--headers` file, repeat `-s/--select`, or combine both.
Each line/value supplies one selection; surrounding whitespace and blank lines
are removed and duplicate selections ignored. Empty selections are errors.
Do not include the FASTA/FASTQ `>`/`@` marker.

- `--match header` (default): exact, case-sensitive **complete-header** matching.
  Descriptions must be included if present. At most one record per requested
  header is emitted across all inputs.
- `--match substring`: literal, case-sensitive substring matching anywhere in
  the complete header; every matching record is emitted.
- `-v/--invert`: emit every record that does not match.

Records retain their complete headers and sequence content, in input order.
FASTA and FASTQ inputs both produce **FASTA**; qualities are discarded.
Omit `-o/--output` or use `-o -` for stdout. A filename ending in `.gz`
automatically produces gzip output. File destinations must be new; parent
directories are created as needed. Input, selection-file, and log-file
collisions are rejected.

Logs, counts, and progress messages go to stderr and the standard log file,
not into the sequence stream. No matches is a successful empty output with a
warning. Failed runs remove newly created files; already streamed stdout
cannot be rolled back.

## Caveats

### Selection is not full-file validation

Exact positive matching stops once all requested complete headers have been
found, so later records are not parsed. Missing selections do not fail the run;
a completely empty result generates a warning.

Matching a first-token ID is not a separate backend mode: use the full header
for exact matches, or choose substring matching explicitly. Selecting one mate
does not automatically select its partner. Memory scales with selections rather
than the input sequence population. Tests use synthetic fixtures, not
production-scale benchmarks.

<!-- BEGIN GENERATED CLI OPTIONS -->
## Options

- `-i`, `--input`: Input file(s) or directory; required.
- `-H`, `--headers`: Plain text file containing complete headers or substrings.
- `-s`, `--select`: Inline selection; repeat as needed.
- `-m`, `--match`: `header` (default) or `substring`.
- `-v`, `--invert`: Write nonmatching records.
- `-o`, `--output`: New FASTA file, or `-`/omitted for stdout.
- `-g`, `--log-file`: Log path (default: `rolypoly.log`).
<!-- END GENERATED CLI OPTIONS -->
