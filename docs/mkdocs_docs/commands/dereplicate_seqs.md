# Dereplicate sequences

A CLI wrapper around the existing streaming `remove_duplicates` utility.
Shared utilities and assembly's `polars_fastx.dereplicate_fasta` are unchanged.

## Usage

```bash
# Print unique FASTA records to stdout.
rolypoly dereplicate-seqs -i sequences.fasta > unique.fasta

# Group reverse complements and gzip file outputs automatically.
rolypoly dereplicate-seqs -i contigs/ --revcomp-as-same \
  -o unique.fasta.gz -d discarded.fasta.gz -m duplicate_groups.tsv.gz

# Keep reverse complements and letter case distinct.
rolypoly dereplicate-seqs -i first.fa,second.fa \
  --revcomp-as-different --case-sensitive -o unique.fa

# Deduplicate by first-token sequence IDs.
rolypoly dereplicate-seqs -i sequences.fasta --by id -o unique_ids.fasta
```

## Inputs and policies

`-i/--input` accepts a FASTA/FASTQ file, comma-separated inputs, or non-recursive
directories. Supported suffixes: `.fa`, `.fasta`, `.fna`, `.faa`, `.fas`, `.fq`,
`.fastq`, and their `.gz` forms. Directory files are sorted; explicit inputs
retain their order. Duplicate paths are removed. Deduplication applies across
all resolved inputs.

`--by seq` (default) compares sequence content, `--by id` compares the first
whitespace-delimited header token, and `--by name` compares complete headers.
`--ignore-case` is the default; `--case-sensitive` preserves comparison case.
The first representative's complete header and sequence are emitted unchanged.

`--revcomp-as-different` (default) retains distinct reverse complements.
`--revcomp-as-same` groups nucleotide reverse complements, including IUPAC
ambiguity codes. It requires `--by seq` and `--ignore-case`: the existing helper
does not correctly complement lowercase bases in case-sensitive mode, so the
wrapper rejects that combination rather than changing the backend.

## Outputs

Both FASTA and FASTQ inputs produce **FASTA**; qualities are not retained.
Omit `-o/--output` or use `-o -` for streaming stdout. Output filenames ending
in `.gz` are gzip compressed. File destinations must be new; parents are
created as needed. Input/output/log collisions and duplicate output paths are
rejected. Counts and diagnostics go to stderr and the standard log file.

- `-d/--save-duplicates`: optional FASTA of discarded records.
- `-m/--save-dup-list`: the utility's existing **headerless** tab-separated
  duplicate list. Each row contains a group count and comma-and-space-separated
  full headers, representative first. Only groups with duplicates appear,
  ordered by decreasing size. With no duplicates, this output is empty.

Failed runs remove newly created files. A stdout prefix already emitted cannot
be retracted.

## Caveats

### Existing backend semantics

This exposes exact hashed deduplication, not approximate clustering, circular
rotation, or quality-based representative selection. Hash collisions are not
verified against full sequences. Memory scales with unique comparison hashes;
the optional duplicate list also retains member headers.

Reverse-complement grouping is intended for nucleotide input, not proteins.
It uses the existing DNA-style complement rules (U complements to A; A to T),
without normalizing RNA U/T spellings. FASTQ mates are independent records,
not synchronized pairs. The duplicate-list member encoding is not lossless
when headers contain commas. Tests use synthetic inputs, not production
benchmarks.

<!-- BEGIN GENERATED CLI OPTIONS -->
## Options

- `-i`, `--input`: Input file(s) or directory; required.
- `-o`, `--output`: New FASTA file, or `-`/omitted for stdout.
- `-b`, `--by`: `seq` (default), `id`, or `name`.
- `--revcomp-as-different`, `--revcomp-as-same`: Keep reverse complements distinct (default) or group them.
- `--ignore-case`, `--case-sensitive`: Ignore case (default) or preserve case distinctions.
- `-d`, `--save-duplicates`: Optional new discarded-record FASTA.
- `-m`, `--save-dup-list`: Optional new legacy duplicate list.
- `-g`, `--log-file`: Log path (default: `rolypoly.log`).
<!-- END GENERATED CLI OPTIONS -->
