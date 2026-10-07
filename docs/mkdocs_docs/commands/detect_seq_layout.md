# Detect sequence layout

Discover FASTA inputs and inspect FASTQ libraries without running filtering,
assembly, or the end-to-end pipeline. `detect-layout` is a shorter alias.
No reference databases or external executables are required.

## Usage

```bash
# Print JSON; diagnostics go to stderr, not into the structured result.
rolypoly detect-seq-layout -i reads/ > layout.json

# Explicit ordered R1/R2 pair.
rolypoly detect-layout -i reads/sample_R1.fq.gz,reads/sample_R2.fq.gz

# Save TOML, creating parent directories but refusing to overwrite an existing file.
rolypoly detect-seq-layout -i sequences/ --fmt toml -o results/layout.toml
```

## Inputs and detection

`-i/--input` accepts one file, comma-separated files, or one directory.
Directory discovery is non-recursive and ignores unsupported extensions.
Supported suffixes are `.fastq`, `.fq`, `.fasta`, `.fa`, `.fna`, `.faa`, and
`.fas`, plus their `.gz` forms. A directory cannot be combined with other inputs.
Paths in results are absolute; duplicate input paths are removed.

FASTA files are discovered by suffix and listed separately; their records and
nucleotide/protein alphabet are not inspected. FASTQ detection uses the shared
helpers used by `filter-reads`, `assemble`, and `roll`, reading up to the first
1,000 records for layout/statistics and 100 records for header metadata.
Parsing failures in the sampled prefix, empty files, and non-FASTQ files
presented as FASTQ fail with a nonzero exit status rather than producing
fallback results.

In directories, R1/R2 associations come from matching filename patterns.
Assembly-style filtered libraries (`dedupe_final_interleaved_SAMPLE.fq[.gz]`
and `dedupe_final_merged_SAMPLE.fq[.gz]`, including historical
`SAMPLE_final_interleaved` / `SAMPLE_final_merged` names) are grouped separately.
Exactly two explicitly listed FASTQ files are treated as an ordered R1/R2 pair,
as in `filter-reads`; FASTA files are excluded before applying this rule.
Other explicit lists follow the existing helper's single-file grouping policy,
while retaining each file's independently sampled layout in `file_details`.

## Output schema

JSON is the default. `-f/--fmt/--format json|toml` selects the format independently
of the output filename. Without `-o/--output`, or with `-o -`, the document goes
to stdout. Logs go to stderr and the standard `-g/--log-file` destination.
Both formats represent the same schema; optional absent fields are omitted
rather than represented as null (TOML has no null value).

| Field | Meaning |
| --- | --- |
| `schema_version` | Output schema version, currently `1`. |
| `sample_size`, `header_sample_size` | Maximum prefix sample sizes, currently `1000` and `100`. |
| `fasta_files` | Discovered FASTA paths, including protein FASTA. |
| `fastq.R1_R2_pairs` | Arrays of ordered R1/R2 paths. |
| `fastq.interleaved_files` | Files classified as interleaved. |
| `fastq.single_end_files` | Single-end files and files the existing grouping policy treats as single-end. |
| `fastq.rolypoly_data` | Filtered libraries keyed by sample, with available `interleaved` and/or `merged` paths. |
| `fastq.file_details` | Per-path detection results described below. |

Each `file_details` entry includes `file_type` (`single`, `interleaved`,
`paired_R1`, `paired_R2`, or `unknown`), `is_gzipped`, `pair_1_count`,
`pair_2_count`, `average_read_length`, `average_read_quality`, and
`header_analysis`. Mean quality is the existing reader's sampled Phred quality
metric. Header metadata includes sample size, header format, sequencer/tile/XY
and barcode indicators and examples, mate counts, format counts, and up to
three example headers. These are header-derived observations, not inferred
biological library properties.

## Caveats

### Sampled inference, not full validation

Interleaving requires adjacent matching mate-labelled template IDs in the sampled
prefix. Equal mate counts alone are insufficient. Unlabelled headers cannot
prove pairing and are guessed single-end. Filename associations and explicit
R1/R2 pairs are not checked for matching templates across files. Unknown layouts
remain visible in `file_details`, even when the shared grouping helper places
them in `single_end_files`. This command does not establish strandedness,
sequencing technology, or whole-file pairing/quality correctness.

FASTA discovery is not FASTA validation. Runtime contract tests use tiny
synthetic fixtures, not production-scale data.

<!-- BEGIN GENERATED CLI OPTIONS -->
## Options

- `-i`, `--input`: FASTA/FASTQ file, comma-separated files, or one directory; required.
- `-o`, `--output`: New output file; omit or use `-` for stdout.
- `-f`, `--fmt`, `--format`: Output format, `json` (default) or `toml`.
- `-g`, `--log-file`: Log file (default: `rolypoly.log`).
<!-- END GENERATED CLI OPTIONS -->
