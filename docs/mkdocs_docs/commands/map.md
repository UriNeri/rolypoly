# Map

<!-- Auto-generated draft from CLI metadata for `rolypoly map`. -->
<!-- Expand this page with command-specific context, examples, and citations. -->

## Summary

Map reads to a reference FASTA with one or more aligners.

## Description

Input libraries are auto-detected from a file, comma-separated file list, or
directory supplied with --input / --input-dir, or declared explicitly via
--paired-end / --single-end / --merged / --long-read. Each input file is
mapped independently; outputs land in per-mapper sub-directories under
--output.

Pair filtering is deliberately optional. In fragmented assemblies, a real
biological fragment may span an unassembled gap and place its mates on two
different contigs; --concordant and --proper will exclude such evidence.

By default, **both mates are retained if either maps**. Unmapped single reads
and fully unmapped pairs are omitted; half-mapped pairs and
secondary/supplementary alignments remain. BBMap's `outm` already retains
unmapped mates of mapped reads. BWA-MEM2 SAM filtering uses the paired (`0x1`),
read-unmapped (`0x4`), and mate-unmapped (`0x8`) flags. The minimap2/mappy adapter
writes an unmapped mate record alongside its mapped partner.
Headers and retained records are preserved by the line-based filters.
This is intentional record removal, not compression.

Use `--include-unmapped` to also retain unmapped single reads and fully
unmapped pairs with BBMap, BWA-MEM2, and minimap2/mappy.
MMseqs searches reads independently and emits only hits; it cannot retain
unmapped mates or fully unmapped reads, and logs this limitation for paired
input or `--include-unmapped`.
Explicit `--concordant` / `--proper` filtering still takes precedence, excluding
half-mapped pairs.

### Lossless SAM compression

**SAM output is gzipped by default** (`--compression-policy sam_only`).
Use `-cp` / `--compression-policy none` for plain SAM. The old `over_10gb`
policy has been removed. `--compressed` remains an alias for `sam_only`.
Gzip uses the existing `pigz` executable and the requested thread count.

| Backend | Output path |
| --- | --- |
| BWA-MEM2 | SAM stdout is filtered line-by-line and written through `pigz`, without a plain SAM intermediate. Mapper/compressor failures prevent publication of the final file. |
| Minimap2/mappy | Records are generated read-by-read and written through `pigz`, without a plain SAM intermediate. |
| BBMap | Writes gzip directly with native `pigz` support, including appended libraries, without a plain SAM intermediate. |
| MMseqs | Writes its SAM result file, then gzips it; the plain SAM is removed only after successful compression. |

With BBMap and mappy, explicit `--concordant` / `--proper` filters still use
the existing post-hoc filtering path where supported. This may rewrite gzip
output but does not require a plain SAM intermediate. BWA applies these
filters directly to its stdout stream.

MMseqs `easy-search` takes **query, target, output, temp directory** in that
order; the output path is not the last positional argument. `/dev/stdout`
does not work with the installed MMseqs2: its writer tries to open
`/dev/stdout.0` and fails. Its stdout can also contain workflow messages.
The command therefore uses a real SAM file rather than a shell pipeline.
MMseqs's prefilter receives the requested memory limit through
`--split-memory-limit`.

Compression is lossless for the retained SAM content, including headers
and optional tags; it does not parse/reserialize records with Polars-Bio.
Existing destinations are not overwritten. MMseqs and compression of
resumed plain SAMs temporarily require space for both plain and compressed
representations. Other output types are unchanged. See
[Roll](roll.md#reducing-mapping-storage) for resume behavior.

### Mapper specifications and extension

`MAPPER_SPECS` in `src/rolypoly/commands/reads/map.py` is the shared, typed
capability table for `map` and `roll`. Every field is commented, and each
mapper entry explains the assumptions of **our adapter**, which may differ
from the native tool's full capabilities.

The table controls CLI mapper choices, invocation, native/stream/file
SAM output, multi-library append behavior, support for unmapped mates and
fully unmapped templates, strict pair-filter validation/routing, and temporary
directory cleanup. There is no central mapper-name or adapter-name dispatcher:
the command calls the spec's `run` callable and composes the processing path
from the remaining capabilities.
Specifications are logged at debug level. Defaults remain gzip output,
both mates retained when either maps, and strict pair filters disabled.

To add a mapper:

1. Add a fully populated `MapperSpec` entry, with comments explaining its
   pairing and output assumptions. Do not claim native filtering is equivalent
   to our record predicates without testing that equivalence.
2. Set `run` to an existing compatible invocation callable, or implement one in
   the same module. It accepts a typed `MapperRun` and its `MapperSpec`, and
   returns the output paths. `MapperRun` supplies normalized inputs, the
   spec-derived output suffix, resources, and stream-filter settings. Invocation
   callables must honor those settings and propagate tool failures.
   Tool-specific calls remain necessary (Python bindings, indexing, native
   arguments), but registering a callable requires no new dispatcher branch.
3. Keep shared capability and processing policy in the table rather than
   adding mapper-name checks to validation or post-processing. File-producing
   specs receive a plain `.sam` suffix and use shared fallback compression;
   other specs receive the requested plain/gzip suffix. The strict filter stage
   selects stream predicates or shared post-hoc filtering. List temporary
   directories as mapper-relative glob patterns in `temporary_patterns`.
4. Add tiny CLI contracts covering outputs, gzip/plain behavior, mate
   retention, supported/unsupported strict filters, multiple libraries, and
   tool failures. New mapper names are automatically offered by both commands
   when their modules load; `none` remains a roll-only skip choice.

## Usage

```bash
rolypoly map [OPTIONS]
```

<!-- BEGIN GENERATED CLI OPTIONS -->
## Options

- `-i`, `--input`: Input FASTQ file, comma-separated FASTQ files, or directory to auto-detect libraries from (type: `TEXT`)
- `-id`, `--input-dir`: Input directory to scan for FASTQ files (alias for --input) (type: `DIRECTORY`)
- `-r`, `--reference`: Reference FASTA to map against (type: `TEXT`; required; default: `Sentinel.UNSET`)
- `-o`, `--output`: Output directory (type: `DIRECTORY`; default: `RP_mapping_output`)
- `--paired-end`: Explicit paired-end library: <lib_num> <R1> <R2> (type: `TEXT`; default: ``)
- `--single-end`: Explicit single-end library: <lib_num> <fastq> (type: `TEXT`; default: ``)
- `--merged`: Explicit merged-read library: <lib_num> <fastq> (type: `TEXT`; default: ``)
- `--long-read`: Long-read FASTQ file(s) (type: `TEXT`; default: ``)
- `-m`, `--mapper`: Mapper (read aligner) choice. Use multiple -m flags to run multiple mappers. (type: `CHOICE`; default: `bbmap`)
- `-ow`, `--overwrite`: Overwrite existing output directory (type: `BOOLEAN`; default: `False`)
- `--concordant`, `--concord`: Keep only paired alignments whose mates map to the same reference in inward-facing FR orientation. Does not enforce insert size. (type: `BOOLEAN`; default: `False`)
- `--proper`: Keep only alignments marked proper-pair (SAM flag 0x2) by the mapper. Supported by bbmap and bwa-mem2. (type: `BOOLEAN`; default: `False`)
- `--include-unmapped`: Also retain fully unmapped reads/pairs where supported. By default, retain both mates if either maps. (type: `BOOLEAN`; default: `False`)
- `-z`, `--compressed`: Alias for --compression-policy sam_only (the default). (type: `BOOLEAN`; default: `False`)
- `-cp`, `--compression-policy`: SAM output compression: sam_only (gzip, default) or none (plain SAM). (type: `CHOICE`; default: `sam_only`)
- `--bwa-mem2-all`, `--no-bwa-mem2-all`: bwa-mem2: pass -a, which reports all alignments for single-end or unpaired paired-end reads. (type: `BOOLEAN`; default: `False`)
- `--bwa-mem2-extra-flags`: Additional bwa-mem2 mem flags (e.g. '-Y -q'); -o is not allowed with streamed output. (type: `TEXT`; default: ``)
- `-t`, `--threads`: Number of worker threads. (type: `INTEGER RANGE`; default: `1`)
- `-M`, `--memory`: Memory limit, for example 8g. (type: `MEMORY`; default: `8g`)
- `-k`, `--keep-tmp`: Keep temporary files. (type: `BOOLEAN`; default: `False`)
- `-tmp`, `--temp-dir`: Temporary working directory. (type: `DIRECTORY`)
- `-g`, `--log-file`: Path to the log file. (type: `FILE`; default: `rolypoly.log`)
<!-- END GENERATED CLI OPTIONS -->
