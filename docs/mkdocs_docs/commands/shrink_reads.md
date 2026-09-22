# Shrink Reads

<!-- Auto-generated draft from CLI metadata for `rolypoly shrink-reads`. -->
<!-- Expand this page with command-specific context, examples, and citations. -->

## Summary

Subset FASTQ reads by count or fraction for lightweight test datasets.

## Description

Supports deterministic head-style subsampling (`first_n`) and random
sampling (`random`) for single-end, interleaved, and paired-end layouts.

This command is intended for quick dry runs and resource-reduced tests,
not as a full read-normalization strategy.

### BBNorm and paired inputs

With `--subset-type bbnorm`, `--sample-size` specifies the target k-mer
depth and `--bbnorm-min-depth` controls the minimum depth. BBNorm's default
two-pass normalization is preserved.

For separate R1/R2 inputs, rolypoly temporarily interleaves the reads before
normalization, keeps BBNorm's intermediate and final reads interleaved, then
splits the result into paired output files. This avoids a BBTools 39.91 reader
failure on valid mate files whose buffered record counts differ. BBNorm rereads
its inputs and does not accept standard input, so a one-pass pipe cannot replace
this temporary file. Temporary files are created beside the output and removed
after success or a Python exception; allow disk space for both the interleaved
input and normalized reads. An abrupt process termination may leave temporary
files behind.

Single-file inputs and `first_n`/`random` sampling do not use this workaround.
It also applies to paired inputs passed through `roll --mini
--mini-subset-type bbnorm`.

## Usage

```bash
rolypoly shrink-reads [OPTIONS]
```

<!-- BEGIN GENERATED CLI OPTIONS -->
## Options

- `-i`, `-in`, `--input`: Input raw reads file(s) or directory containing them. For paired-end reads, you can provide an interleaved file or the R1 and R2 files separated by comma. If a directory is provided, one output per input identified file/pair will be created. (type: `TEXT`; default: `Sentinel.UNSET`)
- `-st`, `--subset-type`: how to sample reads from input. (type: `CHOICE`; default: `first_n`)
- `-sz`, `--sample-size`: For first_n/random, at most this many reads (or proportion if <1). For bbnorm, this is the target k-mer depth. (type: `FLOAT`; default: `1000`)
- `--bbnorm-min-depth`: Minimum depth threshold for bbnorm normalization (min in bbnorm.sh). (type: `INTEGER`; default: `2`)
- `-t`, `--threads`: Number of worker threads. (type: `INTEGER RANGE`; default: `1`)
- `-g`, `--log-file`: Path to the log file. (type: `FILE`; default: `rolypoly.log`)
<!-- END GENERATED CLI OPTIONS -->



