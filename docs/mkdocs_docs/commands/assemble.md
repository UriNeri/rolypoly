# Assemble

<!-- Auto-generated draft from CLI metadata for `rolypoly assemble`. -->
<!-- Expand this page with command-specific context, examples, and citations. -->

## Summary

Assemble reads/contigs with one or more backends and optional dereplication.

## Description

Inputs can be provided explicitly (`--paired-end`, `--single-end`,
`--merged`, `--long-read`, `--raw-fasta`) and/or discovered from
`--input-dir`.

Selected assembler outputs are normalized and optionally dereplicated
before writing final contigs and run metadata to the output directory.

### Read layouts

SPAdes receives a small dataset manifest pointing to the original read files.
MEGAHIT receives comma-separated file lists. Neither path concatenates or
interleaves FASTQ inputs, and `assemble` does not perform trimming or QC.

Use the same library number to associate paired reads with their orphan and
merged reads:

```bash
rolypoly assemble --paired-end 1 R1.fq.gz R2.fq.gz \
  --single-end 1 orphans.fq.gz --merged 1 merged.fq.gz -o assembly
rolypoly assemble --input-dir filter_reads_output -o assembly
```

The directory form is also used by `roll`. It recognizes current
`dedupe_final_interleaved_<sample>` / `dedupe_final_merged_<sample>` outputs,
historical `<sample>_final_<type>` outputs, external R1/R2 pairs, and
interleaved or single FASTQ files. Detection scans the directory itself, not
nested temporary folders. Use explicit library numbers when orphan reads
must be associated with particular external pairs.

metaSPAdes requires one logical paired-end library. When multiple paired
libraries are supplied, RolyPoly pools their file lists into one dataset entry
and logs a warning. Mate pairing is preserved, but the original libraries and
their insert-size distributions are no longer modeled independently. This
uses the original files directly without writing concatenated FASTQ files.
Orphan and merged reads associated with those libraries retain their respective
input categories. Other SPAdes modes keep the libraries separate.

Independent single-read libraries and single-end-only input remain unsupported
in meta mode. Associate orphan/merged reads using the same library number as
their pairs, or choose another supported mode or MEGAHIT.

PenguiN accepts separate R1/R2 file pairs directly. For mixed interleaved,
merged, or single inputs, its file-level pairing interface requires one
single-end stream. RolyPoly pipes these reads through stdin without writing
a concatenated temporary file. This mode does not use pairing information.
An interleaved file alone is also treated as single-end by PenguiN.
`--raw-fasta` is included as ordinary unpaired sequences, not trusted contigs.
A single FASTA is passed directly; multiple FASTA files or FASTA mixed with
reads use the same stream, so mate pairing is not used. No concatenated input
file is written. These inputs remain subject to PenguiN's normal sequence-length
limits and assembly filters; supplying FASTA does not enable a long-read mode.
RolyPoly invokes its protein-guided nucleotide assembly workflow
(`guided_nuclassemble`). This behavior was verified with PenguiN `5.cf8933`;
its [input dispatcher](https://github.com/soedinglab/plass/blob/cf8933/src/workflow/GuidedNuclassembler.cpp)
selects paired mode by file count. A mixed plain/gzip smoke test consumed all
records and matched the single-file baseline's output sequences.

## Usage

```bash
rolypoly assemble [OPTIONS]
```

<!-- BEGIN GENERATED CLI OPTIONS -->
## Options

- `-o`, `--output`: Output path (folder will be created if it doesn't exist) (type: `DIRECTORY`; default: `RP_assembly_output`)
- `-id`, `--input-dir`: Input directory to scan for fastq files (type: `DIRECTORY`)
- `--paired-end`: Library number and paired FASTQ files: <lib_num> <R1> <R2> (type: `TEXT`; default: ``)
- `--single-end`: Library number and single-end FASTQ: <lib_num> <fastq> (type: `TEXT`; default: ``)
- `--merged`: Library number and merged FASTQ: <lib_num> <fastq> (type: `TEXT`; default: ``)
- `--long-read`: path to long read FASTQ: <fastq> Note: long read files are not supported by all assemblers/configurations: SPAdes (standard/meta/rna modes): supported in hybrid assembly via --long-read-type (nanopore default; also pacbio or sanger). metaSPAdes hybrid assembly is experimental per the SPAdes manual, and meta mode pools multiple paired-end libraries into one logical library with a warning (no temporary FASTQ copies). rnaviralSPAdes: long reads are undocumented; a warning is logged and the reads are passed through - remove --long-read if SPAdes errors. MEGAHIT: not supported - long reads are skipped with a warning for the MEGAHIT run. Penguin: not supported - long reads are skipped with a warning for the Penguin run (no ONT/PacBio mode; exact k-mer matching is error-sensitive; reads over 200 kbp would be truncated). Also note: PenguiN pairs FILES as R1/R2 (an interleaved file is read as single-end), separate R1/R2 pairs are passed directly; mixed interleaved/single inputs are streamed through stdin as single-end reads without a concatenated temporary file (pairing is not used in this mode). (type: `TEXT`; default: ``)
- `--long-read-type`: Long read type passed to SPAdes in hybrid assembly (--nanopore, --pacbio, or --sanger). Only used when --long-read is provided. PacBio CLR input should be prefiltered (e.g. circular consensus sequences); PacBio HiFi/CCS reads go through -s/--single-end instead. (type: `CHOICE`; default: `nanopore`)
- `--raw-fasta`: Raw FASTA file(s) to include. SPAdes: trusted contigs. PenguiN: ordinary unpaired sequences (not trusted contigs); mixing with reads uses single-end streaming and loses pairing information. MEGAHIT: not supported. (type: `FILE`; default: ``)
- `-A`, `--assembler`: Assembler choice. For multiple, use multiple -A flags or give a comma-separated list. SPAdes: iterative de bruijn graph assembler - relatively slow and memory heavy, but potentially more accurate. MEGAHIT: multiple kmer based de bruijn graph assembler - Fast and memory light, but potentially less accurate. Penguin: protein-guided nucleotide assembly using guided_nuclassemble. Note1 : RolyPoly uses PenguiN's amino-acid-guided nucleotide assembly mode. Note2 : Without a preset, the default assemblers are SPAdes and MEGAHIT. (type: `CHOICE`; default: `spades, megahit`)
- `--spades-mode`: SPAdes mode for the 'spades' assembler. (type: `CHOICE`; default: `meta`)
- `--preset`: Apply a named assembly preset (overrides --assembler and --dereplicate unless those flags are given explicitly on the command line). 'rna_virus': RNA virus-focused: rnaviralSPAdes + MEGAHIT, broad k-mer range. Removes duplicate contigs (rmdup). Recommended for viral metatranscriptomes. 'metatranscriptome': Metatranscriptome: rnaSPAdes + MEGAHIT, broad k-mer range. Suited for poly-A selected or mixed transcriptome libraries. 'fast': Fast: MEGAHIT only, narrow k-mer range and larger step. Trades an unknown amount of sensitivity for an unknown amount of speed; suitable for quick previews or roll --mini runs. 'complete': Complete: metaSPAdes + rnaviralSPAdes + MEGAHIT with thorough k-mer ranges. Different assemblers may produce better results - the onus of choice is on the user. This will increase the runtime and memory usage significantly 'metag': Metagenomics: metaSPAdes (meta mode) only, broad k-mer range. Suited for DNA-based or mixed metagenomic libraries. (type: `CHOICE`)
- `-op`, `--override-parameters`: JSON-like string of parameters to override. Example: --override-parameters '{"spades": {"k": "21,33,55"}, "megahit": {"k-min": 31}}' (type: `TEXT`; default: `{}`)
- `-ss`, `--skip-steps`: Steps to skip. Repeat the flag to skip multiple steps (comma-separated values are NOT accepted here). Example: --skip-steps dereplicate --skip-steps rename (type: `CHOICE`; default: ``)

    - dereplicate: skip assembler-output dereplication (same as --no-rmdup)
    - rename: skip renaming the concatenated contigs to CID_ ids

- `-ow`, `--overwrite`: Overwrite the output directory if it already exists (deletes the existing output directory first). Without this flag, an existing output directory raises an error. (type: `BOOLEAN`; default: `False`)
- `--dereplicate`, `--no-rmdup`: Dereplicate assembler output by default. Disable with --no-rmdup. (type: `BOOLEAN`; default: `True`)

    - dereplicate: remove identical sequences (same sequence, same length, or its' reverse complement)
    - no-rmdup: do not perform assembler-output dereplication

- `-t`, `--threads`: Number of worker threads. (type: `INTEGER RANGE`; default: `1`)
- `-M`, `--memory`: Memory limit, for example 8g. (type: `MEMORY`; default: `8g`)
- `-k`, `--keep-tmp`: Keep temporary files. (type: `BOOLEAN`; default: `False`)
- `-tmp`, `--temp-dir`: Temporary working directory. (type: `DIRECTORY`)
- `-g`, `--log-file`: Path to the log file. (type: `FILE`; default: `rolypoly.log`)
<!-- END GENERATED CLI OPTIONS -->
