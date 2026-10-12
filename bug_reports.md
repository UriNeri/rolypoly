# RolyPoly `roll` Bug Overview

## Reproduction example

The workflow was launched using:

```bash
for dir in raw_reads/sample_*/; do
    mkdir -p "results/$(basename "$dir")"

    rolypoly roll \
        -i "$dir" \
        -o "results/$(basename "$dir")" \
        --assembler megahit \
        --mapper bwa-mem2 \
        --threads 10 \
        --memory 30gb \
        --log-file "results/$(basename "$dir")/roll.log" \
        --log-level debug
done
```

Script location:

`/home/neri/Documents/post2/scratch_work/kranzler-scratch/pupcycle2/roll_each.sh`

The relevant input directory was:

```text
/home/neri/Documents/post2/scratch_work/kranzler-scratch/pupcycle2/raw_reads/sample_1/
```

The output and log were written to:

```text
/home/neri/Documents/post2/scratch_work/kranzler-scratch/pupcycle2/results/sample_1/
```

Log:

`/home/neri/Documents/post2/scratch_work/kranzler-scratch/pupcycle2/results/sample_1/roll.log`

---

## Bug 1: The first R1/R2 pair is discarded during concatenation

### Input detected

The `sample_1` directory contained two paired-end libraries:

```text
1-1A_R1_001.fastq.gz
1-1A_R2_001.fastq.gz

1-1B_R1_001.fastq.gz
1-1B_R2_001.fastq.gz
```

The log reports two R1/R2 pairs:

```text
R1/R2 pairs: 2
```

The relevant source code is in:

`rolypoly/src/rolypoly/commands/reads/filter_reads.py`

### Observed behavior

The first pair was reformatted into a temporary file:

```text
1-1A_R1_001.fastq.gz + 1-1A_R2_001.fastq.gz
→ temp_concat_interleaved.fq.gz
→ 41,838,104 reads
```

The second pair was written to the final file:

```text
1-1B_R1_001.fastq.gz + 1-1B_R2_001.fastq.gz
→ concat_interleaved.fq.gz
→ 41,210,974 reads
```

The filtering workflow then received only:

```text
Input: 41210974 reads
```

That is exactly the read count from pair `1-1B`, not the expected combined count:

```text
41,838,104 + 41,210,974 = 83,049,078 reads
```

The first filtering command in the log is `decontaminate_rrna`, and it processes only 41,210,974 reads.

### Likely root cause

The paired-file loop in `process_input_fastq()` writes the first pair to `temp_concat_interleaved.fq.gz` and subsequent pairs directly to `concat_interleaved.fq.gz`.

Afterward, the cleanup logic effectively does:

```python
if temp_interleaved.exists():
    if final_interleaved.exists():
        temp_interleaved.unlink()
```

Therefore, when there are multiple pairs:

1. Pair A is written to the temporary file.
2. Pair B is written to the final file.
3. The temporary file containing pair A is deleted.
4. Filtering starts from pair B only.

The relevant logic is approximately at:

`filter_reads.py:1046`

and:

`filter_reads.py:1125`

### Expected behavior

All detected libraries should be concatenated into one interleaved FASTQ before filtering:

```text
pair A + pair B
→ concat_interleaved.fq.gz
→ approximately 83,049,078 reads
→ one filter-reads pipeline
```

The outer script correctly starts one independent `roll` run per `sample_*` directory. The problem is inside the single `filter-reads` invocation for a directory containing multiple R1/R2 pairs.

---

## Bug 2: `roll --assembler megahit` is overridden by the default roll preset

### User command

The command explicitly requested:

```text
--assembler megahit
```

However, no `--preset` was supplied, so `roll` used its default:

```text
--preset rna_virus
```

The log confirms:

```text
Applied roll preset 'rna_virus': filter=rna_virus_metat assembly=rna_virus
```

The roll preset mapping is defined in:

`./src/rolypoly/commands/misc/end_2_end.py`

The default preset is configured near:

`end_2_end.py:83`

### Observed effective configuration

The resulting assembly configuration contains:

```json
"assembler": "['spades_rnaviral', 'megahit']",
"preset": "rna_virus"
```

Configuration file:

`/home/neri/Documents/post2/scratch_work/kranzler-scratch/pupcycle2/results/sample_1/assembly/assembly_config.json`

The log then shows:

```text
Applied assembly preset 'rna_virus'
```

followed by:

```text
Running SPAdes with command:
['spades.py', '--rnaviral', ...]
```

The relevant log section is around:

`roll.log:790`

and:

`roll.log:799`

The assembly preset definition is in:

`./src/rolypoly/commands/assembly/assemble.py`

Specifically, `apply_assembly_preset()` replaces the assembler list when it does not see `assembler` as an explicitly supplied parameter:

`assemble.py:127`

### Important clarification: this was not caused by library count

The log reports:

```text
Found 1 libraries
```

but this occurs after the assembly preset has already been applied.

The sequence is:

```text
Applied assembly preset 'rna_virus'
Saving assembly config
Found 1 libraries
Running SPAdes --rnaviral
```

Therefore, the library count did not select or override the assembler.

The one-library result is also expected from the current filtering design: the filtered reads are represented as one combined sample-level interleaved library:

```text
Libraries: {
  'sample_1': {
    'interleaved': '.../dedupe_final_interleaved_sample_1.fq.gz',
    'merged': '.../dedupe_final_merged_sample_1.fq.gz'
  }
}
```

The issue is precedence handling between:

```text
roll --assembler megahit
```

and:

```text
roll --preset rna_virus
```

### Expected behavior

An assembler explicitly supplied to `roll` should take precedence over the assembler selected by the roll/assembly preset:

```text
rolypoly roll --assembler megahit
→ run MEGAHIT only
```

Alternatively, if presets are intended to override the CLI, the command should clearly document that behavior and log it prominently. The current behavior is surprising because the `--assembler` option appears to be a direct user override.

---

## Summary of expected fixes

### Fix 1: Preserve all input pairs

The input preparation stage should append every detected R1/R2 pair to the same final interleaved output, without deleting the first pair.

Expected validation for this example:

```text
Initial combined input:
approximately 83,049,078 reads
```

The first filtering step should report approximately that count, rather than:

```text
41,210,974 reads
```

### Fix 2: Preserve explicit `roll` assembler arguments

The assembler selected by an explicit `roll --assembler` argument should be passed through as an explicit override when `roll` invokes the internal assembly command.

For this example, the effective configuration should be:

```json
"assembler": "['megahit']"
```

and the log should not contain:

```text
Running SPAdes ... --rnaviral
```

unless the user explicitly requested a SPAdes-based assembler or selected a preset that is documented to override the command-line value.

### Relevant source files

- `./src/rolypoly/commands/reads/filter_reads.py`
- `./src/rolypoly/commands/misc/end_2_end.py`
- `./src/rolypoly/commands/assembly/assemble.py`
- `/home/neri/Documents/post2/scratch_work/kranzler-scratch/pupcycle2/results/sample_1/roll.log`
- `/home/neri/Documents/post2/scratch_work/kranzler-scratch/pupcycle2/results/sample_1/assembly/assembly_config.json`
- `/home/neri/Documents/post2/scratch_work/kranzler-scratch/pupcycle2/roll_each.sh`