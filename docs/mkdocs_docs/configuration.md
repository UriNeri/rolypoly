Configuration
=============

RolyPoly uses a configuration file (`rpconfig.json`) to store settings
such as the data directory. This file is automatically updated when
using `rolypoly get-data`. You can view or edit it manually if needed, it's located in the `src/rolypoly/` directory.  

Example configuration:
```json
    {
    "ROLYPOLY_DATA": "/home/neri/projects/rolypoly/data/",
    "ROLYPOLY_TEST_DIR": "$RP_DIR/testing_folder/",
    "ROLYPOLY_REMIND_CITATIONS": "False"
    }
```
The database path supports a leading environment variable, such as
`${RP_DATA_ROOT}`, with an optional path suffix. See the
[path expansion caveats](#environment-variables-and-path-expansion)
for expansion limits and [storage caveats](commands/get_data.md#caveats)
for fast-storage recommendations and temporary execution permissions. Use
`get-data` to set up the initial configuration and download resources. For an
unset leading variable, the resolver may use a path relative
to the RolyPoly code directory; this is not a general fallback for a missing
absolute database path.

Recommended setup command:

```bash
rolypoly get-data --rolypoly-data /path/to/rolypoly_data
```

## Useful Tidbits

### Silence citation reminders

RolyPoly citation reminder printing is controlled by the `ROLYPOLY_REMIND_CITATIONS`
```json
{
    "ROLYPOLY_REMIND_CITATIONS": "False"
}
```

Notes:
- `"False"` disables citation reminder output to the console.
- `"True"` enables citation reminder output.

### Enable debug logging for troubleshooting
Most commands expose a hidden log-level flag used heavily in tests and debugging.

Examples:

```bash
rolypoly filter-reads -i reads.fq -o out/ -ll DEBUG
rolypoly marker-search -i contigs.fasta -o marker_out/ -log-level DEBUG
```

Tip: if a command supports `--log-file`, set it explicitly so debug output is
persisted in a predictable location.

### Keep temporary/intermediate files

Many commands create temporary intermediates and remove them by default.

Use `--keep-tmp` when available to preserve those files for inspection:

```bash
rolypoly assemble -id filtered_reads/ -o assembly_out/ --keep-tmp
rolypoly filter-contigs -i contigs.fasta -d host.fasta --keep-tmp
```

### Force a specific temp directory

Some commands allow overriding temp paths (for example, `--temp-dir` or
`--tmpdir`). This is useful on HPC systems when you want local scratch I/O.

Examples:

```bash
rolypoly marker-search -i contigs.fasta -o marker_out/ --temp-dir /tmp/rp_marker_tmp
rolypoly mask-dna -i host.fasta -o host_masked.fasta --tmpdir /tmp/rp_mask_tmp
```

### Overwrite existing output files
If the output path provided to a command exists, the commands may behave weird. to delete the existing output, use `--overwrite` .

### skip-existing
If the output path provided to a command exists, some intermediary steps may be skipped if the output files appear to exist. This is useful for resuming a pipeline after an interruption. Note, it may require setting the same temp directory as before, and this is INCOMPATIBLE with `--overwrite`.

### Override step parameters safely

Several pipeline-style commands support JSON overrides for internal tool
parameters. This is useful for tuning without editing source code.

Example:

```bash
rolypoly filter-reads \
    -i reads.fq \
    -o filtered/ \
    --override-parameters '{"dedupe": {"passes": 2}, "trim_adapters": {"minlen": 55}}'
```

## Caveats

### Environment variables and path expansion

The configuration file is named **`rpconfig.json`**. You can export one or two
shell variables for convenient storage roots and reuse them in commands:

```bash
export RP_DATA_ROOT=/fast/storage/rolypoly_data
export RP_WORK_ROOT=/fast/scratch/rolypoly
```

In `rpconfig.json`, set the database field to reference the exported variable
(keep the other settings):

```json
"ROLYPOLY_DATA": "${RP_DATA_ROOT}"
```

The current database resolver supports a leading `$VAR` or `${VAR}`, optionally
followed by a path suffix, and expands a leading `~`. Export variables in the
shell launching RolyPoly, or in its job script; a shell startup-file setting is
not necessarily inherited by a batch job. Setting a variable alone does not
replace an unrelated absolute database path already stored in the configuration.

For command-line paths, let the shell expand variables explicitly, for example
`--temp-dir "$RP_WORK_ROOT/tmp"` and `--output "$RP_WORK_ROOT/results"`, on
commands exposing those options. Create the required parent directories first.
JSON is not shell code: general variable substitution, nested variables and
expansion in every configuration field or command option are not guaranteed.
Literal `~` or `$HOME` paths may therefore remain unexpanded outside the database
resolver; prefer absolute paths or shell-expanded `"$HOME/..."` arguments.

Relative paths can work, but usually depend on the directory from which the
command is launched. Absolute paths are more reliable across scripts, batch jobs
and resumed runs.  
Honestly just use absolute paths. 
