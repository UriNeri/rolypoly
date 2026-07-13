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
Environment variables should work. Note: I recommend using the `get-data` command to set
up the initial configuration and download necessary resources. there's a fallback, if that path is missing, it checks a path relative to the `rolypoly` code directory. 

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

