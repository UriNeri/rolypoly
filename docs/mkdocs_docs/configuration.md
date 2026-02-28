Configuration
=============

RolyPoly uses a configuration file (`rpconfig.json`) to store settings
such as the data directory. This file is automatically updated when
using `rolypoly get-data`.

To view or modify the configuration:

1.  Locate `rpconfig.json` in `src/rolypoly/`.
2.  Open the file with a text editor.
3.  Modify the settings as needed.
4.  Save the file.

Example configuration:

    {
            "ROLYPOLY_DATA": "/clusterfs/jgi/scratch/science/metagen/neri/projects/rolypoly/data/"
    }

Note: It's recommended to use the `get-data` command to set
up the initial configuration and download necessary resources.

Recommended setup command:

```bash
rolypoly get-data --ROLYPOLY_DATA /path/to/rolypoly_data
```

## Useful Tidbits

### Silence citation reminders

RolyPoly citation reminder printing is controlled by the `ROLYPOLY_REMIND_CITATIONS`
value in `rpconfig.json`.

Use string values (`"True"` / `"False"`):

```json
{
    "ROLYPOLY_DATA": "/path/to/rolypoly_data",
    "ROLYPOLY_REMIND_CITATIONS": "False"
}
```

Notes:

- `"False"` disables citation reminder output to the console.
- `"True"` enables citation reminder output.
- For normal CLI use, `rpconfig.json` is the source of truth because RolyPoly
    reads it at startup and exports these values into the process environment.

### Enable debug logging for troubleshooting

Most commands expose a hidden log-level flag used heavily in tests and debugging.

Examples:

```bash
rolypoly filter-reads -i reads.fq -o out/ -ll DEBUG
rolypoly marker-search -i contigs.fasta -o marker_out/ -ll DEBUG
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

Tip: keep JSON keys aligned to the command's documented step names.
