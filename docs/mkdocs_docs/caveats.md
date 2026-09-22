# Caveats and practical limitations

## Command-specific caveats

Additional limitations are documented alongside the relevant commands:

- [Report caveats](commands/report.md#caveats)
- [Taxonomy assignment (`mmtax`) caveats](commands/mmtax.md#caveats)
- [Nucleic-search caveats](commands/nucleic_search.md#caveats)

## Poly(A) trimming and genome termini

Terminal poly(A)/poly(T) trimming can remove genuine viral terminal sequence,
leaving an assembled genome with shortened or missing termini. A trimmed
assembly alone cannot establish the original tail length or a complete genome
end. Experimental poly(A) selection and computational tail trimming are separate
steps; choosing a poly(A)-selected preset can enable trimming.

Keep the reads from before poly(A) trimming if terminal sequence matters. Mapping
those reads back to contigs may help investigate missing termini, but homopolymer
length and ambiguous alignments require careful interpretation. Mapping can also
help assess termini when no poly(A) trimming was performed.

Automatic mapping-based termini rescue remains an open, deferred task; its
placement in `extend`, `termini`, or a precursor step is undecided. Do not assume
these commands currently restore trimmed tails. See the
[scientific background](background.md) for library-preparation context.

## Storage paths and temporary execution

Place the reference database, temporary directory and active output directory on
storage with high I/O throughput and low latency, ideally local SSD/NVMe or fast
local scratch with sufficient free space. We do not recommend network-mounted
folders for these active workloads: repeated database access and many temporary
files can make network latency and shared storage contention expensive. Copy
finished results to shared storage afterwards if needed.

Some tools need to execute files in their working or temporary directories.
MMseqs2 workflows can create and execute `.sh` scripts; RolyPoly's official
ORFfinder installer may execute a downloaded binary from a temporary directory
when an environment installation is unavailable. These locations need directory
traversal permissions, appropriate file execution permissions, and a filesystem
that permits execution. A `noexec` mount can cause `Permission denied` / Linux
`EACCES` (`errno 13`); this is not necessarily the subprocess's exit status.
Changing file permissions with `chmod` does not override `noexec`. Select a
writable, execution-enabled temporary location instead. See the
[Linux execution-error documentation](https://www.man7.org/linux/man-pages/man2/execve.2.html).

## Environment variables and path expansion

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
and resumed runs. See [configuration](configuration.md) for setup instructions.
