# Get Data

<!-- Auto-generated draft from CLI metadata for `rolypoly get-data`. -->
<!-- Expand this page with command-specific context, examples, and citations. -->

## Summary

Download pre-built external data required for RolyPoly.

## Description

This command downloads pre-built databases and reference data from
a public repository. Reproducible builders are maintained in UriNeri/rolypoly-db.

## Usage

```bash
rolypoly get-data [OPTIONS]
```

<!-- BEGIN GENERATED CLI OPTIONS -->
## Options

- `--info`: Display current RolyPoly version, installation type, and configuration paths (type: `BOOLEAN`; default: `False`)
- `-rd`, `--rolypoly-data`, `--data-dir`: If you do not want to download the the data to same location as the rolypoly code, specify an alternative path. TODO: remind user to provide such alt path in other scripts? envirometnal variable maybe (type: `TEXT`; default: `Sentinel.UNSET`)
- `-g`, `--log-file`: Path to the log file. (type: `FILE`; default: `rolypoly.log`)
<!-- END GENERATED CLI OPTIONS -->

## Caveats

### Storage paths and temporary execution

Ideally, place the rolypoly bundled database, temporary directory and active output directory on storage with high I/O throughput and low latency, like a local SSD/NVMe or fast local scratch with sufficient free space.  
We do not recommend network-mounted folders for these active workloads: repeated database access and many temporary files can make network latency and shared storage contention expensive.  

Some tools need to execute files in their working or temporary directories.
MMseqs2 workflows can create and execute `.sh` scripts; RolyPoly's usage of NCBI's
ORFfinder may downloaded and execute a binary from a temporary directory
when an environment installation is unavailable. These locations need directory
traversal permissions, appropriate file execution permissions, and a filesystem
that permits execution. A `noexec` mount can cause `Permission denied` / Linux
`EACCES` (`errno 13`); this is not necessarily the subprocess's exit status.
Changing file permissions with `chmod` does not override `noexec`. Select a
writable, execution-enabled temporary location instead. See the
[Linux execution-error documentation](https://www.man7.org/linux/man-pages/man2/execve.2.html). This error mostly arises when a $HOME folder is used for working/temporary directories, on a large HPC cluster where homes are network mounted. 
