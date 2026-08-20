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

## Options

- `--info`: Display current RolyPoly version, installation type, and configuration paths (type: `BOOLEAN`; default: `False`)
- `-rd`, `--rolypoly-data`, `--data-dir`: If you do not want to download the the data to same location as the rolypoly code, specify an alternative path. TODO: remind user to provide such alt path in other scripts? envirometnal variable maybe (type: `TEXT`; default: `Sentinel.UNSET`)
- `-g`, `--log-file`: Path to the log file. (type: `FILE`; default: `rolypoly.log`)




