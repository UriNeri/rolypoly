# Setup and Data Commands

## Get Data

The `get-data` command downloads and prepares necessary external databases and resources required by RolyPoly.

## Usage

```bash
rolypoly get-data [OPTIONS]
```

## Options

In addition to [common options](index.md#common-options), this command accepts:

- `--info`: Print current version/configuration information and exit
- `--ROLYPOLY_DATA`: Override the data directory used by RolyPoly
- `--log-file`: Path to command log file
- `--log-level`: Hidden testing/debug option

## Description

This command:

1. Downloads pre-built databases
2. Prepares necessary reference data
3. Sets up required directory structure
4. Validates downloaded resources

## Example

```bash
# Basic usage - download to default location
rolypoly get-data

# Use alternative data directory
rolypoly get-data --ROLYPOLY_DATA /path/to/data

# Show current install/data configuration
rolypoly get-data --info
```

## Version

The `version` command displays version and data information for RolyPoly.

### Usage

```bash
rolypoly version
# or
rolypoly --version
```

This command prints:
- Code version (commit hash or semantic version)
- Data version (date)
- Other relevant version information 