# Prepare External Data

The `prepare-external-data` command downloads and prepares necessary external databases and resources required by RolyPoly.

## Usage

```bash
rolypoly prepare-external-data [OPTIONS]
```

## Options

In addition to [common options](index.md#common-options), this command accepts:

- `--try-hard`: Attempt to recreate all databases from scratch instead of downloading pre-built ones (bool, default: False)
- `--data_dir`: Specify an alternative path for data storage (str, optional)

## Description

This command:

1. Downloads pre-built databases (unless `--try-hard` is specified)
2. Prepares necessary reference data
3. Sets up required directory structure
4. Validates downloaded resources

## Example

```bash
# Basic usage - download to default location
rolypoly prepare-external-data

# Use alternative data directory
rolypoly prepare-external-data --data_dir /path/to/data

# Rebuild databases from scratch
rolypoly prepare-external-data --try-hard
``` 