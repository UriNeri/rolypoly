# Contributing to RolyPoly

Contributions welcome! Whether it's bug fixes, new features, documentation improvements, packaging, tests, or reports of your exprience / resource usage for your samples - all help is appreciated. Pull requests or forks are the preferred way to contribute and will be considered, and you can also open issues for discussion or contact one of the developers directly.

## Project Roadmap & TODO List
Check out our [project roadmap and TODO list](https://docs.google.com/spreadsheets/d/1udNbxtK1QMfOhVgxHyhrgw7U1hHFeIazlcLM6VIcbJo/edit?gid=0#gid=0) to see what features and improvements are planned.

## Contribution guidelines
- **Primary Language**: Python >=3.9
- **Secondary Languages**: Some system calls to Bash are allowed.
- **Dependency Management**: via pixi (development)
  - Prefer using existing dependencies over adding new ones.
  - Avoid pandas, and use polars
  - Avoid biopython if possible, check if an existing feature is implemented in `src/rolypoly/utils/bio/*` or use polars-bio.
- **lazy / eager**: the CLI commands are lazy evaluted (see `/src/rolypoly/utils/lazy_group.py`), and need to be explictly added to the `src/rolypoly/rolypoly.py` file. This makes debugging/tracing slightly harder, but it also isolates the commands, so we can break one of them without worrying on it effect it may have on others.

## Code Organization
1. **File Structure**:
   - `src/rolypoly/utils/`: Utility functions and helpers
   - `src/rolypoly/commands/`: Command-line interface modules (using click). 
  - `rolypoly.utils.various` for general-purpose functions that don't fit into other categories (e.g. dataframe operations)
  - `rolypoly.utils.logging` for logging, configuration, output tracking etc

2. **Naming Conventions**:
   - CLI arguments: No positional arguments unless absolutly necceray. Instead, prefer 'decalerd' and explict named arguemnt. Must support both short and long options, e.g. `-s` and `--skip-existing`. Optionally, provide support for json file (`--config config.json`) or json string (`--override-params '{"skip_existing": true}`).
   - Functions and Internal variables: Snake case (e.g., `skip_existing`). Try and reuse variable names from other commands for the same purpose. Long descriptive names are ok.
   - Classes: PascalCase (though use classes sparingly).
   - Environment or Global variables: UPPERCASE or CamelCase.

3. **Temporary Files**:
   - Optionally, create temp directory (hidden argument in some commands `--temp-dir`, if not specified it's within user's output path).  
   - When done, move only final output files to user's output path, or rename the temp-dir if it's easier (same parent path maybe).
   - Try to clean up tmp files unless `--keep-tmp` flag is used.

4. **Calling external tools**:
   - Use `rolypoly.utils.command_runner.run_command_comp()` to run external commands.
   - If that is not possible, use `subprocess.run()`.

5. **Shared Code**:
   - **Avoid creating intermediate helper modules** in `commands/` - utilities belong in `utils/`
   - Place reusable functions in appropriate `utils/` subdirectories (e.g., `utils/bio/` for biological sequence operations)
   - Check existing utilities before implementing new functionality

## Testing & Benchmarking
1. **Testing**:
   - Add tests under `src/tests/*`.
   - Prefer `pytest` for new tests, and keep command smoke tests in `src/tests/test_cli_contracts.py` with scenarios in `src/tests/cli_scenarios.json`.
   - Use small/local fixtures from `testing_folder/` when possible.
   - You can also use `/clusterfs/jgi/scratch/science/metagen/neri/tests/rp_tests/` (on dori), which contains larger example data for different commands.
   - **Run standardized CLI tests**: `pixi run -e dev pytest -q src/tests/test_cli_contracts.py`
   - **Run all tests**: `pixi run -e dev pytest -q src/tests`
   - Legacy ad-hoc scripts under `testing_folder/*.sh` are still useful for manual debugging, but new command validation should be added to the pytest flow above.
2. **Benchmarking**:
   - Use `/usr/bin/time` for resource monitoring. Alternatively, hyperfine is great too but. Ideallt - use SLURM and keep track of the job IDs for later analysis with seff/pyseff.

## Example Workflow: Adding a New Command

Here's a high-level workflow for adding a new command to RolyPoly:

1. **Check for existing utilities**: Search `src/rolypoly/utils/` for existing functions that might help (especially `utils/bio/` for sequence operations)

2. **Create the command file**: Add your command in the appropriate subdirectory under `src/rolypoly/commands/` (e.g., `commands/misc/my_command.py`)
   - Use `@click.command()` decorator
   - Follow naming conventions (short + long options, snake_case for parameters)
   - Import and reuse existing utilities from `utils/` where possible

3. **Add shared utilities if needed**: If you create reusable functions, place them in `src/rolypoly/utils/` (NOT in `commands/`)
   - Use existing modules when appropriate (e.g., `utils/bio/polars_fastx.py` for FASTA/FASTQ operations)

4. **Register the command**: **CRITICAL** - Add your command to `src/rolypoly/rolypoly.py` in the appropriate lazy_subcommands group
   - Format: `"command-name": "rolypoly.commands.subdir.my_command.my_command_function"`
   - The command won't appear in the CLI without this step!

5. **Test the command**:
   - Run `pixi run rolypoly <command-name> --help` to verify it loads
   - Test with actual data
   - Add test cases to `src/tests/` if appropriate

6. **Document**: Update help strings and consider adding examples to README or docs
   - Add a markdown file in the appropriate location under `docs/`
   - Update the `mkdocs.yml` configuration file to include your new documentation
   - Add to the index or relevant navigation section if needed

## **Note**
This project is governed under the LBNL IP office. By contributing, you agree that your contributions will be subject to the terms of the GPLv3 license.
