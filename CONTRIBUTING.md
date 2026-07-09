# Contributing to RolyPoly

Contributions welcome! Whether it's bug fixes, new features, documentation improvements, packaging, tests, or reports of your exprience / resource usage for your samples - all help is appreciated. Pull requests or forks are the preferred way to contribute and will be considered, and you can also open issues for discussion or contact one of the developers directly.

## Project Roadmap & TODO List
Check out our [project roadmap and TODO list](https://docs.google.com/spreadsheets/d/1udNbxtK1QMfOhVgxHyhrgw7U1hHFeIazlcLM6VIcbJo/edit?gid=0#gid=0) to see what features and improvements are planned.

## Contribution guidelines
- **Primary Language**: Python >=3.10
- **Secondary Languages**: Some system calls to shell/Bash are allowed (via run_command_comp, or at at least logged before execution).
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
   - Avoid "_" prefix for functions ("private" or otherwise). if somthing is explictly not meant to ever at all be used outside its scope, that should be in a comment or docstring, but in generally we want to avoid these `"_..."` code blocks and there shouldn't be "private" breaking stuff.

2.1 **Docstrings**:
   - Add a docstring to all user-facing command functions (click entry points) and reusable utility functions.
   - Keep simple helpers concise (one-line docstring is fine).
   - For non-trivial logic, use a multi-line docstring and include sections like `Args`, `Returns`, `Raises`, and `Note` when useful.
   - Prefer the same style already used in the codebase (plain-language summary first, then structured sections if needed).
   - Do not remove existing docstrings unless they are incorrect; update them when behavior or parameters change.
   - Module-level docstrings are recommended for larger utility modules, especially when they contain multiple related functions.

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
   - For most (ideally all) click commands, include a hidden log-level option so tests can consistently enable debug logging:
     - `@click.option("-ll", "--log-level", hidden=True, default="INFO", help="Log level")`
   - Use small/local fixtures from `testing_folder/` when possible.
   - **Run standardized CLI tests**: `pixi run -e dev pytest -q src/tests/test_cli_contracts.py`
   - **Run fast help-only smoke tests** (just `--help` for top-level + each command): `pixi run -e dev pytest -q src/tests/test_cli_help_smoke.py`
   - **Run one command's scenarios**: `pixi run -e dev pytest -q src/tests/test_cli_contracts.py --cli-commands fetch-sra`
   - **Run multiple commands' scenarios**: `pixi run -e dev pytest -q src/tests/test_cli_contracts.py --cli-commands annotate,assemble,marker-search`
   - **Run specific scenario IDs**: `pixi run -e dev pytest -q src/tests/test_cli_contracts.py --cli-scenarios marker_search_runtime_genomad,assemble_megahit_runtime`
   - **Run by text match (id/description/command)**: `pixi run -e dev pytest -q src/tests/test_cli_contracts.py --cli-match fetch,identify`
   - **Environment-variable based selection** (useful in CI/shell scripts):
     - `RP_CLI_COMMANDS=fetch-sra,marker-search pixi run -e dev pytest -q src/tests/test_cli_contracts.py`
     - `RP_CLI_SCENARIOS=assemble_megahit_runtime pixi run -e dev pytest -q src/tests/test_cli_contracts.py`
     - `RP_CLI_MATCH=identify pixi run -e dev pytest -q src/tests/test_cli_contracts.py`
   - **Run all command scenarios + unit tests**: `pixi run -e dev pytest -q src/tests`
   - **Run all tests**: `pixi run -e dev pytest -q src/tests`
   - Legacy ad-hoc scripts under `testing_folder/*.sh` are still useful for manual debugging, but new command validation should be added to the pytest flow above.
2. **Benchmarking**:
   - Use `/usr/bin/time` for resource monitoring. Alternatively, hyperfine is great too but. Ideallt - use SLURM and keep track of the job IDs for later analysis with seff/pyseff.

## Documentation workflow

- Docs source pages are in `docs/mkdocs_docs/`.
- Docs site navigation is configured in `docs/mkdocs.yml` (`nav:` section).
- Command docs are under `docs/mkdocs_docs/commands/`.
- Keep command links in `README.md` aligned with pages listed in `docs/mkdocs.yml`.

Use pixi docs tasks:
- Serve locally (live reload): `pixi run -e dev docs-serve`
- Build static docs: `pixi run -e dev docs-build`
- Auto-generate command help pages:
   - create missing pages: `pixi run -e dev python src/setup/export_command_help_to_docs.py`
   - refresh existing auto-generated pages: `pixi run -e dev python src/setup/export_command_help_to_docs.py --overwrite`

For command pages that need rich/static sections (mermaid, tables, links), add a
per-command scaffold at:
- `src/setup/help_export_scaffolds/<command_name>.md`

The exporter injects scaffold content into generated pages under **Pinned Sections**.

When adding a new command page:
1. Add the markdown page in `docs/mkdocs_docs/commands/`.
2. Add it to `nav` in `docs/mkdocs.yml`.
3. Add/update links in `README.md` and relevant command pages under `docs/mkdocs_docs/commands/`.

## PyPI / TestPyPI release automation

Releases are automated via GitHub Actions using trusted publishing (OIDC), with this flow:
1. Build wheel + sdist and run help-smoke tests (`src/tests/test_cli_help_smoke.py`)
2. Validate package metadata with `twine check`
3. Publish to TestPyPI
4. Install from TestPyPI and run import + CLI smoke check
5. Publish the same artifacts to PyPI

Version bumping is **manual** and happens before CI publishing.

Workflow file: `.github/workflows/pypi-release.yml`

The workflow uses concurrency cancellation per ref (`pypi-release-${ref}`), so newer runs automatically cancel older in-progress runs on the same branch/tag.

**Important**: publishing (steps 2-5 above) only ever runs for an actual GitHub Release event, or a manual `workflow_dispatch`. A plain push to the `release` branch only runs the build/help-smoke job (step 1) - it does not publish anything. This keeps "promote main into release" and "cut a release" as two distinct, non-conflicting actions, so you can never get a duplicate TestPyPI upload from pushing the branch and then publishing a GitHub Release for the same version.

### One-time setup (maintainers)

Configure trusted publishers in both PyPI and TestPyPI for project `rolypoly-tk`:
- Owner/repo: `UriNeri/rolypoly`
- Workflow name: `pypi-release.yml`
- Environment names: `testpypi` and `pypi`

Use environment protection rules in GitHub for safer releases (recommended):
- `testpypi`: optional reviewers
- `pypi`: required reviewer(s)

### Triggering releases

- Push to deployment branch `release`: builds + runs help-smoke tests only. No publish. Safe to re-run/re-push.
- Create/publish a GitHub Release (tag `vX.Y.Z` targeting `release`): the **only** trigger that publishes, running the full build -> TestPyPI -> smoke-install -> PyPI pipeline in one shot.
- `workflow_dispatch` (manual run): builds/tests, and also publishes to TestPyPI; pass `publish_pypi: true` to additionally publish to PyPI (useful for dry-runs/recovery).
- Both publish steps pass `skip-existing: true`, so re-running the workflow (e.g. after a transient failure) won't hard-fail if that version was already uploaded.

- **Bioconda note**: our Bioconda recipe automatically uses the source tar.gz from GitHub Releases; the GitHub Release created below (tagged source) is what Bioconda fetches.

## Branch and release flow (recommended)

- Use `main` or dedicated "dev_nnn" branches when adding features, testing, working, PR review, docs updates.
- Keep `release` as a deployment branch that mirrors what gets published; pushing to it alone is a no-op for publishing.
- At release time, promote `main` into `release`, push `release`, then create a GitHub Release targeting `release` - that release is what actually triggers publishing.
- Avoid long-lived drift between `main` and `release`; if a hotfix lands on `release`, merge it back into `main` asap.

Typical release promotion sequence:
- `git checkout main && git pull origin main`
- `git checkout release && git pull origin release`
- `git merge --ff-only origin/main`
- `git push origin release` (build/smoke-test only - does not publish)
Then create the GitHub Release that triggers publishing and gives Bioconda a tagged source archive:
- `gh release create v0.n.nn --target release --title "v0.n.nn" --generate-notes`

`--generate-notes` has GitHub auto-draft the release notes (merged PRs/commits and any first-time contributors) since the previous tag - prefer this over a hand-written `--notes` string so the notes and contributor list stay accurate without manual upkeep.

### One-command release prep (bump + commit + release)

Use the pixi task:
- `pixi run -e dev bump-commit-publish`

This task runs `src/setup/bump_commit_publish.sh` and by default:
- bumps version in `src/rolypoly/__init__.py` (`micro` by default; or `major`/`minor`/explicit `X.Y.Z`)
- refreshes `src/setup/env_big.yaml` from `pixi workspace export conda-environment -e complete`, with cleanup for micromamba compatibility
- runs help-smoke tests locally
- commits `src/rolypoly/__init__.py` and `src/setup/env_big.yaml`, and pushes to `origin/release` (build/smoke-test only, no publish)
- creates the GitHub Release `vX.Y.Z` targeting `release` via `gh release create ... --generate-notes`, which is what actually triggers the publish workflow

Common options:
- `pixi run -e dev bump-commit-publish -- --bump minor`
- `pixi run -e dev bump-commit-publish -- --bump 0.7.0`
- `pixi run -e dev bump-commit-publish -- --branch release --remote origin`
- `pixi run -e dev bump-commit-publish -- --skip-smoke`
- `pixi run -e dev bump-commit-publish -- --skip-release` (push the branch only; create the GitHub Release yourself later)

### Local fallback (manual upload)

If needed, manual upload with twine is still supported:
- Build: `pixi run -e dev python -m build --sdist --wheel --outdir dist`
- Check: `pixi run -e dev twine check dist/*`
- Upload: `pixi run -e dev twine upload dist/* --verbose`


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
