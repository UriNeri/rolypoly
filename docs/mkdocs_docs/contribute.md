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
   - Ideally, please use `rolypoly.utils.command_runner.run_command_comp()` to run external commands, especially if a logger or output tracking is needed.
   - If that is not possible, use `subprocess.run()`.
   - If there is a `tools` list global variable, update it accordingly, that would expose it if the citation reminder is called.

5. **Shared Code**:
   - **Avoid creating intermediate helper modules** in `commands/` - utilities belong in `utils/`
   - Place reusable functions in appropriate `utils/` subdirectories (e.g., `utils/bio/` for biological sequence operations)
   - Check existing utilities before implementing new functionality

## Testing & Benchmarking
1. **Testing**:
   - Add persistent tests only for stable behavior that is important to users or
     scientific correctness: CLI contracts, output schemas, backend adapters,
     resumability, shared command policy, and regressions likely to recur.
   - Keep one-off migration checks, documentation-generation checks, exploratory
     data audits, benchmarks, and trivial implementation-detail assertions local
     and transient. Record the command and result in the change/PR instead of
     adding them to the repository test suite.
   - Prefer the lowest-cost level that expresses the contract. Put command
     workflow scenarios in `src/tests/cli_scenarios.json`, exercised by
     `src/tests/test_cli_contracts.py`; use focused pytest modules under
     `src/tests/` for stable scientific transformations or output behavior that
     is clearer to validate directly.
   - For most (ideally all) click commands, include a hidden log-level option so tests can consistently enable debug logging:
     - `@click.option("-ll", "--log-level", hidden=True, default="INFO", help="Log level")`
   - Keep committed fixtures synthetic, deterministic, and as small as practical.
     Do not commit generated indexes, command outputs, external databases, or
     large real datasets for smoke tests. Generate them temporarily, download
     them explicitly, or document the external data source instead.
   - Every committed fixture should be referenced by a persistent test or a
     documented reproducible example. Remove its fixture when removing the last
     consumer, unless it has a separately documented purpose.
   - **Run standardized CLI tests**: `pixi run -e dev pytest -q src/tests/test_cli_contracts.py`(or just pixi run test-cli)
   - **Run fast help-only smoke tests** (just `--help` for top-level + each command): `pixi run -e dev pytest -q src/tests/test_cli_help_smoke.py`
   - **Run one command's scenarios**: `pixi run -e dev pytest -q src/tests/test_cli_contracts.py --cli-commands fetch-sra`
   - **Run multiple commands' scenarios**: `pixi run -e dev pytest -q src/tests/test_cli_contracts.py --cli-commands annotate,assemble,marker-search`
   - **Run specific scenario IDs**: `pixi run -e dev pytest -q src/tests/test_cli_contracts.py --cli-scenarios marker_search_runtime_genomad,assemble_megahit_runtime`
   - **Run by text match (id/description/command)**: `pixi run -e dev pytest -q src/tests/test_cli_contracts.py --cli-match fetch,identify`
   - **Environment-variable based selection** (useful in CI/shell scripts):
     - `RP_CLI_COMMANDS=fetch-sra,marker-search pixi run -e dev pytest -q src/tests/test_cli_contracts.py`
     - `RP_CLI_SCENARIOS=assemble_megahit_runtime pixi run -e dev pytest -q src/tests/test_cli_contracts.py`
     - `RP_CLI_MATCH=identify pixi run -e dev pytest -q src/tests/test_cli_contracts.py`
   - **Run all tests**: `pixi run -e dev pytest -q src/tests`
   - Temporary validation scripts should normally live outside the repository
     (for example under `/tmp`) and should not be left as untracked test files.
     Promote one to a persistent pytest or CLI scenario only when it protects a
     stable contract under the criteria above.
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

### Three-step release commands

Release preparation and publication are deliberately separate Pixi tasks. Run them
from a clean `main` working tree in this order:

1. `pixi run -e dev bump` — update local release files only.
2. Review `git diff`, especially the version and exported environment.
3. `pixi run -e dev commit-release` — test, commit, and push `main` and `release`.
4. Wait for the build/help-smoke workflow triggered by the `release` push to pass.
5. `pixi run -e dev publish-release -- --dry-run` — validate the release without creating it.
6. `pixi run -e dev publish-release` — create the GitHub Release and start package publishing.

#### What `bump` does internally

`bump` runs `src/setup/bump.sh`. It requires a clean working tree and defaults to a
micro version increment. It accepts a positional value or `--bump`:

- `pixi run -e dev bump` or `pixi run -e dev bump -- micro`: `0.7.17` -> `0.7.18`
- `pixi run -e dev bump -- patch`: alias for `micro`
- `pixi run -e dev bump -- minor`: `0.7.17` -> `0.8.0`
- `pixi run -e dev bump -- major`: `0.7.17` -> `1.0.0`
- `pixi run -e dev bump -- 0.7.18`: use that explicit `X.Y.Z` version

Internally, the command:

1. Parses `__version__` from `src/rolypoly/__init__.py` and removes any local
   suffix after `+` before calculating the next version.
2. Prepares a replacement `__version__ = "X.Y.Z"` line in a temporary file.
   `pyproject.toml` is not edited because Hatch reads its dynamic version from
   this Python file.
3. Runs `pixi workspace export conda-environment -e complete -n rolypoly-tk`.
4. Normalizes the export for micromamba compatibility: it separates conda and
   pip dependencies, de-duplicates them, preserves the stronger top-level
   constraints, ensures `pip` is present, and removes the editable `-e .` entry.
5. Appends `rolypoly-tk >=X.Y.Z,<1` to the exported pip dependencies and replaces
   `src/setup/env_big.yaml`.
6. Replaces both release files only after the version calculation and environment
   export succeed.

`bump` does not test, stage, commit, switch branches, push, tag, create a GitHub
Release, or publish a package. Its only intended working-tree changes are
`src/rolypoly/__init__.py` and `src/setup/env_big.yaml`.

#### What `commit-release` does

`commit-release` runs `src/setup/commit_release.sh`. It requires `main` to contain
exactly the two unstaged files produced by `bump`. It fetches the remote branch
state, refuses to proceed if local `main` is behind or diverged, runs
`src/tests/test_cli_help_smoke.py`, commits the two files as
`release: bump version to vX.Y.Z`, and pushes `main`. It refreshes `origin/main`,
checks out and updates `release`, runs `git merge --ff-only origin/main`, pushes
`release`, and checks `main` out again. The release branch push runs CI
build/smoke checks but does not publish a package.

Options:

- `pixi run -e dev commit-release -- --skip-smoke`
- `pixi run -e dev commit-release -- --source-branch main --branch release --remote origin`

#### What `publish-release` does

`publish-release` runs `src/setup/publish_release.sh`. It requires a clean working
tree, fetches the deployment branch and tags, and verifies that the local version
matches the version on `origin/release`. It then uses `gh release create` with a
`vX.Y.Z` tag, the `release` target, and generated release notes. Publishing that
GitHub Release triggers the build -> TestPyPI -> install smoke test -> PyPI workflow
and provides the tagged source archive used by Bioconda.

Use `pixi run -e dev publish-release -- --dry-run` to perform the checks, including
GitHub CLI authentication, and print the `gh` command without creating a tag or release.

### Local fallback (manual upload)

If needed, manual upload with twine is still supported:
- Build: `pixi run -e dev python -m build --sdist --wheel --outdir dist`
- Check: `pixi run -e dev twine check dist/*`
- Upload: `pixi run -e dev twine upload dist/* --verbose`


## Example Workflow: Adding a New Command

1. **Check for existing utilities**: Search `src/rolypoly/utils/` for existing functions that might help (especially `utils/bio/` for sequence operations)

2. **Create the command file**: Add your command in the appropriate subdirectory under `src/rolypoly/commands/` (e.g., `commands/misc/my_command.py`)
   - Use `@click.command()` decorator
   - Follow naming conventions (short + long options, snake_case for parameters)
   - Import and reuse existing utilities from `utils/` where possible
   - You might reuse some of the defaults click.option decorators from `utils.cli_options.py`

3. **Add shared utilities if needed**: If you create reusable functions, place them in `src/rolypoly/utils/` (NOT in `commands/`)
   - Use existing modules when appropriate (e.g., `utils/bio/polars_fastx.py` for FASTA/FASTQ operations)

4. **Register the command**: **CRITICAL** - Add your command to `src/rolypoly/rolypoly.py` in the appropriate lazy_subcommands group
   - Format: `"command-name": "rolypoly.commands.subdir.my_command.my_command_function"`
   - The command won't appear in the CLI without this step!

5. **Test the command**:
   - Run `pixi run rolypoly <command-name> --help` to verify it loads
   - Test with actual data
   - Add test cases to `src/tests/` (ONLY if appropriate)

6. **Document**: Update help strings and consider adding examples to README or docs
   - Add a markdown file in the appropriate location under `docs/`
   - Update the `mkdocs.yml` configuration file to include your new documentation
   - Add to the index or relevant navigation section if needed

## **Note**
This project is governed under the LBNL IP office. By contributing, you agree that your contributions will be subject to the terms of the GPLv3 license.
