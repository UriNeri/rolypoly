# Test suite layout

Use the lowest-cost test that expresses the stable contract.

- `cli_scenarios.json` contains black-box CLI workflow scenarios. These should exercise command-line behavior, expected files, table schemas, row counts, and small content checks. Keep monkeypatching and direct Python setup out of this file.
- `test_cli_contracts.py` is only the runner for `cli_scenarios.json` and its shared scenario helpers.
- `test_cli_help_smoke.py` covers command wiring and help output. Do not add help-only scenarios to `cli_scenarios.json` unless they assert command-specific behavior that the generic help smoke cannot cover.
- `regressions/` contains focused Python regression tests for bugs likely to recur. Monkeypatches and small synthetic fixtures are acceptable here when they isolate an external tool or cached pipeline state.
- Other focused `test_*.py` modules cover stable scientific transformations, backend adapters, report output, and shared command policy. Avoid tests for internal helpers unless the behavior is part of a user-facing or scientific contract.

Keep fixtures synthetic, deterministic, and as small as practical. Put one-off audits and migration checks in `/tmp` and record the command/result in the change instead of committing them.
