def pytest_addoption(parser):
    group = parser.getgroup("rolypoly-cli")
    group.addoption(
        "--cli-scenarios",
        action="store",
        default="",
        help="Comma-separated scenario ids to run from cli_scenarios.json",
    )
    group.addoption(
        "--cli-commands",
        action="store",
        default="",
        help="Comma-separated top-level command names to run (e.g. annotate,assemble)",
    )
    group.addoption(
        "--cli-match",
        action="store",
        default="",
        help="Comma-separated substrings to match in scenario id/description/command",
    )
