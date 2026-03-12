from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

import click

from rolypoly.rolypoly import rolypoly


DEFAULT_TEMPLATE = """# __TITLE__

<!-- Auto-generated draft from CLI metadata for `rolypoly __COMMAND__`. -->
<!-- Expand this page with command-specific context, examples, and citations. -->

## Summary

__SUMMARY__

## Description

__DESCRIPTION__

## Usage

```bash
rolypoly __COMMAND__ [OPTIONS]
```

## Options

__OPTIONS_MD__

__EPILOG_MD__

__PINNED_MD__
"""

SKIP_COMMANDS = {"help"}
AUTO_GENERATED_HEADER_PATTERN = re.compile(
    r"Auto-generated draft from CLI metadata for `rolypoly ([a-z0-9][a-z0-9-]*)`\."
)
CONTRIBUTING_SOURCE = Path("CONTRIBUTING.md")
CONTRIBUTING_DOCS_TARGET = Path("docs/mkdocs_docs/contribute.md")


def clean_click_text(text: str) -> str:
    """Normalize Click help text for markdown rendering."""
    normalized = text.replace("\b", "")
    normalized = textwrap.dedent(normalized)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.splitlines())
    normalized = re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", "", normalized)
    return normalized.strip()


def strip_docstring_sections(description: str) -> str:
    """Remove structured docstring sections that duplicate CLI options.

    Keeps the high-level description while dropping sections such as Args,
    Returns, Raises, Parameters, and Examples that tend to repeat option-level
    details already rendered from Click metadata.
    """
    if not description:
        return description

    section_headers = {
        "args:",
        "arguments:",
        "parameters:",
        "params:",
        "returns:",
        "return:",
        "raises:",
    }

    kept_lines: list[str] = []
    for line in description.splitlines():
        normalized = line.strip().lower()
        if normalized in section_headers:
            break
        kept_lines.append(line)

    cleaned = "\n".join(kept_lines).rstrip()
    return cleaned or description


def dedupe_summary_from_description(summary: str, description: str) -> str:
    """Remove duplicated lead sentence when it matches the summary."""
    if not summary or not description:
        return description

    lines = description.splitlines()
    if not lines:
        return description

    first_line = lines[0].strip()
    summary_norm = re.sub(r"\s+", " ", summary.strip()).lower()
    first_line_norm = re.sub(r"\s+", " ", first_line).lower()

    if first_line_norm != summary_norm:
        return description

    trimmed_lines = lines[1:]
    while trimmed_lines and not trimmed_lines[0].strip():
        trimmed_lines.pop(0)

    trimmed = "\n".join(trimmed_lines).strip()
    return trimmed or description


def command_to_title(command_name: str) -> str:
    """Convert a CLI command name to a simple title."""
    return command_name.replace("-", " ").title()


def get_click_command(command_name: str) -> click.Command | None:
    """Return Click command object by name."""
    context = click.Context(rolypoly)
    command = rolypoly.get_command(context, command_name)
    return command


def option_display_names(parameter: click.Option) -> str:
    """Format option names for markdown output."""
    names = list(parameter.opts) + list(parameter.secondary_opts)
    return ", ".join(f"`{name}`" for name in names)


def option_type_text(parameter: click.Option) -> str:
    """Format option type for markdown output."""
    type_name = getattr(parameter.type, "name", None)
    if type_name:
        return type_name.upper()
    return "VALUE"


def option_default_text(
    command: click.Command, parameter: click.Option
) -> str | None:
    """Format default value text for markdown output.

    Resolves callable defaults (for example lambda-based cwd paths) when
    possible, and includes defaults even when Click's show_default is False.
    """
    default_value = None

    context = click.Context(command)
    try:
        default_value = parameter.get_default(context, call=True)
    except Exception:
        default_value = parameter.default

    if callable(default_value):
        return None

    if default_value is None:
        return None

    if isinstance(default_value, tuple):
        joined = ", ".join(str(item) for item in default_value)
        return joined

    if isinstance(default_value, list):
        joined = ", ".join(str(item) for item in default_value)
        return joined

    return str(default_value)


def option_help_text(parameter: click.Option) -> str:
    """Get clean help text for an option."""
    raw_help = parameter.help or "No description provided."
    compact = re.sub(r"\s+", " ", clean_click_text(raw_help)).strip()
    return compact


def render_options_markdown(command: click.Command) -> str:
    """Render markdown bullet list from Click options."""
    option_lines: list[str] = []
    for parameter in command.params:
        if not isinstance(parameter, click.Option):
            continue
        if getattr(parameter, "hidden", False):
            continue

        names = option_display_names(parameter)
        type_text = option_type_text(parameter)
        description = option_help_text(parameter)

        tags: list[str] = [f"type: `{type_text}`"]
        if parameter.required:
            tags.append("required")

        default_text = option_default_text(command, parameter)
        if default_text is not None:
            tags.append(f"default: `{default_text}`")

        tag_text = "; ".join(tags)
        option_lines.append(f"- {names}: {description} ({tag_text})")

    if not option_lines:
        return "No documented CLI options found."

    return "\n".join(option_lines)


def render_epilog_markdown(command: click.Command) -> str:
    """Render epilog section if present."""
    epilog = clean_click_text(command.epilog or "")
    if not epilog:
        return ""
    return f"## Additional Notes\n\n```text\n{epilog}\n```"


def load_pinned_markdown(scaffold_dir: Path, command_name: str) -> str:
    """Load optional pinned markdown scaffold for a command."""
    scaffold_path = scaffold_dir / f"{command_name.replace('-', '_')}.md"
    if not scaffold_path.exists():
        return ""

    pinned = scaffold_path.read_text(encoding="utf-8").strip()
    if not pinned:
        return ""

    return f"## Pinned Sections\n\n{pinned}"


def list_registered_commands() -> list[str]:
    """List command names registered on the top-level rolypoly CLI."""
    context = click.Context(rolypoly)
    return sorted(
        command_name
        for command_name in set(rolypoly.list_commands(context))
        if command_name not in SKIP_COMMANDS
    )


def discover_documented_commands(
    docs_commands_dir: Path, command_names: list[str], template_name: str
) -> dict[str, set[Path]]:
    """Find which commands are already mentioned in docs pages.

    A command is considered documented if one of these appears in a markdown page:
    - `rolypoly <command>`
    - `<command>` in backticks
    """
    command_set = set(command_names)
    documented: dict[str, set[Path]] = {}

    for markdown_file in sorted(docs_commands_dir.glob("*.md")):
        if markdown_file.name == template_name:
            continue

        content = markdown_file.read_text(encoding="utf-8")

        rolypoly_refs = set(re.findall(r"rolypoly\s+([a-z0-9][a-z0-9-]*)", content))
        backtick_refs = set(re.findall(r"`([a-z0-9][a-z0-9-]*)`", content))

        mentioned = (rolypoly_refs | backtick_refs) & command_set
        for command_name in mentioned:
            documented.setdefault(command_name, set()).add(markdown_file)

    return documented


def discover_auto_generated_pages(docs_commands_dir: Path) -> dict[str, Path]:
    """Find auto-generated command pages and map command -> file path."""
    discovered: dict[str, Path] = {}

    for markdown_file in sorted(docs_commands_dir.glob("*.md")):
        content = markdown_file.read_text(encoding="utf-8")
        match = AUTO_GENERATED_HEADER_PATTERN.search(content)
        if not match:
            continue

        command_name = match.group(1)
        discovered[command_name] = markdown_file

    return discovered


def sync_contributing_docs_page(repo_root: Path, dry_run: bool) -> bool:
    """Sync docs contributing page from root CONTRIBUTING.md.

    Returns True when target would be updated (or was updated), else False.
    """
    source_path = repo_root / CONTRIBUTING_SOURCE
    target_path = repo_root / CONTRIBUTING_DOCS_TARGET

    if not source_path.exists():
        raise FileNotFoundError(f"Contributing source not found: {source_path}")

    source_content = source_path.read_text(encoding="utf-8")
    target_content = (
        target_path.read_text(encoding="utf-8") if target_path.exists() else ""
    )

    if source_content == target_content:
        return False

    if not dry_run:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(source_content, encoding="utf-8")

    return True


def render_page(
    template_text: str,
    command_name: str,
    summary: str,
    description: str,
    options_md: str,
    epilog_md: str,
    pinned_md: str,
) -> str:
    """Render a docs page from template placeholders."""
    deduped_description = dedupe_summary_from_description(summary, description)
    return (
        template_text.replace("__TITLE__", command_to_title(command_name))
        .replace("__COMMAND__", command_name)
        .replace("__SUMMARY__", summary)
        .replace("__DESCRIPTION__", deduped_description)
        .replace("__OPTIONS_MD__", options_md)
        .replace("__EPILOG_MD__", epilog_md)
        .replace("__PINNED_MD__", pinned_md)
    )


def get_command_summary(command_name: str) -> str:
    """Get short help text from click command object."""
    command = get_click_command(command_name)
    if command is None:
        return "Auto-generated command help page."
    summary = command.get_short_help_str(limit=float("inf")).strip()
    return summary or "Auto-generated command help page."


def get_command_description(command_name: str) -> str:
    """Get command description from click command help/docstring."""
    command = get_click_command(command_name)
    if command is None:
        return "Auto-generated command help page."

    description = clean_click_text(command.help or "")
    description = strip_docstring_sections(description)
    if description:
        return description

    return "Auto-generated command help page."


def ensure_docs_pages(
    docs_commands_dir: Path,
    scaffold_dir: Path,
    template_text: str,
    missing_commands: list[str],
    output_paths_by_command: dict[str, Path],
    overwrite: bool,
    dry_run: bool,
) -> list[Path]:
    """Create markdown pages for missing commands and return created paths."""
    created: list[Path] = []

    for command_name in missing_commands:
        output_file = output_paths_by_command.get(
            command_name,
            docs_commands_dir / f"{command_name.replace('-', '_')}.md",
        )
        if output_file.exists() and not overwrite:
            continue

        command = get_click_command(command_name)
        if command is None:
            continue

        summary = get_command_summary(command_name)
        description = get_command_description(command_name)
        options_md = render_options_markdown(command)
        epilog_md = render_epilog_markdown(command)
        pinned_md = load_pinned_markdown(scaffold_dir, command_name)
        page_content = render_page(
            template_text,
            command_name,
            summary,
            description,
            options_md,
            epilog_md,
            pinned_md,
        )

        if not dry_run:
            output_file.write_text(page_content, encoding="utf-8")
        created.append(output_file)

    return created


def print_report(
    command_names: list[str],
    documented_map: dict[str, set[Path]],
    missing_commands: list[str],
) -> None:
    """Print command coverage report."""
    print(f"Registered commands in rolypoly.py: {len(command_names)}")
    print(f"Commands with docs mentions: {len(documented_map)}")
    print(f"Commands missing docs pages/mentions: {len(missing_commands)}")
    print()

    print("Documented commands:")
    for command_name in sorted(documented_map):
        files = ", ".join(sorted(path.name for path in documented_map[command_name]))
        print(f"  - {command_name}: {files}")
    print()

    if missing_commands:
        print("Missing commands:")
        for command_name in missing_commands:
            print(f"  - {command_name}")
    else:
        print("Missing commands: none")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze command docs coverage and export missing `rolypoly <command> --help` "
            "outputs to markdown command pages."
        )
    )
    parser.add_argument(
        "--docs-commands-dir",
        default="docs/mkdocs_docs/commands",
        help="Path to docs commands directory",
    )
    parser.add_argument(
        "--template",
        default="src/setup/help_export_template.md",
        help="Markdown template file path",
    )
    parser.add_argument(
        "--scaffold-dir",
        default="src/setup/help_export_scaffolds",
        help="Directory with optional per-command pinned markdown scaffolds",
    )
    parser.add_argument(
        "--commands",
        default="",
        help=(
            "Comma-separated command names to process. "
            "When omitted: default is all missing commands, or all existing "
            "auto-generated command pages if --overwrite is set."
        ),
    )
    parser.add_argument(
        "--all-commands",
        action="store_true",
        help="Process all registered CLI commands (create/update <command>.md pages)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing generated output files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and report, but do not write any files",
    )
    parser.add_argument(
        "--no-sync-contributing",
        action="store_true",
        help="Do not sync docs/mkdocs_docs/contribute.md from root CONTRIBUTING.md",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for docs help exporter."""
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    docs_commands_dir = repo_root / args.docs_commands_dir
    template_path = repo_root / args.template
    scaffold_dir = repo_root / args.scaffold_dir

    if not docs_commands_dir.exists():
        raise FileNotFoundError(f"Docs commands directory not found: {docs_commands_dir}")

    template_text = (
        template_path.read_text(encoding="utf-8")
        if template_path.exists()
        else DEFAULT_TEMPLATE
    )

    command_names = list_registered_commands()
    documented_map = discover_documented_commands(
        docs_commands_dir, command_names, template_path.name
    )
    auto_generated_pages = discover_auto_generated_pages(docs_commands_dir)

    missing_commands = sorted(
        command_name for command_name in command_names if command_name not in documented_map
    )

    target_commands: list[str]
    if args.all_commands:
        target_commands = list(command_names)
    elif args.commands.strip():
        requested = {item.strip() for item in args.commands.split(",") if item.strip()}
        target_commands = [command_name for command_name in command_names if command_name in requested]
    elif args.overwrite:
        target_commands = [
            command_name
            for command_name in command_names
            if command_name in auto_generated_pages
        ]
    else:
        target_commands = list(missing_commands)

    output_paths_by_command = {
        command_name: auto_generated_pages.get(
            command_name,
            docs_commands_dir / f"{command_name.replace('-', '_')}.md",
        )
        for command_name in target_commands
    }

    print_report(command_names, documented_map, missing_commands)

    contributing_sync_changed = False
    if not args.no_sync_contributing:
        contributing_sync_changed = sync_contributing_docs_page(
            repo_root=repo_root, dry_run=args.dry_run
        )

        if contributing_sync_changed:
            status_text = "would be synced" if args.dry_run else "synced"
            print(
                f"\nContributing docs page {status_text}: "
                f"{CONTRIBUTING_DOCS_TARGET} <- {CONTRIBUTING_SOURCE}"
            )

    if args.commands.strip() or args.all_commands or args.overwrite:
        print()
        print("Requested commands to export:")
        for command_name in target_commands:
            print(f"  - {command_name}")

    if args.dry_run:
        print("\nDry-run mode: no files were written.")
        return

    created = ensure_docs_pages(
        docs_commands_dir=docs_commands_dir,
        scaffold_dir=scaffold_dir,
        template_text=template_text,
        missing_commands=target_commands,
        output_paths_by_command=output_paths_by_command,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    if created:
        print("\nCreated/updated pages:")
        for path in created:
            print(f"  - {path.relative_to(repo_root)}")
    elif not contributing_sync_changed:
        print("\nNo pages created (nothing missing, or files already existed).")


if __name__ == "__main__":
    main()
