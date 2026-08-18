"""Canonical options automatically attached to RolyPoly CLI commands."""

import inspect
import re
from pathlib import Path

import rich_click as click

from rolypoly.utils.logging.config import DEFAULT_MEMORY


class MemoryType(click.ParamType):
    """Validate memory values and normalize units such as ``8GB`` to ``8g``."""

    name = "memory"

    def convert(self, value, param, ctx):
        """Return a canonical lower-case memory value."""
        if value is None:
            return None

        normalized = str(value).strip().lower().replace(" ", "")
        match = re.fullmatch(r"(\d+)([kmgt]?)(?:b)?", normalized)
        if match is None:
            self.fail(
                "expected NUMBER optionally followed by K, M, G, or T",
                param,
                ctx,
            )

        number, unit = match.groups()
        return f"{number}{unit}"


MEMORY = MemoryType()

SHARED_OPTION_SPECS: dict[str, tuple[tuple[str, ...], dict[str, object]]] = {
    "threads": (
        ("-t", "--threads"),
        {
            "type": click.IntRange(min=1),
            "default": 1,
            "help": "Number of worker threads.",
        },
    ),
    "memory": (
        ("-M", "--memory"),
        {
            "type": MEMORY,
            "default": DEFAULT_MEMORY,
            "help": "Memory limit, for example 8g.",
        },
    ),
    "keep_tmp": (
        ("-k", "--keep-tmp"),
        {"is_flag": True, "default": False, "help": "Keep temporary files."},
    ),
    "temp_dir": (
        ("-tmp", "--temp-dir"),
        {
            "type": click.Path(file_okay=False, path_type=Path),
            "default": None,
            "help": "Temporary working directory.",
        },
    ),
    "log_file": (
        ("-g", "--log-file"),
        {
            "type": click.Path(dir_okay=False, path_type=Path),
            "default": "rolypoly.log",
            "help": "Path to the log file.",
        },
    ),
    "log_level": (
        ("-ll", "--log-level"),
        {
            "type": click.Choice(
                ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                case_sensitive=False,
            ),
            "default": "INFO",
            "hidden": True,
            "help": "Log level.",
        },
    ),
}

LEGACY_PARAM_DECLS: dict[tuple[str, str], tuple[str, ...]] = {
    ("annotate-rna", "log_level"): ("-l", "-ll", "--log-level"),
    ("filter-reads", "memory"): ("-M", "-mem", "--memory"),
    ("marker-search", "memory"): ("-m", "-M", "--memory"),
    ("marker-search", "temp_dir"): ("-td", "-tempdir", "-tmp", "--temp-dir"),
    ("rdrp-motif-search", "temp_dir"): ("-td", "-tmp", "--temp-dir"),
    ("report", "log_file"): ("-lf", "-g", "--log-file"),
}


def add_shared_options(command: click.Command) -> click.Command:
    """Attach applicable shared options based on the callback's parameters.

    Existing parameters win, allowing an exceptional command to declare a
    command-specific option explicitly without receiving a duplicate.
    """
    if command.callback is None:
        return command

    callback_parameters = inspect.signature(command.callback).parameters
    existing_parameters = {parameter.name for parameter in command.params}

    for parameter_name, (
        param_decls,
        attributes,
    ) in SHARED_OPTION_SPECS.items():
        if (
            parameter_name in callback_parameters
            and parameter_name not in existing_parameters
        ):
            declarations = LEGACY_PARAM_DECLS.get(
                (command.name or "", parameter_name), param_decls
            )
            command.params.append(click.Option(declarations, **attributes))

    return command


def shared_command_context(
    command: click.Command, **context_kwargs: object
) -> click.Context:
    """Create a context for an in-process command with shared defaults attached.

    Direct ``Context.invoke`` calls bypass :class:`LazyGroup`, so workflow
    commands must use this helper to receive the same shared option defaults as
    commands invoked from the top-level CLI.
    """
    return click.Context(add_shared_options(command), **context_kwargs)
