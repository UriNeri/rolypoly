from __future__ import annotations

import importlib
import logging
from pathlib import Path

import click

from rolypoly.rolypoly import rolypoly
from rolypoly.utils.logging import loggit
from rolypoly.utils.logging.loggit import setup_logging


def clear_rolypoly_handlers() -> logging.Logger:
    logger = logging.getLogger()
    for handler in list(logger.handlers):
        if getattr(handler, loggit.ROLYPOLY_HANDLER_ATTR, None) in {
            "console",
            "file",
        }:
            logger.removeHandler(handler)
            handler.close()
    return logger


def get_rolypoly_handlers(logger: logging.Logger) -> tuple[list, list]:
    console_handlers = []
    file_handlers = []
    for handler in logger.handlers:
        handler_type = getattr(handler, loggit.ROLYPOLY_HANDLER_ATTR, None)
        if handler_type == "console":
            console_handlers.append(handler)
        elif handler_type == "file":
            file_handlers.append(handler)
    return console_handlers, file_handlers


def test_setup_logging_reconfigures_without_duplicate_handlers(
    tmp_path: Path,
) -> None:
    root_logger = clear_rolypoly_handlers()

    first_log = tmp_path / "first.log"
    second_log = tmp_path / "second.log"

    setup_logging(first_log, "INFO")
    setup_logging(second_log, "DEBUG")

    console_handlers, file_handlers = get_rolypoly_handlers(root_logger)

    assert len(console_handlers) == 1
    assert len(file_handlers) == 1
    assert root_logger.level == logging.DEBUG
    assert file_handlers[0].level == logging.DEBUG
    assert Path(file_handlers[0].baseFilename).resolve() == second_log.resolve()


def test_fetch_sra_import_does_not_setup_handlers() -> None:
    root_logger = clear_rolypoly_handlers()

    module = importlib.import_module(
        "rolypoly.commands.misc.fetch_sra_fastq"
    )
    importlib.reload(module)

    console_handlers, file_handlers = get_rolypoly_handlers(root_logger)
    assert len(console_handlers) == 0
    assert len(file_handlers) == 0


def test_all_registered_commands_support_log_level_option() -> None:
    ctx = click.Context(rolypoly)
    command_names = sorted(set(rolypoly.list_commands(ctx)))

    missing = []
    for command_name in command_names:
        command = rolypoly.get_command(ctx, command_name)
        if command is None:
            missing.append(command_name)
            continue

        has_log_level = False
        for parameter in command.params:
            if not isinstance(parameter, click.Option):
                continue
            if "--log-level" in parameter.opts or "-ll" in parameter.opts:
                has_log_level = True
                break

        if not has_log_level:
            missing.append(command_name)

    assert not missing, (
        "Commands missing --log-level/-ll option: " + ", ".join(missing)
    )
