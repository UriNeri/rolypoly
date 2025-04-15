from pathlib import Path

import rich_click as click
from rich.console import Console

console = Console()


@click.command()
@click.option(
    "--package-src",
    default=lambda: str(Path(__file__).parents[4]),
    help="The source location of the package",
)
@click.option("--category", help="Command category (subfolder in commands/)")
@click.option("--name", help="Command name")
@click.option(
    "--common-args",
    default="all",
    help='Common arguments to include (comma-separated: threads,input,output,memory,logfile,loglevel or "all")',
)
@click.option(
    "--packages", default="all", help='Packages to import (comma-separated or "all")'
)
@click.option("--description", help="Command description")
@click.option(
    "--args-json",
    help='JSON string of custom arguments in format [{"name": "arg1", "type": "str", "required": true}, ...]',
)
@click.option(
    "--interactive/--no-interactive",
    default=True,
    help="Whether to run in interactive mode if CLI args are missing",
)
def add_command(
    package_src,
    category,
    name,
    common_args,
    packages,
    description,
    args_json,
    interactive,
):
    """Interactive utility to create new RolyPoly commands [dev only].

    This tool guides developers through creating a new command by prompting for
    various details and generating the appropriate command file structure.

    Can be run either interactively or with CLI arguments.
    """
    import json

    package_src = Path(package_src).resolve()
    commands_dir = package_src / "src" / "rolypoly" / "commands"

    if not commands_dir.exists():
        console.print(f"[red]Commands directory {commands_dir} does not exist.[/red]")
        return

    # If any required argument is missing and interactive mode is enabled, fall back to prompts
    if interactive and (not category or not name or not description):
        if not category:
            category = click.prompt(
                'Enter the command category (subfolder of "src/rolypoly/commands/")'
            )
        if not name:
            name = click.prompt("Enter the command name")
        if not description:
            description = click.prompt("Enter a short description for the command")

    # If not interactive and missing required args, fail
    if not all([category, name, description]):
        console.print(
            "[red]Error: category, name, and description are required in non-interactive mode[/red]"
        )
        return

    args = []
    if args_json:
        try:
            args = [
                (arg["name"], arg["type"], arg["required"])
                for arg in json.loads(args_json)
            ]
        except json.JSONDecodeError:
            console.print("[red]Error: Invalid JSON format for args[/red]")
            return
    elif interactive:
        while True:
            arg_name = click.prompt(
                'Enter argument name (or "stoparg" to finish)', default="stoparg"
            )
            if arg_name == "stoparg":
                break
            arg_type = click.prompt(
                f'Enter type for argument "{arg_name}" (e.g., str, int, bool)'
            )
            arg_required = click.confirm(f'Is argument "{arg_name}" mandatory?')
            args.append((arg_name, arg_type, arg_required))

    console.print(f"[green]Command category: {category}[/green]")
    console.print(f"[green]Command name: {name}[/green]")
    console.print(f"[green]Common arguments: {common_args}[/green]")
    console.print(f"[green]Packages: {packages}[/green]")

    command_dir = commands_dir / category
    command_dir.mkdir(parents=True, exist_ok=True)

    command_file = command_dir / f"{name}.py"
    with open(command_file, "w") as f:
        # Write imports
        f.write(f"""import rich_click as click
from rolypoly.utils.loggit import setup_logging
""")
        if packages == "all":
            f.write("""
from rolypoly.utils.citation_reminder import remind_citations
from rolypoly.utils.various import ensure_memory
from rolypoly.utils.loggit import log_start_info
from rolypoly.utils.config import BaseConfig
""")
        else:
            for pkg in packages.split(","):
                if "rich_click" in pkg:
                    f.write("import rich_click as click\n")
                if "remind_citations" in pkg:
                    f.write(
                        "from rolypoly.utils.citation_reminder import remind_citations\n"
                    )
                if "ensure_memory" in pkg:
                    f.write("from rolypoly.utils.various import ensure_memory\n")
                if "log_start_info" in pkg:
                    f.write("from rolypoly.utils.loggit import log_start_info\n")
                if "BaseConfig" in pkg:
                    f.write("from rolypoly.utils.config import BaseConfig\n")

        f.write("\n@click.command()\n")
        if common_args == "all":
            f.write("""@click.option('-i', '--input', required=True, help='Input file or directory')
@click.option('-o', '--output', default='output', help='Output directory')
@click.option('-t', '--threads', default=1, help='Number of threads')
@click.option('-M', '--memory', default='6g', help='Memory allocation')
@click.option('--log-file', default='command.log', help='Path to log file')
@click.option('--log-level', default='INFO', help='Log level')
""")
        else:
            for arg in common_args.split(","):
                if "input" in arg:
                    f.write(
                        "@click.option('-i', '--input', required=True, help='Input file or directory')\n"
                    )
                if "output" in arg:
                    f.write(
                        "@click.option('-o', '--output', default='output', help='Output directory')\n"
                    )
                if "threads" in arg:
                    f.write(
                        "@click.option('-t', '--threads', default=1, help='Number of threads')\n"
                    )
                if "memory" in arg:
                    f.write(
                        "@click.option('-M', '--memory', default='6g', help='Memory allocation')\n"
                    )
                if "logfile" in arg:
                    f.write(
                        "@click.option('--log-file', default='command.log', help='Path to log file')\n"
                    )
                if "loglevel" in arg:
                    f.write(
                        "@click.option('--log-level', default='INFO', help='Log level')\n"
                    )

        for arg_name, arg_type, arg_required in args:
            f.write(
                f"@click.option('--{arg_name}', {'required=True' if arg_required else 'default=None'}, type={arg_type}, help='{arg_name}')\n"
            )

        f.write(f"""def {name}(input, output, threads, memory, log_file, log_level, {", ".join([arg[0] for arg in args])}):
    \"\"\"
    {description}
    \"\"\"
    logger = setup_logging(log_file)
    logger.info(f"Starting {name} with input: {{input}}, output: {{output}}, threads: {{threads}}, memory: {{memory}}, log_level: {{log_level}}")

    # Add the main logic of the command here
    #     

    logger.info("{name} completed successfully!")

if __name__ == "__main__":
    {name}()
""")

    main_script = package_src / "src" / "rolypoly" / "rolypoly.py"
    with open(main_script, "r") as f:
        lines = f.readlines()

    # Find the appropriate section to add the command
    for i, line in enumerate(lines):
        if "lazy_subcommands" in line:
            # Find the appropriate category or create a new one
            category_found = False
            for j in range(i + 1, len(lines)):
                if f'"{category}"' in lines[j]:
                    category_found = True
                    # Find the end of the category's commands
                    for k in range(j + 1, len(lines)):
                        if "}" in lines[k]:
                            lines.insert(
                                k,
                                f'            "{name}": "rolypoly.commands.{category}.{name}.{name}",\n',
                            )
                            break
                    break
            if not category_found:
                # Add new category
                lines.insert(i + 1, f'    "{category}": {{\n')
                lines.insert(i + 2, f'        "name": "{category.capitalize()}",\n')
                lines.insert(i + 3, f'        "commands": {{\n')
                lines.insert(
                    i + 4,
                    f'            "{name}": "rolypoly.commands.{category}.{name}.{name}"\n',
                )
                lines.insert(i + 5, f"        }}\n")
                lines.insert(i + 6, f"    }},\n")
            break

    with open(main_script, "w") as f:
        f.writelines(lines)

    click.echo(
        f"Command {name} created successfully in {command_file} and added to {main_script}."
    )


if __name__ == "__main__":
    add_command()
