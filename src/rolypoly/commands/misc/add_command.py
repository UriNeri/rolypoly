import os
import rich_click as click
from rich.console import Console

console = Console()

@click.command()
@click.option('--package-src', prompt='Package source location', default=os.path.dirname(__file__).split('/src/rolypoly/utils')[0], help='The source location of the package')
def add_command(package_src):
    """Interactive utility to create new RolyPoly commands [dev only].

    This tool guides developers through creating a new command by prompting for
    various details and generating the appropriate command file structure.

    Args:
        package_src (str): Root directory of the RolyPoly package. Defaults to
            the parent directory of the current file.

    The tool will interactively prompt for:
        1. Command category (subfolder in commands/)
        2. Command name
        3. Common arguments to include (threads, input, output, etc.)
        4. Required package imports
        5. Command-specific arguments with types and requirements
        6. Command description

    Generated files include:
        - Command implementation in appropriate category subfolder
        - Click command decorators with specified options
        - Required imports and boilerplate code
        - Basic command structure with logging setup

    Example:
             # Run interactively
             add_command()
             # Or specify package source directly
             add_command(package_src="/path/to/rolypoly")

    Note:
        This is a development tool and should not be used in production.
        The generated command will need additional implementation details.
    """
    if not os.path.exists(package_src):
        console.print(f"[red]Package source location {package_src} does not exist.[/red]")
        return
    package_src = os.path.abspath(package_src)
    
    commands_dir = os.path.join(package_src, 'src', 'rolypoly', 'commands')
    
    if not os.path.exists(commands_dir):
        console.print(f"[red]Commands directory {commands_dir} does not exist.[/red]")
        return

    command_category = click.prompt('Enter the command category (subfolder of "src/rolypoly/commands/")')
    command_name = click.prompt('Enter the command name')
    # command_config = click.prompt('Enter the command config class name (or "None" if not needed, or "BaseConfig" if it inherits from BaseConfig)')
    common_args = click.prompt('Enter common arguments to use (all or combination of: threads, input, output, memory, logfile, loglevel)', default='all')
    packages = click.prompt('Enter packages to import (all or combination of: rich-click, remind_citations, ensure_memory, log_start_info, BaseConfig)', default='all')
    
    console.print(f"[green]Command category: {command_category}[/green]")
    console.print(f"[green]Command name: {command_name}[/green]")
    console.print(f"[green]Common arguments: {common_args}[/green]")
    console.print(f"[green]Packages: {packages}[/green]")
    
    args = []
    while True:
        arg_name = click.prompt('Enter argument name (or "stoparg" to finish)', default='stoparg')
        if arg_name == 'stoparg':
            break
        arg_type = click.prompt(f'Enter type for argument "{arg_name}" (e.g., str, int, bool)')
        arg_required = click.confirm(f'Is argument "{arg_name}" mandatory?')
        args.append((arg_name, arg_type, arg_required))
    
    description = click.prompt('Enter a short description for the command')

    command_dir = os.path.join(commands_dir, command_category)
    os.makedirs(command_dir, exist_ok=True)
    
    command_file = os.path.join(command_dir, f"{command_name}.py")
    with open(command_file, 'w') as f:
        # Write imports
        f.write(f"""import click
from rolypoly.utils.loggit import setup_logging
""")
        if packages == 'all':
            f.write("""

from rolypoly.utils.citation_reminder import remind_citations
from rolypoly.utils.various import ensure_memory
from rolypoly.utils.loggit import log_start_info
from rolypoly.utils.config import BaseConfig
""")
        else:
            if 'rich_click' in packages:
                f.write("import rich_click as click\n")
            if 'remind_citations' in packages:
                f.write("from rolypoly.utils.citation_reminder import remind_citations\n")
            if 'ensure_memory' in packages:
                f.write("from rolypoly.utils.various import ensure_memory\n")
            if 'log_start_info' in packages:
                f.write("from rolypoly.utils.loggit import log_start_info\n")
            if 'BaseConfig' in packages:
                f.write("from rolypoly.utils.config import BaseConfig\n")

        f.write("\n@click.command()\n")
        if common_args == 'all':
            f.write("""@click.option('-i', '--input', required=True, help='Input file or directory')
@click.option('-o', '--output', default='output', help='Output directory')
@click.option('-t', '--threads', default=1, help='Number of threads')
@click.option('-M', '--memory', default='6g', help='Memory allocation')
@click.option('--log-file', default='command.log', help='Path to log file')
@click.option('--log-level', default='INFO', help='Log level')
""")
        else:
            if 'input' in common_args:
                f.write("@click.option('-i', '--input', required=True, help='Input file or directory')\n")
            if 'output' in common_args:
                f.write("@click.option('-o', '--output', default='output', help='Output directory')\n")
            if 'threads' in common_args:
                f.write("@click.option('-t', '--threads', default=1, help='Number of threads')\n")
            if 'memory' in common_args:
                f.write("@click.option('-M', '--memory', default='6g', help='Memory allocation')\n")
            if 'logfile' in common_args:
                f.write("@click.option('--log-file', default='command.log', help='Path to log file')\n")
            if 'loglevel' in common_args:
                f.write("@click.option('--log-level', default='INFO', help='Log level')\n")
        
        for arg_name, arg_type, arg_required in args:
            f.write(f"@click.option('--{arg_name}', {'required=True' if arg_required else 'default=None'}, type={arg_type}, help='{arg_name}')\n")
        
        f.write(f"""def {command_name}(input, output, threads, memory, log_file, log_level, {" ,".join([arg[0] for arg in args])}):
    \"\"\"
    {description}
    \"\"\"
    logger = setup_logging(log_file)
    logger.info(f"Starting {command_name} with input: {{input}}, output: {{output}}, threads: {{threads}}, memory: {{memory}}, log_level: {{log_level}}")

    # Add the main logic of the command here
    #     

    logger.info("{command_name} completed successfully!")

if __name__ == "__main__":
    {command_name}()
""")
    
    main_script = os.path.join(package_src, 'src', 'rolypoly', 'rolypoly.py')
    #  look for the lazy_subcommands section, insert the new command after it in the format of :                                "assembly":"rolypoly.commands.assembly.assembly.assembly",
    with open(main_script, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if "lazy_subcommands" in line:
            break
    lines.insert(lines.index(line) + 1, f"                                \"{command_name}\": \"rolypoly.commands.{command_category}.{command_name}.{command_name}\",\n")
    with open(main_script, 'w') as f:
        f.writelines(lines)

    click.echo(f"Command {command_name} created successfully in {command_file} and added to {main_script}.")

if __name__ == "__main__":
    add_command()