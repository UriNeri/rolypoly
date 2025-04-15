import rich_click as click
from rich.console import Console

console = Console()


class LazyGroup(click.RichGroup):
    """Click Group subclass that lazily loads commands to improve startup time for large CLI applications.

    Args:
        lazy_subcommands: Mapping of command names to either:
            - A string import path: "module.command-object-name"
            - A dict defining a command group: {
                "name": "Group Display Name",
                "commands": {
                    "cmd1": "module.command1",
                    "cmd2": "module.command2"
                }
            }
            Hidden commands can be specified by prefixing the import path with "hidden:"
    """

    def __init__(self, *args, lazy_subcommands=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lazy_subcommands = lazy_subcommands or {}
        self._command_groups = {}  # Store group info for help display
        self._init_command_groups()

    def _init_command_groups(self):
        """Initialize command groups from lazy_subcommands configuration."""
        for name, value in list(self.lazy_subcommands.items()):
            if isinstance(value, dict) and "name" in value and "commands" in value:
                # Store group info for help display
                self._command_groups[name] = {"name": value["name"], "commands": {}}

                # Add commands directly to main group but track their group
                for cmd_name, cmd_path in value["commands"].items():
                    if not cmd_path.startswith("hidden:"):
                        self.lazy_subcommands[cmd_name] = cmd_path
                        self._command_groups[name]["commands"][cmd_name] = cmd_path

                # Remove the group definition from lazy_subcommands
                del self.lazy_subcommands[name]

    def list_commands(self, ctx):
        """List all available commands, excluding hidden ones."""
        base = super().list_commands(ctx)
        # Only include lazy commands that aren't marked as hidden
        lazy = sorted(
            name
            for name, path in self.lazy_subcommands.items()
            if not path.startswith("hidden:")
        )
        return base + lazy

    def get_command(self, ctx, cmd_name):
        """Get a command by name, loading it if necessary."""
        if cmd_name in self.lazy_subcommands:
            try:
                return self._lazy_load(cmd_name)
            except (ImportError, AttributeError) as e:
                # If loading fails, remove the command from listings
                if not self.lazy_subcommands[cmd_name].startswith("hidden:"):
                    console.print(
                        f"[yellow]Warning:[/yellow] Failed to load command '{cmd_name}': {str(e)}"
                    )
                return None
        return super().get_command(ctx, cmd_name)

    def _lazy_load(self, cmd_name):
        """Load a command lazily from its module."""
        from importlib import import_module, util

        import_path = self.lazy_subcommands[cmd_name]
        # Remove the "hidden:" prefix if present
        if import_path.startswith("hidden:"):
            import_path = import_path[7:]

        try:
            modname, cmd_object_name = import_path.rsplit(".", 1)

            # Check if module exists first
            if util.find_spec(modname) is None:
                raise ImportError(f"Module '{modname}' not found")

            # Try to import the module
            mod = import_module(modname)

            # Try to get the command object
            if not hasattr(mod, cmd_object_name):
                raise AttributeError(
                    f"Module '{modname}' has no attribute '{cmd_object_name}'"
                )

            cmd_object = getattr(mod, cmd_object_name)

            # Verify it's a Click command
            if not isinstance(cmd_object, click.BaseCommand):
                raise ValueError(
                    f"Object '{cmd_object_name}' in module '{modname}' is not a Click command"
                )

            return cmd_object

        except Exception as e:
            if not import_path.startswith("hidden:"):
                console.print(
                    f"[red]Error loading command '{cmd_name}':[/red] {str(e)}"
                )
            raise

    def format_commands(self, ctx, formatter):
        """Custom command formatter to show grouped commands."""
        from rich.panel import Panel
        from rich.table import Table

        # First get all commands that aren't in groups
        ungrouped = []
        for cmd_name in self.list_commands(ctx):
            in_group = False
            for group in self._command_groups.values():
                if cmd_name in group["commands"]:
                    in_group = True
                    break
            if not in_group:
                cmd = self.get_command(ctx, cmd_name)
                if cmd is None or cmd.hidden:
                    continue
                ungrouped.append((cmd_name, cmd))

        # Create a table for each group
        for group_name, group_info in sorted(self._command_groups.items()):
            valid_commands = []
            # Get all commands for this group from the original lazy_subcommands mapping
            for cmd_name, cmd_path in group_info["commands"].items():
                if cmd_path.startswith("hidden:"):
                    continue
                cmd = self.get_command(ctx, cmd_name)
                if cmd is None:
                    continue
                # Get full help string without truncation
                help_str = cmd.get_short_help_str(limit=float("inf"))
                valid_commands.append((cmd_name, help_str))

            if valid_commands:  # Only show groups that have valid commands
                table = Table(
                    show_header=False, box=None, padding=(0, 2), show_edge=False
                )
                table.add_column("Command", style="bold cyan", width=20)
                table.add_column("Description", no_wrap=False)  # Allow text wrapping

                for cmd_name, help_str in sorted(valid_commands):
                    table.add_row(cmd_name, help_str or "")

                # Create a panel for the group
                panel = Panel(
                    table,
                    title=f"[bold]{group_info['name']}[/bold]",
                    title_align="left",
                    border_style="dim",
                )
                console.print(panel)
