"""Lazy loading functionality for (Rich)Click command groups.

LazyGroup class is a subclass of Click's Group that lazily loads commands. Adapted from  https://click.palletsprojects.com/en/stable/complex/ and other sources.
This is only to improve startup time. Tl;dr: only import subcommands when they are needed.

Example:
    ```python
    @click.group(cls=LazyGroup, lazy_subcommands={
        "cmd1": "package.module.cmd1",
        "cmd2": "package.module.cmd2"
    })
    def cli():
        pass
    ```
"""

import importlib
import rich_click as click
from rich.console import Console
import pathlib as pt
console = Console()

class LazyGroup(click.RichGroup):
    """Click Group subclass that lazily loads commands.

    This class extends Click's Group to provide lazy loading of subcommands,
    importing them only when needed. improve startup time
    for large CLI applications with many subcommands.

    Args:
        *args: Variable length argument list passed to Click.Group
        lazy_subcommands (dict, optional): Mapping of command names to their import paths.
            Format: {"command-name": "module.command-object-name"}
        **kwargs: Arbitrary keyword arguments passed to Click.Group

    Example:
        ```python
        lazy_commands = {
            "cmd1": "mypackage.commands.cmd1.main",
            "cmd2": "mypackage.commands.cmd2.main"
        }
        @click.group(cls=LazyGroup, lazy_subcommands=lazy_commands)
        def cli():
            pass
        ```
    """

    def __init__(self, *args, lazy_subcommands=None, **kwargs):
        super().__init__(*args, **kwargs)
        # lazy_subcommands is a map of the form:
        #
        #   {command-name} -> {module-name}.{command-object-name}
        #
        self.lazy_subcommands = lazy_subcommands or {}

    def list_commands(self, ctx):
        """List all available commands, including lazy-loaded ones.

        Args:
            ctx (click.Context): The current Click context

        Returns:
            list: Sorted list of all available command names
        """
        base = super().list_commands(ctx)
        lazy = sorted(self.lazy_subcommands.keys())
        return base + lazy

    def get_command(self, ctx, cmd_name):
        """Get a command by name, loading it if necessary.

        Args:
            ctx (click.Context): The current Click context
            cmd_name (str): Name of the command to get

        Returns:
            click.Command: The requested command object

        Raises:
            ValueError: If lazy loading fails or returns an invalid command object
        """
        if cmd_name in self.lazy_subcommands:
            return self._lazy_load(cmd_name)
        return super().get_command(ctx, cmd_name)

    def _lazy_load(self, cmd_name):
        """Load a command lazily from its module.

        Args:
            cmd_name (str): Name of the command to load

        Returns:
            click.Command: The loaded command object

        Raises:
            ValueError: If the loaded object is not a valid Click command
        """
        import_path = self.lazy_subcommands[cmd_name]
        modname, cmd_object_name = import_path.rsplit(".", 1)
        # do the import
        mod = importlib.import_module(modname)
        # get the Command object from that module
        cmd_object = getattr(mod, cmd_object_name) # type: ignore
        # check the result to make debugging easier
        if not isinstance(cmd_object, click.BaseCommand): # type: ignore
            raise ValueError(
                f"Lazy loading of {import_path} failed by returning "
                "a non-command object"
            )
        return cmd_object


@click.command()
def help_long(**kwargs):
    """Long and detailed description of commands and options (lazy loaded)."""
    console.print("""
    RolyPoly is a command-line tool with subcommands for different stages of the RNA virus identification pipeline. 
    Many commands share common options:

    [bold yellow]- [green]-t[/green], [cyan]--threads[/cyan][/bold yellow]: Number of threads to use. [i](int, default: 1)[/i]
    [bold yellow]- [green]-M[/green], [cyan]--memory[/cyan][/bold yellow]: Memory allocation in GB. [i](string, default: "6gb")[/i]
    [bold yellow]- [green]-o[/green], [cyan]--output[/cyan] or [cyan]--output-dir[/cyan][/bold yellow]: Output file or directory location. [i](string, default: current working directory + command-specific suffix)[/i]
    [bold yellow]- [green]-r[/green], [cyan]--keep-tmp[/cyan][/bold yellow]:  temporary files. [i](bool, default: False)[/i]
    [bold yellow]- [green]-g[/green], [cyan]--log-file[/cyan][/bold yellow]: Path to the log file. [i](string, default: command-specific log file in the current working directory)[/i]
    [bold yellow]- [green]-i[/green], [cyan]--input[/cyan][/bold yellow]: Input file or directory. [i](string, required)[/i]

    For detailed usage of each command, use the [bold]--help[/bold] option:

    ```bash
    rolypoly [COMMAND] --help
    ```

    [bold red]Note:[/bold red] Some features mentioned in the workflow description (such as annotation and host classification) are still in development and may not be available in the current version. The `annotate` command is currently a placeholder and will be implemented in future updates.  
    Currently available commands:
    """)

    # prepare_external_data
    console.print("""
    [bold cyan]### Prepare External Data[/bold cyan]

    ```bash
    rolypoly prepare_external_data [OPTIONS]
    ```

    Downloads and prepares necessary external data for the pipeline. Additional options include:

    - [bold][cyan]--data_dir[/cyan][/bold]: Specify an alternative path for data storage. [i](string, optional)[/i]
    - [bold][cyan]--try-hard[/cyan][/bold]: Attempt to recreate all databases from scratch instead of downloading pre-built ones. [i](flag, default: False)[/i]
    """)

    # filter_reads
    console.print("""
    [bold cyan]### Filter Reads[/bold cyan]

    ```bash
    rolypoly filter_reads [OPTIONS]
    ```

    Processes and filters raw RNA-seq reads. Additional options include:

    - [bold][green]-D[/green], [cyan]--known-dna[/cyan][/bold]: Fasta file of known DNA entities. [i](string, required)[/i]
    - [bold][green]-s[/green], [cyan]--speed[/cyan][/bold]: Set bbduk.sh speed value. [i](int, optional)[/i]
    """)

    # assembly
    console.print("""
    [bold cyan]### Assembly[/bold cyan]

    ```bash
    rolypoly assembly [OPTIONS]
    ```

    Performs assembly on the filtered reads. Additional options include:

    - [bold][green]-A[/green], [cyan]--assembler[/cyan][/bold]: Choice of assembler(s). [i](string, default: "spades,megahit,penguin")[/i]
    - [bold][green]-d[/green], [cyan]--post-cluster[/cyan][/bold]: Perform post-assembly clustering. [i](bool, default: False)[/i]
    """)

    # rdrp_hmmsearch
    console.print("""
    [bold cyan]### Marker Protein Search[/bold cyan]

    ```bash
    rolypoly marker-search [OPTIONS]
    ```

    (pyhmmer) Searches for RdRp sequences in the assembled contigs. Additional options include:

    - [bold][cyan]--db[/cyan][/bold]: Database to search against. [i](string, choices: ["RVMT", "NCBI_Ribovirus", "all", "other"])[/i]
    - [bold][cyan]--db-path[/cyan][/bold]: Path to user-supplied database (if using 'other'). [i](string, optional)[/i]
    """)
    custom_help()

    # filter_contigs
    console.print("""
    [bold cyan]### Filter Assembly[/bold cyan]

    ```bash
    rolypoly filter_contigs [OPTIONS]
    ```

    This command filters the assembly to remove potential host or contamination sequences. Additional options include:

    - [bold][cyan]--host[/cyan][/bold]: Path to the user-supplied host/contamination fasta. [i](string, required)[/i]
    - [bold][green]-F1[/green], [cyan]--filter1[/cyan][/bold]: First set of rules for match filtering. [i](string, optional)[/i]
    - [bold][green]-F2[/green], [cyan]--filter2[/cyan][/bold]: Second set of rules for match filtering. [i](string, optional)[/i]
    - [bold][cyan]--dont-mask[/cyan][/bold]: If set, host fasta won't be masked for potential RNA virus-like sequences. [i](bool, default: False)[/i]
    """)
    
    # mask_dna
    console.print("""
    [bold cyan]### Mask DNA[/bold cyan]

    ```bash
    rolypoly mask_dna [OPTIONS]
    ```

    Masks DNA sequences. Additional options include:

    - [bold][green]-f[/green], [cyan]--flatten[/cyan][/bold]: Attempt to kcompress.sh the masked file. [i](bool, default: False)[/i]
    - [bold][green]-F[/green], [cyan]--Fast[/cyan][/bold]: Use mmseqs2 instead of bbmap.sh. [i](bool, default: False)[/i]
    """)
    
    # Configuration
    console.print(f"""
    [bold cyan]### Configuration[/bold cyan]

    The pipeline uses a configuration file ([i]rpconfig.json[/i]) to store settings such as the data directory.
                   This file is automatically updated when using the [bold]prepare-external-data[/bold] command.
                   On your system, the location of the file is: {pt.Path(__file__).parent.parent / 'rpconfig.json'
}
    """)

    # end-2-end
    console.print(r"""
    [bold cyan]### End-to-End Pipeline Command[/bold cyan]

    The [bold]end-2-end[/bold] command consolidates all the steps in the RolyPoly pipeline into a single execution flow, 
    from raw RNA-seq data to virus identification and reporting. This includes pre-processing, 
    assembly, filtering, marker protein (rdrp) searching, and result summarization.
    This super-command will use default options for all sub commands, but *most* of them could be over ridden via the following.

    Usage example:
    
    ```bash
    rolypoly end-2-end -i /path/to/raw_data --output-dir /path/to/output --threads 8 --memory 16g --host /path/to/host.fasta --db RVMT
    ```

    [bold yellow]- [green]-i[/green], [cyan]--input[/cyan][/bold yellow]: Input path to raw RNA-seq data. [i](string, required)[/i]
    [bold yellow]- [green]-o[/green], [cyan]--output-dir[/cyan][/bold yellow]: Output directory. [i](string, default: current working directory + '_RP_pipeline')[/i]
    [bold yellow]- [green]-t[/green], [cyan]--threads[/cyan][/bold yellow]: Number of threads to use. [i](int, default: 1)[/i]
    [bold yellow]- [green]-M[/green], [cyan]--memory[/cyan][/bold yellow]: Memory allocation in GB. [i](string, default: "6g")[/i]
    [bold yellow]- [green]-D[/green], [cyan]--host[/cyan][/bold yellow]: Path to the user-supplied host/contamination fasta. [i](string, optional)[/i]
    [bold yellow]- [cyan]--keep-tmp[/cyan][/bold yellow]: Keep temporary files. [i](bool, default: False)[/i]
    [bold yellow]- [cyan]--log-file[/cyan][/bold yellow]: Path to the log file. [i](string, default: current working directory + 'rolypoly_pipeline.log')[/i]

    [bold cyan]## Assembly Options[/bold cyan]
    [bold yellow]- [green]-A[/green], [cyan]--assembler[/cyan][/bold yellow]: Assembler choice. For multiple assemblers, provide a comma-separated list. [i](string, default: "spades,megahit,penguin")[/i]
    [bold yellow]- [green]-d[/green], [cyan]--post-cluster[/cyan][/bold yellow]: Perform post-assembly clustering. [i](bool, default: False)[/i]

    [bold cyan]## Filter Contigs Options[/bold cyan]
    [bold yellow]- [green]-Fm1[/green], [cyan]--filter1_nuc[/cyan][/bold yellow]: First set of rules for nucleic filtering by aligned stats. [i](string, default: "alnlen >= 120 & pident>=75")[/i]
    [bold yellow]- [green]-Fm2[/green], [cyan]--filter2_nuc[/cyan][/bold yellow]: Second set of rules for nucleic match filtering. [i](string, default: "qcov >= 0.95 & pident>=95")[/i]
    [bold yellow]- [green]-Fd1[/green], [cyan]--filter1_aa[/cyan][/bold yellow]: First set of rules for amino (protein) match filtering. [i](string, default: "length >= 80 & pident>=75")[/i]
    [bold yellow]- [green]-Fd2[/green], [cyan]--filter2_aa[/cyan][/bold yellow]: Second set of rules for protein match filtering. [i](string, default: "qcovhsp >= 95 & pident>=80")[/i]
    [bold yellow]- [cyan]--dont-mask[/cyan][/bold yellow]: If set, host fasta won't be masked for potential RNA virus-like sequences. [i](bool, default: False)[/i]
    [bold yellow]- [cyan]--mmseqs-args[/cyan][/bold yellow]: Additional arguments to pass to the MMseqs2 search command. [i](string, optional)[/i]
    [bold yellow]- [cyan]--diamond-args[/cyan][/bold yellow]: Additional arguments to pass to the Diamond search command. [i](string, optional)[/i]

    [bold cyan]## RNA viral marker protein Search Options[/bold cyan]
    [bold yellow]- [cyan]--db[/cyan][/bold yellow]: Database to use for marker protein search. [i](string, default: "all")[/i]
    """)


# RdRp_hmmsearch -- long
def custom_help():
    console.print(r""" Workflow: Input can be amino acid (protein) or nucleic (i.e. contigs). If contigs, the suggested mode, it translates them into either ORFs or to six end-to-end frames (with stops replaced by X),  
    then searches for RNA viral hallmark proteins (by default RdRps) using the selected HMMer DBs (or all of them, or user-supplied one). 
    If you use these options, please cite the respective paper.
    DB options are:
        • [cyan]NeoRdRp_v2.1[/cyan] 
            [blue][link=https://github.com/shoichisakaguchi/NeoRdRp]GitHub[/link][/blue]  |  [blue][link=https://doi.org/10.1264/jsme2.ME22001]Paper[/link][/blue] 
        • [cyan]RVMT[/cyan]
            [blue][link=https://github.com/UriNeri/RVMT]GitHub[/link][/blue]  |  [blue][link=https://zenodo.org/record/7368133]Zenodo[/link][/blue]  |  [blue][link=https://doi.org/10.1016/j.cell.2022.08.023]Paper[/link][/blue] 
        • [cyan]RdRp-Scan[/cyan]
            [blue][link=https://github.com/JustineCharon/RdRp-scan]GitHub[/link][/blue]  |  [blue][link=https://doi.org/10.1093/ve/veac082]Paper[/link][/blue] 
                ⤷ which IIRC incorporated PALMdb: [blue][link=https://github.com/rcedgar/palmdb]GitHub[/link][/blue]  |  [blue][link=https://doi.org/10.7717/peerj.14055]Paper[/link][/blue] 
        • [cyan]TSA_Olendraite[/cyan] 
            [blue][link=https://drive.google.com/drive/folders/1liPyP9Qt_qh0Y2MBvBPZQS6Jrh9X0gyZ?usp=drive_link]Data[/link][/blue]  |  [blue][link=https://doi.org/10.1093/molbev/msad060]Paper[/link][/blue]  |  [blue][link=https://www.repository.cam.ac.uk/items/1fabebd2-429b-45c9-b6eb-41d27d0a90c2]Thesis[/link][/blue] 
        • [cyan]Pfam_A_37[/cyan] 
            [blue][link=https://doi.org/10.1093/nar/gkab1063]Paper[/link][/blue] 
            RdRps and RT profiles from PFAM_A v.37 --- PF04197.17,PF04196.17,PF22212.1,PF22152.1,PF22260.1,PF05183.17,PF00680.25,PF00978.26,PF00998.28,PF02123.21,PF07925.16,PF00078.32,PF07727.19,PF13456.11
            Data: https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz 
            Paper: https://doi.org/10.1093/nar/gkab1063
        • [cyan]genomad[/cyan] 
            [blue][link=https://github.com/apcamargo/genomad]GitHub[/link][/blue]  |  [blue][link=https://doi.org/10.1038/s41587-023-01953-y]Paper[/link][/blue] 

""")


if __name__ == "__main__":
    help_long()