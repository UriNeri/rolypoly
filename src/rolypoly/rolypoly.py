# rolypoly.py
import os as os
from importlib import resources
from json import load

import rich_click as click

from .utils.lazy_group import LazyGroup
from .utils.loggit import get_version_info

# Configure rich_click for nice formatting
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.STYLE_COMMANDS_LIST = "bold cyan"
click.rich_click.STYLE_COMMANDS_TABLE_GRID = "dim"
click.rich_click.STYLE_COMMANDS_TABLE_SHOW_LINES = True
click.rich_click.STYLE_COMMANDS_TABLE_PAD_EDGE = True
click.rich_click.STYLE_COMMANDS_TABLE_PADDING = (0, 2)
click.rich_click.STYLE_OPTIONS_LIST_ALIGN_IN_COLUMNS = True

# load config
with resources.files("rolypoly").joinpath("rpconfig.json").open("r") as conff:
    config = load(conff)
data_dir = config["ROLYPOLY_DATA"]
os.environ["ROLYPOLY_DATA"] = data_dir
os.environ["citation_file"] = (
    resources.files("rolypoly")
    .joinpath("../../misc/all_used_tools_dbs_citations.json")
    .as_posix()
)


@click.group(
    cls=LazyGroup,
    context_settings={"help_option_names": ["-h", "--help", "-help"]},
    lazy_subcommands={
        "data": {
            "name": "Setup and Data",
            "commands": {
                "update": "rolypoly.commands.misc.update.update",
                "prepare-data": "rolypoly.commands.misc.prepare_external_data.prepare_external_data",
            },
        },
        "reads": {
            "name": "Raw Reads Processing",
            "commands": {
                "filter-reads": "rolypoly.commands.reads.filter_reads.filter_reads",
                "mask-dna": "rolypoly.utils.fax.mask_dna",  # Keeping this as is since it's in utils
            },
        },
        "annotation": {
            "name": "Genome Annotation",
            "commands": {
                "annotate": "rolypoly.commands.annotation.annotate.annotate",
                "annotate-rna": "rolypoly.commands.annotation.annotate_RNA.annotate_RNA",
                "annotate-prot": "hidden:rolypoly.commands.annotation.annotate_prot.annotate_prot",
            },
        },
        "assembly": {
            "name": "Meta/Genome Assembly",
            "commands": {
                "assemble": "rolypoly.commands.assembly.assemble.assembly",
                "filter-contigs": "rolypoly.commands.assembly.filter_contigs.filter_contigs",
                # Commenting out unimplemented commands
                # "co-assembly": "rolypoly.commands.assembly.co_assembly.co_assembly",
                # "refine": "rolypoly.commands.assembly.refinement.refine"
            },
        },
        "misc": {
            "name": "Miscellaneous",
            "commands": {
                "end2end": "rolypoly.commands.misc.end_2_end.run_pipeline",
                # "add-command": "hidden:rolypoly.commands.misc.add_command.add_command",
                "fetch-sra": "rolypoly.commands.misc.fetch_sra_fastq.fetch_sra",  # Not  a click command (yet?)
                "sequence-stats": "rolypoly.commands.misc.sequence_stats.sequence_stats",
                # "visualize": "rolypoly.commands.virotype.visualize.visualize",
                "quick-taxonomy": "rolypoly.commands.misc.quick_taxonomy.quick_taxonomy",
                "test": "tests.test_cli_commands.test",
            },
        },
        # "characterise": {
        #     "name": "Characterisation",
        #     "commands": {
        #         "characterise": "hidden:rolypoly.commands.virotype.predict_characteristics.predict_characteristics",
        #         "predict-host": "hidden:rolypoly.commands.host.classify.predict_host_range",
        #         "correlate": "hidden:rolypoly.commands.bining.corrolate.corrolate",
        #         # Commenting out unimplemented/broken commands
        #         # "summarize": "rolypoly.commands.virotype.summarize.summarize"
        #     },
        # },
        "identify": {
            "name": "RNA Virus Identification",
            "commands": {
                "marker-search": "rolypoly.commands.identify_virus.marker_search.marker_search",
                "search-viruses": "rolypoly.commands.identify_virus.search_viruses.virus_mapping",
            },
        },
    },
)
@click.version_option(version=get_version_info(), prog_name="rolypoly")
def rolypoly():
    """RolyPoly: RNA Virus analysis tookit.\n
    Use rolypoly `command` --help for more details \n"""
    pass


if __name__ == "__main__":
    rolypoly()
