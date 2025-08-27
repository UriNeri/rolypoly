import os as os
from importlib import resources
from json import load

import rich_click as click

from .utils.lazy_group import LazyGroup
from .utils.logging.loggit import get_version_info
from .utils.various import flat_dict

click.rich_click.USE_RICH_MARKUP = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.STYLE_COMMANDS_TABLE_PAD_EDGE = True
click.rich_click.STYLE_COMMANDS_TABLE_PADDING = (0, 2)

# load rolypoly config
with resources.files("rolypoly").joinpath("rpconfig.json").open("r") as conff:
    config = load(conff)
data_dir = config["ROLYPOLY_DATA"]
os.environ["ROLYPOLY_DATA"] = data_dir # export to env just in case
os.environ["citation_file"] = str(
    resources.files("rolypoly")
    .joinpath("../../misc/all_used_tools_dbs_citations.json")
)

@click.group(
    cls=LazyGroup,
    context_settings={"help_option_names": ["-h", "--help", "-help", "--hepl"]},
    lazy_subcommands={
        "data": {
            "name": "Setup and Data",
            "commands": {
                "get-data": "rolypoly.commands.misc.get_external_data.get_data",
                # "build-data": "rolypoly.commands.misc.build_data.build_data",
            },
        },
        "reads": {
            "name": "Raw Reads Processing",
            "commands": {
                "filter-reads": "rolypoly.commands.reads.filter_reads.filter_reads",
                "shrink-reads": "rolypoly.commands.reads.shrink_reads.shrink_reads",
                "mask-dna": "rolypoly.utils.bio.masking.mask_dna",  # Keeping this as is since it's in utils
            },
        },
        "annotation": {
            "name": "Genome Annotation",
            "commands": {
                "annotate": "rolypoly.commands.annotation.annotate.annotate",
                "annotate-rna": "rolypoly.commands.annotation.annotate_RNA.annotate_RNA",
                "annotate-prot": "rolypoly.commands.annotation.annotate_prot.annotate_prot",
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
                "sequence-stats": "rolypoly.utils.bio.sequences.sequence_stats",
                # "visualize": "rolypoly.commands.virotype.visualize.visualize",
                "quick-taxonomy": "rolypoly.commands.misc.quick_taxonomy.quick_taxonomy",
                # "test": "tests.test_cli_commands.test",
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
@click.version_option(version=flat_dict(get_version_info(),sep="\n"), prog_name="rolypoly")
def rolypoly():
    """RolyPoly: RNA Virus analysis tookit.\n
    Use rolypoly `command` --help for more details \n"""
    pass

if __name__ == "__main__":
    rolypoly()
