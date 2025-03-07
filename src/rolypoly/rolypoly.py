# rolypoly.py
import os as os
import rich_click as click
from json import load
from .utils.lazy_group import LazyGroup
from importlib import resources
from .utils.loggit import get_version_info

# load config
with resources.files("rolypoly").joinpath("rpconfig.json").open("r") as conff:
    config = load(conff)
data_dir = config['ROLYPOLY_DATA']
os.environ['ROLYPOLY_DATA'] =  data_dir   
os.environ['citation_file'] =  resources.files("rolypoly").joinpath("../../misc/all_used_tools_dbs_citations.json").as_posix()

@click.group(name="Main",
             cls=LazyGroup,
             context_settings={'show_default': True, "help_option_names" : ['-h',"-H", '--help']},
             lazy_subcommands={
                                "help" : "rolypoly.utils.lazy_group.help_long",
                                "update" : "rolypoly.commands.misc.update.update",
                                "end-2-end" : "rolypoly.commands.misc.end_2_end.run_pipeline",
                                "prepare-external-data" : "rolypoly.commands.misc.prepare_external_data.prepare_external_data",
                                "filter-reads" : "rolypoly.commands.reads.filter_reads.filter_reads",
                                "mask-dna": "rolypoly.utils.fax.mask_dna",
                                "search-viruses" : "rolypoly.commands.identify_virus.search_viruses.virus_mapping",
                                "assembly":"rolypoly.commands.assembly.assembly.assembly",
                                "marker-search" : "rolypoly.commands.identify_virus.marker_search.marker_search",
                                "filter-contigs" : "rolypoly.commands.assembly.filter_contigs.filter_contigs",
                                "add-command" : "rolypoly.commands.misc.add_command.add_command",
                                # "filter_assembly_nuc": "rolypoly.commands.assembly.filter_assembly_nuc.filter_assembly_nuc",
                                # "filter_assembly_prot": "rolypoly.commands.assembly.filter_assembly_aa.filter_assembly_aa",
                                "annotate": "rolypoly.commands.annotation.annotate.annotate",
                                # "annotate_proteins": "rolypoly.commands.annotation.annotate_prot.annotate_prot",
                                "annotate-rna": "rolypoly.commands.annotation.annotate_RNA.annotate_RNA",
                                # "characterise": "rolypoly.commands.virotype.predict_characteristics.predict_characteristics",
                                # "predict_host_range": "rolypoly.commands.host.classify.predict_host_range",
                                # "dummy": "rolypoly.utils.dummy.dummy",
                                # "corrolate": "rolypoly.commands.bining.corrolate.corrolate",
                                # "summarize": "rolypoly.commands.virotype.summarize.summarise"
                                #
                               }
             )
@click.version_option(version=get_version_info(), prog_name="rolypoly")
def rolypoly():
    """RolyPoly: RNA Virus analysis tookit.\n
    Use rolypoly `command` --help for more details \n"""
    # RolyPoly version: {get_version_info()}"""
    pass

if __name__ == "__main__":
    rolypoly()

    