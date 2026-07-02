import os
from pathlib import Path
from typing import Union

import rich_click as click
from rich.console import Console

from rolypoly.commands.annotation.annotate_prot import (
    ProteinAnnotationConfig,
    process_protein_annotations,
)
from rolypoly.commands.annotation.annotate_RNA import (
    RNAAnnotationConfig,
    process_RNA_annotations,
)
from rolypoly.utils.logging.citation_reminder import remind_citations
from rolypoly.utils.logging.config import BaseConfig

# from rolypoly.utils.logging.loggit import log_start_info

console = Console(width=150)
global tools
tools = []


class AnnotationConfig(BaseConfig):
    def __init__(self, *args, **kwargs):
        # Extract BaseConfig parameters
        base_config_params = {
            "input": kwargs.get("input", ""),
            "output": kwargs.get("output", ""),
            "threads": kwargs.get("threads", 1),
            "log_file": kwargs.get("log_file", ""),
            "log_level": kwargs.get("log_level", "INFO"),
            "memory": kwargs.get("memory", "6gb"),
        }
        super().__init__(**base_config_params)

        # Create subdirectories first
        (self.output_dir / "rna_annotation").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "protein_annotation").mkdir(
            parents=True, exist_ok=True
        )

        self.skip_steps = kwargs.get("skip_steps", [])
        self.rna_config: Union[RNAAnnotationConfig, None] = None
        self.protein_config: Union[ProteinAnnotationConfig, None] = None


@click.command(name="annotate")
@click.option(
    "-i",
    "--input",
    required=True,
    type=click.Path(exists=True),
    help="Input nucleotide sequence file (fasta, fna, fa, or faa)",
)
@click.option(
    "-o",
    "--output",
    default="rolypoly_annotation",
    help="Output file location.",
)
@click.option("-t", "--threads", default=1, help="Number of threads")
@click.option(
    "-g",
    "--log-file",
    default=lambda: Path(os.getcwd()) / "annotate_logfile.txt",
    help="Path to log file",
)
@click.option(
    "-M", "--memory", default="6gb", help="Memory in GB. Example: -M 8gb"
)
@click.option(
    "--override-parameters",
    default="{}",
    help='JSON-like string of parameters to override. Example: --override-parameters \'{"RNAfold": {"temperature": 37}, "ORFfinder": {"minimum_length": 150}}\'',
)
@click.option(
    "--skip-steps",
    default="",
    help="Comma-separated list of steps to skip. Example: --skip-steps RNA_annotation,protein_annotation or --skip-steps IRESfinder,RNAMotif or --skip-steps ORFfinder,hmmsearch",
)
@click.option(
    "--secondary-structure-tool",
    default="LinearFold",
    type=click.Choice(
        ["LinearFold", "RNAfold"]
    ),  # , "SQUARNA", "RNAstructure", "IPknot"
    help="Tool for secondary structure prediction",
)
@click.option(
    "--ires-tool",
    default="IRESfinder",
    type=click.Choice(["IRESfinder", "IRESpy"]),
    help="Tool for IRES identification",
)
@click.option(
    "--trna-tool",
    default="tRNAscan-SE",
    type=click.Choice(["tRNAscan-SE", "ARAGORN"]),
    help="Tool for tRNA identification. ARAGORN NOT FULLY SUPPORTED YET",
)
@click.option(
    "--rnamotif-tool",
    hidden=True,
    default="lightmotif",
    type=click.Choice(["lightmotif", "pymeme"], case_sensitive=False),
    help="Tool for RNA sequence motif identification (PSSM search). Not fully supported yet.",
)
@click.option(
    "--gene-prediction-tool",
    default="pyrodigal",
    type=click.Choice(
        ["pyrodigal", "orffinder", "six_frames", "MetaGeneAnnotator"]
    ),
    help="Tool for gene prediction",
)
@click.option(
    "--domain-db",
    default="Pfam",
    type=click.Choice(
        [
            "Pfam",
            "Vfam",
            # "InterPro",
            # "Phrogs",
            "RVMT",
            "genomad",
            "all",
            "custom",
        ]
    ),
    multiple=True,
    help="Database for domain detection",
)
@click.option(
    "--custom-domain-db",
    default="",
    help="Path to a custom domain database in HMM format (for use with --domain-db custom)",
)
@click.option(
    "--min-orf-length",
    default=30,
    help="Minimum ORF length for gene prediction",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]),
    help="Log level",
    hidden=True,
)
@click.option(
    "--search-tool",
    default="hmmsearch",
    type=click.Choice(
        ["hmmsearch", "hmmscan", "mmseqs2", "DIAMOND", "nail"],
        case_sensitive=False,
    ),
    help="Tool/command for protein domain detection. hmmer commands are via pyhmmer bindings. ",
)
@click.option(
    "--temp-dir",
    default=None,
    type=click.Path(),
    help="Base temporary directory for intermediate files. If not specified, a timestamped temp directory is created inside each sub-step's output directory.",
)
@click.option(
    "--keep-tmp",
    is_flag=True,
    default=False,
    help="Keep temporary files instead of removing them after each sub-step completes.",
)
def annotate(
    input,
    output,
    threads,
    log_file,
    memory,
    override_parameters,
    skip_steps,
    secondary_structure_tool,
    ires_tool,
    trna_tool,
    rnamotif_tool,
    gene_prediction_tool,  # TODO: ADD SUPPORT FOR THIS
    domain_db,
    custom_domain_db,
    min_orf_length,
    log_level,
    search_tool,
    temp_dir=None,
    keep_tmp=False,
):
    """Run combined RNA + protein annotation on nucleotide viral sequences.

    This command orchestrates `annotate-rna` and `annotate-prot` into a single
    workflow and writes results into `rna_annotation/` and
    `protein_annotation/` subdirectories under the selected output path.

    Use `--skip-steps` to disable specific stages and `--override-parameters`
    to forward JSON overrides to sub-tools.
    """
    import json

    from rolypoly.utils.logging.loggit import log_start_info

    override_parameters = (
        json.loads(override_parameters) if override_parameters else {}
    )
    skip_steps_list = skip_steps.split(",") if skip_steps else []
    output_path = Path(output).resolve()
    temp_dir_path = Path(temp_dir).resolve() if temp_dir else None

    # Create main config first
    config = AnnotationConfig(
        input=input,
        output=output_path,
        threads=threads,
        log_level=log_level,
        log_file=log_file,
        memory=memory,
        skip_steps=skip_steps_list,
    )

    # Create RNA config
    rna_config = RNAAnnotationConfig(
        input=Path(config.input).absolute().resolve(),  # type: ignore
        output_dir=config.output_dir / "rna_annotation",
        threads=threads,
        log_level=log_level,
        log_file=config.logger,  # Pass the logger directly
        memory=memory,
        override_parameters=override_parameters,
        skip_steps=skip_steps_list,
        secondary_structure_tool=secondary_structure_tool,
        ires_tool=ires_tool,
        trna_tool=trna_tool,
        rnamotif_tool=rnamotif_tool,
        overwrite=True,  # Prevent directory checks
        temp_dir=str(temp_dir_path / "rna_annotation") if temp_dir_path else None,
        keep_tmp=keep_tmp,
    )

    # Create protein config
    protein_config = ProteinAnnotationConfig(
        input=input,
        output_dir=config.output_dir / "protein_annotation",
        threads=threads,
        log_file=config.logger,  # Pass the logger directly
        memory=memory,
        override_parameters=override_parameters,
        skip_steps=skip_steps_list,
        gene_prediction_tool=gene_prediction_tool,
        search_tool=search_tool,
        domain_db=domain_db,
        custom_domain_db=custom_domain_db,
        min_orf_length=min_orf_length,
        genetic_code=11,  # Default genetic code
        overwrite=True,  # Prevent directory checks
        temp_dir=str(temp_dir_path / "protein_annotation") if temp_dir_path else None,
        keep_tmp=keep_tmp,
    )

    # Attach sub-configs to main config
    config.rna_config = rna_config
    config.protein_config = protein_config

    log_start_info(config.logger, config.__dict__)
    config.logger.info("Starting annotation process    ")

    if "protein_annotation" not in config.skip_steps:
        process_protein_annotations(config.protein_config)
        if config.protein_config.search_tool == "hmmsearch":
            tools.append("HMMER")
        else:
            tools.append(config.protein_config.search_tool)

        tools.append(config.protein_config.domain_db)
    else:
        config.logger.info("Skipping protein annotation")
    if "RNA_annotation" not in config.skip_steps:
        process_RNA_annotations(config.rna_config)
        tools.append(config.rna_config.secondary_structure_tool)
        tools.append(config.rna_config.ires_tool)
        tools.append(config.rna_config.trna_tool)
        tools.append(config.rna_config.rnamotif_tool)
        tools.append(
            "rfam"
        )  # TODO: add other needed domain_db to the citation reminder
    else:
        config.logger.info("Skipping RNA annotation")

    # TODO: logic to combine RNA and protein annotation results - either stack the tables or combine the gff3s (both?)

    config.logger.info("Annotation process completed.")

    # remind_citations(tools)
    if config.log_level != 10:
        with open(f"{config.log_file}", "a") as f_out:
            f_out.write(remind_citations(tools, return_bibtex=True) or "")


if __name__ == "__main__":
    annotate()
