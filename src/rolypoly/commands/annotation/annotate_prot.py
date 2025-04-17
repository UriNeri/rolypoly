from pathlib import Path

import rich_click as click
from rich.console import Console

from rolypoly.utils.config import BaseConfig
from rolypoly.utils.various import run_command_comp


class ProteinAnnotationConfig(BaseConfig):
    """Configuration for protein annotation pipeline"""

    def __init__(
        self,
        input: Path,
        output_dir: Path,
        threads: int,
        log_file: Path,
        memory: str,
        override_params: dict[str, any] = None,
        skip_steps: list[str] = None,
        search_tool: str = "hmmsearch",
        domain_db: str = "Pfam",
        custom_domain_db: str = "",
        min_orf_length: int = 30,
        genetic_code: int = 11,
        gene_prediction_tool: str = "ORFfinder",
        **kwargs,
    ):
        # Extract BaseConfig parameters
        base_config_params = {
            "input": input,
            "output": output_dir,
            "threads": threads,
            "log_file": log_file,
            "memory": memory,
        }
        super().__init__(**base_config_params)

        self.skip_steps = skip_steps or []
        self.search_tool = search_tool
        self.domain_db = domain_db
        self.custom_domain_db = custom_domain_db
        self.min_orf_length = min_orf_length
        self.genetic_code = genetic_code
        self.gene_prediction_tool = gene_prediction_tool
        self.step_params = {
            "ORFfinder": {"minimum_length": 30},
            "hmmsearch": {"E": 1e-5},
            "InterProScan": {},  # TODO: decide if using InterProScan as is or if using pyhmmer on it's member dbs.
        }

        if override_params:
            for step, params in override_params.items():
                if step in self.step_params:
                    self.step_params[step].update(params)
                else:
                    print(
                        f"Warning: Unknown step '{step}' in override_parameters. Ignoring."
                    )


console = Console(width=150)


@click.command()
@click.option(
    "-i",
    "--input",
    required=True,
    help="Input directory containing rolypoly's virus identification results",
)
@click.option(
    "-o", "--output-dir", default="./annotate_prot_output", help="Output directory path"
)
@click.option("-t", "--threads", default=1, help="Number of threads")
@click.option(
    "-g", "--log-file", default="./annotate_prot_logfile.txt", help="Path to log file"
)
@click.option("-M", "--memory", default="8gb", help="Memory in GB. Example: -M 8gb")
@click.option(
    "-op",
    "--override-params",
    "--override-parameters",
    default="{}",
    help='JSON-like string of parameters to override. Example: --override-parameters \'{"ORFfinder": {"minimum_length": 150}, "hmmsearch": {"E": 1e-3}}\'',
)
@click.option(
    "--skip-steps",
    default="",
    help="Comma-separated list of steps to skip. Example: --skip-steps ORFfinder,hmmsearch",
)
@click.option(
    "--gene-prediction-tool",
    default="ORFfinder",
    type=click.Choice(
        ["ORFfinder", "pyrodigal", "six-frame", "MetaGeneAnnotator"],
        case_sensitive=False,
    ),
    help="Tool for gene prediction",
)
@click.option(
    "--search-tool",
    default="hmmsearch",
    type=click.Choice(
        ["hmmsearch", "mmseqs2", "DIAMOND", "nail"], case_sensitive=False
    ),
    help="Tool/command for protein domain detection. hmmer commands are used via pyhmmer bindings",
)
@click.option(
    "--domain-db",
    default="Pfam",
    type=click.Choice(
        ["Pfam", "Vfam", "InterPro", "Phrogs", "dbCAN", "all", "none", "custom"],
        case_sensitive=False,
    ),
    help="Database for domain detection",
)
@click.option(
    "--custom-domain-db",
    default="",
    help="Path to a custom domain database in HMM format (for use with --domain-db custom)",
)
@click.option(
    "--min-orf-length", default=30, help="Minimum ORF length for gene prediction"
)
@click.option("--genetic-code", default=11, help="Genetic code for ORFfinder")
def annotate_prot(
    input,
    output_dir,
    threads,
    log_file,
    memory,
    override_parameters,
    skip_steps,
    gene_prediction_tool,
    search_tool,
    domain_db,
    custom_domain_db,
    min_orf_length,
    genetic_code,  # TODO: ADD SUPPORT FOR THIS
):
    """Identify coding sequences (ORFs) in a genome, and attempt to predict their translated seq (protein) function via (py)hmmsearches against a collection of DBs (or user supplied one)"""
    import json

    from rolypoly.utils.various import ensure_memory

    config = ProteinAnnotationConfig(
        input=input,
        output_dir=output_dir,
        threads=threads,
        log_file=log_file,
        memory=ensure_memory(memory)["giga"],
        override_parameters=json.loads(override_parameters)
        if override_parameters
        else {},
        skip_steps=skip_steps.split(",") if skip_steps else [],
        search_tool=search_tool,
        domain_db=domain_db,
        custom_domain_db=custom_domain_db,
        min_orf_length=min_orf_length,
        gene_prediction_tool=gene_prediction_tool,
        genetic_code=genetic_code,
    )

    try:
        process_protein_annotations(config)
    except Exception as e:
        console.print(f"An error occurred during protein annotation: {str(e)}")
        raise


def process_protein_annotations(config):
    """Process protein annotations"""
    config.logger.info("Starting protein annotation process")

    steps = [
        predict_orfs,  # i.e. call genes
        search_protein_domains,
        predict_protein_function,
    ]

    for step in steps:
        step_name = step.__name__
        if step_name not in config.skip_steps:
            config.logger.info(f"Starting step: {step_name}")
            step(config)
        else:
            config.logger.info(f"Skipping step: {step_name}")

    combine_results(config)
    config.logger.info("Protein annotation process completed successfully")


def predict_orfs(config):
    """Predict open reading frames using selected tool"""
    if config.gene_prediction_tool == "ORFfinder":
        predict_orfs_with_orffinder(config)
    elif config.gene_prediction_tool == "pyrodigal":
        predict_orfs_with_pyrodigal(config)
    elif config.gene_prediction_tool == "six-frame":
        predict_orfs_with_six_frame(config)
    elif config.gene_prediction_tool == "MetaGeneAnnotator":
        config.logger.info("MetaGeneAnnotator not implemented yet")
        # predict_orfs_with_metagenannotator(config)
    else:
        config.logger.info(
            f"Skipping ORF prediction as {config.gene_prediction_tool} is not supported"
        )


def predict_orfs_with_pyrodigal(config):
    """Predict ORFs using pyrodigal"""
    from rolypoly.utils.fax import pyro_predict_orfs

    pyro_predict_orfs(
        input_file=config.input,
        output_file=config.output_dir / "predicted_orfs.faa",
        threads=config.threads,
        gv_or_else="gv",
        genetic_code=config.genetic_code,
    )


def predict_orfs_with_six_frame(config):
    """Translate 6-frame reading frames of a DNA sequence using seqkit."""
    from rolypoly.utils.fax import translate_6frx_seqkit

    output_file = str(config.output_dir / "predicted_orfs.faa")
    translate_6frx_seqkit(str(config.input), output_file, config.threads)
    return output_file


def search_protein_domains_hmmscan(config):
    """Search protein domains using hmmscan."""
    from rolypoly.utils.fax import search_hmmdb

    search_hmmdb(
        config.input,
        config.domain_db,
        config.output_dir / "protein_domains.tsv",
        output_format="modomtblout",
        threads=config.threads,
        logger=config.logger,
        match_region=True,
        full_qseq=True,
        ali_str=True,
    )


def predict_orfs_with_orffinder(config):
    """Predict ORFs using ORFfinder."""
    import os
    from shutil import which

    from rolypoly.utils.various import run_command_comp

    if not which("ORFfinder"):
        config.logger.error(
            "ORFfinder not found. Please install ORFfinder and add it to your PATH (it isn't a conda/mamba installable package, but you can do the following:  wget ftp://ftp.ncbi.nlm.nih.gov/genomes/TOOLS/ORFfinder/linux-i64/ORFfinder.gz; gunzip ORFfinder.gz; chmod a+x ORFfinder; mv ORFfinder $CONDA_PREFIX/bin)."
        )
        lazy = input(
            "Do you want to install ORFfinder for you (i.e. ran the above commands)? [yes/no]  "
        )
        if lazy.lower() == "yes":
            os.system(
                "wget ftp://ftp.ncbi.nlm.nih.gov/genomes/TOOLS/ORFfinder/linux-i64/ORFfinder.gz; gunzip ORFfinder.gz; chmod a+x ORFfinder; mv ORFfinder $CONDA_PREFIX/bin"
            )
            config.logger.info("ORFfinder installed successfully")
        else:
            config.logger.error(
                "ORFfinder not found, you don't want me to install it, and you don't want to use another tool         seriously. Exiting    "
            )
            exit(1)

    config.logger.info("Predicting ORFs")
    input_fasta = Path(config.input)
    output_file = config.output_dir / "predicted_orfs.faa"

    run_command_comp(
        "ORFfinder",
        params={
            "in": str(input_fasta),
            "out": str(output_file),
            "ml": config.min_orf_length,
            "s": config.genetic_code,
            "n": "true",
            "outfmt": 0,
        },
        logger=config.logger,
    )


def search_protein_domains(config):
    config.logger.info("Searching for protein domains")
    input_fasta = config.output_dir / "predicted_orfs.faa"
    output_file = config.output_dir / "protein_domains.out"

    if config.search_tool == "hmmsearch":
        search_protein_domains_hmmsearch(config, input_fasta, output_file)
    elif config.search_tool == "hmmscan":
        search_protein_domains_hmmscan(config, input_fasta, output_file)
    elif config.search_tool == "mmseqs2":
        search_protein_domains_mmseqs2(config, input_fasta, output_file)
    elif config.search_tool == "DIAMOND":
        search_protein_domains_diamond(config, input_fasta, output_file)
    else:
        config.logger.info(
            f"Skipping protein domain search as {config.search_tool} is not supported"
        )


def search_protein_domains_hmmsearch(config, input_fasta, output_file):
    """Search protein domains using hmmsearch."""
    run_command_comp(
        "hmmsearch",
        params={"cpu": config.threads, "tblout": output_file},
        positional_args=[str(config.domain_db), str(input_fasta)],
        logger=config.logger,
    )


def search_protein_domains_mmseqs2(config, input_fasta, output_file):
    """Search protein domains using mmseqs2."""
    run_command_comp(
        "mmseqs",
        positional_args=[
            "easy-search",
            str(input_fasta),
            str(config.domain_db),
            str(output_file),
            str(config.output_dir / "tmp"),
        ],
        params={"threads": config.threads},
        logger=config.logger,
    )


def search_protein_domains_diamond(config, input_fasta, output_file):
    """Search protein domains using DIAMOND."""
    run_command_comp(
        "diamond",
        positional_args=["blastp"],
        params={
            "query": str(input_fasta),
            "db": str(config.domain_db),
            "out": str(output_file),
            "threads": config.threads,
            "outfmt": 6,
        },
        logger=config.logger,
    )


def predict_protein_function(config):
    config.logger.info("Predicting protein function")
    # TODO: decide how this should be done


def combine_results(config):
    import polars as pl

    config.logger.info("Combining annotation results")

    orf_file = config.output_dir / "predicted_orfs.faa"
    domain_file = config.output_dir / "protein_domains.out"

    orf_data = process_orf_data(orf_file)
    domain_data = process_domain_data(domain_file, config.search_tool)

    combined_data = pl.join(orf_data, domain_data, on="orf_id", how="left")

    output_file = config.output_dir / "combined_protein_annotations.csv"
    combined_data.write_csv(output_file)

    config.logger.info(f"Combined annotation results written to {output_file}")


def process_orf_data(orf_file):
    import polars as pl

    # Process the ORF file and return a DataFrame
    orf_data = pl.read_csv(
        orf_file,
        separator="\t",
        columns=[
            "orf_id",
            "orf_start",
            "orf_end",
            "orf_length",
            "orf_score",
            "orf_e_value",
            "orf_frame",
            "orf_direction",
            "orf_description",
        ],
    )
    return orf_data


def process_domain_data(domain_file, search_tool):
    import polars as pl

    if search_tool == "hmmsearch":
        return process_hmmsearch_data(domain_file)
    elif search_tool == "mmseqs2":
        return process_mmseqs2_data(domain_file)
    elif search_tool == "DIAMOND":
        return process_diamond_data(domain_file)
    else:
        return pl.DataFrame()


def process_hmmsearch_data(domain_file):
    import polars as pl

    df = pl.read_csv(
        domain_file,
        separator="\t",
        columns=[
            "orf_id",
            "orf_start",
            "orf_end",
            "orf_length",
            "orf_score",
            "orf_e_value",
            "orf_frame",
            "orf_direction",
            "orf_description",
        ],
    )
    return df


def process_mmseqs2_data(domain_file):
    import polars as pl

    df = pl.read_csv(
        domain_file,
        separator="\t",
        columns=[
            "orf_id",
            "orf_start",
            "orf_end",
            "orf_length",
            "orf_score",
            "orf_e_value",
            "orf_frame",
            "orf_direction",
            "orf_description",
        ],
    )
    return df


def process_diamond_data(domain_file):
    import polars as pl

    df = pl.read_csv(
        domain_file,
        separator="\t",
        columns=[
            "orf_id",
            "orf_start",
            "orf_end",
            "orf_length",
            "orf_score",
            "orf_e_value",
            "orf_frame",
            "orf_direction",
            "orf_description",
        ],
    )
    return df


if __name__ == "__main__":
    annotate_prot()
