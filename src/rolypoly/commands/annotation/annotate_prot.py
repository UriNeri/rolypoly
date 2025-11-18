from pathlib import Path

import rich_click as click
from rich.console import Console

from rolypoly.utils.logging.config import BaseConfig
from rolypoly.utils.various import run_command_comp
from typing import Union
import logging
import polars as pl

global output_files
output_files = pl.DataFrame(schema={"file": pl.Utf8, "description": pl.Utf8, "db": pl.Utf8,  "tool": pl.Utf8, "params": pl.Utf8, "command": pl.Utf8})


class ProteinAnnotationConfig(BaseConfig):
    """Configuration for protein annotation pipeline"""

    def __init__(
        self,
        input: Path,
        output_dir: Path,
        threads: int,
        log_file: Union[Path, logging.Logger, None],
        memory: str,
        override_parameters: dict[str, object] = {},
        skip_steps: list[str] = [],
        search_tool: str = "hmmsearch",
        domain_db: str = "Pfam",
        min_orf_length: int = 30,
        genetic_code: int = 11,
        gene_prediction_tool: str = "ORFfinder",
        evalue: float = 1e-2,
        db_create_mode: str = "auto",
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
        self.min_orf_length = min_orf_length
        self.genetic_code = genetic_code
        self.gene_prediction_tool = gene_prediction_tool
        self.evalue = evalue
        self.db_create_mode = db_create_mode
        self.step_params = {
            "ORFfinder": {"minimum_length": min_orf_length, "start_codon": 1, "strand": "both", "outfmt": 0, "ignore_nested": False},
            "pyrodigal": {"minimum_length": min_orf_length},
            "six-frame": {"threads": 1, "min_orf_length": min_orf_length},
            "hmmsearch": {"inc_e": evalue, "mscore": 5},
            "diamond": {"evalue": evalue}, 
            "mmseqs2": {"evalue": evalue, "cov": 0.5},
        }

        if override_parameters:
            for step, params in override_parameters.items():
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
    "-o",
    "--output-dir",
    default="./annotate_prot_output",
    help="Output directory path",
)
@click.option(
    "-t",
    "--threads",
    default=1,
    help="Number of threads",
)
@click.option(
    "-g",
    "--log-file",
    default="./annotate_prot_logfile.txt", help="Path to log file"
)
@click.option(
    "-M",
    "--memory",
    default="8gb",
    help="Memory in GB. Example: -M 8gb",
    hidden=True,
)
@click.option(
    "-op",
    "--override-parameters",
    "--override-params",
    default="{}",
    help='JSON-like string of parameters to override. Example: --override-parameters \'{"ORFfinder": {"minimum_length": 150}, "hmmsearch": {"E": 1e-3}}\'',
)
@click.option(
    "-ss",
    "--skip-steps",
    default="",
    help="Comma-separated list of steps to skip. Example: --skip-steps ORFfinder,hmmsearch",
)
@click.option(
    "-gp",
    "--gene-prediction-tool",
    default="ORFfinder",
    type=click.Choice(
        ["ORFfinder", "pyrodigal", "six-frame"], # , "bbmap"``
        case_sensitive=False,
    ),
    help="""Tool for gene prediction. \n
    * pyrodigal-gv: might work well for some viruses, but it's not as well tested for RNA viruses. Includes internal genetic code assignment. \n
    * ORFfinder: The default ORFfinder settings may have some false positives, but it's fast and easy to use. \n
    * six-frame: includes all 6 reading frames, so all possible ORFs are predicted - prediction is quick but will include many false positives, and the input for the domain search will be larger. \n
    """,
)
@click.option(
    "-st",
    "--search-tool",
    default="hmmsearch",
    type=click.Choice(
        ["hmmsearch", "mmseqs2", "diamond"], case_sensitive=False #, "nail"
    ),
    help="Tool/command for protein domain detection. Only one tool can be used at a time.",
)
@click.option(
    "-d",
    "--domain-db",
    default="RVMT",
    type=str,
    help="""Database for domain detection. \n
    * Pfam: Pfam-A \n
    * RVMT: RVMT \n
    * genomad: genomad \n
    * RefSeq_virus: RefSeq_virus \n
    * custom: custom (path to a custom database in HMM format or a directory of MSA/hmms files) \n
    * all: all (all databases) \n
    """,
)
@click.option(
    "-ml",
    "--min-orf-length",
    default=30,
    help="Minimum ORF length for gene prediction",
)
@click.option(
    "-gc",
    "--genetic-code",
    default=11, 
    help="Genetic code (a.k.a. translation table) ",
)
@click.option(
    "-e",
    "--evalue",
    default=1e-1,
    help="E-value for search result filtering. Note, this is for inital filteringg only, you are encouraged to filter the results further using e.g. profile coverage and scores.",
)
@click.option(
    "--db-create-mode",
    default="auto",
    type=click.Choice(["auto", "mmseqs", "hmm"], case_sensitive=False),
    help="How to handle custom database directories: auto=guess, mmseqs=build mmseqs profile DB, hmm=build concatenated HMM",
)
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
    min_orf_length,     
    genetic_code, 
    evalue
    ,
    db_create_mode
):
    """Identify coding sequences (ORFs) from fasta, and predicts their translated seqs putative function via homology search. \n
    Currently supported tools and databases: \n
    * Translations: ORFfinder, pyrodigal, six-frame \n
    * Search engines: \n
    - (py)hmmsearch: Pfam, RVMT, genomad, vfam \n
    - mmseqs2: Pfam, RVMT, genomad, vfam \n
    - diamond: Uniref50, RefSeq (both are subseted to viral sequences) \n
    * custom: user supplied database. Needs to be in tool appropriate format, or a directory of aligned fasta files (for hmmsearch) 
    """
    # - nail: Pfam, RVMT, genomad, custom (via nail) # TODO: add support for nail. https://github.com/TravisWheelerLab/nail
    import json

    from rolypoly.utils.various import ensure_memory

    config = ProteinAnnotationConfig(
        input=input,
        output_dir=output_dir,
        threads=threads,
        log_file=log_file,
        memory=ensure_memory(memory)["giga"],
        override_parameters=json.loads(override_parameters) if override_parameters else {},
        skip_steps=skip_steps.split(",") if skip_steps else [],
        search_tool=search_tool,
        domain_db=domain_db,
        min_orf_length=min_orf_length,
        gene_prediction_tool=gene_prediction_tool,
        genetic_code=genetic_code,
        evalue=evalue,
        db_create_mode=db_create_mode,
    )

    # config.logger.info(f"Using {config.search_tool} for domain search")
    try:
        process_protein_annotations(config)
    except Exception as e:
        console.print(f"An error occurred during protein annotation: {str(e)}")
        raise

def process_protein_annotations(config):
    """Process protein annotations"""
    config.logger.info("Starting protein annotation process")
    # create a "raw_out" subdirectory in output folder
    raw_out_dir = config.output_dir / "raw_out"
    raw_out_dir.mkdir(parents=True, exist_ok=True)
    config.logger.debug(f"created raw_out directory: {raw_out_dir}")

    steps = [
        predict_orfs,  # i.e. call genes
        search_protein_domains,
        combine_results,
    ]

    for step in steps:
        step_name = step.__name__
        if step_name not in config.skip_steps:
            config.logger.info(f"Starting step: {step_name}")
            step(config)
        else:
            config.logger.info(f"Skipping step: {step_name}")

    config.logger.info("Protein annotation process completed successfully")
    output_files.write_csv(config.output_dir / "output_files.tsv", separator="\t")


def predict_orfs(config):
    """Predict open reading frames using selected tool"""
    if config.gene_prediction_tool == "ORFfinder":
        predict_orfs_with_orffinder(config)
    elif config.gene_prediction_tool == "pyrodigal":
        predict_orfs_with_pyrodigal(config)
    elif config.gene_prediction_tool == "six-frame":
        predict_orfs_with_six_frame(config)
    else:
        config.logger.info(
            f"Skipping ORF prediction as {config.gene_prediction_tool} is not supported"
        )


def predict_orfs_with_pyrodigal(config):
    """Predict ORFs using pyrodigal"""
    from rolypoly.utils.bio import pyro_predict_orfs

    output_file = config.output_dir / "raw_out" / "predicted_orfs.faa"
    pyro_predict_orfs(
        input_file=config.input,
        output_file=output_file,
        threads=config.threads,
        # genetic_code=config.step_params["pyrodigal"]["genetic_code"],
        min_gene_length=config.step_params["pyrodigal"]["min_orf_length"],
    )
    global output_files
    output_files = output_files.vstack(pl.DataFrame({"file": [str(output_file)], "description": ["predicted ORFs"], "db": ["pyrodigal"], "tool": ["pyrodigal"], "params": [str(config.step_params["pyrodigal"])], "command": [f"pyrodigal via pyrodigal module: genetic_code={config.step_params['pyrodigal']['genetic_code']}, threads={config.threads}"]}))
    return output_file


def predict_orfs_with_six_frame(config):
    """Translate 6-frame reading frames of a DNA sequence using seqkit."""
    from rolypoly.utils.bio.translation import translate_6frx_seqkit

    output_file = str(config.output_dir / "raw_out" / "predicted_orfs.faa")
    translate_6frx_seqkit(str(config.input), output_file, config.threads)
    global output_files
    output_files = output_files.vstack(pl.DataFrame({"file": [output_file], "description": ["predicted ORFs"], "db": ["six-frame"], "tool": ["six-frame"], "params": [str(config.step_params["six-frame"])], "command": [f"ext. call seqkit: seqkit -w0 translate -j {config.threads} {config.input} > {output_file}"]}))
    return output_file


def get_database_paths(config, tool_name):
    """Get database paths for the specified tool with validation"""
    import os
    
    hmmdbdir = Path(os.environ["ROLYPOLY_DATA"]) / "hmmdbs"
    mmseqs2_dbdir = Path(os.environ["ROLYPOLY_DATA"]) / "mmseqs2"
    diamond_dbdir = Path(os.environ["ROLYPOLY_DATA"]) / "diamond"
    
    # Database paths for different tools
    DB_PATHS = {
        "hmmsearch": {
            "RVMT".lower(): hmmdbdir / "RVMT/NVPC.hmm",
            "Pfam".lower(): hmmdbdir / "pfam/Pfam-A.hmm",
            "genomad".lower(): hmmdbdir / "genomad.hmm",
            "vfam".lower(): hmmdbdir / "vfam.hmm",
        },
        "mmseqs2": {
            "RVMT".lower(): mmseqs2_dbdir / "RVMT/nvpc",  
            "Pfam".lower(): mmseqs2_dbdir / "pfam/pfamA37",
            "genomad".lower(): mmseqs2_dbdir / "genomad/genomad_vv",
        },
        "diamond": {
            "RefSeq_virus".lower(): diamond_dbdir / "RefSeq_virus.dmnd",
            "Uniref50".lower(): diamond_dbdir / "uniref50_viral.dmnd",  # TODO: create these DBs, add them to data, and push new data version.
        }
    }
    
    if tool_name not in DB_PATHS:
        config.logger.warning(f"No predefined databases for tool {tool_name}")
        return {}
    
    tool_db_paths = DB_PATHS[tool_name]
    
    if config.domain_db == "all":
        database_paths = tool_db_paths
    elif config.domain_db.startswith("/") or config.domain_db.startswith("./"):
        custom_database = str(Path(config.domain_db).resolve())
        if not Path(custom_database).exists():
            config.logger.error(f"Custom database path {custom_database} does not exist")
            return {}
        
        # Handle custom database files and directories (mainly for hmmsearch)
        if tool_name == "hmmsearch":
            # check if a file it's an hmm or an msa file
            if custom_database.endswith(".hmm"):
                database_paths = {"Custom": custom_database}
            elif custom_database.endswith((".faa", ".fasta", ".afa")):
                from rolypoly.utils.bio.alignments import hmm_from_msa

                database_paths = {
                    "Custom": hmm_from_msa(
                        msa_file=config.domain_db,
                        output=config.domain_db.replace(".faa", ".hmm"),
                        name=Path(config.domain_db).stem,
                    )
                }
            # if it's a directory:
            elif Path(custom_database).is_dir():
                from rolypoly.utils.bio.library_detection import validate_database_directory
                
                db_info = validate_database_directory(custom_database, logger=config.logger)
                config.logger.info(f"Database directory analysis: {db_info['message']}")
                
                if db_info["type"] == "hmm_directory":
                    # concatenate all hmms into one file
                    with open(Path(custom_database) / "concatenated.hmm", "w") as f:
                        for hmm_file in db_info["files"]:
                            with open(hmm_file, "r") as hmm_file_obj:
                                f.write(hmm_file_obj.read())
                    database_paths = {"Custom": str(Path(custom_database) / "concatenated.hmm")}
                elif db_info["type"] == "msa_directory":
                    from rolypoly.utils.bio.alignments import hmmdb_from_directory

                    hmmdb_from_directory(
                        msa_dir=custom_database,
                        output=Path(custom_database) / "all_msa_built.hmm",
                        # alphabet="aa",
                    )
                    database_paths = {"Custom": str(Path(custom_database) / "all_msa_built.hmm")}
                else:
                    config.logger.error(f"Unsupported database directory type: {db_info['type']}")
                    return {}
            else:
                config.logger.error(f"Invalid custom database path: {custom_database}")
                return {}
        else:
            # For other tools, just use the path as is
            database_paths = {"Custom": custom_database}
    
    # Additional handling: if the user requested mmseqs2 and provided a directory
    # with MSAs, optionally build an mmseqs profile DB from that directory.
    if tool_name == "mmseqs2":
        # If the config indicates a directory, and db_create_mode requests mmseqs
        try:
            db_create_mode = config.db_create_mode
        except Exception:
            db_create_mode = "auto"
        for key, path in list(database_paths.items()):
            p = Path(str(path))
            if p.is_dir():
                # Decide whether to build mmseqs profile DB
                build_mmseqs = False
                if db_create_mode == "mmseqs":
                    build_mmseqs = True
                elif db_create_mode == "hmm":
                    build_mmseqs = False
                else:  # auto: if dir contains MSAs (.faa/.msa), build mmseqs profiles
                    msa_files = list(p.glob("*.faa")) + list(p.glob("*.msa")) + list(p.glob("*.afa"))
                    if len(msa_files) > 0:
                        build_mmseqs = True

                if build_mmseqs:
                    from rolypoly.utils.bio.alignments import mmseqs_profile_db_from_directory
                    mm_out = Path(os.environ.get("ROLYPOLY_DATA", ".")) / "mmseqs2" / p.name
                    mm_out_parent = mm_out.parent
                    mm_out_parent.mkdir(parents=True, exist_ok=True)
                    # default info table column names used by geNomad outputs
                    name_col = "MARKER"
                    accs_col = "ANNOTATION_ACCESSIONS"
                    desc_col = "ANNOTATION_DESCRIPTION"
                    mmseqs_profile_db_from_directory(
                        msa_dir=str(p),
                        output=str(mm_out),
                        msa_pattern="*.faa",
                        info_table=None,
                        name_col=name_col,
                        accs_col=accs_col,
                        desc_col=desc_col,
                    )
                    database_paths[key] = str(mm_out)
    else:
        requested_dbs = config.domain_db.split(",")
        database_paths = {}
        for db in requested_dbs:
            db_key = db.lower()
            if db_key in tool_db_paths:
                database_paths[db] = tool_db_paths[db_key]
            else:
                config.logger.warning(f"Database '{db}' is not supported for {tool_name}. Supported databases: {', '.join(tool_db_paths.keys())}")
    
    return database_paths





def search_protein_domains_hmmsearch(config):
    """Search protein domains using hmmsearch."""
    from rolypoly.utils.bio.alignments import search_hmmdb

    # Use the standard ORF prediction output location
    translation_output = config.output_dir / "raw_out" / "predicted_orfs.faa"
    if not translation_output.exists():
        config.logger.error(f"Translation output not found: {translation_output}. Make sure ORF prediction step completed successfully.")
        return

    # Get database paths
    database_paths = get_database_paths(config, "hmmsearch")
    if not database_paths:
        return

    global output_files
    config.logger.info(f"Using {', '.join(database_paths.keys())} for domain search")
    for db in database_paths.keys():
        config.logger.info(f"Searching {db} for domains")
        search_hmmdb(
            amino_file=translation_output,
            db_path=database_paths[db],
            output=config.output_dir / f"{db}_protein_domains.tsv",
            output_format="modomtblout",
            threads=config.threads,
            logger=config.logger,
            match_region=True,
            full_qseq=True,
            ali_str=True,
            inc_e=config.step_params["hmmsearch"]["inc_e"],
            mscore=config.step_params["hmmsearch"]["mscore"],
        )
        output_files = output_files.vstack(pl.DataFrame({"file": [str(config.output_dir / f"{db}_protein_domains.tsv")], "description": [f"protein domains for {db}"], "db": [db], "tool": ["hmmsearch"], "params": [str(config.step_params["hmmsearch"])], "command": [f"builtin via pyhmmer bindings: hmmsearch -E {config.step_params['hmmsearch']['inc_e']} -m {config.step_params['hmmsearch']['mscore']} {database_paths[db]} {translation_output}"]}))
        config.logger.info(f"Finished searching {db} for domains")


def predict_orfs_with_orffinder(config):
    """Predict ORFs using ORFfinder."""
    import os
    from shutil import which

    from rolypoly.utils.bio.translation import predict_orfs_orffinder

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
    output_file = config.output_dir / "raw_out" / "predicted_orfs.faa"
    predict_orfs_orffinder(
        input_fasta=config.input,
        output_file=config.output_dir / "raw_out" / "predicted_orfs.faa",
        genetic_code=config.genetic_code,
        min_orf_length=config.step_params["ORFfinder"]["minimum_length"],
        start_codon=config.step_params["ORFfinder"]["start_codon"],
        strand=config.step_params["ORFfinder"]["strand"],
        outfmt=config.step_params["ORFfinder"]["outfmt"],
        ignore_nested=config.step_params["ORFfinder"]["ignore_nested"],
    )
    global output_files
    output_files = output_files.vstack(pl.DataFrame({"file": [str(output_file)], "description": ["predicted ORFs"], "db": ["ORFfinder"], "tool": ["ORFfinder"], "params": [str(config.step_params["ORFfinder"])], "command": [f"ORFfinder -m {config.step_params['ORFfinder']['minimum_length']} -s {config.step_params['ORFfinder']['start_codon']} -l {config.step_params['ORFfinder']['strand']} -o {output_file} {config.input}"]}))

def search_protein_domains(config):
    config.logger.info("Searching for protein domains")

    if config.search_tool == "hmmsearch":
        search_protein_domains_hmmsearch(config)
    elif config.search_tool == "mmseqs2":
        search_protein_domains_mmseqs2(config)
    elif config.search_tool == "diamond":
        search_protein_domains_diamond(config)
    else:
        config.logger.info(
            f"Skipping protein domain search as {config.search_tool} is not supported"
        )

def search_protein_domains_mmseqs2(config):
    """Search protein domains using mmseqs2."""
    
    # Use the standard ORF prediction output location
    translation_output = config.output_dir / "raw_out" / "predicted_orfs.faa"
    if not translation_output.exists():
        config.logger.error(f"Translation output not found: {translation_output}. Make sure ORF prediction step completed successfully.")
        return

    # Get database paths
    database_paths = get_database_paths(config, "mmseqs2")
    if not database_paths:
        return

    global output_files
    config.logger.info(f"Using {', '.join(database_paths.keys())} for domain search")
    for db_name, db_path in database_paths.items():
        config.logger.info(f"Searching {db_name} for domains")
        output_file = config.output_dir / f"{db_name}_mmseqs2_domains.tsv"
        run_command_comp(
            "mmseqs",
            positional_args=[
                "easy-search",
                str(translation_output),
                str(db_path),
                str(output_file),
                str(config.output_dir / "tmp"),
            ],
            params={"threads": config.threads, "e": config.step_params["mmseqs2"]["evalue"], "c": config.step_params["mmseqs2"]["cov"]},
            logger=config.logger,
        )
        output_files = output_files.vstack(pl.DataFrame({"file": [str(output_file)], "description": [f"protein domains for {db_name}"], "db": [db_name], "tool": ["mmseqs2"], "params": [str(config.step_params["mmseqs2"])], "command": [f"ext. call mmseqs2: mmseqs easy-search {translation_output} {db_path} {output_file} {config.output_dir / 'tmp'} -t {config.threads} -e {config.step_params['mmseqs2']['evalue']} -c {config.step_params['mmseqs2']['cov']}"]}))
        config.logger.info(f"Finished searching {db_name} for domains")

def search_protein_domains_diamond(config):
    """Search protein domains using DIAMOND."""
    
    # Use the standard ORF prediction output location
    translation_output = config.output_dir / "raw_out" / "predicted_orfs.faa"
    if not translation_output.exists():
        config.logger.error(f"Translation output not found: {translation_output}. Make sure ORF prediction step completed successfully.")
        return

    # Get database paths
    database_paths = get_database_paths(config, "diamond")
    if not database_paths:
        return

    global output_files
    config.logger.info(f"Using {', '.join(database_paths.keys())} for domain search")
    for db_name, db_path in database_paths.items():
        config.logger.info(f"Searching {db_name} for domains")
        output_file = config.output_dir / f"{db_name}_diamond_domains.tsv"
        run_command_comp(
            "diamond",
            positional_args=["blastp"],
            params={
                "query": str(translation_output),
                "db": str(db_path),
                "out": str(output_file),
                "threads": config.threads,
                "outfmt": 6,
                "evalue": config.step_params["diamond"]["evalue"],
            },
            logger=config.logger,
        )
        output_files = output_files.vstack(pl.DataFrame({"file": [str(output_file)], "description": [f"protein domains for {db_name}"], "db": [db_name], "tool": ["diamond"], "params": [str(config.step_params["diamond"])], "command": [f"ext. call diamond: diamond blastp -d {db_path} -q {translation_output} -o {output_file} -t {config.threads} -e {config.step_params['diamond']['evalue']}"]}))
        config.logger.info(f"Finished searching {db_name} for domains")

def combine_results(config):
    import polars as pl

    config.logger.info("Combining annotation results")

    # Get translation output files
    translation_files = output_files.filter(
        pl.col("description").str.contains("predicted ORFs")
    )
    
    # Get domain search output files
    domain_files = output_files.filter(
        pl.col("description").str.contains("protein domains")
    )
    
    if translation_files.height == 0:
        config.logger.error("No translation files found for combining results")
        return
    
    if domain_files.height == 0:
        config.logger.warning("No domain search files found for combining results")
        # Still create a summary with just translation info
        translation_summary = translation_files.select([
            "file", "description", "db", "tool", "params", "command"
        ])
        output_file = config.output_dir / "annotation_results_summary.csv"
        translation_summary.write_csv(output_file)
        config.logger.info(f"Translation summary written to {output_file}")
        return

    # Create a comprehensive summary of all results
    all_results = pl.concat([translation_files, domain_files])
    
    # Add metadata about the analysis
    analysis_summary = pl.DataFrame({
        "analysis_type": ["protein_annotation"] * all_results.height,
        "timestamp": [pl.datetime.now()] * all_results.height,
        "input_file": [str(config.input)] * all_results.height,
        "output_dir": [str(config.output_dir)] * all_results.height,
    })
    
    # Combine everything
    combined_results = pl.concat([all_results, analysis_summary], how="horizontal")
    
    # Write comprehensive results
    output_file = config.output_dir / "combined_protein_annotations.csv"
    combined_results.write_csv(output_file)
    
    # Also create a simpler summary focusing on the domain search results
    if domain_files.height > 0:
        domain_summary = domain_files.select([
            "file", "description", "db", "tool"
        ]).with_columns([
            pl.col("file").map_elements(lambda x: Path(x).name, return_dtype=pl.Utf8).alias("filename"),
            pl.lit("Domain search results").alias("result_type")
        ])
        
        domain_summary_file = config.output_dir / "domain_search_summary.csv"
        domain_summary.write_csv(domain_summary_file)
        config.logger.info(f"Domain search summary written to {domain_summary_file}")

    config.logger.info(f"Combined annotation results written to {output_file}")
    config.logger.info(f"Total files processed: {all_results.height}")
    
    # Log summary statistics
    translation_count = translation_files.height
    domain_count = domain_files.height
    config.logger.info(f"Translation files: {translation_count}")
    config.logger.info(f"Domain search files: {domain_count}")
    
    # Log which tools and databases were used
    tools_used = all_results.select("tool").unique().to_series().to_list()
    dbs_used = all_results.select("db").unique().to_series().to_list()
    config.logger.info(f"Tools used: {', '.join(tools_used)}")
    config.logger.info(f"Databases used: {', '.join(dbs_used)}")


if __name__ == "__main__":
    annotate_prot()
