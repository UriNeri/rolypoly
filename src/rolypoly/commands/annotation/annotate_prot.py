import logging
import os
import shutil
from pathlib import Path
from typing import Union

import polars as pl
import rich_click as click
from polars.exceptions import NoDataError

from rolypoly.utils.bio.sequences import guess_fasta_alpha
from rolypoly.utils.logging.config import BaseConfig
from rolypoly.utils.various import run_command_comp

# global tools # TODO: add support for this directly here not just from annotate.py


def init_output_files_table() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "file": pl.Utf8,
            "description": pl.Utf8,
            "db": pl.Utf8,
            "tool": pl.Utf8,
            "params": pl.Utf8,
            "command": pl.Utf8,
        }
    )


global output_files
output_files = init_output_files_table()

#schemas
INFO_TABLE_SPECS = {
    "nvpc": {
        "relative_path": Path("profiles") / "NVPC_descriptions.csv.gz",
        "fallback_relative_paths": [Path("profiles") / "NVPC_descriptions.csv"],
        "join_column": "profile_accession",
        "prefix": "nvpc_meta",
        "columns": ["profile_accession", "Name", "Description", "neff", "nseq"],
        "read_csv_kwargs": {"separator": ",", "has_header": True},
    },
    "genomad": {
        "relative_path": Path("profiles")
        / "genomad_rna_viral_markers_with_annotation.csv.gz",
        "fallback_relative_paths": [
            Path("profiles") / "genomad_rna_viral_markers_with_annotation.csv"
        ],
        "join_column": "MARKER",
        "prefix": "genomad_meta",
        "columns": [
            "MARKER",
            "ANNOTATION_ACCESSIONS",
            "ANNOTATION_DESCRIPTION",
            "TAXONOMY",
        ],
        "read_csv_kwargs": {"separator": ",", "has_header": True},
    },
    "vfam": {
        "relative_path": Path("profiles") / "vfam.annotations.tsv.gz",
        "fallback_relative_paths": [Path("profiles") / "vfam.annotations.tsv"],
        "join_column": "GroupName",
        "prefix": "vfam_meta",
        "columns": [
            "GroupName",
            "ProteinCount",
            "SpeciesCount",
            "FunctionalCategory",
            "ConsensusFunctionalDescription",
        ],
        "rename_columns": {"#GroupName": "GroupName"},
        "read_csv_kwargs": {"separator": ",", "has_header": True},
    },
    "uniref50": {
        "relative_path": Path("reference_seqs")
        / "uniref"
        / "uniref50_viral.tsv.gz",
        "fallback_relative_paths": [
            Path("reference_seqs") / "uniref" / "uniref50_viral.tsv"
        ],
        "join_column": "Cluster_ID",
        "prefix": "uniref50_meta",
        "columns": [
            "Cluster_ID",
            "Cluster Name",
            "Types",
            "Size",
            "Organisms",
            "Length",
            "Identity",
            "Cluster members",
        ],
        "rename_columns": {"Cluster ID": "Cluster_ID"},
        "read_csv_kwargs": {"separator": "\t", "has_header": True},
    },
}


class ProteinAnnotationConfig(BaseConfig):
    """Configuration for protein annotation pipeline"""

    def __init__(
        self,
        input: Path,
        output_dir: Path,
        threads: int,
        log_file: Union[Path, logging.Logger, None],
        log_level: str = "INFO",
        memory: str | None = None,
        override_parameters: dict[str, object] = {},
        skip_steps: list[str] = [],
        search_tool: str = "hmmsearch",
        domain_db: str = "Pfam",
        min_orf_length: int = 30,
        genetic_code: int = 11,
        gene_prediction_tool: str = "ORFfinder",
        evalue: float = 1e-2,
        db_create_mode: str = "auto",
        output_format: str = "tsv",
        resolve_mode: str = "simple",
        min_overlap_positions: int = 10,
        include_alignment_strings: bool = True,
        temp_dir: Union[Path, str, None] = None,
        keep_tmp: bool = False,
        **kwargs,
    ):
        # Extract BaseConfig parameters
        base_config_params = {
            "input": input,
            "output": output_dir,
            "threads": threads,
            "log_file": log_file,
            "log_level": log_level,
            "memory": memory,
            "temp_dir": temp_dir,
            "keep_tmp": keep_tmp,
        }
        super().__init__(**base_config_params)

        self.skip_steps = skip_steps or []
        self.search_tool = search_tool
        self.domain_db = normalize_domain_db_value(domain_db)
        self.min_orf_length = min_orf_length
        self.genetic_code = genetic_code
        self.gene_prediction_tool = gene_prediction_tool
        self.evalue = evalue
        self.db_create_mode = db_create_mode
        self.output_format = output_format
        self.resolve_mode = resolve_mode
        self.min_overlap_positions = min_overlap_positions
        self.include_alignment_strings = include_alignment_strings
        self.step_params = {
            "ORFfinder": {
                "minimum_length": min_orf_length,
                "start_codon": 1,
                "strand": "both",
                "outfmt": 0,
                "ignore_nested": False,
            },
            "pyrodigal": {"minimum_length": min_orf_length},
            "six-frame": {"threads": 1, "minimum_length": min_orf_length},
            "hmmsearch": {"inc_e": evalue, "mscore": 8, "min_ali_len": 10},
            "diamond": {"evalue": evalue},
            "mmseqs2": {"evalue": evalue, "cov": 0.1},
        }

        if override_parameters:
            for step, params in override_parameters.items():
                if step in self.step_params:
                    self.step_params[step].update(params)
                else:
                    print(
                        f"Warning: Unknown step '{step}' in override_parameters. Ignoring."
                    )


def normalize_domain_db_value(domain_db: object) -> str:
    if isinstance(domain_db, (tuple, list, set)):
        values = [str(item) for item in domain_db if str(item)]
        return ",".join(values) if values else "Pfam"
    if domain_db is None:
        return "Pfam"
    return str(domain_db)


def stage_protein_input_as_orfs(config) -> bool:
    input_path = Path(config.input)
    if not input_path.exists() or not input_path.is_file():
        return False

    if guess_fasta_alpha(str(input_path)) != "amino":
        return False

    output_file = config.output_dir / "predicted_orfs.faa"
    if input_path.resolve() != output_file.resolve():
        shutil.copyfile(input_path, output_file)

    global output_files
    output_files = output_files.vstack(
        pl.DataFrame(
            {
                "file": [str(output_file)],
                "description": ["provided amino-acid input (used as ORFs)"],
                "db": ["input"],
                "tool": ["input_protein"],
                "params": ["{}"],
                "command": [
                    f"copy input protein FASTA: {input_path} -> {output_file}"
                ],
            }
        )
    )
    return True


@click.command()
@click.option(
    "-i",
    "--input",
    required=True,
    help="Fasta file or input directory containing rolypoly's virus identification results",
)
@click.option(
    "-o",
    "--output-dir",
    default="./annotate_prot_output",
    help="Output directory path",
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
    default="pyrodigal",
    type=click.Choice(
        ["ORFfinder", "pyrodigal", "six-frame"],  # , "bbmap"``
        case_sensitive=False,
    ),
    help="""Tool for gene prediction. \n
    * pyrodigal-rv: might work well for some viruses, but it's not as well tested for RNA viruses. Includes internal genetic code assignment. \n
    * ORFfinder: The default ORFfinder settings may have some false positives, but it's fast and easy to use. \n
    * six-frame: includes all 6 reading frames, so all possible ORFs are predicted - prediction is quick but will include many false positives, and the input for the domain search will be larger. \n
    """,
)
@click.option(
    "-st",
    "--search-tool",
    default="hmmsearch",
    type=click.Choice(
        ["hmmsearch", "mmseqs2", "diamond"],
        case_sensitive=False,  # , "nail"
    ),
    help="Tool/command for protein domain detection. Only one tool can be used at a time.",
)
@click.option(
    "-d",
    "--domain-db",
    default="Pfam,NVPC",
    type=str,
    help="""comma-separated list of database(s) for domain detection. \n
    * Pfam: Pfam-A (only hmmsearch) \n
    * RVMT: RVMT RdRp profiles \n
    * NVPC: RVMT's New Viral Profile Clusters, filtered to remove "hypothetical" proteins \n
    * genomad: genomad virus-specific markers - note these can be good for identification but not ideal for annotation. \n
    * vfam: VFam profiles from VOGDB release 236, filtered to remove low-information profiles \n
    * uniref50: UniRef50 viral subset (for diamond only) \n
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
    help="Genetic code (a.k.a. translation table) NOT REALLY USED CURRENTLY",
)
@click.option(
    "-e",
    "--evalue",
    default=1e-3,
    help="E-value for search result filtering. Note, this is for inital filteringg only, you are encouraged to filter the results further using e.g. profile coverage and scores.",
)
@click.option(
    "--db-create-mode",
    default="auto",
    type=click.Choice(["auto", "mmseqs", "hmm"], case_sensitive=False),
    help="How to handle custom database directories: auto=guess, mmseqs=build mmseqs profile DB, hmm=build concatenated HMM",
)
@click.option(
    "--output-format",
    default="tsv",
    type=click.Choice(["tsv", "csv", "gff3"], case_sensitive=False),
    help="Output format for the combined results",
)
@click.option(
    "-rm",
    "--resolve-mode",
    default="simple",
    type=click.Choice(
        [
            "merge",
            "one_per_range",
            "one_per_query",
            "split",
            "drop_contained",
            "none",
            "simple",
        ]
    ),
    help="""How to deal with overlapping domain hits in the same query sequence. \n
        - merge: all overlapping hits are merged into one range \n
        - one_per_range: one hit per range (ali_from-ali_to) is reported \n
        - one_per_query: one hit per query sequence is reported \n
        - split: each overlapping domain is split into a new row \n
        - drop_contained: hits that are contained within (i.e. enveloped by) other hits are dropped \n
        - none: no resolution of overlapping hits is performed \n
        - simple: heuristic-based approach - chains drop_contained with adaptive overlap detection for polyproteins \n
        """,
)
@click.option(
    "-mo",
    "--min-overlap-positions",
    default=10,
    help="Minimal number of overlapping positions between two intersecting ranges before they are considered as overlapping (used in some resolve_mode(s)). With 'simple' mode, this is adaptively adjusted for polyprotein detection.",
)
@click.option(
    "--alignment-strings/--no-alignment-strings",
    default=True,
    help="Include alignment identity strings in hmmsearch outputs (applies to modomtblout format).",
)
def annotate_prot(
    input,
    output_dir,
    threads,
    log_file,
    log_level,
    memory,
    override_parameters,
    skip_steps,
    gene_prediction_tool,
    search_tool,
    domain_db,
    min_orf_length,
    genetic_code,
    evalue,
    db_create_mode,
    output_format,
    resolve_mode,
    min_overlap_positions,
    alignment_strings,
):
    """Identify coding sequences (ORFs) from fasta, and predicts their translated seqs putative function via homology search. \n
    Currently supported tools and databases: \n
    * Translations: ORFfinder, pyrodigal, six-frame \n
    * Search engines: \n
    - (py)hmmsearch: Pfam, NVPC, RVMT, genomad, vfam \n
    - mmseqs2: NVPC, RVMT, genomad, vfam \n
    - diamond: Uniref50 (viral subset) \n
    * custom: user supplied database. Needs to be in tool appropriate format, or a directory of aligned fasta files (for hmmsearch)
    """
    # - nail: Pfam, RVMT, genomad, custom (via nail) # TODO: add support for nail. https://github.com/TravisWheelerLab/nail
    import json

    from rolypoly.utils.various import ensure_memory

    domain_db = normalize_domain_db_value(domain_db)

    config = ProteinAnnotationConfig(
        input=input,
        output_dir=output_dir,
        threads=threads,
        log_file=log_file,
        log_level=log_level,
        memory=ensure_memory(memory)["giga"],
        override_parameters=(
            json.loads(override_parameters) if override_parameters else {}
        ),
        skip_steps=skip_steps.split(",") if skip_steps else [],
        search_tool=search_tool,
        domain_db=domain_db,
        min_orf_length=min_orf_length,
        gene_prediction_tool=gene_prediction_tool,
        genetic_code=genetic_code,
        evalue=evalue,
        db_create_mode=db_create_mode,
        output_format=output_format,
        resolve_mode=resolve_mode,
        min_overlap_positions=min_overlap_positions,
        include_alignment_strings=alignment_strings,
    )

    # config.logger.info(f"Using {config.search_tool} for domain search")
    try:
        process_protein_annotations(config)
    except Exception as e:
        config.logger.warning(
            f"An error occurred during protein annotation: {str(e)}"
        )
        raise


def process_protein_annotations(config):
    """MAIN LOGIC HERE."""
    global output_files
    output_files = init_output_files_table()

    config.logger.info("Starting protein annotation process")

    if stage_protein_input_as_orfs(config):
        config.logger.info(
            "Detected amino-acid input; using it directly as predicted ORFs and skipping ORF prediction"
        )
        if "predict_orfs" not in config.skip_steps:
            config.skip_steps.append("predict_orfs")

    # create a "raw_out" subdirectory in output folder
    raw_out_dir = config.output_dir / "raw_out"
    raw_out_dir.mkdir(parents=True, exist_ok=True)
    config.logger.debug(f"created raw_out directory: {raw_out_dir}")

    steps = [
        predict_orfs,  # i.e. call genes
        search_protein_domains,
        resolve_domain_overlaps,  # Resolve overlapping domain hits
        combine_results,
    ]

    # if config.search_tool in ["diamond", "mmseqs2"]:
    #     config.skip_steps.append("resolve_domain_overlaps")

    for step in steps:
        step_name = step.__name__
        if step_name not in config.skip_steps:
            config.logger.info(f"Starting step: {step_name}")
            step(config)
        else:
            config.logger.info(f"Skipping step: {step_name}")

    config.logger.info("Protein annotation process completed successfully")
    output_files.write_csv(
        config.output_dir / "output_files.tsv", separator="\t"
    )


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
    from rolypoly.utils.bio.translation import pyro_predict_orfs

    output_file = config.output_dir / "predicted_orfs.faa"
    pyro_predict_orfs(
        input_file=config.input,
        output_file=output_file,
        threads=config.threads,
        # genetic_code=config.step_params["pyrodigal"]["genetic_code"],
        min_gene_length=config.step_params["pyrodigal"]["minimum_length"],
    )
    global output_files
    output_files = output_files.vstack(
        pl.DataFrame(
            {
                "file": [str(output_file)],
                "description": ["predicted ORFs"],
                "db": ["pyrodigal"],
                "tool": ["pyrodigal"],
                "params": [str(config.step_params["pyrodigal"])],
                "command": [
                    f"pyrodigal via pyrodigal module: threads={config.threads}"
                ],
            }
        )
    )
    return output_file


def predict_orfs_with_six_frame(config):
    """Translate 6-frame reading frames of a DNA sequence using seqkit."""
    from rolypoly.utils.bio.translation import translate_6frx_seqkit

    output_file = str(config.output_dir / "predicted_orfs.faa")
    translate_6frx_seqkit(str(config.input), output_file, config.threads)
    global output_files
    output_files = output_files.vstack(
        pl.DataFrame(
            {
                "file": [output_file],
                "description": ["predicted ORFs"],
                "db": ["six-frame"],
                "tool": ["six-frame"],
                "params": [str(config.step_params["six-frame"])],
                "command": [
                    f"ext. call seqkit: seqkit -w0 translate -j {config.threads} {config.input} > {output_file}"
                ],
            }
        )
    )
    return output_file


def get_database_paths(config, tool_name):
    """Get database paths for the specified tool with validation"""
    import os

    hmmdbdir = Path(os.environ["ROLYPOLY_DATA"]) / "profiles" / "hmmdbs"
    mmseqs2_dbdir = (
        Path(os.environ["ROLYPOLY_DATA"]) / "profiles" / "mmseqs_dbs"
    )
    reference_seqs_dir = Path(os.environ["ROLYPOLY_DATA"]) / "reference_seqs"
    # diamond_dbdir = Path(os.environ["ROLYPOLY_DATA"]) / "profiles" / "diamond" # not needed really , will just use the fasta as input cause diamond accepts fasta directly

    # Database paths for different tools
    DB_PATHS = {
        "hmmsearch": {
            "NVPC".lower(): hmmdbdir / "nvpc.hmm",
            "RVMT".lower(): hmmdbdir / "rvmt.hmm",
            "Pfam".lower(): hmmdbdir / "Pfam-A.hmm",
            "genomad".lower(): hmmdbdir / "genomad_rna_viral_markers.hmm",
            "vfam".lower(): hmmdbdir / "vfam.hmm",
        },
        "mmseqs2": {
            "NVPC".lower(): mmseqs2_dbdir / "nvpc/nvpc",
            "RVMT".lower(): mmseqs2_dbdir / "RVMT/RVMT",
            "vfam".lower(): mmseqs2_dbdir / "vfam/vfam",
            "Pfam".lower(): mmseqs2_dbdir / "pfam_a/pfam_a_38_seed",
            "genomad".lower(): mmseqs2_dbdir / "genomad/rna_viral_markers",
        },
        "diamond": {
            "uniref50".lower(): reference_seqs_dir
            / "uniref/uniref50_viral.fasta.gz",
            "RVMT".lower(): reference_seqs_dir / "RVMT/RVMT_cleaned_orfs.faa.gz",
        },
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
            config.logger.error(
                f"Custom database path {custom_database} does not exist"
            )
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
                # determine if the directory contains hmms or msas, look at file extensions
                list_of_files = list(Path(custom_database).glob("*"))
                unique_extensions = set(
                    [f.suffix.lower() for f in list_of_files if f.is_file()]
                )
                if ".hmm" in unique_extensions:
                    db_type = "hmm_directory"
                elif unique_extensions.intersection(
                    {".faa", ".msa", ".afa", ".fasta"}
                ):
                    db_type = "msa_directory"
                config.logger.info(
                    f"Database directory analysis: {db_type} detected based on file extensions"
                )
                # concatenate into the same path as the input directory, but with .hmm suffix
                db_info = {
                    "type": db_type,
                    "path": custom_database.rstrip("/") + ".hmm",
                }
                if db_type == "hmm_directory":
                    # concatenate all hmms into one file
                    with open(Path(db_info["path"]), "w") as f_out:
                        for hmm_file in list_of_files:
                            with open(hmm_file, "r") as f_in:
                                f_out.write(f_in.read())
                    database_paths = {"Custom": str(Path(db_info["path"]))}
                elif db_info["type"] == "msa_directory":
                    from rolypoly.utils.bio.alignments import (
                        hmmdb_from_directory,
                    )

                    hmmdb_from_directory(
                        msa_dir=custom_database,
                        output=Path(db_info["path"]),
                        # alphabet="aa",
                    )
                    database_paths = {"Custom": str(Path(db_info["path"]))}
                else:
                    config.logger.error(
                        f"Unsupported database directory type: {db_info['type']}"
                    )
                    return {}
            else:
                config.logger.error(
                    f"Invalid custom database path: {custom_database}"
                )
                return {}
        else:
            # For other tools, just use the path as is
            database_paths = {"Custom": custom_database}

        # Additional handling: if the user requested mmseqs2 and provided a directory
        # with MSAs, optionally build an mmseqs profile DB from that directory. # TODO: test this
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
                        msa_files = (
                            list(p.glob("*.faa"))
                            + list(p.glob("*.msa"))
                            + list(p.glob("*.afa"))
                        )
                        if len(msa_files) > 0:
                            build_mmseqs = True

                    if build_mmseqs:
                        from rolypoly.utils.bio.alignments import (
                            mmseqs_profile_db_from_directory,
                        )

                        mm_out = (
                            Path(os.environ.get("ROLYPOLY_DATA", "."))
                            / "mmseqs2"
                            / p.name
                        )
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
            db_key = db.lower()  # remember to lower case for matching!!!
            if db_key in tool_db_paths:
                database_paths[db] = tool_db_paths[db_key]
            else:
                config.logger.warning(
                    f"Database '{db}' is not supported for {tool_name}. Supported databases: {', '.join(tool_db_paths.keys())}"
                )

    return database_paths


def search_protein_domains_hmmsearch(config):
    """Search protein domains using hmmsearch."""
    from rolypoly.utils.bio.alignments import search_hmmdb

    # Use the standard ORF prediction output location
    translation_output = config.output_dir / "predicted_orfs.faa"
    if not translation_output.exists():
        config.logger.error(
            f"Translation output not found: {translation_output}. Make sure ORF prediction step completed successfully."
        )
        return

    # Get database paths
    database_paths = get_database_paths(config, "hmmsearch")
    if not database_paths:
        config.logger.error(
            f"No valid databases found for hmmsearch. Requested: {config.domain_db}. "
            f"Supported: Pfam, NVPC, RVMT, genomad, vfam. Please check your --domain-db parameter."
        )
        raise ValueError("No valid databases found for hmmsearch search")

    global output_files
    config.logger.info(
        f"Using {', '.join(database_paths.keys())} for domain search"
    )
    for db in database_paths.keys():
        config.logger.info(f"Searching with {db}...")
        search_hmmdb(
            amino_file=translation_output,
            db_path=database_paths[db],
            output=config.output_dir / f"{db}_protein_domains.tsv",
            output_format="modomtblout",
            threads=config.threads,
            logger=config.logger,
            match_region=False,
            full_qseq=False,
            ali_str=config.include_alignment_strings,
            inc_e=config.step_params["hmmsearch"]["inc_e"],
            mscore=config.step_params["hmmsearch"]["mscore"],
            min_ali_len=config.step_params["hmmsearch"]["min_ali_len"]
        )
        output_files = output_files.vstack(
            pl.DataFrame(
                {
                    "file": [
                        str(config.output_dir / f"{db}_protein_domains.tsv")
                    ],
                    "description": [f"protein domains for {db}"],
                    "db": [db],
                    "tool": ["hmmsearch"],
                    "params": [str(config.step_params["hmmsearch"])],
                    "command": [
                        f"builtin via pyhmmer bindings: hmmsearch -E {config.step_params['hmmsearch']['inc_e']} -m {config.step_params['hmmsearch']['mscore']} {database_paths[db]} {translation_output}"
                    ],
                }
            )
        )
        config.logger.info(f"Finished searching {db} for domains")


def predict_orfs_with_orffinder(config):
    """Predict ORFs using ORFfinder."""
    from shutil import which

    from rolypoly.utils.bio.translation import predict_orfs_orffinder

    if not which("ORFfinder"):
        config.logger.error(
            "ORFfinder not found. Please install ORFfinder and add it to your PATH (it isn't a conda/mamba installable package, but you can do the following:  wget ftp://ftp.ncbi.nlm.nih.gov/genomes/TOOLS/ORFfinder/linux-i64/ORFfinder.gz; gunzip ORFfinder.gz; chmod a+x ORFfinder; mv ORFfinder $CONDA_PREFIX/bin)."
        )
        # lazy = input(
        #     "Do you want to install ORFfinder for you (i.e. ran the above commands)? [yes/no]  "
        # )
        lazy = "yes"  # most people don't care
        if lazy.lower() == "yes":
            import subprocess as sp
            sp.run(
                "wget ftp://ftp.ncbi.nlm.nih.gov/genomes/TOOLS/ORFfinder/linux-i64/ORFfinder.gz; gunzip ORFfinder.gz; chmod a+x ORFfinder; mv ORFfinder $CONDA_PREFIX/bin",
                shell=True,
                check=True,
            )
            config.logger.info("ORFfinder installed successfully")
        else:
            config.logger.error(
                "ORFfinder not found, you don't want me to install it, and you don't want to use another tool         seriously. Exiting    "
            )
            exit(1)

    config.logger.info("Predicting ORFs")
    output_file = config.output_dir / "predicted_orfs.faa"
    predict_orfs_orffinder(
        input_fasta=config.input,
        output_file=config.output_dir / "predicted_orfs.faa",
        genetic_code=config.genetic_code,
        min_orf_length=config.step_params["ORFfinder"]["minimum_length"],
        start_codon=config.step_params["ORFfinder"]["start_codon"],
        strand=config.step_params["ORFfinder"]["strand"],
        outfmt=config.step_params["ORFfinder"]["outfmt"],
        ignore_nested=config.step_params["ORFfinder"]["ignore_nested"],
    )
    global output_files
    output_files = output_files.vstack(
        pl.DataFrame(
            {
                "file": [str(output_file)],
                "description": ["predicted ORFs"],
                "db": ["ORFfinder"],
                "tool": ["ORFfinder"],
                "params": [str(config.step_params["ORFfinder"])],
                "command": [
                    f"ORFfinder -m {config.step_params['ORFfinder']['minimum_length']} -s {config.step_params['ORFfinder']['start_codon']} -l {config.step_params['ORFfinder']['strand']} -o {output_file} {config.input}"
                ],
            }
        )
    )


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
    translation_output = config.output_dir / "predicted_orfs.faa"
    if not translation_output.exists():
        config.logger.error(
            f"Translation output not found: {translation_output}. Make sure ORF prediction step completed successfully."
        )
        return

    # Get database paths
    database_paths = get_database_paths(config, "mmseqs2")
    if not database_paths:
        config.logger.error(
            f"No valid databases found for mmseqs2. Requested: {config.domain_db}. "
            f"Supported: NVPC, RVMT, vfam, Pfam, genomad. Please check your --domain-db parameter."
        )
        raise ValueError("No valid databases found for mmseqs2 search")

    global output_files
    config.logger.info(
        f"Using {', '.join(database_paths.keys())} for domain search"
    )
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
            positional_args_location="start",
            params={
                "threads": config.threads,
                "e": config.step_params["mmseqs2"]["evalue"],
                "c": config.step_params["mmseqs2"]["cov"],
                "format-mode": 2,
            },
            logger=config.logger,
            output_file=str(output_file),
        )
        if output_file.exists() and output_file.stat().st_size > 0:
            mmseqs_columns = [
                "qseqid",
                "sseqid",
                "pident",
                "length",
                "mismatch",
                "gapopen",
                "qstart",
                "qend",
                "sstart",
                "send",
                "evalue",
                "bitscore",
                "qlen",
                "tlen",
            ]
            suffix_pattern = r"(?:\.faa\.msa\.Cons\.msa|\.msa\.Cons\.msa|\.Cons\.msa|\.faa\.msa|\.msa)$"
            try:
                cleaned = pl.read_csv(
                    output_file,
                    separator="\t",
                    has_header=False,
                    new_columns=mmseqs_columns,
                ).with_columns(
                    pl.col("sseqid")
                    .str.replace_all(suffix_pattern, "", literal=False)
                    .alias("sseqid")
                )
                cleaned.write_csv(
                    output_file, separator="\t", include_header=False
                )
            except NoDataError:
                config.logger.debug(
                    f"No data found in {output_file}, skipping suffix cleanup"
                )
        output_files = output_files.vstack(
            pl.DataFrame(
                {
                    "file": [str(output_file)],
                    "description": [f"protein domains for {db_name}"],
                    "db": [db_name],
                    "tool": ["mmseqs2"],
                    "params": [str(config.step_params["mmseqs2"])],
                    "command": [
                        f"ext. call mmseqs2: mmseqs easy-search {translation_output} {db_path} {output_file} {config.output_dir / 'tmp'} -t {config.threads} -e {config.step_params['mmseqs2']['evalue']} -c {config.step_params['mmseqs2']['cov']}"
                    ],
                }
            )
        )
        config.logger.info(f"Finished searching {db_name} for domains")


def search_protein_domains_diamond(config):
    """Search protein domains using DIAMOND."""

    # Use the standard ORF prediction output location
    translation_output = config.output_dir / "predicted_orfs.faa"
    if not translation_output.exists():
        config.logger.error(
            f"Translation output not found: {translation_output}. Make sure ORF prediction step completed successfully."
        )
        return

    # Get database paths
    database_paths = get_database_paths(config, "diamond")
    if not database_paths:
        config.logger.error(
            f"No valid databases found for diamond. Requested: {config.domain_db}. "
            f"Supported: uniref50, RVMT. Please check your --domain-db parameter."
        )
        raise ValueError("No valid databases found for diamond search")

    global output_files
    config.logger.info(
        f"Using {', '.join(database_paths.keys())} for domain search"
    )
    for db_name, db_path in database_paths.items():
        config.logger.info(f"Searching {db_name} for domains")
        output_file = config.output_dir / f"{db_name}_diamond_domains.tsv"
        run_command_comp(
            "diamond",
            positional_args=["blastp"],
            positional_args_location="start",
            params={
                "query": str(translation_output),
                "db": str(db_path),
                "out": str(output_file),
                "threads": config.threads,
                "outfmt": "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen",
                "evalue": config.step_params["diamond"]["evalue"],
            },
            logger=config.logger,
            output_file=str(output_file),
        )
        output_files = output_files.vstack(
            pl.DataFrame(
                {
                    "file": [str(output_file)],
                    "description": [f"protein domains for {db_name}"],
                    "db": [db_name],
                    "tool": ["diamond"],
                    "params": [str(config.step_params["diamond"])],
                    "command": [
                        f"ext. call diamond: diamond blastp -d {db_path} -q {translation_output} -o {output_file} -t {config.threads} -e {config.step_params['diamond']['evalue']}"
                    ],
                }
            )
        )
        config.logger.info(f"Finished searching {db_name} for domains")


def resolve_domain_overlaps(config):
    """Resolve overlapping domain hits using consolidate_hits."""
    import polars as pl

    from rolypoly.utils.bio.interval_ops import consolidate_hits

    global output_files  # Declare at the start of function

    config.logger.info("Resolving overlapping domain hits")

    # Get domain search output files
    domain_files = output_files.filter(
        pl.col("description").str.contains("protein domains")
    )

    if domain_files.height == 0:
        config.logger.info("No domain files to process for overlap resolution")
        return

    # Process each domain file
    for row in domain_files.iter_rows(named=True):
        domain_file = Path(row["file"])
        tool = row["tool"]

        if not domain_file.exists() or domain_file.stat().st_size == 0:
            config.logger.warning(
                f"Domain file {domain_file} is empty or doesn't exist, skipping"
            )
            continue

        config.logger.info(f"Resolving overlaps in {domain_file.name}")

        try:
            # Read domain hits with appropriate headers based on tool
            if tool == "diamond":
                # Diamond BLAST format with qlen/slen for adaptive overlap detection
                diamond_columns = [
                    "query_id",
                    "subject_id",
                    "pident",
                    "length",
                    "mismatch",
                    "gapopen",
                    "qstart",
                    "qend",
                    "sstart",
                    "send",
                    "evalue",
                    "bitscore",
                    "qlen",
                    "slen",
                ]
                domain_df = pl.read_csv(
                    domain_file,
                    separator="\t",
                    has_header=False,
                    new_columns=diamond_columns,
                )
                column_specs = "query_id,subject_id"
                rank_columns = "-bitscore,+evalue"
            elif tool == "mmseqs2":
                # MMSeqs2 BLAST-TAB format with qlen/tlen for adaptive overlap detection
                mmseqs_columns = [
                    "qseqid",
                    "sseqid",
                    "pident",
                    "length",
                    "mismatch",
                    "gapopen",
                    "qstart",
                    "qend",
                    "sstart",
                    "send",
                    "evalue",
                    "bitscore",
                    "qlen",
                    "tlen",
                ]
                domain_df = pl.read_csv(
                    domain_file,
                    separator="\t",
                    has_header=False,
                    new_columns=mmseqs_columns,
                )
                column_specs = "qseqid,sseqid"
                rank_columns = "-bitscore,+evalue"
            else:
                # HMMER or other format - assume headers are present
                domain_df = pl.read_csv(domain_file, separator="\t")
                column_specs = "query_full_name,hmm_full_name"
                rank_columns = "-full_hmm_score,+full_hmm_evalue,-hmm_cov"

            if domain_df.height == 0:
                config.logger.info(f"No hits in {domain_file.name}, skipping")
                continue

            # Resolve overlaps based on user-specified mode
            if config.resolve_mode == "simple":
                # Use adaptive 'simple' mode for overlap resolution with polyprotein detection
                resolved_df = consolidate_hits(
                    input=domain_df,
                    column_specs=column_specs,
                    rank_columns=rank_columns,
                    one_per_query=False,
                    one_per_range=True,
                    min_overlap_positions=config.min_overlap_positions,
                    merge=False,
                    split=False,
                    drop_contained=True,
                    alphabet="aa",
                    adaptive_overlap=True,
                )
            elif config.resolve_mode != "none":
                # Use specified resolve mode
                resolve_mode_dict = {
                    "split": False,
                    "one_per_range": False,
                    "one_per_query": False,
                    "merge": False,
                    "drop_contained": False,
                }
                resolve_mode_dict[config.resolve_mode] = True
                resolved_df = consolidate_hits(
                    input=domain_df,
                    min_overlap_positions=config.min_overlap_positions,
                    column_specs=column_specs,
                    rank_columns=rank_columns,
                    alphabet="aa",
                    **resolve_mode_dict,
                )
            else:
                # No resolution
                resolved_df = domain_df

            # Write resolved results
            resolved_file = (
                domain_file.parent / f"{domain_file.stem}_resolved.tsv"
            )
            resolved_df.write_csv(resolved_file, separator="\t")

            config.logger.info(
                f"Resolved {domain_df.height} hits to {resolved_df.height} non-overlapping hits. "
                f"Output: {resolved_file}"
            )

            # Update output_files to include resolved file
            output_files = output_files.vstack(
                pl.DataFrame(
                    {
                        "file": [str(resolved_file)],
                        "description": [f"resolved {row['description']}"],
                        "db": [row["db"]],
                        "tool": [f"{row['tool']}_resolved"],
                        "params": [row["params"]],
                        "command": [
                            f"{row['command']} | consolidate_hits(adaptive_overlap=True)"
                        ],
                    }
                )
            )

        except Exception as e:
            config.logger.error(
                f"Error resolving overlaps in {domain_file}: {e}"
            )
            continue

    config.logger.info("Domain overlap resolution completed")


def combine_results(config):
    """Combine annotation results and write in requested format."""
    import shutil

    import polars as pl

    config.logger.info("Combining annotation results")

    # Get domain search output files (prefer resolved versions)
    resolved_files = output_files.filter(
        pl.col("description").str.contains("resolved")
    )

    if resolved_files.height > 0:
        # Use resolved files if available
        domain_files = resolved_files
        config.logger.info(
            f"Using {resolved_files.height} resolved domain files"
        )
    else:
        # Fall back to unresolved domain files
        domain_files = output_files.filter(
            pl.col("description").str.contains("protein domains")
        )
        config.logger.info(
            f"Using {domain_files.height} unresolved domain files"
        )

    if domain_files.height == 0:
        config.logger.warning(
            "No domain search files found for combining results"
        )
        return

    # Load and combine all domain search results
    all_domain_data = []
    for row in domain_files.iter_rows(named=True):
        try:
            df = pl.read_csv(row["file"], separator="\t")

            if config.search_tool in ["diamond", "mmseqs2"]:
                # Add headers to diamond/mmseqs2 output
                # diamond format: qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen
                # mmseqs2 format (format-mode 2): qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen tlen
                if config.search_tool == "diamond":
                    col_names = [
                        "qseqid",
                        "sseqid",
                        "pident",
                        "length",
                        "mismatch",
                        "gapopen",
                        "qstart",
                        "qend",
                        "sstart",
                        "send",
                        "evalue",
                        "bitscore",
                        "qlen",
                        "slen",
                    ]
                else:  # mmseqs2
                    col_names = [
                        "qseqid",
                        "sseqid",
                        "pident",
                        "length",
                        "mismatch",
                        "gapopen",
                        "qstart",
                        "qend",
                        "sstart",
                        "send",
                        "evalue",
                        "bitscore",
                        "qlen",
                        "tlen",
                    ]
                df.columns = col_names

            # Add metadata columns
            df = df.with_columns(
                [
                    pl.lit(row["db"]).alias("database"),
                    pl.lit(row["tool"]).alias("search_tool"),
                ]
            )
            if "profile_accession" not in df.columns:
                accession_source = None
                for candidate in [
                    "target_accession",
                    "accession",
                    "sseqid",
                    "target_name",
                ]:
                    if candidate in df.columns:
                        accession_source = candidate
                        break
                if accession_source:
                    df = df.with_columns(
                        pl.col(accession_source).alias("profile_accession")
                    )
                else:
                    df = df.with_columns(
                        pl.lit(row["db"]).alias("profile_accession")
                    )
            all_domain_data.append(df)
        except Exception as e:
            config.logger.warning(f"Could not read {row['file']}: {e}")
            continue

    if not all_domain_data:
        config.logger.error("No valid domain search data to combine")
        return

    # Combine all domain data
    combined_data = pl.concat(all_domain_data, how="diagonal")

    if "profile_accession" in combined_data.columns:
        combined_data = combined_data.with_columns(
            pl.col("profile_accession").cast(pl.Utf8).str.strip_chars()
        )

    combined_data = enrich_with_info_tables(combined_data, config.logger)

    from rolypoly.utils.bio.polars_fastx import (
        add_missing_gff_columns,
        normalize_column_names,
    )

    combined_data = normalize_column_names(combined_data)

    # Write output in requested format
    if config.output_format == "gff3":
        combined_data = add_missing_gff_columns(
            combined_data, default_type="protein_domain", default_score=0.0
        )
        write_combined_results_to_gff(config, combined_data)
    elif config.output_format == "csv":
        output_file = config.output_dir / "combined_annotations.csv"
        combined_data.write_csv(output_file)
        config.logger.info(
            f"Combined annotation results written to {output_file}"
        )
    else:  # tsv (default)
        output_file = config.output_dir / "combined_annotations.tsv"
        combined_data.write_csv(output_file, separator="\t")
        config.logger.info(
            f"Combined annotation results written to {output_file}"
        )

    # Log summary statistics
    config.logger.info(f"Total annotations: {combined_data.height}")
    if "database" in combined_data.columns:
        dbs_used = (
            combined_data.select("database").unique().to_series().to_list()
        )
        config.logger.info(f"Databases used: {', '.join(dbs_used)}")
    if "search_tool" in combined_data.columns:
        tools_used = (
            combined_data.select("search_tool").unique().to_series().to_list()
        )
        config.logger.info(f"Search tools used: {', '.join(tools_used)}")

    # Cleanup temporary directories
    tmp_dir = config.output_dir / "tmp"

    if tmp_dir.exists():
        try:
            shutil.rmtree(tmp_dir)
            config.logger.info(
                f"Cleaned up mmseqs2 temporary directory: {tmp_dir}"
            )
        except Exception as e:
            config.logger.warning(f"Could not remove tmp directory: {e}")

    # Clean up rolypoly temp_dir (created by BaseConfig)
    if (
        not config.keep_tmp
        and hasattr(config, "temp_dir")
        and config.temp_dir.exists()
    ):
        try:
            shutil.rmtree(config.temp_dir)
            config.logger.info(
                f"Cleaned up rolypoly temporary directory: {config.temp_dir}"
            )
        except Exception as e:
            config.logger.warning(f"Could not remove temp_dir: {e}")

    raw_out_dir = config.output_dir / "raw_out"
    if raw_out_dir.exists() and not any(raw_out_dir.iterdir()):
        try:
            raw_out_dir.rmdir()
            config.logger.info("Removed empty raw_out directory")
        except Exception as e:
            config.logger.warning(f"Could not remove raw_out directory: {e}")


def enrich_with_info_tables(dataframe: pl.DataFrame, logger: logging.Logger):
    """Join domain metadata from known info tables onto combined annotations."""

    if (
        "database" not in dataframe.columns
        or "profile_accession" not in dataframe.columns
    ):
        return dataframe

    log = logger or logging.getLogger(__name__)

    try:
        data_root = Path(os.environ["ROLYPOLY_DATA"])
    except KeyError:
        log.warning("ROLYPOLY_DATA is not set; skipping metadata enrichment")
        return dataframe

    db_values = dataframe.get_column("database").drop_nulls().unique().to_list()
    db_keys = {
        str(value).lower() for value in db_values if isinstance(value, str)
    }
    matched_specs = db_keys.intersection(INFO_TABLE_SPECS.keys())
    if not matched_specs:
        return dataframe

    enriched = dataframe
    for db_key in matched_specs:
        spec = INFO_TABLE_SPECS[db_key]
        candidate_paths = [
            data_root / path
            for path in [
                spec["relative_path"],
                *spec.get("fallback_relative_paths", []),
            ]
        ]
        info_path = next((path for path in candidate_paths if path.exists()), None)
        if info_path is None:
            log.warning(
                f"Metadata table for '{db_key}' not found at "
                f"{', '.join(str(path) for path in candidate_paths)}, skipping join"
            )
            continue

        read_kwargs = {"separator": ",", "has_header": True}
        read_kwargs.update(spec.get("read_csv_kwargs", {}))
        try:
            info_df = pl.read_csv(info_path, **read_kwargs)
        except Exception as exc:
            log.warning(f"Failed to read metadata table {info_path}: {exc}")
            continue

        rename_map = spec.get("rename_columns", {})
        if rename_map:
            info_df = info_df.rename(rename_map)

        join_column = spec["join_column"]
        if join_column not in info_df.columns:
            log.warning(
                f"Join column '{join_column}' missing in {info_path}, skipping join"
            )
            continue

        requested_cols = spec.get("columns")
        if requested_cols:
            ordered_unique = []
            for col in requested_cols:
                if col in info_df.columns and col not in ordered_unique:
                    ordered_unique.append(col)
            if join_column not in ordered_unique:
                ordered_unique.insert(0, join_column)
            info_df = info_df.select(ordered_unique)

        info_df = info_df.with_columns(
            pl.col(join_column)
            .cast(pl.Utf8)
            .str.strip_chars()
            .alias(join_column)
        )
        info_df = info_df.rename({join_column: "profile_accession"})

        meta_cols = [
            col for col in info_df.columns if col != "profile_accession"
        ]
        if not meta_cols:
            continue

        prefix = spec.get("prefix", db_key)
        rename_targets = {col: f"{prefix}_{col}" for col in meta_cols}
        info_df = info_df.rename(rename_targets)
        selected_cols = ["profile_accession"] + list(rename_targets.values())
        info_df = info_df.select(selected_cols)

        enriched = enriched.join(info_df, on="profile_accession", how="left")

    return enriched


def write_combined_results_to_gff(config, combined_data):
    """Write combined results to GFF3 format."""
    from rolypoly.utils.bio.polars_fastx import write_gff3_dataframe

    output_file = config.output_dir / "combined_annotations.gff3"
    write_gff3_dataframe(
        combined_data,
        output_file,
        input_fasta=config.input,
        default_type="protein_domain",
        default_score=0.0,
    )
    config.logger.info(f"Combined annotation results written to {output_file}")


if __name__ == "__main__":
    annotate_prot()
