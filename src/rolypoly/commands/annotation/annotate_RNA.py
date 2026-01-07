import logging
import os
from pathlib import Path
from typing import Union

import rich_click as click
from rich.console import Console

from rolypoly.utils.logging.citation_reminder import remind_citations
from rolypoly.utils.logging.config import BaseConfig
from rolypoly.utils.various import ensure_memory

global tools
tools = [""]

console = Console(width=150)


class RNAAnnotationConfig(BaseConfig):
    """Configuration for RNA annotation pipeline"""

    def __init__(
        self,
        input: Path,
        output_dir: Path,
        threads: int,
        log_file: Union[Path, logging.Logger, None],
        log_level: str,
        memory: str,
        override_parameters: dict[str, object] = {},
        skip_steps: list[str] = [],
        rnamotif_tool=None,
        secondary_structure_tool: str = "RNAfold",
        ires_tool: str = "IRESfinder",
        trna_tool: str = "tRNAscan-SE",
        cm_db: str = "Rfam",
        custom_cm_db: str = "",
        output_format: str = "tsv",
        motif_db: str = "jaspar_rna",
        resolve_mode: str = "simple",
        min_overlap_positions: int = 10,
        **kwargs,  # TODO: decide if this is really needed.
    ):
        # Extract BaseConfig parameters
        base_config_params = {
            "input": input,
            "output": output_dir,
            "threads": threads,
            "log_file": log_file,
            "log_level": log_level,
            "memory": memory,
        }
        super().__init__(**base_config_params)

        self.secondary_structure_tool = secondary_structure_tool
        self.ires_tool = ires_tool
        self.trna_tool = trna_tool
        self.rnamotif_tool = rnamotif_tool
        self.cm_db = cm_db
        self.custom_cm_db = custom_cm_db
        self.output_format = output_format
        self.motif_db = motif_db
        self.skip_steps = skip_steps or []
        self.resolve_mode = resolve_mode
        self.min_overlap_positions = min_overlap_positions

        self.step_params = {
            "RNAfold": {
                "temperature": 25
            },  # 37 is unreasonably hot for most enviroments
            "LinearFold": {},  # Fixed parameter name
            "RNAstructure": {"temperature": 25},
            "cmsearch": {
                "cut_ga": True,
                "noali": True,
            },  # Removed E-value as it's incompatible with cut_ga. User can still set it manually..
            "IRESfinder": {"min_score": 0.5},
            "IRESpy": {"min_score": 0.6},
            "tRNAscan-SE": {"forceow": True, "G": True},  #
            "lightmotif": {},
            "aragorn": {"l": True},
        }
        if override_parameters:
            for step, params in override_parameters.items():
                if step in self.step_params:
                    self.step_params[step].update(params)
                else:
                    print(
                        f"Warning: Unknown step '{step}' in override_parameters. Ignoring."
                    )


@click.command(name="annotate_RNA")
@click.option(
    "-i",
    "--input",
    required=True,
    type=click.Path(exists=True),
    help="Input nucleotide sequence file (fasta, fna, fa, or faa)",
)
@click.option(
    "-o",
    "--output-dir",
    default="./annotate_RNA_output",
    help="Output directory path",
)
@click.option("-t", "--threads", default=1, help="Number of threads")
@click.option(
    "-g",
    "--log-file",
    default="./annotate_RNA_logfile.txt",
    help="Path to log file",
)
@click.option(
    "-l",
    "--log-level",
    default="INFO",
    help="Log level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
)
@click.option(
    "-M", "--memory", default="4gb", help="Memory in GB. Example: -M 8gb"
)
@click.option(
    "-op",
    "--override_parameters",
    "--override-parameters",
    default="{}",
    help='JSON-like string of parameters to override. Example: --override-parameters \'{"RNAfold": {"temperature": 37}, "cmscan": {"E": 1e-5}}\'',
)
@click.option(
    "--skip-steps",
    default="",
    help="Comma-separated list of steps to skip. Example: --skip-steps RNAfold,cmsearch",
)
@click.option(
    "--secondary-structure-tool",
    default="LinearFold",
    type=click.Choice(["RNAfold", "LinearFold"], case_sensitive=False),
    help="Tool for secondary structure prediction. LinearFold is faster but less configurable.",
)
@click.option(
    "--ires-tool",
    default="IRESfinder",
    type=click.Choice(["IRESfinder", "IRESpy"], case_sensitive=False),
    help="Tool for IRES identification",
)
@click.option(
    "--trna-tool",
    default="tRNAscan-SE",
    type=click.Choice(["tRNAscan-SE", "aragorn"], case_sensitive=False),
    help="Tool for tRNA identification",
)
@click.option(
    "--rnamotif-tool",
    default="RNAMotif",
    type=click.Choice(["RNAMotif", "aragorn"], case_sensitive=False),
    help="Tool for RNAmotif identification",
)
@click.option(
    "--cm-db",
    default="Rfam",  # TODO: in the fuiture Add the exprimental RVMT extended Rfam set (maybe check it first with Rscape )
    type=click.Choice(["Rfam", "custom"], case_sensitive=False),
    help="Database for cmscan",
)
@click.option(
    "--custom-cm-db",
    default="",
    help="Path to a custom cm database in nhmmer/cm format (mandatory to use with --cm-db custom)",
)
@click.option(
    "--output-format",
    default="tsv",
    type=click.Choice(["tsv", "csv", "gff3"], case_sensitive=False),
    help="Output format for the combined results",
)
@click.option(
    "--motif-db",
    default="RolyPoly",
    help="Database to use for RNA motif scanning - RolyPoly, jaspar_core, or a path to a folder containg a pwm/msa files",
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
    help="""How to deal with overlapping RNA element hits in the same sequence. \n
        - merge: all overlapping hits are merged into one range \n
        - one_per_range: one hit per range is reported \n
        - one_per_query: one hit per query sequence is reported \n
        - split: each overlapping element is split into a new row \n
        - drop_contained: hits that are contained within other hits are dropped \n
        - none: no resolution of overlapping hits is performed \n
        - simple: heuristic-based approach using drop_contained \n
        """,
)
@click.option(
    "-mo",
    "--min-overlap-positions",
    default=10,
    help="Minimal number of overlapping positions between two intersecting ranges before they are considered as overlapping (used in some resolve_mode(s)).",
)
def annotate_RNA(
    input,
    output_dir,
    threads,
    log_file,
    log_level,
    memory,
    override_parameters,
    skip_steps,
    secondary_structure_tool,
    ires_tool,
    trna_tool,
    rnamotif_tool,
    cm_db,
    custom_cm_db,
    output_format,
    motif_db,
    resolve_mode,
    min_overlap_positions,
):
    """Predict viral sequence RNA secondary structure, search for ribozymes, IRES, tRNAs, and other RNA structural elements.
    By default, the following steps are run in the following order: predict_secondary_structure (LinearFold), search_ribozymes (Rfam via cmscan), predict_trnas (tRNAscan-SE).
    Additional steps (detect_ires, search_rna_motifs, search_rna_elements) are currently disabled as they are under development.
    Use --skip-steps to skip specific steps."""
    import json

    config = RNAAnnotationConfig(
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
        secondary_structure_tool=secondary_structure_tool,
        ires_tool=ires_tool,
        trna_tool=trna_tool,
        rnamotif_tool=rnamotif_tool,
        cm_db=cm_db,
        custom_cm_db=custom_cm_db,
        output_format=output_format,
        motif_db=motif_db,
        resolve_mode=resolve_mode,
        min_overlap_positions=min_overlap_positions,
    )
    # config.logger.info("Starting RNA annotation process")

    try:
        process_RNA_annotations(config)
    except Exception as e:
        console.print(f"An error occurred during RNA annotation: {str(e)}")
        raise

    # remind_citations(tools)
    if config.log_level != "DEBUG":
        with open(f"{config.log_file}", "a") as f_out:
            f_out.write(remind_citations(tools, return_bibtex=True) or "")


def process_RNA_annotations(config):
    config.logger.info("Starting RNA annotation process")

    # Set up logging for sh commands
    # logging.basicConfig(level=logging.INFO)

    steps = [
        predict_secondary_structure,  # LinearFold
        search_ribozymes,  # Rfam via cmscan
        # detect_ires,  # Not yet tested/finalized
        predict_trnas,  # tRNAscan-SE
        # search_rna_motifs,  # Not yet tested/finalized
        # search_rna_elements,  # RNAsselem - not yet tested
        resolve_rna_element_overlaps,  # Resolve overlapping RNA element hits
    ]

    # TODO: Consider, maybe, use ChangeDirectory into a temp dir which on a "finally" leaves it and copies only whats needed
    for step in steps:
        step_name = step.__name__
        if step_name not in config.skip_steps:
            config.logger.info(f"Starting step: {step_name}")
            step(config)
        else:
            config.logger.info(f"Skipping step: {step_name}")

    combine_results(config)

    config.logger.info("RNA annotation process completed successfully")


def predict_secondary_structure(config):
    config.logger.info("Predicting RNA secondary structure")
    # config.logger.info(f"{config.input}")

    input_path = Path(config.input)
    if input_path.is_file():
        if input_path.suffix in [".fasta", ".fa", ".fna", ".faa"]:
            input_fasta = input_path
        else:
            raise ValueError(f"Input file {input_path} is not a FASTA file")
    elif input_path.is_dir():
        from rolypoly.utils.bio.library_detection import find_fasta_files

        fasta_files = find_fasta_files(input_path, logger=config.logger)
        if not fasta_files:
            raise ValueError(f"No FASTA files found in directory {input_path}")
        input_fasta = fasta_files[0]
    else:
        raise ValueError(
            f"Input path {input_path} is neither a file nor a directory"
        )

    output_file = config.output_dir / "secondary_structure.fold"

    if config.secondary_structure_tool == "RNAfold":
        predict_secondary_structure_rnafold(config, input_fasta, output_file)
    elif config.secondary_structure_tool == "RNAstructure":
        predict_secondary_structure_rnastructure(
            config, input_fasta, output_file
        )
    elif config.secondary_structure_tool == "LinearFold":
        predict_secondary_structure_linearfold(config, input_fasta, output_file)


def predict_secondary_structure_rnafold(config, input_fasta, output_file):
    """Predict RNA secondary structure using RNAfold."""
    import RNA
    from needletail import parse_fastx_file

    # from needletail import Record as record

    with open(output_file, "w") as out_f:
        for record in parse_fastx_file(str(input_fasta)):
            sequence = str(record.seq)  # pyright: ignore

            # convert to RNA
            sequence = sequence.replace("T", "U")
            # Set folding parameters
            md = RNA.md()
            md.temperature = config.step_params["RNAfold"]["temperature"]
            # print(sequence)

            # Predict MFE structure
            (ss, mfe) = RNA.fold(sequence, str(md))

            # Calculate partition function and base pair probabilities
            (ss_pf, fe) = RNA.pf_fold(sequence, md)

            # Generate dot-plot
            RNA.plot_structure_svg(
                data=sequence,
                filename=config.output_dir / f"{record.id}_plot.svg",  # pyright: ignore
                sequence=sequence,
                structure=ss,
            )  # pyright: ignore

            # Write results
            out_f.write(f">{record.id}\n")  # pyright: ignore
            out_f.write(f"Sequence: {sequence}\n")
            out_f.write(f"MFE structure: {ss}\n")
            out_f.write(f"MFE: {mfe:.2f} kcal/mol\n")
            out_f.write(f"Ensemble structure: {ss_pf}\n")
            out_f.write(f"Ensemble free energy: {fe:.2f} kcal/mol\n\n")

    config.logger.info(
        f"Secondary structure prediction completed. Output written to {output_file}"
    )
    tools.append("rnafold")


def predict_secondary_structure_rnastructure(config, input_fasta, output_file):
    """Predict RNA secondary structure using RNAstructure."""
    import subprocess

    from needletail import parse_fastx_file

    with open(output_file, "w") as out_f:
        for record in parse_fastx_file(str(input_fasta)):
            sequence = str(record.seq).replace("T", "U")  # pyright: ignore

            # Write sequence to temporary file
            temp_seq_file = config.output_dir / f"{record.id}_temp.seq"  # pyright: ignore
            with open(temp_seq_file, "w") as temp_f:
                temp_f.write(sequence)

            # Run RNAstructure Fold
            ct_file = config.output_dir / f"{record.id}_temp.ct"  # pyright: ignore
            params = config.step_params["RNAstructure"]
            cmd = [
                "Fold",
                "--temperature",
                str(params.get("temperature", 25)),
                str(temp_seq_file),
                str(ct_file),
            ]
            subprocess.run(cmd)

            # Parse CT file to extract structure
            with open(ct_file, "r") as ct_f:
                lines = ct_f.readlines()
                structure = "".join(
                    [
                        (
                            "("
                            if int(line.split()[4]) > int(line.split()[0])
                            else ")"
                            if int(line.split()[4]) != 0
                            else "."
                        )
                        for line in lines[1:]
                    ]
                )

            # Write results
            out_f.write(f">{record.id}\n")  # pyright: ignore
            out_f.write(f"Sequence: {sequence}\n")
            out_f.write(f"Structure: {structure}\n\n")

            # Clean up temporary files
            temp_seq_file.unlink()
            ct_file.unlink()

    config.logger.info(
        f"RNAstructure prediction completed. Output written to {output_file}"
    )


def predict_secondary_structure_linearfold(config, input_fasta, output_file):
    """Predict RNA secondary structure using LinearFold."""
    import subprocess
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from needletail import parse_fastx_file

    def process_sequence(record):
        """Process a single sequence with LinearFold."""
        sequence = str(record.seq).replace("T", "U")

        try:
            # Prepare LinearFold command
            params = config.step_params["LinearFold"]
            beamsize = params.get("beamsize", 100)

            # LinearFold expects input via stdin
            cmd = ["linearfold", "--beamsize", str(beamsize)]

            # Run LinearFold with sequence input via stdin
            process = subprocess.run(
                cmd,
                input=sequence,
                text=True,
                capture_output=True,
                timeout=300,  # 5 minute timeout per sequence
            )

            if process.returncode == 0:
                # Parse LinearFold output
                output_lines = process.stdout.strip().split("\n")
                if len(output_lines) >= 2:
                    # LinearFold outputs: sequence, then structure with energy
                    structure_line = output_lines[1]
                    # Extract structure and MFE from output like: "(((...))) (-2.34)"
                    if "(" in structure_line and ")" in structure_line:
                        parts = structure_line.split()
                        if len(parts) >= 2:
                            structure = parts[0]
                            # Extract MFE from parentheses
                            mfe_str = parts[1].strip("()")
                            try:
                                mfe = float(mfe_str)
                            except ValueError:
                                mfe = 0.0
                            return (record.id, sequence, structure, mfe)

                config.logger.warning(
                    f"Unexpected LinearFold output format for {record.id}: {process.stdout}"
                )
                return (record.id, sequence, "Error", 0.0)
            else:
                config.logger.error(
                    f"LinearFold failed for sequence {record.id}: {process.stderr}"
                )
                return (record.id, sequence, "Error", 0.0)

        except subprocess.TimeoutExpired:
            config.logger.error(
                f"LinearFold timed out for sequence {record.id}"
            )
            return (record.id, sequence, "Error", 0.0)
        except Exception as e:
            config.logger.error(
                f"Error processing sequence {record.id}: {str(e)}"
            )
            return (record.id, sequence, "Error", 0.0)

    # Process sequences in parallel
    with (
        open(output_file, "w") as out_f,
        ThreadPoolExecutor(max_workers=config.threads) as executor,
    ):
        futures = []
        for record in parse_fastx_file(str(input_fasta)):
            futures.append(executor.submit(process_sequence, record))

        for future in as_completed(futures):
            seq_id, sequence, structure, mfe = future.result()
            out_f.write(f">{seq_id}\n")
            out_f.write(f"{sequence}\n")
            if structure != "Error":
                out_f.write(f"{structure}\n")
                out_f.write(f"({mfe:.2f})\n")
            else:
                out_f.write("Error in structure prediction\n")

    config.logger.info(
        f"LinearFold prediction completed. DBN file with MFE predictions written to {output_file}"
    )
    tools.append("LinearFold")


def search_ribozymes(config):
    """Search for ribozymes using Rfam or a custom cm database."""
    import os
    from pathlib import Path

    from rolypoly.utils.various import run_command_comp

    config.logger.info(f"Searching for ribozymes using {config.cm_db}")
    input_fasta = config.input
    output_file = config.output_dir / "ribozymes.out"

    if config.cm_db == "Rfam":
        cm_db_path = os.path.join(
            Path(os.environ.get("ROLYPOLY_DATA", "")), "profiles/cm/Rfam.cm"
        )
        tools.append("Rfam")
    elif config.cm_db == "custom":
        cm_db_path = os.path.abspath(config.custom_cm_db)
    else:
        raise ValueError(f"Invalid cm database: {config.cm_db}")

    # Prepare parameters
    params = config.step_params["cmsearch"].copy()

    # Filter out False boolean values and only keep True boolean flags
    filtered_params = {}
    for param, value in params.items():
        if isinstance(value, bool):
            if value:  # Only add True boolean values as flags
                filtered_params[param] = True
        else:
            filtered_params[param] = value

    # Set up positional arguments in correct order
    positional_args = [str(cm_db_path), str(input_fasta)]

    # Add required parameters
    filtered_params.update(
        {"cpu": config.threads, "tblout": str(output_file), "o": "/dev/null"}
    )

    success = run_command_comp(
        base_cmd="cmscan",
        positional_args=positional_args,
        params=filtered_params,
        logger=config.logger,
        output_file=output_file,
        skip_existing=True,
    )
    if success:
        tools.append("cmsearch")
    return success


def search_rna_elements(config):
    """Search for RNA structural elements using RNAsselem.

    Note: This function is not yet tested and is commented out for now.
    """
    config.logger.warning(
        "search_rna_elements (RNAsselem) is not yet tested and is currently disabled"
    )
    return
    # config.logger.info("Searching for RNA elements")
    # input_fasta = config.input
    # output_file = config.output_dir / "rna_elements.out"
    # tools.append("RNAsselem")

    # # Prepare parameters
    # params = config.step_params["RNAsselem"].copy()

    # # Add required parameters
    # params.update({"i": str(input_fasta), "o": str(output_file)})


def detect_ires(config):
    config.logger.info("Detecting IRES elements")
    input_fasta = config.input
    output_file = config.output_dir / "ires.out"

    if config.ires_tool == "IRESfinder":
        detect_ires_iresfinder(config, input_fasta, output_file)
    elif config.ires_tool == "IRESpy":
        detect_ires_irespy(config, input_fasta, output_file)


def detect_ires_iresfinder(config, input_fasta, output_file):
    import shutil

    from rolypoly.utils.various import run_command_comp

    # Check if IRESfinder is available in PATH
    if not shutil.which("IRESfinder.py"):
        config.logger.error(
            "IRESfinder.py not found in PATH. Please ensure it is installed correctly."
        )
        return False

    # Prepare parameters
    params = config.step_params["IRESfinder"].copy()

    # Add required parameters
    params.update({"i": str(input_fasta), "o": str(output_file)})

    success = run_command_comp(
        base_cmd="IRESfinder.py",  # Use full script name
        positional_args=[],  # IRESfinder uses named parameters for everything
        params=params,
        logger=config.logger,
        output_file=output_file,
        skip_existing=True,
    )
    if success:
        tools.append("IRESfinder")
    return success


def detect_ires_irespy(config, input_fasta, output_file):
    """Detect IRES elements using IRESpy."""
    import polars as pl
    from needletail import parse_fastx_file

    from rolypoly.utils.various import run_command_comp

    sequences = parse_fastx_file(str(input_fasta))
    with open(output_file, "w") as out:
        out.write("Sequence Name\tIRES Score\tStart\tEnd\n")
        for seq in sequences:
            # Prepare parameters
            params = config.step_params["IRESpy"].copy()

            # Add required parameters
            params.update({"i": str(seq.seq), "o": str(output_file)})  # pyright: ignore

            success = run_command_comp(
                base_cmd="irespy",
                positional_args=[],  # IRESpy uses named parameters for everything
                params=params,
                logger=config.logger,
                output_file=output_file,
                skip_existing=True,
                prefix_style="auto",
            )
            if success:
                tools.append("IRESpy")
                ires_data = pl.read_csv(
                    output_file, separator="\t", has_header=False
                ).select(
                    [
                        pl.col("column_1").alias("sequence_id"),
                        pl.col("column_2").alias("ires_score"),
                        pl.col("column_3").alias("ires_start"),
                        pl.col("column_4").alias("ires_end"),
                    ]
                )
                out.write(
                    f"{seq.id}\t{ires_data['ires_score']}\t{ires_data['ires_start']}\t{ires_data['ires_end']}\n"  # pyright: ignore
                )

    config.logger.info(f"IRESpy completed. Output written to {output_file}")


def predict_trnas(config):
    if config.trna_tool == "tRNAscan-SE":
        predict_trnas_with_tRNAscan(config)
    elif config.trna_tool == "aragorn":
        predict_trnas_with_aragorn(config)
    else:
        config.logger.info(
            f"Skipping tRNA prediction as {config.trna_tool} is not supported"
        )


def predict_trnas_with_aragorn(config):
    """Predict tRNAs using Aragorn."""
    from rolypoly.utils.various import run_command_comp

    config.logger.info("Predicting tRNAs")
    input_fasta = config.input
    output_file = config.output_dir / "trnas.out"

    # Prepare parameters
    params = config.step_params["aragorn"].copy()

    # Filter out False boolean values and only keep True boolean flags
    filtered_params = {}
    for param, value in params.items():
        if isinstance(value, bool):
            if value:  # Only add True boolean values as flags
                filtered_params[param] = True
        else:
            filtered_params[param] = value

    # Add required parameters
    filtered_params.update({"o": str(output_file)})

    # Set up positional arguments
    positional_args = [str(input_fasta)]

    success = run_command_comp(
        base_cmd="aragorn",
        positional_args=positional_args,
        params=filtered_params,
        logger=config.logger,
        output_file=output_file,
        skip_existing=True,
    )
    if success:
        tools.append("aragorn")
    return success


def predict_trnas_with_tRNAscan(config):
    """Predict tRNAs using tRNAscan-SE."""
    from rolypoly.utils.various import run_command_comp

    config.logger.info("Predicting tRNAs")
    input_fasta = config.input
    output_file = config.output_dir / "trnas.out"

    if os.path.exists(output_file):
        config.logger.info(
            f"tRNAscan-SE output file {output_file} already exists. removing it as trnascan-se will get confused"
        )
        os.remove(output_file)

    # Prepare parameters
    params = config.step_params["tRNAscan-SE"].copy()

    # Filter out False boolean values and only keep True boolean flags
    filtered_params = {}
    for param, value in params.items():
        if isinstance(value, bool):
            if value:  # Only add True boolean values as flags
                filtered_params[param] = True
        else:
            filtered_params[param] = value

    # Add required parameters
    filtered_params.update(
        {
            "thread": config.threads,  # Use --thread instead of -threads
            "o": str(output_file),
        }
    )

    # Set up positional arguments
    positional_args = [str(input_fasta)]

    success = run_command_comp(
        base_cmd="tRNAscan-SE",
        positional_args=positional_args,
        params=filtered_params,
        logger=config.logger,
        output_file=output_file,
        skip_existing=True,
    )
    tools.append("trnascan-se")
    # if success:
    return success


def search_rna_motifs(config):  # py
    """PSSM search using lightmotif."""

    config.logger.warning("RNA motif search is still in development.")
    # from pathlib import Path

    # import lightmotif
    # from needletail import parse_fastx_file

    # config.logger.info(
    #     f"Searching for RNA structural elements using {config.motif_db} database"
    # )

    # # Check ROLYPOLY_DATA environment variable
    # if "ROLYPOLY_DATA" not in os.environ:
    #     config.logger.error("ROLYPOLY_DATA environment variable is not set")
    #     return False

    # datadir = Path(os.environ.get("ROLYPOLY_DATA", ""))
    # if not datadir.exists():
    #     config.logger.error(f"ROLYPOLY_DATA directory {datadir} does not exist")
    #     return False

    # if config.motif_db == "RolyPoly":
    #     motifs_dir = datadir / "RNA_motifs" / "rolypoly"
    #     if not motifs_dir.exists():
    #         config.logger.warning(f"RolyPoly RNA motifs search is still in development. You can {motifs_dir} with your own motifs, but no guarantees that it will work.")
    #         config.logger.warning("Skipping RNA motif search.")
    #         # config.logger.warning(f"RolyPoly RNA motifs directory {motifs_dir} does not exist. Skipping RNA motif search.")
    #         return False
    # elif config.motif_db == "jaspar_core":
    #     motifs_dir = datadir / "RNA_motifs" / "jaspar_core"
    #     if not motifs_dir.exists():
    #         config.logger.warning(f"RolyPoly RNA motifs search is still in development. You can {motifs_dir} with your own motifs, but no guarantees that it will work.")
    #         # config.logger.warning(f"JASPAR core RNA motifs directory {motifs_dir} does not exist. Skipping RNA motif search.")
    #         return False
    # else:
    #     motifs_dir = Path(config.motif_db)
    #     # check if motifs_dir exists
    #     if not motifs_dir.exists():
    #         config.logger.error(f"Motifs directory {motifs_dir} does not exist")
    #         return False
    #     config.logger.info(f"Loading motifs from {motifs_dir}")

    # config.logger.info(f"Loading motifs from {config.motif_db} database")

    # input_fasta = config.input
    # output_file = config.output_dir / "rna_motifs.out"

    # # Load motifs from the database
    # motifs = []
    # try:
    #     # For built-in databases, try to load from lightmotif
    #     if config.motif_db in ["RolyPoly", "jaspar_core"]:
    #         # Try loading the database name directly first
    #         try:
    #             loader = lightmotif.lib.Loader(config.motif_db) # pyright: ignore
    #             motifs.extend(list(loader))
    #         except Exception:
    #             # If that fails, skip motif search
    #             config.logger.warning(f"Could not load {config.motif_db} database. This may not be available in your lightmotif installation. Skipping RNA motif search.")
    #             return False
    #     else:
    #         # For custom paths, load from directory
    #         loader = lightmotif.lib.Loader(str(motifs_dir)) # pyright: ignore
    #         motifs.extend(list(loader))

    #     if motifs:
    #         config.logger.info(
    #             f"Successfully loaded {len(motifs)} motifs from {config.motif_db}"
    #         )
    #         tools.append(f"lightmotif_{config.motif_db}")
    #     else:
    #         config.logger.warning(f"No motifs found in {config.motif_db} database")
    #         return False
    # except ValueError as e:
    #     config.logger.warning(f"Invalid motif database '{config.motif_db}': {e}. Skipping RNA motif search.")
    #     return False
    # except Exception as e:
    #     config.logger.warning(f"Error loading motifs from {config.motif_db}: {e}. Skipping RNA motif search.")
    #     return False

    # with open(output_file, "w") as out:
    #     out.write("Sequence\tMotif\tStart\tEnd\tScore\tStrand\n")

    #     # Process each sequence
    #     for record in parse_fastx_file(str(input_fasta)):
    #         sequence = str(record.seq) # pyright: ignore

    #         try:
    #             # Create an encoded sequence for scanning
    #             encoded = lightmotif.lib.create(sequence)
    #             if encoded is None:
    #                 config.logger.warning(
    #                     f"Could not encode sequence {record.id} - skipping" # pyright: ignore
    #                 )
    #                 continue

    #             # Create striped sequence for scanning
    #             striped = lightmotif.lib.stripe(encoded)

    #             # Scan for each motif
    #             for motif in motifs:
    #                 try:
    #                     # Configure striped sequence for this motif
    #                     striped(motif.matrix)

    #                     # Score all positions
    #                     scores = motif.matrix.score(striped)

    #                     # Find positions above threshold
    #                     params = config.step_params["lightmotif"]
    #                     min_score = params.get("min_score", 0.5)

    #                     # Get scores as a list
    #                     score_list = scores.unstripe()

    #                     # Find positions above threshold
    #                     for pos, score in enumerate(score_list):
    #                         if score >= min_score:
    #                             out.write(
    #                                 f"{record.id}\t{motif.name}\t{pos + 1}\t{pos + motif.matrix.width}\t{score:.3f}\t+\n"
    #                             )
    #                 except Exception as e:
    #                     config.logger.warning(
    #                         f"Error scanning sequence {record.id} with motif {motif.name}: {e}"
    #                     )
    #                     continue
    #         except Exception as e:
    #             config.logger.warning(f"Error processing sequence {record.id}: {e}")
    #             continue

    # config.logger.info(f"RNA motif scanning completed. Output written to {output_file}")
    # return True


def process_ribozymes_data(config, ribozymes_file):
    """Process ribozymes data from cmscan output.
    Returns an empty DataFrame if the file is empty or has no valid data.
    """
    import polars as pl

    from rolypoly.utils.various import read_fwf

    if (
        not os.path.exists(ribozymes_file)
        or os.path.getsize(ribozymes_file) == 0
    ):
        config.logger.warning(
            f"Ribozymes file {ribozymes_file} is empty or does not exist"
        )
        return pl.DataFrame()

    try:
        # First check if file has any non-comment lines
        with open(ribozymes_file, "r") as f:
            has_data = any(
                not line.startswith("#") and line.strip() for line in f
            )

        if not has_data:
            config.logger.warning(
                f"Ribozymes file {ribozymes_file} contains no data (only comments)"
            )
            return pl.DataFrame()

        fwf = "------------------- --------- -------------------- --------- --- -------- -------- -------- -------- ------ ----- ---- ---- ----- ------ --------- --- ---------------------"
        # Convert fixed-width format to list of (start, width) tuples
        widths = []
        start = 0
        for segment in fwf.split(" "):
            width = len(segment)
            widths.append((start, width + 1))
            start += width + 1  # +1 for the space between columns

        columns = [
            "profile_name",
            "rfam_id",
            "sequence_ID",
            "qaccession",
            "mdl",
            "mdl_from",
            "mdl_to",
            "seq_from",
            "seq_to",
            "strand",
            "trunc",
            "pass",
            "gc",
            "bias",
            "score",
            "evalue",
            "inc",
            "ribozyme_description",
        ]
        dtypes = [
            pl.Utf8,
            pl.Utf8,
            pl.Utf8,
            pl.Utf8,
            pl.Utf8,
            pl.Int64,
            pl.Int64,
            pl.Int64,
            pl.Int64,
            pl.Utf8,
            pl.Utf8,
            pl.Utf8,
            pl.Utf8,
            pl.Utf8,
            pl.Float64,
            pl.Float64,
            pl.Utf8,
            pl.Utf8,
        ]

        raw_data = read_fwf(
            ribozymes_file,
            widths=widths,
            columns=columns,
            dtypes=dtypes,
            comment_prefix="#",
        )
        if raw_data.is_empty():
            config.logger.warning(
                f"No valid data found in ribozymes file {ribozymes_file}"
            )
            return pl.DataFrame()

        config.logger.debug(
            f"Raw ribozymes data from {ribozymes_file}: {raw_data}"
        )

        # Normalize to minimal schema, keeping important ribozyme-specific columns
        from rolypoly.utils.bio.polars_fastx import (
            create_minimal_annotation_schema,
        )

        data = create_minimal_annotation_schema(
            raw_data,
            annotation_type="ribozyme",
            source=config.cm_db,
            tool_specific_cols=[
                "profile_name",
                "evalue",
                "ribozyme_description",
            ],
        )
        return data
    except Exception as e:
        config.logger.error(f"Error processing ribozymes data: {str(e)}")
        return pl.DataFrame()


def process_ires_iresfinder(ires_file):
    import polars as pl

    if ires_file.is_file():
        raw_data = pl.read_csv(ires_file, separator="\t")
        from rolypoly.utils.bio.polars_fastx import (
            create_minimal_annotation_schema,
        )

        # Rename columns for normalization
        if "Sequence Name" in raw_data.columns:
            raw_data = raw_data.rename({"Sequence Name": "sequence_id"})
        if "Start" in raw_data.columns:
            raw_data = raw_data.rename({"Start": "start"})
        if "End" in raw_data.columns:
            raw_data = raw_data.rename({"End": "end"})
        if "IRES Score" in raw_data.columns:
            raw_data = raw_data.rename({"IRES Score": "score"})

        return create_minimal_annotation_schema(
            raw_data, annotation_type="IRES", source="IRESfinder"
        )
    return pl.DataFrame()


def process_ires_irespy(ires_file):
    import polars as pl

    if ires_file.is_file():
        raw_data = pl.read_csv(ires_file, separator="\t")
        from rolypoly.utils.bio.polars_fastx import (
            create_minimal_annotation_schema,
        )

        # Rename columns for normalization
        if "Sequence Name" in raw_data.columns:
            raw_data = raw_data.rename({"Sequence Name": "sequence_id"})
        if "Start" in raw_data.columns:
            raw_data = raw_data.rename({"Start": "start"})
        if "End" in raw_data.columns:
            raw_data = raw_data.rename({"End": "end"})
        if "IRES Score" in raw_data.columns:
            raw_data = raw_data.rename({"IRES Score": "score"})

        return create_minimal_annotation_schema(
            raw_data, annotation_type="IRES", source="IRESpy"
        )
    return pl.DataFrame()


def process_trnas_data_tRNAscan_SE(trnas_file):
    import polars as pl

    if trnas_file.is_file() and os.path.getsize(trnas_file) > 0:
        # tRNAscan-SE output has a 3-line header that needs to be skipped
        # Line 1: Column headers (Sequence, tRNA, Bounds, etc.)
        # Line 2: More specific headers (Name, tRNA #, Begin, End, etc.)
        # Line 3: Dashes separator (---------, ------, etc.)
        # Then actual data starts

        try:
            # Read the file and skip header lines
            with open(trnas_file, "r") as f:
                lines = f.readlines()

            # Find where actual data starts (after the dashes line)
            data_start = 0
            for i, line in enumerate(lines):
                if line.strip().startswith("--------"):
                    data_start = i + 1
                    break

            if data_start == 0 or data_start >= len(lines):
                # No data found
                return pl.DataFrame()

            # Extract data lines
            data_lines = lines[data_start:]
            data_lines = [line.strip() for line in data_lines if line.strip()]

            if not data_lines:
                return pl.DataFrame()

            # Parse the data manually since the format is space-separated with variable spacing
            records = []
            for line in data_lines:
                parts = line.split()
                if len(parts) >= 9:  # Ensure we have enough columns
                    begin_pos = int(parts[2])  # Begin position
                    end_pos = int(parts[3])  # End position

                    # Determine strand from Begin/End positions
                    # If Begin > End, it's on minus strand
                    # If Begin < End, it's on plus strand
                    if begin_pos > end_pos:
                        strand = "-"
                        start = end_pos  # Start is the smaller coordinate
                        end = begin_pos  # End is the larger coordinate
                    else:
                        strand = "+"
                        start = begin_pos  # Start is the smaller coordinate
                        end = end_pos  # End is the larger coordinate

                    record = {
                        "sequence_id": parts[0],  # Sequence name
                        "type": "tRNA",
                        "start": start,  # Corrected start position
                        "end": end,  # Corrected end position
                        "score": float(parts[8]),  # Infernal score
                        "source": "tRNAscan-SE",
                        "strand": strand,  # Correctly determined strand
                        "phase": ".",
                        "tRNA_type": parts[4],  # tRNA type (Gln, Leu, etc.)
                        "anticodon": parts[5],  # Anticodon
                    }
                    records.append(record)

            if records:
                raw_data = pl.DataFrame(records)
                from rolypoly.utils.bio.polars_fastx import (
                    create_minimal_annotation_schema,
                )

                return create_minimal_annotation_schema(
                    raw_data,
                    annotation_type="tRNA",
                    source="tRNAscan-SE",
                    tool_specific_cols=["tRNA_type", "anticodon"],
                )

        except Exception as e:
            # If parsing fails, return empty DataFrame
            print(f"Warning: Failed to parse tRNAscan-SE output: {e}")
            return pl.DataFrame()

    return pl.DataFrame()


def process_trnas_data_aragorn(trnas_file):
    import polars as pl

    if trnas_file.is_file() and os.path.getsize(trnas_file) > 0:
        return pl.read_csv(trnas_file, separator="\t", has_header=False).select(
            [
                pl.col("column_1").alias("sequence_id"),
                pl.lit("tRNA").alias("type"),
                pl.col("column_2").cast(pl.Int64).alias("start"),
                pl.col("column_3").cast(pl.Int64).alias("end"),
                pl.col("column_5").alias("score"),
                pl.lit("aragorn").alias("source"),
                pl.lit("+").alias("strand"),
                pl.lit(".").alias("phase"),
                pl.col("column_4").alias("tRNA_type"),
            ]
        )
    return pl.DataFrame()


def process_rna_elements_data(rna_elements_file):
    import polars as pl

    if rna_elements_file.is_file():
        return pl.read_csv(rna_elements_file, separator="\t").select(
            [
                pl.col("Sequence").alias("sequence_id"),
                pl.lit("Element").alias("type"),
                pl.col("Start").cast(pl.Int64).alias("start"),
                pl.col("End").cast(pl.Int64).alias("end"),
                pl.lit("0").alias("score"),
                pl.lit("RNAsselem").alias("source"),
                pl.lit("+").alias("strand"),
                pl.lit(".").alias("phase"),
            ]
        )
    return pl.DataFrame()


def process_rna_motifs_data(config, rna_motifs_file):
    import polars as pl

    if rna_motifs_file.is_file():
        return pl.read_csv(rna_motifs_file, separator="\t").select(
            [
                pl.col("sequence_id").alias("sequence_id"),
                pl.lit("RNA_motif").alias("type"),
                pl.col("start").cast(pl.Int64).alias("start"),
                pl.col("end").cast(pl.Int64).alias("end"),
                pl.col("score").alias("score"),
                pl.lit("RNAMotif").alias("source"),
                pl.lit("+").alias("strand"),
                pl.lit(".").alias("phase"),
                pl.col("motif_type").alias("motif_type"),
            ]
        )
    return pl.DataFrame()


def read_multiDBN_to_dataframe(MultiDBN_file):
    import polars as pl

    if not MultiDBN_file.is_file():
        return pl.DataFrame()

    records = []
    with open(MultiDBN_file, "r") as f_in:
        for record in f_in.read().split(">"):
            if not record.strip():
                continue

            lines = record.strip().split("\n")
            sequence_id = lines[0]
            sequence = lines[1]
            structure_line = lines[2].split()
            structure = structure_line[0]

            # Try to parse energy value, handle LinearFold errors
            mfe = -0.1  # default
            if len(structure_line) > 1:
                try:
                    # Energy should be in format "(value)"
                    energy_str = structure_line[1].strip("()")
                    mfe = float(energy_str)
                except ValueError:
                    # LinearFold error message like "Error in structure prediction"
                    mfe = -0.1

            records.append(
                {
                    "sequence_id": sequence_id,
                    "type": "RNA_secondary_structure",
                    "start": 1,
                    "end": len(sequence),
                    "score": mfe,
                    "source": "LinearFold",
                    "strand": "+",
                    "phase": ".",
                    "sequence": sequence,
                    "structure": structure,
                }
            )

    return pl.DataFrame(records)


def resolve_rna_element_overlaps(config):
    """Resolve overlapping RNA element hits using consolidate_hits."""
    from pathlib import Path

    import polars as pl

    from rolypoly.utils.bio.interval_ops import consolidate_hits

    config.logger.info("Resolving overlapping RNA element hits")

    # Files that might have overlapping hits
    element_files = [
        (config.output_dir / "ribozymes.out", "cmscan")
        # Note: IRES, tRNA, and motif files typically don't need overlap resolution
        # as they represent discrete elements, but can be added if needed
    ]

    for element_file, tool_name in element_files:
        if not element_file.exists() or element_file.stat().st_size == 0:
            config.logger.debug(
                f"Element file {element_file} doesn't exist or is empty, skipping"
            )
            continue

        config.logger.info(f"Resolving overlaps in {element_file.name}")

        try:
            # Read element hits
            element_df = pl.read_csv(
                element_file, separator="\t", comment_prefix="#"
            )

            if element_df.height == 0:
                config.logger.info(f"No hits in {element_file.name}, skipping")
                continue

            # Resolve overlaps based on user-specified mode
            if config.resolve_mode == "simple":
                # Use 'simple' mode for RNA elements (no adaptive overlap for nucleotides)
                resolved_df = consolidate_hits(
                    input=element_df,
                    column_specs="target_name,query_name",
                    rank_columns="-score,+e_value",
                    one_per_query=False,
                    one_per_range=True,
                    min_overlap_positions=config.min_overlap_positions,
                    merge=False,
                    split=False,
                    drop_contained=True,
                    alphabet="nucl",
                    adaptive_overlap=False,  # No polyprotein detection for RNA
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
                    input=element_df,
                    min_overlap_positions=config.min_overlap_positions,
                    column_specs="target_name,query_name",
                    rank_columns="-score,+e_value",
                    alphabet="nucl",
                    **resolve_mode_dict,
                )
            else:
                # No resolution
                resolved_df = element_df

            # Write resolved results
            resolved_file = (
                element_file.parent / f"{element_file.stem}_resolved.out"
            )
            resolved_df.write_csv(resolved_file, separator="\t")

            config.logger.info(
                f"Resolved {element_df.height} hits to {resolved_df.height} non-overlapping hits. "
                f"Output: {resolved_file}"
            )

        except Exception as e:
            config.logger.warning(
                f"Could not resolve overlaps in {element_file}: {e}"
            )
            # Non-critical error, continue processing
            continue

    config.logger.info("RNA element overlap resolution completed")


def combine_results(config):
    """Combine annotation results from different steps."""
    import polars as pl

    config.logger.info("Combining annotation results")

    all_results = []

    try:
        # Process ribozymes
        if "search_ribozymes" not in config.skip_steps:
            ribozymes_file = config.output_dir / "ribozymes.out"
            if (
                os.path.exists(ribozymes_file)
                and os.path.getsize(ribozymes_file) > 0
            ):
                ribozymes_data = process_ribozymes_data(config, ribozymes_file)
                if not ribozymes_data.is_empty():
                    all_results.append(("ribozyme", ribozymes_data))
                    config.logger.debug(
                        f"Added ribozyme data:\n{ribozymes_data.head()}"
                    )

        # Process IRES
        if "detect_ires" not in config.skip_steps:
            ires_file = config.output_dir / "ires.out"
            if ires_file.is_file():
                if config.ires_tool == "IRESfinder":
                    ires_data = process_ires_iresfinder(ires_file)
                else:
                    ires_data = process_ires_irespy(ires_file)
                if not ires_data.is_empty():
                    all_results.append(("ires", ires_data))
                    config.logger.debug(f"Added IRES data:\n{ires_data.head()}")

        # Process tRNAs
        if "predict_trnas" not in config.skip_steps:
            trnas_file = config.output_dir / "trnas.out"
            if trnas_file.is_file():
                if config.trna_tool == "tRNAscan-SE":
                    trnas_data = process_trnas_data_tRNAscan_SE(trnas_file)
                else:
                    # trnas_data = process_trnas_data_aragorn(trnas_file)
                    trnas_data = pl.DataFrame()
                    config.logger.warning(
                        "integration of Aragorn output is not supported - the raw output file is available at %s",
                        trnas_file,
                    )
                if not trnas_data.is_empty():
                    all_results.append(("trna", trnas_data))
                    config.logger.debug(
                        f"Added tRNA data:\n{trnas_data.head()}"
                    )

        # Process RNA elements
        if "search_rna_elements" not in config.skip_steps:
            rna_elements_file = config.output_dir / "rna_elements.out"
            if rna_elements_file.is_file():
                rna_elements_data = process_rna_elements_data(rna_elements_file)
                if not rna_elements_data.is_empty():
                    all_results.append(("rna_element", rna_elements_data))
                    config.logger.debug(
                        f"Added RNA elements data:\n{rna_elements_data.head()}"
                    )

        # Process RNA motifs
        if "search_rna_motifs" not in config.skip_steps:
            rna_motifs_file = config.output_dir / "rna_motifs.out"
            if rna_motifs_file.is_file():
                rna_motifs_data = process_rna_motifs_data(
                    config, rna_motifs_file
                )
                if not rna_motifs_data.is_empty():
                    all_results.append(("rna_motif", rna_motifs_data))
                    config.logger.debug(
                        f"Added RNA motifs data:\n{rna_motifs_data.head()}"
                    )

        # Process secondary structure
        if "predict_secondary_structure" not in config.skip_steps:
            structure_file = config.output_dir / "secondary_structure.fold"
            if structure_file.is_file():
                structure_data = read_multiDBN_to_dataframe(structure_file)
                if not structure_data.is_empty():
                    all_results.append(("secondary_structure", structure_data))
                    config.logger.debug(
                        f"Added secondary structure data:\n{structure_data.head()}"
                    )

        # Combine all results using unified schema
        if all_results:
            from rolypoly.utils.bio.polars_fastx import ensure_unified_schema

            # Ensure all dataframes have the same schema
            unified_dataframes = ensure_unified_schema(all_results)

            if unified_dataframes:
                # Stack all dataframes directly since they now have the same schema
                combined_data = pl.concat(unified_dataframes, how="vertical")
                config.logger.debug(f"Combined data:\n{combined_data.head()}")
                if config.output_format == "gff3":
                    combined_data = add_missing_gff_columns(combined_data)
                    write_combined_results_to_gff(config, combined_data)
                elif config.output_format == "csv":
                    output_file = config.output_dir / "combined_annotations.csv"
                    combined_data.write_csv(output_file)
                    config.logger.info(
                        f"Combined annotation results written to {output_file}"
                    )
                elif config.output_format == "tsv":
                    output_file = config.output_dir / "combined_annotations.tsv"
                    combined_data.write_csv(output_file, separator="\t")
                    config.logger.info(
                        f"Combined annotation results written to {output_file}"
                    )
            else:
                config.logger.warning(
                    "Failed to create unified schema for results"
                )
        else:
            config.logger.warning(
                "No results to combine - no valid data found in any output files"
            )

    except Exception as e:
        config.logger.error(f"Error combining results: {str(e)}")
        raise


def write_combined_results_to_gff(config, combined_data):
    from rolypoly.utils.bio.sequences import add_fasta_to_gff

    output_file = config.output_dir / "combined_annotations.gff3"
    with open(output_file, "w") as f:
        f.write("##gff-version 3\n")
        for row in combined_data.iter_rows(named=True):
            record = convert_record_to_gff3_record(row)
            config.logger.debug(f"Writing record:\n{row}")
            f.write(f"{record}\n")

    # Optionally add FASTA section
    add_fasta_to_gff(config, output_file)
    config.logger.info(f"Combined annotation results written to {output_file}")


def convert_record_to_gff3_record(
    row,
):  # for dict objects expected to be coherced into a gff3
    # try to identify a sequence_id columns (query, qseqid, contig_id, contig, id, name)
    sequence_id_columns = [
        "sequence_id",
        "query",
        "qseqid",
        "contig_id",
        "contig",
        "id",
        "name",
        "query_full_name",
    ]
    sequence_id_col = next(
        (col for col in sequence_id_columns if col in row.keys()), None
    )
    if sequence_id_col is None:
        raise ValueError(
            f"No sequence ID column found in row. Available columns: {list(row.keys())}"
        )

    # try to identify a score column (score, Score, bitscore, qscore, bit)
    score_columns = ["score", "Score", "bitscore", "qscore", "bit", "bits"]
    score_col = next(
        (col for col in score_columns if col in row.keys()), "score"
    )

    # try to identify a source column (source, Source, db, DB)
    source_columns = ["source", "Source", "db", "DB"]
    source_col = next(
        (col for col in source_columns if col in row.keys()), "source"
    )

    # try to identify a type column (type, Type, feature, Feature)
    type_columns = ["type", "Type", "feature", "Feature"]
    type_col = next((col for col in type_columns if col in row.keys()), "type")

    # try to identify a strand column (strand, Strand, sense, Sense)
    strand_columns = ["strand", "Strand", "sense", "Sense"]
    strand_col = next(
        (col for col in strand_columns if col in row.keys()), "strand"
    )

    # try to identify a phase column (phase, Phase)
    phase_columns = ["phase", "Phase"]
    phase_col = next(
        (col for col in phase_columns if col in row.keys()), "phase"
    )

    # Build GFF3 attributes string
    attrs = []
    # Define columns that should not be included in attributes
    excluded_cols = [
        sequence_id_col,
        source_col,
        score_col,
        type_col,
        strand_col,
        phase_col,
        "start",
        "end",  # Also exclude start/end since they're separate GFF3 fields
    ]

    for key, value in row.items():
        if key not in excluded_cols:
            # Skip empty values (empty strings, None, ".", etc.)
            if (
                value
                and str(value).strip()
                and str(value) != "."
                and str(value) != ""
            ):
                attrs.append(f"{key}={value}")

    # Get values, using defaults for missing columns
    sequence_id = row[sequence_id_col]
    source = row.get(source_col, "rp")
    score = row.get(score_col, "0")
    feature_type = row.get(type_col, "feature")
    strand = row.get(strand_col, "+")
    phase = row.get(phase_col, ".")

    # Format GFF3 record
    gff3_fields = [
        sequence_id,
        source,
        feature_type,
        str(row.get("start", "1")),
        str(row.get("end", "1")),
        str(score),
        strand,
        phase,
        ";".join(attrs) if attrs else ".",
    ]

    return "\t".join(gff3_fields)


def add_missing_gff_columns(dataframe):
    import polars as pl

    # check if the dataframe has the columns 'attributes',"source", "type", "score", "strand", "phase", if not add them
    # cols_to_check = ['attributes', 'source', 'type', 'score', 'strand', 'phase']
    if "source" not in dataframe.columns:
        dataframe = dataframe.with_columns(pl.lit("rp").alias("source"))
    if "type" not in dataframe.columns:
        dataframe = dataframe.with_columns(pl.lit("feature").alias("type"))
    if "score" not in dataframe.columns:
        dataframe = dataframe.with_columns(pl.lit("0").alias("score"))
    if "strand" not in dataframe.columns:
        dataframe = dataframe.with_columns(pl.lit("+").alias("strand"))
    if "phase" not in dataframe.columns:
        dataframe = dataframe.with_columns(pl.lit(".").alias("phase"))
    if "attributes" not in dataframe.columns:
        dataframe = dataframe.with_columns(pl.lit(".").alias("attributes"))
    if "start" not in dataframe.columns:
        dataframe = dataframe.with_columns(pl.lit(0).alias("start"))
    if "end" not in dataframe.columns:
        dataframe = dataframe.with_columns(pl.lit(1).alias("end"))
    return dataframe


if __name__ == "__main__":
    annotate_RNA()
    # TODO: prediction of ribosomal shunt
    # TODO: prediction of IRES
    # TODO: prediction of ribosomal slippage / ribosomal frameshifts
    # TODO: Identification of Kozak sequence motifs
