import os
from pathlib import Path

from rich.console import Console
from rich_click import Choice, command, option

from rolypoly.utils.config import BaseConfig


class RVirusSearchConfig(BaseConfig):
    def __init__(self, **kwargs):
        # Always treat output as a directory
        output_path = Path(kwargs.get("output", ""))
        kwargs["output_dir"] = str(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        super().__init__(
            input=kwargs.get("input"),
            output=kwargs.get("output"),
            keep_tmp=kwargs.get("keep_tmp"),
            log_file=kwargs.get("log_file"),
            threads=kwargs.get("threads"),
            memory=kwargs.get("memory"),
            config_file=kwargs.get("config_file"),
            overwrite=kwargs.get("overwrite"),
            log_level=kwargs.get("log_level"),
            temp_dir=kwargs.get("temp_dir"),
        )  # initialize the BaseConfig class
        # initialize the rest of the parameters (i.e. the ones that are not in the BaseConfig class)
        self.database = kwargs.get("database")
        self.inc_evalue = kwargs.get("inc_evalue") or 0.05
        self.score = kwargs.get("score") or 20
        self.aa_method = kwargs.get("aa_method") or "six_frame"
        self.resolve_mode = kwargs.get("resolve_mode") or "simple"
        self.min_overlap_positions = kwargs.get("min_overlap_positions") or 10
        self.name = kwargs.get("name") or None


global tools
tools = []

console = Console(width=150)


@command()
@option(
    "-i",
    "--input",
    required=True,
    help="Input fasta file. Preferably nucleotide contigs, but you can provide amino acid input too (the script would skip 6 frame translation)",
)
@option(
    "-o",
    "--output",
    default=lambda: f"{os.getcwd()}/marker_search_out",
    help="Path to output directory. Note - if multiple DBs are used and the resolve-mode is `none`, multiple outputs are made (DB name appended as suffix).",
)
@option(
    "-rm",
    "--resolve-mode",
    default="simple",
    type=Choice(
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
    help="""How to deal with regions in your query that match multiple profiles? \n
        - merge: all overlapping hits are merged into one range \n
        - one_per_range: one hit per range (ali_from-ali_to) is reported \n
        - one_per_query: one hit per query sequence is reported \n
        - split: each overlapping domain is split into a new row \n
        - drop_contained: hits that are contained within (i.e. enveloped by) other hits are dropped. \n
        - none: no resolution of overlapping hits is performed. \n
        - simple: heuristic/personal observation based - chains drop_contained output with split mode. \n
        """,
)
@option(
    "-mo",
    "--min-overlap-positions",
    default=10,
    help="Minimal number of overlapping positions between two intersecting ranges before they are considered as overlapping (used in some resolve_mode(s)",
)
@option(
    "-ie",
    "--inc-evalue",
    default=0.05,
    help="Maximal e-value for including a domain match in the results",
)  #  for HMM reporting
@option(
    "-s",
    "--score",
    default=20,
    help="Minimal score for including a domain match in the results",
)
@option(
    "-am",
    "--aa-method",
    default="six_frame",
    type=Choice(["six_frame", "pyrodigal", "bbmap"]),
    help="Method to translate nucleotide sequences into amino acids. Options: six frame translation using seqkit, pyrodigal-gv uses pyrodigal-meta with additional genetic codes, bbmap callgenes.sh (quick but less accurate for metagenomic data)",
)
@option(
    "-db",
    "--database",
    type=str,
    default="NeoRdRp_v2.1,genomad",
    help="""comma separated list of databases to search against (or `all`), or path to a custom database. \n
        options: NeoRdRp_v2.1, RdRp-scan, RVMT, TSA_2018, Pfam_A_37, genomad, all, \n
        For custom path, either an .hmm file, a directory with .hmm files, or a folder with MSA files (which would be used to build an HMM DB)""",
)
@option("-t", "--threads", default=1, help="Number of threads to use for searching")
@option(
    "-g",
    "--log-file",
    default="./marker_search_logfile.txt",
    help="Absolute path to logfile",
)
@option(
    "-m", "--memory", hidden=True, default="6g", help="Memory limit for the job in GB"
)
@option(
    "-cf",
    "--config-file",
    hidden=True,
    default=None,
    help="path to a json config file with parameters for the search - overrides command line parameters",
)
@option(
    "-n",
    "--name",
    hidden=True,
    default=None,
    help="basename for the output files (default is the basename of the input file)",
)
@option(
    "-k",
    "--keep-tmp",
    hidden=True,
    is_flag=True,
    default=False,
    help="Keep the temporary files",
)
@option(
    "-ow",
    "--overwrite",
    is_flag=True,
    default=False,
    help="Do not overwrite the output directory if it already exists",
)
@option(
    "-ll",
    "--log-level",
    default="info",
    hidden=True,
    help="Log level. Options: debug, info, warning, error, critical",
)
@option(
    "-td",
    "-tempdir",
    "--temp-dir",
    default="./marker_search_tmp/",
    help="Path to temporary directory",
)
def marker_search(
    input,
    output,
    resolve_mode,
    min_overlap_positions,
    inc_evalue,
    score,
    aa_method,
    database,
    threads,
    log_file,
    memory,
    config_file,
    name,
    keep_tmp,
    overwrite,
    log_level,
    temp_dir,
):
    """RNA virus marker protein search - using pre-made/user-supplied DBs.
    Most pre-made DBs are based on RdRp domain (except for geNomad).
    Input can be nucleotide contigs or amino acid seqs.
    If nucleotide, by default all contigs will be translated to six end-to-end frames (with stops replaced by `X`), or into ORFs called by pyrodigal (meta) or callgenes.sh
    Please cite accordingly based on the DBs you select. Pre-compiled options are: \n
    • NeoRdRp2.1 \n
        GitHub: https://github.com/shoichisakaguchi/NeoRdRp  | Paper: https://doi.org/10.1264/jsme2.ME22001 \n
    • RVMT \n
        GitHub: https://github.com/UriNeri/RVMT  | Zenodo: https://zenodo.org/record/7368133  |  Paper: https://doi.org/10.1016/j.cell.2022.08.023 \n
    • RdRp-Scan \n
        GitHub: https://github.com/JustineCharon/RdRp-scan  |  Paper: https://doi.org/10.1093/ve/veac082 \n
            ⤷ (which IIRC incorporated PALMdb, GitHub: https://github.com/rcedgar/palmdb, Paper: https://doi.org/10.7717/peerj.14055 \n
    • TSA_Olendraite (TSA_2018) \n
        Data: https://drive.google.com/drive/folders/1liPyP9Qt_qh0Y2MBvBPZQS6Jrh9X0gyZ?usp=drive_link  |  Paper: https://doi.org/10.1093/molbev/msad060 \n
        Thesis: https://www.repository.cam.ac.uk/items/1fabebd2-429b-45c9-b6eb-41d27d0a90c2
    • Pfam_A_37 \n
        RdRps and RT profiles from PFAM_A v.37 --- PF04197.17,PF04196.17,PF22212.1,PF22152.1,PF22260.1,PF05183.17,PF00680.25,PF00978.26,PF00998.28,PF02123.21,PF07925.16,PF00078.32,PF07727.19,PF13456.11
        Data: https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam37.0/Pfam-A.hmm.gz | Paper https://doi.org/10.1093/nar/gkaa913
    • geNomad \n
        RNA virus marker genes from geNomad v1.9 --- https://zenodo.org/records/14886553
    For custom path, either an .hmm file, a directory with .hmm files, or a folder with MSA files (which would be used to build an HMM DB).
    """
    import json

    import polars as pl

    from rolypoly.utils.citation_reminder import remind_citations
    from rolypoly.utils.fax import (
        guess_fasta_alpha,
        hmm_from_msa,
        hmmdb_from_directory,
        pyro_predict_orfs,
        search_hmmdb,
        translate_6frx_seqkit,
        translate_with_bbmap,
    )
    from rolypoly.utils.interval_ops import consolidate_hits
    from rolypoly.utils.loggit import log_start_info


    # Determine if output should be treated as directory based on resolve_mode and path
    output = str(Path(output).absolute())
    is_directory_output = resolve_mode == "none" or output.endswith("/")

    if is_directory_output:
        # Ensure output ends with '/' to signal directory to config
        if not output.endswith("/"):
            output = output + "/"
    else:
        # Ensure parent directory exists for file output
        Path(output).parent.mkdir(parents=True, exist_ok=True)

    if config_file:
        config = RVirusSearchConfig(**json.load(open(config_file)))
    else:
        if not name:
            name = Path(input).stem
        config = RVirusSearchConfig(
            input=input,
            output=output,
            inc_evalue=inc_evalue,
            score=score,
            aa_method=aa_method,
            temp_dir=temp_dir,
            database=database,
            overwrite=overwrite,
            log_level=log_level,
            threads=threads,
            log_file=log_file,
            name=name,
            keep_tmp=keep_tmp,
            resolve_mode=resolve_mode,
            min_overlap_positions=min_overlap_positions,
            memory=memory,
        )

    # Logging
    log_start_info(config.logger, config.to_dict())

    config.logger.info(f"Starting RNA virus marker protein search with: {config.input}")

    # Determine the databases to use
    hmmdbdir = Path(os.environ["ROLYPOLY_DATA"]) / "hmmdbs"

    DB_PATHS = {
        "NeoRdRp_v2.1".lower(): hmmdbdir / "neordrp/zenodo/NeoRdRp.2.1.hmm",
        "RdRp-scan".lower(): hmmdbdir / "RdRp-scan.hmm",
        "RVMT".lower(): hmmdbdir / "RVMT.hmm",
        "TSA_2018".lower(): hmmdbdir / "TSA_Olendraite.hmm",
        "Pfam_A_37".lower(): hmmdbdir / "rt_rdrp_pfamA37.hmm",
        "genomad".lower(): hmmdbdir / "genomad.hmm",
    }

    if database == "all":
        database_paths = DB_PATHS
    elif database.startswith("/") or database.startswith("./"):
        custom_database = str(Path(database).resolve())
        if not Path(custom_database).exists():
            config.logger.error(f"Custom database path {custom_database} does not exist")
            return
        else:
            # check if a file it's an hmm or an msa file
            if custom_database.endswith(".hmm"):
                database_paths = {"Custom": custom_database}
            elif custom_database.endswith((".faa", ".fasta", ".afa")):
                from rolypoly.utils.fax import hmm_from_msa

                database_paths = {
                    "Custom": hmm_from_msa(
                        msa_file=database,
                        output=database.replace(".faa", ".hmm"),
                        name=Path(database).stem,
                    )
                }
            # if it's a directory:
            elif Path(custom_database).is_dir():
                # cehck if all files are hmm
                if all(f.endswith((".hmm")) for f in Path(custom_database).glob("*")):
                    # concatenate all hmms into one file
                    hmm_files = Path(custom_database).glob("*.hmm")
                    with open(Path(custom_database) / "concatenated.hmm", "w") as f:
                        for hmm_file in hmm_files:
                            with open(hmm_file, "r") as hmm_file_obj:
                                f.write(hmm_file_obj.read())
                    database_paths = {"Custom": str(Path(custom_database) / "concatenated.hmm")}
                elif all(
                    f.endswith((".faa", ".fasta", ".afa"))
                    for f in Path(custom_database).glob("*")
                ):
                    from rolypoly.utils.fax import hmmdb_from_directory

                    hmmdb_from_directory(
                        msa_dir=custom_database,
                        output=Path(custom_database) / "all_msa_built.hmm",
                        alphabet="aa",
                    )
                    database_paths = {"Custom": str(Path(custom_database) / "all_msa_built.hmm")}
            else:
                config.logger.error(f"Invalid custom database path: {custom_database}")
                return
    else:
        databases = database.split(",")
        database_paths = {db: DB_PATHS[db.lower()] for db in databases}

    input_alpha = guess_fasta_alpha(input)

    if input_alpha == "nucl":
        config.logger.info(f"Input identified as nucl")
        amino_file = str(config.temp_dir / f"{config.name}")
        if config.aa_method == "pyrodigal":
            config.logger.info("Predicting ORFs using pyrodigal-gv")
            amino_file = amino_file + "_pyro.faa"
            pyro_predict_orfs(input, amino_file, threads)
            tools.append("pyrodigal")
        elif config.aa_method == "bbmap":
            config.logger.info("Using BBMap's callgenes.sh for translation")
            amino_file = amino_file + "_cg.faa"
            translate_with_bbmap(input, amino_file, threads)
            tools.append("bbmap")
        else:
            config.logger.info("Using seqkit for 6 frames translation")
            amino_file = amino_file + "_6frx.faa"
            translate_6frx_seqkit(input, amino_file, threads)
            tools.append("seqkit")
    elif input_alpha == "aa":
        config.logger.info("Using supplied amino acid fasta file, skipping translation")
        amino_file = input
    else:
        config.logger.error(
            "Input is not in fasta format or seqs not recognized as nucleotide or amino acid"
        )
        return

    all_outputs = []
    config.logger.info(f"Searching with {amino_file}")
    for db_name, db_path in database_paths.items():
        # Search translated sequences against viral marker databases
        config.logger.info(f"Searching {db_name}")
        tools.append(f"{db_name}")
        tmp_output = config.temp_dir / f"raw_{config.name}_vs_{db_name}.tsv"
        output_database = search_hmmdb(
            amino_file=amino_file,
            db_path=db_path,
            output=tmp_output,
            threads=threads,
            logger=config.logger,
            inc_e=config.inc_evalue,
            mscore=config.score,
            output_format="modomtblout",
            ali_str=True,
            full_qseq=True,
            match_region=True,
        )
        all_outputs.append(tmp_output)

    # read all output files, stack them, and resolve overlaps
    config.logger.debug(f"Reading {len(all_outputs)} output files")
    stack_df = pl.scan_csv(
        all_outputs, separator="\t", infer_schema_length=123123
    ).collect()
    config.logger.debug(stack_df)
    if stack_df.is_empty():
        config.logger.info(f"No hits found in any DB")
        config.logger.info(f"skipping resolution of overlaps")
        config.resolve_mode = "none"

    results_file = Path(output) / "marker_search_results.tsv"

    if config.resolve_mode == "simple":
        testdf = consolidate_hits(
            input=stack_df,
            one_per_query=False,
            one_per_range=True,
            min_overlap_positions=config.min_overlap_positions,
            merge=False,
            split=False,
            column_specs="query_full_name,hmm_full_name",
            rank_columns="-full_hmm_score,+full_hmm_evalue",
            drop_contained=True,
        )

        testdf = consolidate_hits(
            input=testdf,
            one_per_query=False,
            one_per_range=False,
            min_overlap_positions=config.min_overlap_positions,
            merge=False,
            split=True,
            column_specs="query_full_name,hmm_full_name",
            rank_columns="-full_hmm_score,+full_hmm_evalue",
            drop_contained=False,
        )

    elif config.resolve_mode != "none":
        resolve_mode_dict = {
            "split": False,
            "one_per_range": False,
            "one_per_query": False,
            "merge": False,
            "drop_contained": False,
            "simple": False,
        }
        resolve_mode_dict[config.resolve_mode] = True
        testdf = consolidate_hits(
            input=stack_df,
            min_overlap_positions=config.min_overlap_positions,
            column_specs="query_full_name,hmm_full_name",
            rank_columns="-full_hmm_score,+full_hmm_evalue",
            **resolve_mode_dict,
        )

    # Write to a file in the output directory instead of the directory itself
    testdf.write_csv(results_file, separator="\t")

    # Remove temporary directory if keep_tmp is False
    if not config.keep_tmp:
        import shutil

        shutil.rmtree(config.temp_dir)
        config.logger.info(f"Removed temporary directory: {config.temp_dir}")

    config.logger.info(f"""Finished RNA virus marker protein search using : {input}""")
    output_files = [ix.absolute() for ix in Path(output).glob("*.tsv")]
    config.logger.info(f"""Outputs saved to {output_files}""")

    tools.append("pyhmmer")
    tools.append("hmmer")

    with open(f"{config.log_file}", "w") as f_out:
        f_out.write(remind_citations(tools, return_bibtex=True))

if __name__ == "__main__":
    marker_search
