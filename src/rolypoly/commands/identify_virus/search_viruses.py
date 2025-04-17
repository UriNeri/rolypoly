import os
from pathlib import Path as pt

import rich_click as click
from rich.console import Console

global tools
tools = []

console = Console()


@click.command()
@click.option("-t", "--threads", default=1, help="Threads (all) [1]")
@click.option("-M", "--memory", default="6g", help="MEMORY in gb (more) [6]")
@click.option(
    "-o",
    "--output",
    default=lambda: f"{os.getcwd()}_RP_mapping",
    help="output file location - set suffix to .tab, .sam or html [default: .tab]",
)
@click.option("--keep-tmp", is_flag=True, default=False, help="Keep temporary files")
@click.option(
    "--db",
    type=click.Choice(["RVMT", "NCBI_Ribovirus", "all", "other"]),
    default="all",
    help="""Select the database to search against.""",
)
@click.option(
    "--db-path",
    default="",
    help="Path to the user-supplied source (required if --db is 'other'). Either a fasta or a path to formatted MMseqs2 virus database",
)
@click.option(
    "-g",
    "--log-file",
    default=lambda: f"{os.getcwd()}/search_viruses_logfile.txt",
    help="Abs path to logfile",
)
@click.option(
    "-i",
    "--input",
    required=True,
    help="Input path to nucl fasta file OR preformatted mmseqs db",
)
def virus_mapping(threads, memory, output, keep_tmp, db, db_path, log_file, input):
    """MMseqs2 Virus mapping/search wrapper - takes in reads/contigs (i.e. nucs), and search them against precompiled virus databases OR user-supplied databases."""
    import shutil
    import subprocess

    from rolypoly.utils.citation_reminder import remind_citations
    from rolypoly.utils.loggit import log_start_info, setup_logging

    # TODO: functionalize / use wrappers for mmseqs2.
    input = pt(input).absolute().resolve()
    og_input = input
    output = pt(output).absolute().resolve()
    # Logging
    logger = setup_logging(log_file)

    log_start_info(
        logger,
        {
            "input": input,
            "output": output,
            "db": db,
            "db_path": db_path,
            "threads": threads,
            "memory": memory,
            "keep_tmp": keep_tmp,
            "log_file": log_file,
        },
    )
    logger.info(f"Input : {input}")
    logger.info(f"Virus db: {db}")

    # Get environment
    datadir = pt(os.environ["ROLYPOLY_DATA"])

    os.environ["MEMORY"] = memory
    os.environ["THREADS"] = str(threads)

    # Main logic
    output = pt(output).absolute().resolve()
    output_path = output.parent
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        logger.warning(f"Output path already exists: {output_path}")

    output_format = output.suffix
    logger.info(f"Started virus mapping for: {input}")

    # TODO: functionalize / use wrappers for mmseqs2.

    # Create folders for MMseqs2 to use
    tmpdir = output_path / "tmp"
    os.makedirs(tmpdir, exist_ok=True)
    res_path = tmpdir / "results_virus_mmdb/"
    shutil.rmtree(res_path, ignore_errors=True)
    os.makedirs(res_path, exist_ok=True)

    # if the input is fasta Convert the input  into an mmseqs DB
    if pt(input).suffix in [
        ".faa",
        ".fasta",
        ".fas",
        ".fa",
        ".fna",
        ".fastq",
        ".fq",
        ".gz",
        ".fastq.gz",
        ".fq.gz",
        ".fasta.gz",
        ".fa.gz",
        ".fas.gz",
        ".faa.gz",
    ]:
        logger.info(f"Converting input to mmseqs DB")
        tmp = pt(tmpdir) / "pl_sv_contig_db"
        os.makedirs(tmp, exist_ok=True)
        mmseqs_createdb_cmd = f"mmseqs createdb {str(input)} {tmp}/mmdb  --dbtype 2"
        subprocess.run(mmseqs_createdb_cmd, shell=True, check=True)
        input = tmp / "mmdb"  # Ensure the path is updated correctly after creation

    DB_PATHS = {
        "NCBI_Ribovirus": datadir / "NCBI_ribovirus/mmdb/refseq_ribovirus",
        "RVMT": datadir / "RVMT/mmdb/RVMT_mmseqs_db2",
    }

    # Determine the databases to use
    if db == "all":
        db_paths = DB_PATHS
    elif db == "other":
        if not db_path:
            console.print(
                "[bold red]Error:[/bold red] Please provide a path to the user-supplied database with --db-path"
            )
            return
        if pt(db_path).suffix in [".faa", ".fasta", ".fas", ".fa", ".fna"]:
            logger.info(f"Converting target db to mmseqs DB")
            tmp = pt(tmpdir) / "rp_sv_custom_db"
            os.makedirs(tmp, exist_ok=True)
            mmseqs_createdb_cmd = f"mmseqs createdb {db_path} {tmp}/cmmdb  --dbtype 2"
            subprocess.run(mmseqs_createdb_cmd, shell=True, check=True)
            db_path = (
                tmp / "cmmdb"
            )  # Ensure the path is updated correctly after creation
        db_paths = {"Custom": db_path}
    else:
        db_paths = {db: DB_PATHS[db]}

    for db_name, db_path in db_paths.items():
        logger.info(f"Searching against {db_name}")
        this_resdb = res_path / db_name
        os.makedirs(this_resdb, exist_ok=True)

        # Perform the MMseqs2 search
        mmseqs_search_cmd = (
            f"mmseqs search {input} {db_path} {this_resdb}/res {tmpdir} "
            f"--min-seq-id 0.5 --threads {threads} -a --search-type 3 -s 8 --strand 2"
        )
        subprocess.run(mmseqs_search_cmd, shell=True, check=True)

        # Convert results to desired format
        if output_format == ".tab":
            mmseqs_convertalis_cmd = (
                f"mmseqs convertalis {input} {db_path} {this_resdb}/res "
                f"{output.with_suffix('')}_vs_{db_name}.tab --format-mode 4 "
                f"--format-output qheader,theader,qlen,tlen,qstart,qend,tstart,tend,alnlen,mismatch,qcov,tcov,bits,evalue,gapopen,pident,nident"
            )
            subprocess.run(mmseqs_convertalis_cmd, shell=True, check=True)
        elif output_format == ".sam":
            mmseqs_convertalis_cmd = (
                f"mmseqs convertalis {input} {db_path} {this_resdb}/res "
                f"{output.with_suffix('')}_vs_{db_name}.sam --format-mode 1 --search-type 3"
            )
            subprocess.run(mmseqs_convertalis_cmd, shell=True, check=True)
        elif output_format == ".html":
            mmseqs_convertalis_cmd = (
                f"mmseqs convertalis {input} {db_path} {this_resdb}/res "
                f"{output.with_suffix('')}_vs_{db_name}.html --format-mode 3 --search-type 3"
            )
            subprocess.run(mmseqs_convertalis_cmd, shell=True, check=True)

    # Clean up
    # Remove intermediate files
    if not keep_tmp:
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)
        if os.path.exists(res_path):
            shutil.rmtree(res_path, ignore_errors=True)
        for tmp_file in pt(".").glob("tmp*"):
            if tmp_file.is_dir():
                shutil.rmtree(tmp_file)
            else:
                tmp_file.unlink()
        for tmp_file in pt(".").glob("tmp*/*"):
            if tmp_file.is_dir():
                shutil.rmtree(tmp_file)
            else:
                tmp_file.unlink()
        for tmp_file in pt(".").glob("search_virus_mmdb*"):
            if tmp_file.is_dir():
                shutil.rmtree(tmp_file)
            else:
                tmp_file.unlink()

    # subprocess.run(f"bgzip -@{threads} *_virus_mapping_out.tab", shell=True, check=True)

    logger.info(f"Finished virus mapping for: {og_input}")
    logger.info(f"Final output: {output}")
    tools.append("mmseqs2")
    # remind_citations(tools)
    with open(f"{log_file}", "a") as f_out:
        f_out.write(remind_citations(tools, return_bibtex=True))


if __name__ == "__main__":
    virus_mapping()
