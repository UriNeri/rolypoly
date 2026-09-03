import os
from pathlib import Path as pt

import rich_click as click


global tools
tools = []
global matched_tabb
matched_tabb = []

BUILT_IN_DB_PATHS = {
    "ncbi_ribovirus": (
        "reference_seqs/ncbi_virus/mmseqs/refseq_ribovirus_genomes_cleaned"
    ),
    "ncbi_non_riboviria": (
        "reference_seqs/ncbi_virus/mmseqs/refseq_non_riboviria_genomes"
    ),
    "rvmt": "reference_seqs/RVMT/mmseqs/RVMT_cleaned",
}


def get_builtin_virus_db_paths(datadir):
    """Resolve built-in nucleic-search database paths below ROLYPOLY_DATA."""
    datadir = pt(datadir)
    return {name: datadir / path for name, path in BUILT_IN_DB_PATHS.items()}


@click.command(name="nucleic-search")
@click.option(
    "-o",
    "--output",
    default=lambda: f"{os.getcwd()}_RP_mapping",
    help="output file location - set suffix to .tab, .sam or html",
)
@click.option(
    "--db",
    "--database",
    type=click.Choice(
        ["RVMT", "NCBI_Ribovirus", "NCBI_Non_Riboviria", "all", "other"]
    ),
    default="all",
    help=(
        "Select the database to search against. 'all' retains its historical "
        "meaning: the two RNA-virus databases (RVMT and NCBI_Ribovirus)."
    ),
)
@click.option(
    "--db-path",
    default="",
    help="Path to the user-supplied source (required if --db is 'other'). Either a fasta or a path to formatted MMseqs2 virus database",
)
@click.option(
    "-i",
    "--input",
    required=True,
    help=(
        "Input FASTA/FASTQ file, comma-separated sequence files, directory of "
        "sequence files, or one preformatted MMseqs2 database prefix"
    ),
)
@click.option(
    "-mo",
    "--matched-output",
    # default=lambda: f"{os.getcwd()}/matched_virus_contigs.fasta",
    help="Output path for matched virus contigs. set to 'no' to skip writing matched contigs",
)
@click.option(
    "-e",
    "--mmseqs-evalue",
    default=1e-1,
    help="E-value threshold for MMseqs2 search)",
)
@click.option(
    "-id",
    "--mmseqs-identity",
    default=0.7,
    help="minimum Identity threshold for MMseqs2 search)",
)
@click.option(
    "-al",
    "--mmseqs-min-aln-len",
    default=95,
    help="Minimum alignment length for MMseqs2 search)",
)
def nucleic_search(
    threads,
    memory,
    output,
    keep_tmp,
    db,
    db_path,
    log_file,
    log_level,
    input,
    matched_output,
    mmseqs_evalue,
    mmseqs_identity,
    mmseqs_min_aln_len,
    temp_dir,
):
    """Search nucleotide reads or contigs against virus reference databases.

    Input can be one FASTA/FASTQ file, a comma-separated list, a directory, or
    an existing MMseqs2 database. Sequence inputs are combined into one MMseqs2
    query database. Records are searched independently; paired-read
    concordance is not evaluated here. Use ``rolypoly map`` for pair-aware read
    mapping.
    """
    import shutil
    import subprocess

    import polars as pl

    from rolypoly.utils.bio.interval_ops import (
        consolidate_hits,
        derive_strand_from_coordinates,
    )
    from rolypoly.utils.bio.library_detection import (
        is_fasta_file,
        is_sequence_file,
        resolve_sequence_inputs,
    )
    from rolypoly.utils.bio.sequences import filter_fasta_by_headers
    from rolypoly.utils.logging.citation_reminder import remind_citations
    from rolypoly.utils.logging.loggit import log_start_info, setup_logging

    # TODO: functionalize / use wrappers for mmseqs2.
    try:
        input_paths = resolve_sequence_inputs(
            input, allow_single_nonsequence=True
        )
    except (FileNotFoundError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    original_input_paths = input_paths.copy()
    output = pt(output).absolute().resolve()
    # Logging
    logger = setup_logging(log_file, log_level)

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
            "log_level": log_level,
            "temp_dir": temp_dir,
        },
    )
    invocation_name = click.get_current_context(silent=True)
    if invocation_name and invocation_name.info_name == "virus-mapping":
        logger.warning(
            "'virus-mapping' is deprecated; use 'nucleic-search' instead."
        )
    logger.info("Input: %s", ", ".join(str(path) for path in input_paths))
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

    output_suffix = output.suffix.lower()
    valid_output_formats = {".tab", ".sam", ".html"}
    if output_suffix == "":
        logger.info(
            "No output suffix provided for --output; defaulting to '.tab'."
        )
        output_format = ".tab"
        output_prefix = output
    elif output_suffix in valid_output_formats:
        output_format = output_suffix
        output_prefix = output.with_suffix("")
    else:
        raise click.BadParameter(
            "Unsupported --output suffix. Use one of: .tab, .sam, .html"
        )
    logger.info(
        "Started nucleic search for: %s",
        ", ".join(str(path) for path in input_paths),
    )

    # Create folders for MMseqs2 to use
    if temp_dir:
        tmpdir = pt(temp_dir).absolute().resolve()
        os.makedirs(tmpdir, exist_ok=True)
    else:
        tmpdir = output_path / "tmp"
        os.makedirs(tmpdir, exist_ok=True)
    res_path = tmpdir / "results_virus_mmdb/"
    shutil.rmtree(res_path, ignore_errors=True)
    os.makedirs(res_path, exist_ok=True)

    input_is_sequence_file = all(is_sequence_file(path) for path in input_paths)

    # If the input is FASTA/FASTQ, combine it into one MMseqs2 query DB.
    if input_is_sequence_file:
        logger.info(
            "Converting %d sequence input file(s) to an MMseqs2 DB",
            len(input_paths),
        )
        tmp = pt(tmpdir) / "pl_sv_contig_db"
        os.makedirs(tmp, exist_ok=True)
        subprocess.run(
            [
                "mmseqs",
                "createdb",
                *(str(path) for path in input_paths),
                str(tmp / "mmdb"),
                "--dbtype",
                "2",
            ],
            check=True,
        )
        input = tmp / "mmdb"
    else:
        input = input_paths[0]
    db = db.lower()  # Normalize db_name to lowercase for comparison
    db_paths_available = get_builtin_virus_db_paths(datadir)

    # Determine the databases to use
    if db == "all":
        db_paths = {
            name: db_paths_available[name]
            for name in ("ncbi_ribovirus", "rvmt")
        }
    elif db == "other":
        if not db_path:
            logger.warning(
                "Please provide a path to the user-supplied database with --db-path"
            )
            return
        if is_fasta_file(db_path):
            logger.info("Converting target db to mmseqs DB")
            tmp = pt(tmpdir) / "rp_sv_custom_db"
            os.makedirs(tmp, exist_ok=True)
            mmseqs_createdb_cmd = (
                f"mmseqs createdb {db_path} {tmp}/cmmdb  --dbtype 2"
            )
            subprocess.run(mmseqs_createdb_cmd, shell=True, check=True)
            db_path = (
                tmp / "cmmdb"
            )  # Ensure the path is updated correctly after creation
        db_paths = {"Custom": db_path}
    else:
        db_paths = {db: db_paths_available[db]}

    for db_name, db_path in db_paths.items():
        logger.info(f"Searching against {db_name}")
        this_resdb = res_path / db_name
        os.makedirs(this_resdb, exist_ok=True)

        # Perform the MMseqs2 search
        mmseqs_search_cmd = (
            f"mmseqs search {input} {db_path} {this_resdb}/res {tmpdir} "
            f"--min-seq-id {mmseqs_identity} --threads {threads} -a --search-type 3 -s 8 --strand 2"
            f" --max-seqs 10000 -e {mmseqs_evalue} --cov-mode 0 --min-aln-len {mmseqs_min_aln_len}"
        )
        subprocess.run(mmseqs_search_cmd, shell=True, check=True)

        # Convert results to desired format
        if output_format == ".tab":
            mmseqs_convertalis_cmd = (
                f"mmseqs convertalis {input} {db_path} {this_resdb}/res "
                f"{output_prefix}_vs_{db_name}.tab --format-mode 4 "
                f"--format-output qheader,theader,qlen,tlen,qstart,qend,tstart,tend,alnlen,mismatch,qcov,tcov,bits,evalue,gapopen,pident,nident"
            )
            subprocess.run(mmseqs_convertalis_cmd, shell=True, check=True)

            # Apply hit resolution with strand awareness
            logger.info(f"Resolving overlapping hits for {db_name}")
            result_file = pt(f"{output_prefix}_vs_{db_name}.tab")

            # Read results and derive strand from coordinates
            hits_df = pl.read_csv(result_file, separator="\t")
            hits_df = derive_strand_from_coordinates(
                hits_df, qstart_col="qstart", qend_col="qend"
            )

            # Resolve overlapping hits per-strand
            resolved_hits = consolidate_hits(
                input=hits_df,
                one_per_range=True,
                strand_col="strand",
                column_specs="qheader,theader",
                rank_columns="-bits,+evalue,-qcov",
                alphabet="nucl",
            )

            # Write resolved results
            resolved_hits.write_csv(
                str(result_file), separator="\t", include_header=True
            )
            logger.info(
                f"Wrote {len(resolved_hits)} resolved hits to {result_file}"
            )
        elif output_format == ".sam":
            mmseqs_convertalis_cmd = (
                f"mmseqs convertalis {input} {db_path} {this_resdb}/res "
                f"{output_prefix}_vs_{db_name}.sam --format-mode 1 --search-type 3"
            )
            subprocess.run(mmseqs_convertalis_cmd, shell=True, check=True)
        elif output_format == ".html":
            mmseqs_convertalis_cmd = (
                f"mmseqs convertalis {input} {db_path} {this_resdb}/res "
                f"{output_prefix}_vs_{db_name}.html --format-mode 3 --search-type 3"
            )
            subprocess.run(mmseqs_convertalis_cmd, shell=True, check=True)
        matched_tabb.append(
            f"{output_prefix}_vs_{db_name}.{output_format.lstrip('.')}"
        )

    matched_output_opt_out = (
        matched_output is None
        or str(matched_output).strip() == ""
        or str(matched_output).strip().lower() == "no"
    )

    if not matched_output_opt_out:
        logger.info(f"Writing matched virus contigs to {matched_output}")
        matched_output = pt(matched_output).absolute().resolve()
        matched_output.parent.mkdir(parents=True, exist_ok=True)

        if input_is_sequence_file:
            # Get matched query headers and recover records across every input.
            matched_headers = set()
            for db_name, db_path in db_paths.items():
                this_resdb = res_path / db_name
                header_tsv = tmpdir / f"matched_query_headers_{db_name}.tsv"
                mmseqs_headers_cmd = (
                    f"mmseqs convertalis {input} {db_path} {this_resdb}/res "
                    f"{header_tsv} --format-output qheader"
                )
                subprocess.run(mmseqs_headers_cmd, shell=True, check=True)

                if header_tsv.exists():
                    with open(header_tsv, "r") as fin:
                        for line in fin:
                            qheader = line.strip().split("\t")[0]
                            if qheader:
                                matched_headers.add(qheader)

            filter_counts = filter_fasta_by_headers(
                original_input_paths,
                matched_headers,
                matched_output,
                return_counts=True,
            )
            written_count = filter_counts["records_written"]
            logger.info(
                "Wrote %d unique matched nucleotide sequences to %s",
                written_count,
                matched_output,
            )
        else:
            logger.warning(
                "Input appears to be an MMseqs DB; falling back to MMseqs-only matched-output extraction."
            )
            matched_fasta_parts = []
            for db_name in db_paths:
                this_resdb = res_path / db_name
                subdb_path = tmpdir / f"matched_contigs_db_{db_name}"
                part_fasta = tmpdir / f"matched_contigs_{db_name}.fasta"

                mmseqs_extract_cmd = (
                    f"mmseqs createsubdb {this_resdb}/res {input} {subdb_path}"
                )
                subprocess.run(mmseqs_extract_cmd, shell=True, check=True)

                mmseqs_convertdb_cmd = (
                    f"mmseqs convert2fasta {subdb_path} {part_fasta}"
                )
                subprocess.run(mmseqs_convertdb_cmd, shell=True, check=True)

                if part_fasta.exists():
                    matched_fasta_parts.append(part_fasta)

            seen_headers = set()
            with open(matched_output, "w") as fout:
                for fasta_part in matched_fasta_parts:
                    write_record = False
                    with open(fasta_part, "r") as fin:
                        for line in fin:
                            if line.startswith(">"):
                                header = line[1:].strip()
                                write_record = header not in seen_headers
                                if write_record:
                                    seen_headers.add(header)
                            if write_record:
                                fout.write(line)

            logger.info(
                f"Wrote {len(seen_headers)} unique matched virus contigs to {matched_output}"
            )

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

    logger.info(
        "Finished nucleic search for: %s",
        ", ".join(str(path) for path in original_input_paths),
    )
    logger.info(f"Final output: {matched_tabb}")
    tools.append("mmseqs2")
    # remind_citations(tools)
    if logger.level != 10:  # If not DEBUG level
        with open(f"{log_file}", "a") as f_out:
            f_out.write(remind_citations(tools, return_bibtex=True) or "")


# Python-level compatibility for integrations importing the historical name.
virus_mapping = nucleic_search


if __name__ == "__main__":
    nucleic_search()
