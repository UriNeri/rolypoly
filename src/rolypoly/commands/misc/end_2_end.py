import os
from pathlib import Path

import rich_click as click


@click.command()
@click.option(
    "-i",
    "--input",
    required=True,
    help="Input path to raw RNA-seq data (fastq/gz file or directory with fastq/gz files)",
)
@click.option(
    "-o",
    "--output-dir",
    default=lambda: f"{os.getcwd()}_rp_e2e",
    help="Output directory",
)
@click.option("-t", "--threads", default=1, help="Number of threads")
@click.option("-M", "--memory", default="6g", help="Memory allocation")
@click.option(
    "-D",
    "--host",
    help="Path to the user-supplied host/contamination fasta /// Fasta file of known DNA entities expected in the sample",
)
@click.option("--keep-tmp", is_flag=True, help="Keep temporary files")
@click.option(
    "--log-file",
    default=lambda: f"{os.getcwd()}/rolypoly_pipeline.log",
    help="Path to log file",
)
@click.option(
    "-ll",
    "--log-level",
    default="INFO",
    help="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
)
@click.option(
    "--skip-existing", is_flag=True, help="Skip commands if output files already exist"
)
# Assembly options
@click.option(
    "-A",
    "--assembler",
    default="spades,megahit,penguin",
    help="Assembler choice. For multiple, give a comma-separated list",
)
@click.option(
    "-d", "--post-cluster", is_flag=True, help="Perform post-assembly clustering"
)
# Filter contigs options
@click.option(
    "-Fm1",
    "--filter1_nuc",
    default="alnlen >= 120 & pident>=75",
    help="First set of rules for nucleic filtering by aligned stats",
)
@click.option(
    "-Fm2",
    "--filter2_nuc",
    default="qcov >= 0.95 & pident>=95",
    help="Second set of rules for nucleic match filtering",
)
@click.option(
    "-Fd1",
    "--filter1_aa",
    default="length >= 80 & pident>=75",
    help="First set of rules for amino (protein) match filtering",
)
@click.option(
    "-Fd2",
    "--filter2_aa",
    default="qcovhsp >= 95 & pident>=80",
    help="Second set of rules for protein match filtering",
)
@click.option(
    "--dont-mask",
    is_flag=True,
    help="If set, host fasta won't be masked for potential RNA virus-like seqs",
)
@click.option(
    "--mmseqs-args", help="Additional arguments to pass to MMseqs2 search command"
)
@click.option(
    "--diamond-args",
    default="--id 50 --min-orf 50",
    help="Additional arguments to pass to Diamond search command",
)
# Marker gene search options
@click.option("--db", default="all", help="Database to use for marker gene search")
def run_pipeline(
    input,
    output_dir,
    threads,
    memory,
    host,
    keep_tmp=False,
    log_file=None,
    assembler="spades,megahit,penguin",
    post_cluster=False,  # TODO: add or remove this, decide.
    filter1_nuc="alnlen >= 120 & pident>=75",
    filter2_nuc="qcov >= 0.95 & pident>=95",
    filter1_aa="length >= 80 & pident>=75",
    filter2_aa="qcovhsp >= 95 & pident>=80",
    dont_mask=False,
    mmseqs_args=None,
    diamond_args="--id 50 --min-orf 50",
    db="all",
    skip_existing=False,
    log_level="INFO",
):
    """End-to-end pipeline for RNA virus discovery from raw sequencing data.

    This pipeline performs a complete analysis workflow including:
    1. Read filtering and quality control
    2. De novo assembly
    3. Contig filtering
    4. Marker gene search (default: RdRps)
    5. Genome annotation
    6. Virus characteristics prediction

    Args:
        input (str): Path to raw RNA-seq data (fastq/gz file or directory)
        output_dir (str): Output directory path (default: current_dir_rp_e2e)
        threads (int): Number of CPU threads to use (default: 1)
        memory (str): Memory allocation with units (e.g., "6g") (default: "6g")
        host (str): Path to host/contamination FASTA file
        keep_tmp (bool): Keep temporary files if True (default: False)
        log_file (str): Path to log file (default: rolypoly_pipeline.log)
        assembler (str): Comma-separated list of assemblers (default: "spades,megahit,penguin")
        post_cluster (bool): Perform post-assembly clustering if True (default: False)
        filter1_nuc (str): First nucleotide filtering rules (default: "alnlen >= 120 & pident>=75")
        filter2_nuc (str): Second nucleotide filtering rules (default: "qcov >= 0.95 & pident>=95")
        filter1_aa (str): First amino acid filtering rules (default: "length >= 80 & pident>=75")
        filter2_aa (str): Second amino acid filtering rules (default: "qcovhsp >= 95 & pident>=80")
        dont_mask (bool): Skip masking host FASTA for RNA virus-like sequences (default: False)
        mmseqs_args (str): Additional MMseqs2 search arguments
        diamond_args (str): Additional Diamond search arguments (default: "--id 50 --min-orf 50")
        db (str): Marker gene search database name (default: "neordrp")

    Returns:
        None: Results are written to the specified output directory
    """
    from rolypoly.commands.annotation.annotate import annotate
    from rolypoly.commands.assembly.assemble import assembly
    from rolypoly.commands.assembly.filter_contigs import filter_contigs
    from rolypoly.commands.identify_virus.marker_search import (
        marker_search as marker_search,
    )
    from rolypoly.commands.reads.filter_reads import filter_reads
    from rolypoly.commands.virotype.predict_characteristics import (
        predict_characteristics,
    )
    from rolypoly.rolypoly import get_version_info
    from rolypoly.utils.loggit import (  # , check_file_exists, check_file_size
        log_start_info,
        setup_logging,
    )

    known_dna = host
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "rolypoly_pipeline.log"
    logger = setup_logging(log_file, log_level.upper())

    logger.info("Starting RolyPoly pipeline version: %s", get_version_info())
    if log_level == "debug":
        logger.debug("Debug mode enabled")
        import sys

        log_start_info(logger, dict(zip(sys.argv[1::2], sys.argv[2::2])))

    # Step 1: Filter Reads
    logger.info("Step 1: Filtering reads    ")
    filtered_reads = output_dir / "filtered_reads"
    if skip_existing:
        if filtered_reads.exists():
            logger.info("Filtered reads already exist, skipping step")
            pass
    else:
        filtered_reads.mkdir(parents=True, exist_ok=True)
        ctx = click.Context(filter_reads)
        ctx.invoke(
            filter_reads,
            input=input,
            output=str(filtered_reads),
            threads=threads,
            memory=memory,
            known_dna=known_dna,
            keep_tmp=keep_tmp,
            log_file=logger,
            speed=15,
        )

    # Step 2: Assembly
    logger.info("Step 2: Performing assembly    ")
    assembly_output = output_dir / "assembly"
    final_assembly = assembly_output / "final_assembly.fasta"
    if skip_existing and final_assembly.exists():
        logger.info("Assembly output already exists, skipping step")
    else:
        ctx = click.Context(assembly)
        ctx.invoke(
            assembly,
            threads=threads,
            memory=memory,
            output=str(assembly_output),
            keep_tmp=keep_tmp,
            log_file=str(log_file),
            input=str(filtered_reads),
            assembler=assembler,
            overwrite=skip_existing,
        )

    # Step 3: Filter Assembly
    logger.info("Step 3: Filtering assembly    ")
    filtered_assembly = output_dir / "assemblies" / "filtered_assembly.fasta"
    if skip_existing and filtered_assembly.exists():
        logger.info("Filtered assembly already exists, skipping step")
    else:
        ctx = click.Context(filter_contigs)
        ctx.invoke(
            filter_contigs,
            input=str(final_assembly),
            host=host,
            output=str(filtered_assembly),
            mode="both",
            threads=threads,
            memory=memory,
            keep_tmp=keep_tmp,
            log_file=str(log_file),
            filter1_nuc=filter1_nuc,
            filter2_nuc=filter2_nuc,
            filter1_aa=filter1_aa,
            filter2_aa=filter2_aa,
            dont_mask=dont_mask,
            mmseqs_args=mmseqs_args,
            diamond_args=diamond_args,
        )

    # Step 4: marker protein Search
    logger.info("Step 4: Searching for marker protein sequences    ")
    marker_output = output_dir / "marker_search_results"
    if skip_existing and marker_output.exists():
        logger.info("Marker search results already exist, skipping step")
    else:
        ctx = click.Context(marker_search)
        ctx.invoke(
            marker_search,
            input=str(filtered_assembly),
            output=str(marker_output),
            threads=threads,
            memory=memory,
            db=db,
            keep_tmp=keep_tmp,
            log_file=str(log_file),
        )

    # Step 5: Annotation
    logger.info("Step 5: Annotation")
    annotation_output = output_dir / "annotation_results"
    if skip_existing and annotation_output.exists():
        logger.info("Annotation results already exist, skipping step")
    else:
        ctx = click.Context(annotate)
        ctx.invoke(
            annotate,
            input=str(marker_output),
            output=str(annotation_output),
            threads=threads,
            memory=memory,
            keep_tmp=keep_tmp,
            log_file=str(log_file),
        )

    # Step 6: Predict Virus Characteristics
    logger.info("Step 6: Predicting virus characteristics    ")
    characteristics_output = output_dir / "virus_characteristics.tsv"
    if skip_existing and characteristics_output.exists():
        logger.info("Virus characteristics already exist, skipping step")
    else:
        ctx = click.Context(predict_characteristics)
        ctx.invoke(
            predict_characteristics,
            input=str(output_dir),
            output=str(characteristics_output),
            database=os.path.join(
                os.environ["datadir"], "virus_literature_database.tsv"
            ),
            threads=threads,
            log_file=str(log_file),
        )

    logger.info("RolyPoly pipeline completed successfully!")


if __name__ == "__main__":
    run_pipeline()
