import rich_click as click
from pathlib import Path
import os

@click.command()
@click.option("-i", "--input", required=True, help="Input path to raw RNA-seq data (fastq/gz file or directory with fastq/gz files)")
@click.option("-o", "--output-dir", default=lambda: f"{os.getcwd()}_rp_e2e", help="Output directory")
@click.option("-t", "--threads", default=1, help="Number of threads")
@click.option("-M", "--memory", default="6g", help="Memory allocation")
@click.option("-D","--host", help="Path to the user-supplied host/contamination fasta /// Fasta file of known DNA entities expected in the sample")
@click.option("--keep-tmp", is_flag=True, help="Keep temporary files")
@click.option("--log-file", default=lambda: f"{os.getcwd()}/rolypoly_pipeline.log", help="Path to log file")

# Assembly options
@click.option("-A", "--assembler", default="spades,megahit,penguin", help="Assembler choice. For multiple, give a comma-separated list")
@click.option("-d", "--post-cluster", is_flag=True, help="Perform post-assembly clustering")

# Filter contigs options
@click.option("-Fm1", "--filter1_nuc", default="alnlen >= 120 & pident>=75", help="First set of rules for nucleic filtering by aligned stats")
@click.option("-Fm2", "--filter2_nuc", default="qcov >= 0.95 & pident>=95", help="Second set of rules for nucleic match filtering")
@click.option("-Fd1", "--filter1_aa", default="length >= 80 & pident>=75", help="First set of rules for amino (protein) match filtering")
@click.option("-Fd2", "--filter2_aa", default="qcovhsp >= 95 & pident>=80", help="Second set of rules for protein match filtering")
@click.option("--dont-mask", is_flag=True, help="If set, host fasta won't be masked for potential RNA virus-like seqs")
@click.option("--mmseqs-args", help="Additional arguments to pass to MMseqs2 search command")
@click.option("--diamond-args",default="--id 50 --min-orf 50", help="Additional arguments to pass to Diamond search command")

# Marker gene search options
@click.option("--db", default="all", help="Database to use for marker gene search")

def run_pipeline(input, output_dir, threads, memory,  host, keep_tmp, log_file,
                 assembler, post_cluster, filter1_nuc, filter2_nuc, filter1_aa, filter2_aa, dont_mask, mmseqs_args, diamond_args, db):
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

    Example:
             run_pipeline(
                 input="reads.fastq",
                 output_dir="results",
                 threads=8,
                 memory="16g",
                 host="host.fasta"
             )
    """
    from rolypoly.commands.reads.filter_reads import filter_reads
    from rolypoly.commands.assembly.assembly import assembly
    from rolypoly.commands.assembly.filter_contigs import filter_contigs
    from rolypoly.commands.identify_virus.marker_search import marker_search as marker_search
    from rolypoly.utils.loggit import setup_logging #, check_file_exists, check_file_size
    from rolypoly.commands.virotype.predict_characteristics import predict_characteristics
    from rolypoly.commands.annotation.annotate import annotate  
    known_dna=host
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "rolypoly_pipeline.log"
    logger = setup_logging(log_file)

    logger.info("Starting RolyPoly pipeline    ")
    logger.info(f"Input: {input}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Threads: {threads}")
    logger.info(f"Memory: {memory}")
    logger.info(f"Known DNA: {known_dna}")
    logger.info(f"Host: {host}")
    logger.info(f"Keep temporary files: {keep_tmp}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Assembler: {assembler}")
    logger.info(f"Post-cluster: {post_cluster}")
    logger.info(f"Filter1 (nucleotide): {filter1_nuc}")
    logger.info(f"Filter2 (nucleotide): {filter2_nuc}")
    logger.info(f"Filter1 (amino acid): {filter1_aa}")
    logger.info(f"Filter2 (amino acid): {filter2_aa}")
    logger.info(f"Don't mask: {dont_mask}")
    logger.info(f"MMseqs2 args: {mmseqs_args}")
    logger.info(f"Diamond args: {diamond_args}")
    logger.info(f"Marker protein search database: {db}")

    # Step 1: Filter Reads
    logger.info("Step 1: Filtering reads    ")
    filtered_reads = output_dir / "filtered_reads"
    filtered_reads.mkdir(parents=True, exist_ok=True)
    ctx = click.Context(filter_reads)
    ctx.invoke(filter_reads,
        input=input,
        output_dir=str(filtered_reads),
        threads=threads,
        memory=memory,
        known_dna=known_dna,
        keep_tmp=keep_tmp,
        log_file=logger,
        speed=15
    )

    # Step 2: Assembly
    logger.info("Step 2: Performing assembly    ")
    assembly_output = output_dir / "assembly"
    assembly_output.mkdir(parents=True, exist_ok=True)
    ctx = click.Context(assembly)
    ctx.invoke(assembly,
        threads=threads,
        memory=memory,
        output_dir=str(assembly_output),
        keep_tmp=keep_tmp,
        log_file=str(log_file),
        input=str(filtered_reads),
        assembler=assembler
    )
    final_assembly = assembly_output / "final_assembly.fasta"

    # Step 3: Filter Assembly
    logger.info("Step 3: Filtering assembly    ")
    filtered_assembly = output_dir / "assemblies" / "filtered_assembly.fasta"
    ctx = click.Context(filter_contigs)
    ctx.invoke(filter_contigs,
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
        diamond_args=diamond_args
    )

    # Step 4: marker protein Search
    logger.info("Step 4: Searching for marker protein sequences    ")
    marker_output = output_dir / "marker_search_results"
    ctx = click.Context(marker_search)
    ctx.invoke(marker_search,
        input=str(filtered_assembly),
        output=str(marker_output),
        threads=threads,
        memory=memory,
        db=db,
        keep_tmp=keep_tmp,
        log_file=str(log_file)
    )

    # Step 5: Annotation (TODO: finish implementing)
    logger.info("Step 5: Annotation (not yet implemented)    ")
    annotation_output = output_dir / "annotation_results"
    ctx = click.Context(annotate)
    ctx.invoke(annotate,
        input=str(marker_output),
        output=str(annotation_output),
        threads=threads,
        memory=memory,
        keep_tmp=keep_tmp,
        log_file=str(log_file)
    )

    # Step 6: Predict Virus Characteristics
    logger.info("Step 6: Predicting virus characteristics    ")
    characteristics_output = output_dir / "virus_characteristics.tsv"
    ctx = click.Context(predict_characteristics)
    ctx.invoke(predict_characteristics,
        input=str(output_dir),
        output=str(characteristics_output),
        database=os.path.join(os.environ['datadir'], "virus_literature_database.tsv"),
        threads=threads,
        log_file=str(log_file)
    )

    logger.info("RolyPoly pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()