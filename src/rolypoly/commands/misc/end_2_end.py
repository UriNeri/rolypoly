import os
import shutil
from datetime import datetime
from pathlib import Path

import rich_click as click

from rolypoly.commands.assembly.assemble import ASSEMBLY_PRESETS
from rolypoly.commands.identify_virus.search_viruses import nucleic_search
from rolypoly.commands.reads.filter_reads import FILTER_READS_PRESETS
from rolypoly.utils.cli_options import shared_command_context

### TODOs:
### simplify the presets / align preset name and description with those in the read-filtering and assembly commands

global tools
tools = []

FILTER_PRESET_NAMES = sorted(FILTER_READS_PRESETS.keys())
ASSEMBLY_PRESET_NAMES = sorted(ASSEMBLY_PRESETS.keys())

# Maps roll preset name → (filter_reads preset, assemble preset, description).
# The description is used in --help text; it should convey library prep type and key behaviour.
ROLL_PRESET_MAP: dict[str, tuple[str, str, str]] = {
    "rna_virus": (
        "rna_virus_metat",
        "rna_virus",
        "RNA virus metatranscriptome (default): rRNA removal (mincovfraction=0.6), host + identified-DNA filtering, no polyA trim; rnaviralSPAdes+MEGAHIT assembly",
    ),
    "ribodepleted": (
        "total_rna_ribodepleted",
        "rna_virus",
        "Total-RNA ribo-depleted: stricter rRNA removal (mincovfraction=0.65), host + identified-DNA filtering, no polyA trim; rnaviralSPAdes+MEGAHIT assembly",
    ),
    "poly_a": (
        "poly_a_selected",
        "metatranscriptome",
        "Poly-A selected mRNA: polyA tail trimming (trimpolya=18), stricter quality trim (trimq=12); rnaSPAdes+MEGAHIT assembly",
    ),
    "all_virus_metat": (
        "all_virus_metat",
        "rna_virus",
        "All-virus metatranscriptome / RNA virome: relaxed rRNA removal (mincovfraction=0.5), skips identified-DNA filter; rnaviralSPAdes+MEGAHIT assembly",
    ),
    "DNA_virus": (
        "all_virus_metag",
        "metag",
        "DNA virome / metagenomics: skips rRNA and identified-DNA filtering entirely; metaSPAdes only",
    ),
    "complete": (
        "rna_virus_metat",
        "complete",
        "Expansive: rna_virus_metat read filtering + all three assemblers (metaSPAdes+rnaviralSPAdes+MEGAHIT)",
    ),
    "fast": (
        "fast",
        "fast",
        "Quick preview / mini mode: subsamples reads, skips error correction and identified-DNA filter; MEGAHIT only with narrow k-mer range; narrowed marker/nucleic/annotation databases",
    ),
}


@click.command(name="roll")
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
@click.option(
    "-D",
    "--host",
    default=None,
    help="Path to the user-supplied host/contamination fasta /// Fasta file of known DNA entities expected in the sample. If not provided some steps will be skippted.",
)
@click.option(
    "--preset",
    default="rna_virus",
    show_default=True,
    type=click.Choice(sorted(ROLL_PRESET_MAP.keys())),
    help=(
        "Preset that selects both a filter-reads and an assemble preset suited to the library type. "
        + "  ".join(
            f"'{name}': {desc}"
            for name, (_, _, desc) in ROLL_PRESET_MAP.items()
        )
    ),
)
@click.option(
    "--filter-preset",
    default=None,
    type=click.Choice(FILTER_PRESET_NAMES),
    help="Override the read-filtering preset chosen by --preset.",
)
@click.option(
    "--assembly-preset",
    default=None,
    type=click.Choice(ASSEMBLY_PRESET_NAMES),
    help="Override the assembly preset chosen by --preset.",
)
# Mini subsampling
@click.option(
    "--mini",
    is_flag=True,
    help="Enable mini mode for quick testing. This will subsample the input reads and use a faster assembly preset.",
)
@click.option(
    "-sz",
    "--sample-size",
    default=50000,
    type=int,
    help="Total reads (>1) OR proportion (0-1) of total reads to be used by --mini subsampling. NOTE: this is ignored if --mini-subset-type is set to bbnorm",
)
@click.option(
    "-ml",
    "--min-len",
    "--minimum-length",
    default=200,
    type=int,
    help="Contigs shorter than this will not be used during virus identification (i.e. in marker search OR nucleic search)",
)
@click.option(
    "-mst",
    "--mini-subset-type",
    default="random",
    type=click.Choice(["random", "first", "bbnorm"]),
    help="Subset type used if --mini is set. note: first is quicker than random which is quicker than bbnorm, but bbnorm is the only one that might be useful in a non 'quick and dirty' attempt. 'first' assumes your input isn't sorted by anything.",
)
@click.option(
    "--skip-existing",
    is_flag=True,
    help="Skip commands if output files already exist",
)
@click.option(
    "-ow",
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite roll output directory if it already exists",
)
# Assembly options
@click.option(
    "-A",
    "--assembler",
    default="spades,megahit",
    help="Assembler choice (spades,megahit,penguin). For multiple, give a comma-separated list",
)
@click.option(
    "--no-rmdup",
    is_flag=True,
    help="Disable default assembly dereplication before downstream analysis.",
)
@click.option(
    "-m",
    "--mapper",
    multiple=True,
    default=("bbmap",),
    type=click.Choice(["bbmap", "mmseqs", "bwa-mem2", "minimap2", "none"]),
    help="Mapper backend(s) for standalone read mapping. Use multiple -m flags for multiple mappers. select 'none' to skip mapping. ",
)
@click.option(
    "--cluster-backend",
    type=click.Choice(
        ["linclust", "mmseqs", "pyskani", "pyfastani", "blastn", "kmer", "none"]
    ),
    default="linclust",
    show_default=True,
    help="ANI backend for contig clustering. Use 'none' to skip clustering entirely.",
)
@click.option(
    "--cluster-method",
    type=click.Choice(["centroid", "connected-components", "leiden"]),
    default="centroid",
    show_default=True,
    help="Clustering method passed to the cluster command.",
)
@click.option(
    "--cluster-min-identity",
    type=float,
    default=99.0,
    show_default=True,
    help="Minimum identity threshold for clustering.",
)
@click.option(
    "--cluster-min-target-coverage",
    type=float,
    default=99.0,
    show_default=True,
    help="Minimum target coverage threshold for clustering.",
)
@click.option(
    "--cluster-min-query-coverage",
    type=float,
    default=0.0,
    show_default=True,
    help="Minimum query coverage threshold for clustering.",
)
@click.option(
    "--cluster-min-alignment-fraction",
    type=float,
    default=0.0,
    show_default=True,
    help="Minimum min(query,target) coverage threshold for clustering.",
)
@click.option(
    "--cluster-mmseqs-sensitivity",
    type=float,
    default=7.5,
    show_default=True,
    help="MMseqs sensitivity when cluster backend uses mmseqs.",
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
    help="Second set of rules for protein match filtering (out potential host/contamination sequences)",
)
@click.option(
    "--dont-mask",
    is_flag=True,
    help="If set, host fasta won't be masked for potential RNA virus-like seqs",
)
@click.option(
    "--mmseqs-args",
    default="--min-seq-id 0.5 --min-aln-len 80",
    help="Additional arguments to pass to MMseqs2 search command during filtering of potential host/contamination sequences",
)
@click.option(
    "--diamond-args",
    default="--id 50 --min-orf 50",
    help="Additional arguments to pass to Diamond search command during filtering of potential host/contamination sequences",
)
@click.option(
    "--skip-steps",
    default=None,
    hidden=True,
    help="Skip these steps in the workflow: filter_reads,assemble,filter_contigs,cluster,marker_search,nucleic_search,map_reads,annotate,rdrp_motif_search,taxonomy,report. Provide a comma-separated list of step names to skip. Note: skipping both filter_reads and assemble treats the input as contigs and also skips map_reads.",
)
# Marker gene search options
@click.option(
    "--dbm",
    "--db-marker",
    default="rvmt,genomad,Pfam_RTs_RdRp",
    help="Database(s) to use for marker gene search",
)
@click.option(
    "--dbn",
    "--db-nucleic",
    default="all",
    help="Database(s) to use for nucleic acid search",
)
@click.option(
    "--dba",
    "--db-annotation",
    default="all",
    help="Database(s) to use for protein annotation.",
)
@click.option("-txb", "--taxonomy-backend", type=click.Choice(["mmseqs", "diamond"]),
              default="mmseqs", show_default=True)
@click.option("-txd", "--taxonomy-db", default="ncbi_virus", show_default=True,
              help="Built-in mmtax database name or custom backend database path.")
@click.option("-txt", "--taxonomy-taxdump", type=click.Path(exists=True, file_okay=False),
              default=None, help="Taxdump required when --taxonomy-db is a custom path.")
@click.option("-txs", "--taxonomy-sensitivity", default="normal", show_default=True,
              help="Shared mmtax sensitivity preset or level 1-8.")
# Pretty report options
@click.option(
    "--report/--no-report",
    "make_report",
    default=True,
    show_default=True,
    help="Write an interactive HTML roll report (roll_report.html) from the "
    "annotation results (marker/protein hits + RNA track) at the end of the run.",
)
@click.option(
    "--report-best-by",
    default="score",
    show_default=True,
    type=click.Choice(["score", "evalue", "longest", "source"]),
    help="Initial 'best hit per range' criterion shown in the report "
    "(toggleable in the viewer).",
)
def roll(
    input,
    output_dir,
    threads,
    memory,
    host,
    min_len,
    preset=None,
    filter_preset=None,
    assembly_preset=None,
    mini=False,
    sample_size=100000,
    mini_subset_type="random",
    keep_tmp=False,
    temp_dir=None,
    log_file=None,
    assembler="spades,megahit,penguin",
    mapper=("bbmap",),
    no_rmdup=False,
    cluster_backend="linclust",
    cluster_method="centroid",
    cluster_min_identity=99.0,
    cluster_min_target_coverage=99.0,
    cluster_min_query_coverage=0.0,
    cluster_min_alignment_fraction=0.0,
    cluster_mmseqs_sensitivity=7.5,
    filter1_nuc="alnlen >= 120 & pident>=75",
    filter2_nuc="qcov >= 0.95 & pident>=95",
    filter1_aa="length >= 80 & pident>=75",
    filter2_aa="qcovhsp >= 95 & pident>=80",
    dont_mask=False,
    mmseqs_args=None,
    diamond_args="--id 50 --min-orf 50",
    skip_steps=None,
    dbn="all",
    dbm="all",
    dba="all",
    taxonomy_backend="mmseqs",
    taxonomy_db="ncbi_virus",
    taxonomy_taxdump=None,
    taxonomy_sensitivity="normal",
    make_report=True,
    report_best_by="score",
    skip_existing=False,
    overwrite=False,
    log_level="INFO",
):
    """End-to-end pipeline for RNA virus discovery from raw sequencing data.

    This pipeline performs a complete analysis workflow including:
    1. Read filtering and quality control (optionally, subsampling too)
    2. De novo assembly
    3. Contig filtering
    4. Marker gene search (default: RdRps + genomad) and nucleic search (default: known RNA viruses)
    5. Genome annotation (default: NVPC + Pfam for proteins, and Rfam+linerfold for catalytic/structural RNAs)
    6. Optional ICTV taxonomy assignment with mmtax
    7. Virus characteristics prediction - NOT IMPLEMENTED YET

    Returns:
        None: Results are written to the specified output directory
    """
    # Normalise skip_steps early, mirroring filter-reads' ReadFilterConfig handling:
    # it may arrive as None, a list of step names, or a comma-separated string.
    # We keep roll config-free (no BaseConfig) and just work with a plain list.
    if isinstance(skip_steps, str):
        skip_steps = [s.strip() for s in skip_steps.split(",") if s.strip()]
    elif isinstance(skip_steps, (list, tuple, set)):
        skip_steps = [str(s).strip() for s in skip_steps if str(s).strip()]
    else:
        skip_steps = []

    # If the user opts out of BOTH read filtering and assembly, assume the input
    # is already contigs / ready for downstream use: use it as the assembly and
    # also skip read mapping (there are no processed reads to map back).
    skip_filter_reads = "filter_reads" in skip_steps
    skip_assemble = "assemble" in skip_steps
    input_is_contigs = skip_filter_reads and skip_assemble
    if input_is_contigs:
        for implied in ("filter_reads", "assemble", "map_reads"):
            if implied not in skip_steps:
                skip_steps.append(implied)

    import sys
    import shlex

    command_args = list(sys.argv)
    if command_args and Path(command_args[0]).name == "rolypoly":
        command_args[0] = "rolypoly"
    command_line = shlex.join(command_args)

    import polars as pl

    from rolypoly.commands.annotation.annotate import annotate
    from rolypoly.commands.assembly.assemble import assembly
    from rolypoly.commands.assembly.filter_contigs import filter_contigs
    from rolypoly.commands.bining.cluster import cluster as cluster_sequences
    from rolypoly.commands.identify_virus.marker_search import (
        marker_search as marker_search,
    )
    from rolypoly.commands.reads.filter_reads import filter_reads
    from rolypoly.commands.reads.map import map as map_reads
    from rolypoly.commands.reads.shrink_reads import shrink_reads

    # from rolypoly.commands.virotype.predict_characteristics import (
    #     predict_characteristics,
    # )
    from rolypoly.utils.bio.library_detection import handle_input_fastq
    from rolypoly.utils.logging.loggit import (  # , check_file_exists, check_file_size
        log_start_info,
        setup_logging,
    )

    output_dir = Path(output_dir).absolute()

    if overwrite and output_dir.exists():
        print(f"Warning: removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir, ignore_errors=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    if log_file is None:
        log_file = output_dir / "rolypoly_pipeline.log"
    logger = setup_logging(log_file, log_level.upper())
    log_start_info(logger, dict(zip(sys.argv[1::2], sys.argv[2::2])))

    if temp_dir is None:
        temp_dir = (
            output_dir / f"rolypoly__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ).absolute()

    if Path(temp_dir).exists() and not Path(temp_dir).is_dir():
        raise ValueError(
            f"Provided temp_dir path exists and is not a directory: {temp_dir}"
        )
    if Path(temp_dir).exists() and not overwrite:
        logger.warning(
            f"Provided temp_dir {temp_dir} path already exists, and --overwrite wasn't set, this may cause issues"
        )
    else:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)

    temp_base_dir = Path(temp_dir).resolve()
    logger.info("Setting temp directory: %s", temp_base_dir)

    if input_is_contigs:
        logger.warning(
            "Skipping filter-reads and assemble opts to assume input is contigs "
            "or ready to use downstream (read mapping will also be skipped)."
        )

    known_dna = host
    # handle_input_fastq expects sequencing reads; skip it when the input is
    # already contigs (both filter-reads and assemble were skipped).
    input_info = (
        handle_input_fastq(input, logger=logger) if not input_is_contigs else {}
    )

    # Resolve filter/assembly presets: explicit --filter-preset / --assembly-preset always win.
    mapped_f, mapped_a, _ = ROLL_PRESET_MAP[preset]
    filter_preset = filter_preset or mapped_f
    assembly_preset = assembly_preset or mapped_a
    # 'fast' roll preset is an alias for mini mode (subsamples reads + narrows databases).
    if preset == "fast" and not mini:
        mini = True
        logger.info(
            "Roll preset 'fast' activates mini mode (read subsampling)."
        )
    logger.info(
        "Applied roll preset '%s': filter=%s assembly=%s",
        preset,
        filter_preset,
        assembly_preset,
    )

    input_for_filter = input
    step = 0
    # Suppress per-command citation writes; roll collects and writes them once at the end.
    os.environ["ROLYPOLY_SUPPRESS_CITATIONS_WRITE"] = "1"
    if mini and not input_is_contigs:
        step += 1
        assembly_preset = "fast"
        logger.info("Step %d: Subsampling reads (mini mode)", step)
        logger.info("mini mode: assembly preset forced to 'fast'")
        dbm = "genomad,rvmt"
        dbn = "ncbi_ribovirus"
        dba = "pfam"
        min_len = 500

        mini_input_dir = output_dir / "mini_input"
        mini_sample_size = 10 if mini_subset_type == "bbnorm" else sample_size
        logger.info(
            "Mini mode selected shrink method '%s' with sample_size=%s",
            mini_subset_type,
            mini_sample_size,
        )
        mini_has_output = mini_input_dir.exists() and any(
            mini_input_dir.glob("*.[fq]*")
        )
        if skip_existing and mini_has_output:
            logger.info(
                "Mini input %s already exists, skipping step", mini_input_dir
            )
        else:
            if overwrite and mini_input_dir.exists():
                shutil.rmtree(mini_input_dir, ignore_errors=True)
            mini_input_dir.mkdir(parents=True, exist_ok=True)
            ctx = shared_command_context(shrink_reads)
            ctx.invoke(
                shrink_reads,
                input=input,
                output=str(mini_input_dir),
                subset_type=mini_subset_type,
                sample_size=mini_sample_size,
                log_file=str(log_file),
                log_level=log_level.lower(),
            )
        sampled_info = handle_input_fastq(str(mini_input_dir), logger=logger)
        if sampled_info.get("R1_R2_pairs"):
            first_pair = sampled_info["R1_R2_pairs"][0]
            input_for_filter = f"{first_pair[0]},{first_pair[1]}"
        elif sampled_info.get("interleaved_files") or sampled_info.get(
            "single_end_files"
        ):
            input_for_filter = str(mini_input_dir)
        logger.info("Mini mode input prepared at %s", str(mini_input_dir))

    # Step: Filter Reads
    step += 1
    logger.info("Step %d: Preprocessing reads (`filter-reads`)    ", step)
    filtered_reads = output_dir / "filtered_reads"
    if "filter_reads" in skip_steps:
        logger.info("Step %d: Skipping filter-reads (in --skip-steps)", step)
    elif skip_existing and filtered_reads.exists():
        logger.info(
            "Filtered reads %s already exist, skipping step", filtered_reads
        )
    else:
        filtered_reads.mkdir(parents=True, exist_ok=True)
        ctx = shared_command_context(filter_reads)
        ctx.invoke(
            filter_reads,
            input=input_for_filter,
            output=str(filtered_reads),
            threads=threads,
            memory=memory,
            known_dna=known_dna,
            preset=filter_preset,
            keep_tmp=keep_tmp,
            log_file=str(log_file),
            speed=15,
            temp_dir=str(temp_base_dir / "filter_reads")
            if temp_base_dir
            else None,
        )

    # Step: Assembly
    step += 1
    logger.info("Step %d: Performing assembly (`assemble`)    ", step)
    assembly_output = output_dir / "assembly"
    final_assembly_file = assembly_output / "final_assembly.fasta"
    if input_is_contigs:
        # Both filter-reads and assemble were skipped: treat the input as the
        # assembly. Rename headers to the CID_ convention (as the assemble step
        # would when combining assemblers) so downstream steps see consistent
        # ids; fall back to a plain copy if the rename helpers are unavailable.
        logger.info(
            "Step %d: Using provided input as assembly (contigs mode)", step
        )
        assembly_output.mkdir(parents=True, exist_ok=True)
        try:
            from rolypoly.utils.bio.sequences import (
                process_sequences,
                read_fasta_df,
                rename_sequences,
                write_fasta_file,
            )

            mapping_file = assembly_output / "contigs_id_map.tsv"
            existing_original_ids: dict[str, list[str]] = {}
            if mapping_file.exists():
                existing_mapping = pl.read_csv(mapping_file, separator="\t")
                if {"old_id", "new_id"}.issubset(existing_mapping.columns):
                    for old_id, new_id in existing_mapping.select(
                        "old_id", "new_id"
                    ).iter_rows():
                        original_ids = existing_original_ids.setdefault(
                            str(new_id), []
                        )
                        if str(old_id) not in original_ids:
                            original_ids.append(str(old_id))

            contigs_df = read_fasta_df(str(input))
            renamed_df, id_map = rename_sequences(contigs_df, prefix="CID")
            # Match the assemble step's contigs_id_map schema (old_id, new_id,
            # assembler, length, gc_content, n_count) so downstream steps that
            # reuse the map - length filtering, the run-stats report - find the
            # columns they expect. Contigs came straight from the input, so the
            # assembler is just "input".
            renamed_df = process_sequences(renamed_df)
            write_fasta_file(
                headers=renamed_df["header"].to_list(),
                seqs=renamed_df["seq"].to_list()
                if "seq" in renamed_df.columns
                else renamed_df["sequence"].to_list(),
                output_file=str(final_assembly_file),
            )
            mapping_rows = []
            for index, (input_id, new_id) in enumerate(id_map.items()):
                original_ids = existing_original_ids.get(input_id, [input_id])
                for original_id in original_ids:
                    mapping_rows.append(
                        {
                            "old_id": original_id,
                            "new_id": new_id,
                            "assembler": "input",
                            "length": renamed_df["length"][index],
                            "gc_content": round(renamed_df["gc_content"][index], 2),
                            "n_count": renamed_df["n_count"][index],
                        }
                    )
            pl.DataFrame(mapping_rows).write_csv(mapping_file, separator="\t")
            logger.info(
                "Renamed %d input contigs to CID_ ids for downstream use",
                renamed_df.height,
            )
        except Exception as rename_error:
            logger.warning(
                "Contig renaming unavailable (%s); using input headers as-is.",
                rename_error,
            )
            if final_assembly_file.exists() or final_assembly_file.is_symlink():
                final_assembly_file.unlink()
            try:
                final_assembly_file.symlink_to(Path(input).resolve())
            except OSError:
                shutil.copy(str(input), str(final_assembly_file))
    elif "assemble" in skip_steps:
        logger.info(
            "Step %d: Skipping assemble (in --skip-steps); expecting existing %s",
            step,
            final_assembly_file,
        )
    elif skip_existing and final_assembly_file.exists():
        logger.info(
            "Assembly output %s already exists, skipping step",
            final_assembly_file,
        )
    else:
        ctx = shared_command_context(assembly)
        ctx.invoke(
            assembly,
            threads=threads,
            memory=memory,
            output=str(assembly_output),
            keep_tmp=keep_tmp,
            log_file=str(log_file),
            input_dir=str(filtered_reads),
            assembler=assembler,
            preset=assembly_preset,
            dereplicate=not no_rmdup,
            overwrite=overwrite,
            temp_dir=str(temp_base_dir / "assembly") if temp_base_dir else None,
        )

    if not final_assembly_file.exists():
        raise FileNotFoundError(
            "Expected assembly output was not found: "
            f"{final_assembly_file}. "
            "Ensure the assemble step produced the expected output for the selected preset."
        )
    logger.info(
        "Using assembly output for downstream steps: %s", final_assembly_file
    )
    # filtering 1 (host)
    # Step: Filter Assembly
    step += 1
    logger.info("Step %d: Filtering assembly (`filter-contigs`)    ", step)
    filtered_assembly = assembly_output / "filtered_assembly.fasta"
    if skip_existing and filtered_assembly.exists():
        logger.info(
            "Filtered assembly %s already exists, skipping step",
            filtered_assembly,
        )
        final_assembly_file = filtered_assembly
    else:
        if host is None:
            logger.info(
                "No host fasta provided, skipping assembly filtering step. "
                "Filtered assembly will be the same as final assembly."
            )
            symlink_path = assembly_output / "filtered_assembly.fasta"
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
            symlink_path.symlink_to(final_assembly_file.resolve())
            final_assembly_file = symlink_path
        else:
            ctx = shared_command_context(filter_contigs)
            ctx.invoke(
                filter_contigs,
                input=str(final_assembly_file),
                known_dna=host,
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
                temp_dir=str(temp_base_dir / "filter_contigs")
                if temp_base_dir
                else None,
            )
            final_assembly_file = filtered_assembly

    # Step: Cluster contigs # consider moving this after marker/nucleic search?
    step += 1
    logger.info("Step %d: Cluster contigs (`cluster-sequences`)    ", step)
    clustered_assembly = assembly_output / "clustered_assembly.fasta"
    if skip_existing and clustered_assembly.exists():
        logger.info(
            "Clustered assembly %s already exists, skipping step",
            clustered_assembly,
        )
        final_assembly_file = clustered_assembly
    else:
        if cluster_backend != "none":
            cluster_output = assembly_output / "cluster_memberships.tsv"
            ctx = shared_command_context(cluster_sequences)
            ctx.invoke(
                cluster_sequences,
                input_path=final_assembly_file,
                input_type="fasta",
                preset_name=None,
                ani_backend=cluster_backend,
                clustering_method=cluster_method,
                min_identity=cluster_min_identity,
                min_target_coverage=cluster_min_target_coverage,
                min_query_coverage=cluster_min_query_coverage,
                min_alignment_fraction=cluster_min_alignment_fraction,
                mmseqs_sensitivity=cluster_mmseqs_sensitivity,
                similarity_measure="identity",
                output_path=cluster_output,
                representatives_fasta=clustered_assembly,
                output_format="tsv",
                threads=threads,
                log_file=str(log_file),
                log_level=log_level,
                temp_dir=str(temp_base_dir / "cluster")
                if temp_base_dir
                else None,
            )
            if not clustered_assembly.exists():
                raise FileNotFoundError(
                    "Expected representative FASTA was not found after clustering: "
                    f"{clustered_assembly}"
                )
            final_assembly_file = clustered_assembly
        else:
            if clustered_assembly.exists() or clustered_assembly.is_symlink():
                clustered_assembly.unlink()
            clustered_assembly.symlink_to(final_assembly_file.resolve())
            final_assembly_file = clustered_assembly

    # Filter 2 (min-length) (maybe make this optional or move before clustering?)
    # Step: Minimal contig length > n
    step += 1
    length_filtered = assembly_output / "length_filtered.fasta"
    if skip_existing and length_filtered.exists():
        logger.info(
            "length_filtered assembly %s already exists, skipping step",
            length_filtered,
        )
        final_assembly_file = length_filtered
    else:
        logger.info("Step %d: Length filtering (min_len=%d bp)", step, min_len)
        from rolypoly.utils.bio.sequences import (
            filter_fasta_by_headers,
            read_fasta_df,
        )

        contig_id_map = assembly_output / "contigs_id_map.tsv"
        # Only reuse the map's lengths when it actually carries a length column
        # (the contigs-mode fallback copy, or an older map, may not); otherwise
        # fall through to computing lengths from the FASTA.
        id_map_has_length = contig_id_map.exists() and (
            "length"
            in pl.read_csv(contig_id_map, separator="\t", n_rows=0).columns
        )
        if id_map_has_length:
            # Re-use lengths computed at assembly command time.
            # Intersect with the IDs currently in the assembly (cluster reps,
            # post-dedup and post-host-filter) before applying the length cut-off.
            current_ids = set(
                read_fasta_df(str(final_assembly_file))["header"].to_list()
            )
            keep_ids = (
                pl.read_csv(contig_id_map, separator="\t")
                .filter(
                    pl.col("new_id").is_in(current_ids)
                    & (pl.col("length") >= min_len)
                )["new_id"]
                .to_list()
            )
        else:
            # Fallback: compute lengths directly from the current assembly FASTA.
            from rolypoly.utils.bio.sequences import process_sequences

            keep_ids = (
                process_sequences(read_fasta_df(str(final_assembly_file)))
                .filter(pl.col("length") >= min_len)["header"]
                .to_list()
            )
        filter_fasta_by_headers(
            fasta_file=str(final_assembly_file),
            headers=keep_ids,
            output_file=str(length_filtered),
        )
        logger.info(
            "Length filtering: kept %d contigs with length >= %d bp",
            len(keep_ids),
            min_len,
        )
        final_assembly_file = length_filtered

    # virus identificaiton
    matched_contigs = set()
    # Step: Marker protein Search
    step += 1
    marker_output = output_dir / "marker_search_results"
    marker_hits_path = marker_output / "marker_search_results.tsv"
    if "marker_search" in skip_steps:
        logger.info("Step %d: Skipping marker search (in --skip-steps)", step)
    elif skip_existing and marker_hits_path.is_file():
        logger.info(
            "Marker search results %s already exist, skipping step",
            marker_output,
        )
    else:
        logger.info("Step %d: Searching for marker protein sequences    ", step)
        ctx = shared_command_context(marker_search)
        ctx.invoke(
            marker_search,
            input=str(final_assembly_file),
            output=str(marker_output),
            threads=threads,
            memory=memory,
            database=dbm,
            keep_tmp=keep_tmp,
            log_file=str(log_file),
            temp_dir=str(temp_base_dir / "marker_search")
            if temp_base_dir
            else None,
            # write_matched_input_seqs=True, #no need we do this for both the marker and the nucleic search in the next step
            # matched_input_seqs_output=str(marker_output / "marker_matched_contigs.fasta"),
        )

    if "marker_search" not in skip_steps:
        # Reused results must populate the downstream selection exactly like
        # results produced in this invocation.
        marker_hits = pl.read_csv(marker_hits_path, separator="\t")
        if "marker_role" in marker_hits.columns:
            marker_candidate_hits = marker_hits.filter(
                pl.col("marker_role") == "candidate"
            )
            rt_evidence_count = marker_hits.filter(
                pl.col("marker_role") == "rt_evidence"
            ).height
            if rt_evidence_count:
                logger.info(
                    "Excluded %d RT-evidence marker hits from downstream contig selection",
                    rt_evidence_count,
                )
        elif "profile_class" in marker_hits.columns:
            marker_candidate_hits = marker_hits.filter(
                pl.col("profile_class") != "rt"
            )
        else:
            # Backward compatibility for results made with older data/code,
            # which did not distinguish candidate and RT evidence.
            marker_candidate_hits = marker_hits
            logger.warning(
                "Reused marker results lack marker-role metadata; historical RT hits "
                "cannot be separated without rerunning marker-search"
            )
        marker_matched_contigs = (
            marker_candidate_hits["source_seq_id"].unique().to_list()
        )
        if len(marker_matched_contigs) == 0:
            logger.warning("No marker contigs found =/")
        matched_contigs.update(marker_matched_contigs)

    # Step: Nucleic search
    step += 1
    nucleic_search_dir = output_dir / "nucleic_search_results"
    nucleic_search_dir.mkdir(parents=True, exist_ok=True)
    nucleic_search_output = nucleic_search_dir / "results.tab"
    nucleic_result_files = sorted(nucleic_search_dir.glob("*_vs_*.tab"))
    if "nucleic_search" in skip_steps:
        logger.info("Step %d: Skipping nucleic search (in --skip-steps)", step)
    elif skip_existing and nucleic_result_files:
        logger.info(
            "Nucleic search results already exist (%d files), skipping step",
            len(nucleic_result_files),
        )
    else:
        logger.info("Step %d: Searching for nucleic sequences", step)
        ctx = shared_command_context(nucleic_search)
        ctx.invoke(
            nucleic_search,
            input=str(final_assembly_file),
            output=str(nucleic_search_output),
            threads=threads,
            temp_dir=str(temp_base_dir / "nucleic_search")
            if temp_base_dir
            else None,
            memory=memory,
            db=dbn,
            log_file=str(log_file),
            matched_output="no",
        )

    # get only the contigs that have hits to the known RNA viruses for downstream annotation
    nucleic_result_files = sorted(nucleic_search_dir.glob("*_vs_*.tab"))
    if not nucleic_result_files:
        logger.warning(
            "No nucleic search result files found in %s", nucleic_search_dir
        )
    else:
        contig_hit_table = pl.scan_csv(
            source=str(nucleic_search_dir / "*_vs_*.tab"),
            separator="\t",
            has_header=True,
        )
        nucleic_hits_df = contig_hit_table.collect()  # scan_csv is lazy
        nucleic_matched_contigs = nucleic_hits_df["qheader"].unique().to_list()
        if len(nucleic_matched_contigs) == 0:
            logger.warning("No nucleic contigs found =/")
        else:
            matched_contigs.update(nucleic_matched_contigs)

    matched_contigs_file = output_dir / "all_matched_contigs.fasta"
    has_matched_contigs = len(matched_contigs) > 0
    if has_matched_contigs:
        from rolypoly.utils.bio.sequences import filter_fasta_by_headers

        filter_fasta_by_headers(
            fasta_file=str(final_assembly_file),
            headers=list(matched_contigs),
            output_file=str(matched_contigs_file),
        )
        logger.info("Written matched contigs to %s", matched_contigs_file)
    elif {"marker_search", "nucleic_search"}.issubset(skip_steps):
        logger.warning(
            "Marker and nucleic searches were both skipped; using the final assembly "
            "for annotation input"
        )
        matched_contigs_file = final_assembly_file
        has_matched_contigs = True
    else:
        matched_contigs_file.write_text("", encoding="utf-8")
        logger.warning(
            "No candidate marker or nucleic matched contigs were found; downstream "
            "mapping, annotation, motif search, and taxonomy will be skipped"
        )
        stale_downstream = [
            path
            for path in (
                output_dir / "read_mapping",
                output_dir / "annotation_results",
                output_dir / "rdrp_motif_search",
                output_dir / "taxonomy",
            )
            if path.exists()
        ]
        if stale_downstream:
            logger.warning(
                "Existing downstream outputs were not reused and remain on disk: %s",
                ", ".join(str(path) for path in stale_downstream),
            )

    # Step: Map **original** reads back to the chosen assembly
    step += 1
    mapping_output = output_dir / "read_mapping"
    if not has_matched_contigs:
        logger.info(
            "Step %d: Skipping read mapping (no candidate contigs)", step
        )
    elif "map_reads" in skip_steps:
        # This includes the contigs mode (filter-reads + assemble skipped), where
        # there are no processed reads to map back.
        logger.info(
            "Step %d: Skipping read mapping (in --skip-steps%s)",
            step,
            "; contigs mode" if input_is_contigs else "",
        )
    elif skip_existing and mapping_output.exists():
        logger.info(
            "Read mapping output %s already exists, skipping step",
            mapping_output,
        )
    elif "none" in mapper:
        logger.info(
            "Step %d: Skipping read mapping as 'none' was selected as mapper",
            step,
        )
    else:
        logger.info(
            "Step %d: Mapping original reads to matched contigs    ", step
        )
        ctx = shared_command_context(map_reads)
        ctx.invoke(
            map_reads,
            input=input,
            reference=str(matched_contigs_file),
            output=str(mapping_output),
            mapper=mapper,
            threads=threads,
            memory=memory,
            keep_tmp=keep_tmp,
            log_file=str(log_file),
            overwrite=overwrite,
            log_level=log_level,
            temp_dir=str(temp_base_dir / "read_mapping")
            if temp_base_dir
            else None,
        )

    # Step: Annotation
    step += 1
    logger.info("Step %d: Annotation", step)
    annotation_output = output_dir / "annotation_results"
    if not has_matched_contigs:
        logger.info("Step %d: Skipping annotation (no candidate contigs)", step)
    elif "annotate" in skip_steps:
        logger.info("Step %d: Skipping annotate (in --skip-steps)", step)
    elif skip_existing and annotation_output.exists():
        logger.info(
            "Annotation results %s already exist, skipping step",
            annotation_output,
        )
    else:
        ctx = shared_command_context(annotate)
        ctx.invoke(
            annotate,
            input=str(matched_contigs_file),
            output=str(annotation_output),
            threads=threads,
            memory=memory,
            domain_db=dba,
            keep_tmp=keep_tmp,
            log_file=str(log_file),
            temp_dir=str(temp_base_dir / "annotation")
            if temp_base_dir
            else None,
            search_tool="diamond" if dba == "uniref50" else "hmmsearch",
            # roll writes a single combined report itself (report step below).
            html=False,
        )

    # Step: identify RdRp motifs (TBD call rdrp_motif_search.py)
    step += 1
    logger.info("Step %d: RdRp motifs marking", step)
    rdrp_motif_search_output = output_dir / "rdrp_motif_search"
    if not has_matched_contigs:
        logger.info(
            "Step %d: Skipping RdRp motif search (no candidate contigs)", step
        )
    elif "rdrp_motif_search" in skip_steps:
        logger.info(
            "Step %d: Skipping RdRp motif search (in --skip-steps)", step
        )
    elif skip_existing and rdrp_motif_search_output.exists():
        logger.info(
            "Annotation results %s already exist, skipping step",
            rdrp_motif_search_output,
        )
    else:
        from rolypoly.commands.identify_virus.rdrp_motif_search import (
            rdrp_motif_search,
        )

        ctx = shared_command_context(rdrp_motif_search)
        ctx.invoke(
            rdrp_motif_search,
            input=str(matched_contigs_file),
            output=str(rdrp_motif_search_output),
            threads=threads,
            memory=memory,
            keep_tmp=keep_tmp,
            overwrite=overwrite,
            log_file=str(log_file),
            log_level=log_level,
            temp_dir=str(temp_base_dir / "rdrp_motif_search")
            if temp_base_dir
            else None,
        )

    # Step: ICTV taxonomy assignment
    step += 1
    taxonomy_output_dir = output_dir / "taxonomy"
    taxonomy_output = taxonomy_output_dir / "mmtax.tsv"
    if not has_matched_contigs:
        logger.info("Step %d: Skipping taxonomy (no candidate contigs)", step)
    elif "taxonomy" in skip_steps:
        logger.info("Step %d: Skipping taxonomy (in --skip-steps)", step)
    elif skip_existing and taxonomy_output.exists():
        logger.info("Taxonomy output %s already exists, skipping step", taxonomy_output)
    else:
        logger.info("Step %d: Assigning ICTV taxonomy (`mmtax`)", step)
        from rolypoly.commands.virotype.mmtax import mmtax

        predicted_orfs = annotation_output / "protein_annotation" / "predicted_orfs.faa"
        predicted_orfs_gff = predicted_orfs.with_suffix(".gff")
        taxonomy_output_dir.mkdir(parents=True, exist_ok=True)
        ctx = shared_command_context(mmtax)
        ctx.invoke(
            mmtax,
            input_path=Path(matched_contigs_file),
            query_type="nucl",
            proteins=predicted_orfs if predicted_orfs.exists() else None,
            protein_map=None,
            protein_gff=predicted_orfs_gff if predicted_orfs_gff.exists() else None,
            infer_protein_map=predicted_orfs.exists()
            and not predicted_orfs_gff.exists(),
            output=taxonomy_output,
            database=taxonomy_db,
            taxdump=Path(taxonomy_taxdump) if taxonomy_taxdump else None,
            backend=taxonomy_backend,
            sensitivity=taxonomy_sensitivity,
            threads=threads,
            memory=memory,
            temp_dir=(temp_base_dir / "taxonomy") if temp_base_dir else None,
            keep_tmp=keep_tmp,
            log_file=Path(log_file),
            log_level=log_level,
        )

    # # Step: Predict Virus Characteristics - TBD not yet implemented!
    # logger.info("Step 8: Predicting virus characteristics    ")
    # characteristics_output = output_dir / "virus_characteristics.tsv"
    # if skip_existing and characteristics_output.exists():
    #     logger.info("Virus characteristics already exist, skipping step")
    # else:
    #     ctx = click.Context(predict_characteristics)
    #     ctx.invoke(
    #         predict_characteristics,
    #         input=str(output_dir),
    #         output=str(characteristics_output),
    #         database=os.path.join(
    #             os.environ["datadir"], "virus_literature_database.tsv"
    #         ),
    #         threads=threads,
    #         log_file=str(log_file),
    #     )

    # Step: Report (interactive genome maps)
    step += 1
    report_output = output_dir / "roll_report.html"
    if not has_matched_contigs:
        logger.info("Step %d: Skipping report (no candidate contigs)", step)
    elif "report" in skip_steps or not make_report:
        logger.info("Step %d: Skipping report (disabled)", step)
    elif skip_existing and report_output.exists():
        logger.info(
            "Report %s already exists, skipping step", report_output
        )
    else:
        logger.info("Step %d: Writing interactive genome-map report", step)
        # The shared helper discovers the marker table (hmmsearch or
        # mmseqs2/diamond schema) and the annotate-rna table by header, so exact
        # filenames aren't hard-coded here. Wrapped in try/except so a report
        # failure never breaks the pipeline.
        from rolypoly.utils.viz.genome_maps import write_report_for_dir

        try:
            report_path = write_report_for_dir(
                output_dir,
                report_output,
                title=f"RolyPoly roll — {Path(input).stem}",
                initial_criterion=report_best_by,
                initial_tab="table",
                command_line=command_line,
                log_file=log_file,
            )
            if report_path is not None:
                logger.info("Wrote interactive genome-map report to %s", report_path)
            else:
                logger.warning("No annotation tables found for the report; skipped.")
        except Exception as report_error:
            logger.warning(
                "Report generation failed (%s); continuing.", report_error
            )

    # step: cleanup, just in case:
    if not keep_tmp and temp_base_dir.exists():
        logger.info("Cleaning up temporary files in %s", temp_base_dir)
        shutil.rmtree(temp_base_dir, ignore_errors=True)

    # Defensive sweep: sub-commands create auto-named `rolypoly_tmp_*` dirs (and
    # can leave `rolypoly__*` temp bases from interrupted steps) inside the output
    # tree. Remove any such leftovers unless the user asked to keep temp files.
    if not keep_tmp:
        for leftover in list(output_dir.glob("**/rolypoly_tmp_*")) + list(
            output_dir.glob("rolypoly__*")
        ):
            if leftover.is_dir() and leftover.resolve() != temp_base_dir.resolve():
                logger.debug("Removing leftover temp directory: %s", leftover)
                shutil.rmtree(leftover, ignore_errors=True)

    logger.info("RolyPoly pipeline completed successfully!")
    from rolypoly.utils.logging.citation_reminder import remind_citations

    used_tools = set(tools)
    if logger.level != 10:
        os.environ.pop("ROLYPOLY_SUPPRESS_CITATIONS_WRITE", None)
        with open(f"{log_file}", "a") as f_out:
            f_out.write(remind_citations(used_tools, return_bibtex=True) or "")


if __name__ == "__main__":
    roll()
