"""Cluster sequences by ANI/AAI with multiple backends and algorithms.

Supports computing pairwise identities on the fly from FASTA files
(via pyskani, blastn, or mmseqs2), or importing pre-computed alignment
tables (BLAST outfmt 6, CheckV ANI table, MMseqs2 easy-search output).

Clustering algorithms:
    - centroid : Greedy incremental (CD-HIT / CheckV aniclust style).
                 Sequences sorted by length; each assigned to the first
                 centroid passing thresholds, or becomes a new centroid.
    - connected-components : Union-find connected components.
    - leiden : Leiden community detection (requires igraph + leidenalg).

Standard output columns:
    seq_id, cluster_id, representative_id, is_representative,
    identity_to_representative, coverage_to_representative
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import polars as pl
import rich_click as click

from rolypoly.utils.logging.loggit import log_start_info, setup_logging
from rolypoly.utils.bio.clustering import (
    SIMILARITY_COLUMNS,
    cluster_centroid_greedy,
    cluster_connected_components,
    cluster_leiden,
    compute_ani_blastn,
    compute_ani_mmseqs,
    compute_ani_pyfastani,
    compute_ani_pyskani,
    compute_kmer_overlap_matrix,
    empty_cluster_frame,
    enrich_edges_with_derived_metrics,
    filter_edges,
    flag_repetitive_sequences,
    kmer_prefilter_pairs,
    kmer_to_edge_table,
    parse_ani_table,
    parse_blast6_to_edges,
    parse_mmseqs_table,
    summarise_clusters,
)
from rolypoly.utils.bio.polars_fastx import load_sequences

# Ensure the FASTX plugins are registered
from rolypoly.utils.bio import polars_fastx as _polars_fastx  # noqa: F401


#  Input loaders 


INPUT_TYPE_FASTA = "fasta"
INPUT_TYPE_BLAST6 = "blast6"
INPUT_TYPE_ANI_TABLE = "ani-table"
INPUT_TYPE_MMSEQS = "mmseqs"

ANI_BACKENDS = ["pyskani", "pyfastani", "blastn", "mmseqs", "kmer"]
CLUSTERING_METHODS = ["centroid", "connected-components", "leiden"]
OUTPUT_FORMATS = ["tsv", "csv", "parquet", "jsonl"]
SIMILARITY_MEASURES = sorted(SIMILARITY_COLUMNS.keys())

#  Presets 
# Each preset maps option names (as they appear in the function
# signature) to override values.  Options not listed keep the user's
# explicit value or the CLI default.  Presets are based on the tool
# comparison in the Vclust paper (DOI:10.1038/s41592-025-02701-7).

PRESETS: dict[str, dict[str, object]] = {
    "miuvig-species": {
        # MIUViG species-level vOTU standard (CheckV aniclust defaults).
        # 95 % ANI over 85 % AF, greedy centroid clustering.
        # Uses blastn for alignment-based ANI (gold standard accuracy).
        "ani_backend": "blastn",
        "clustering_method": "centroid",
        "similarity_measure": "identity",
        "min_identity": 95.0,
        "min_target_coverage": 85.0,
        "min_query_coverage": 0.0,
        "min_alignment_fraction": 0.0,
        "min_evalue": 1e-3,
    },
    "checkv": {
        # CheckV anicalc+aniclust: BLASTn all-vs-all, centroid greedy.
        # Identical thresholds to miuvig-species but uses blastn backend.
        "ani_backend": "blastn",
        "clustering_method": "centroid",
        "similarity_measure": "identity",
        "min_identity": 95.0,
        "min_target_coverage": 85.0,
        "min_query_coverage": 0.0,
        "min_alignment_fraction": 0.0,
        "min_alignment_length": 0,
        "min_evalue": 1e-3,
    },
    "pyskani": {
        # skani / pyskani style.  Thresholds on tANI instead
        # of local ANI.  tANI = (ANI1*AF1*LEN1 + ANI2*AF2*LEN2)/(LEN1+LEN2).
        "ani_backend": "pyskani",
        "clustering_method": "centroid",
        "similarity_measure": "tani",
        "min_identity": 95.0,
        "min_target_coverage": 0.0,
        "min_query_coverage": 0.0,
        "min_alignment_fraction": 0.0,
    },
    "pyfastani": {
        # pyfastani (FastANI) style.  Uses tANI with adjusted fragment
        # length and minimum fraction for shorter viral contigs.
        "ani_backend": "pyfastani",
        "clustering_method": "centroid",
        "similarity_measure": "tani",
        "min_identity": 95.0,
        "min_target_coverage": 0.0,
        "min_query_coverage": 0.0,
        "min_alignment_fraction": 0.0,
    },
    "cd-hit": {
        # CD-HIT-EST style: high-identity greedy clustering with a
        # global sequence identity threshold and no explicit AF filter.
        # Uses mmseqs for fast greedy search (closest to CD-HIT's speed).
        "ani_backend": "mmseqs",
        "clustering_method": "centroid",
        "similarity_measure": "identity",
        "min_identity": 95.0,
        "min_target_coverage": 0.0,
        "min_query_coverage": 0.0,
        "min_alignment_fraction": 0.0,
        "mmseqs_sensitivity": 7.5,
    },
    "mmseqs-cluster": {
        # MMseqs2 easy-cluster style: fast MMseqs2 search + centroid,
        # high sensitivity.
        "ani_backend": "mmseqs",
        "clustering_method": "centroid",
        "similarity_measure": "identity",
        "min_identity": 95.0,
        "min_target_coverage": 85.0,
        "mmseqs_sensitivity": 7.5,
    },
    "kmer-fast": {
        # Kmer-db 2 style (inspired...): approximate k-mer overlap only, no alignment.
        # Fast for very large datasets or as a preliminary screen.
        "ani_backend": "kmer",
        "clustering_method": "centroid",
        "similarity_measure": "identity",
        "kmer_k": 15,
        "min_identity": 90.0,
        "min_target_coverage": 0.0,
        "min_query_coverage": 0.0,
        "min_alignment_fraction": 0.0,
    },
    "genus": {
        # Rough genus-level grouping: lower identity, no AF filter,
        # connected-components for transitive closure.
        # Uses blastn for better sensitivity at lower identity thresholds.
        "ani_backend": "blastn",
        "clustering_method": "connected-components",
        "similarity_measure": "identity",
        "min_identity": 70.0,
        "min_target_coverage": 0.0,
        "min_query_coverage": 0.0,
        "min_alignment_fraction": 0.0,
        "min_evalue": 1e-3,
    },
    "leiden-community": {
        # Leiden community detection: flexible resolution-based
        # clustering, useful for exploratory analysis.
        # Uses blastn for accurate edge weights in community detection.
        "ani_backend": "blastn",
        "clustering_method": "leiden",
        "similarity_measure": "identity",
        "min_identity": 90.0,
        "min_target_coverage": 0.0,
        "leiden_resolution": 1.0,
        "min_evalue": 1e-3,
    },
}

PRESET_NAMES = sorted(PRESETS.keys())

# Generated description for the epilog
_PRESET_DESCRIPTIONS = {
    "miuvig-species": (
        "MIUViG species-level vOTU (95% ANI, 85% AF, blastn, centroid)"
    ),
    "checkv": (
        "CheckV anicalc+aniclust (blastn, 95% ANI, 85% AF, centroid)"
    ),
    "pyskani": (
        "skani/pyskani-style (tANI >= 95%, no AF filter, centroid)"
    ),
    "pyfastani": (
        "FastANI/pyfastani-style (tANI >= 95%, adjusted frag len, centroid)"
    ),
    "cd-hit": (
        "CD-HIT-EST-style (mmseqs, 95% ANI, no AF filter, greedy centroid)"
    ),
    "mmseqs-cluster": (
        "MMseqs2 easy-cluster-style (mmseqs backend, 95% ANI, 85% AF)"
    ),
    "kmer-fast": (
        "Kmer-db 2-style (k-mer overlap only, fast approximate, 90% identity)"
    ),
    "genus": (
        "Rough genus-level (blastn, 70% ANI, no AF, connected-components)"
    ),
    "leiden-community": (
        "Leiden community detection (blastn, 90% ANI, resolution=1.0)"
    ),
}

COMMAND_EPILOG = """
\b
Presets (--preset NAME):
  Presets override multiple options at once to match common tool
  configurations.  Explicit CLI flags always take priority over
  preset values.
\b
""" + "\n".join(
    f"  {name:20s} {_PRESET_DESCRIPTIONS[name]}"
    for name in PRESET_NAMES
) + """
\b

Similarity measures (details from Vclust, DOI:10.1038/s41592-025-02701-7):
  identity / ANI    M(A,B) / L(A,B)
                    Nucleotide identity over the aligned region.
  tANI              (ANI1*AF1*LEN1 + ANI2*AF2*LEN2) / (LEN1+LEN2)
                    Total ANI, bidirectional length-weighted average.
  global_ani        ANI * AF = M(A,B) / |A|
                    Identity normalised by full sequence length.
  kmer_overlap      |kmers(A)∩kmers(B)| / min(|kmers(A)|,|kmers(B)|)
                    K-mer overlap coefficient (fast, approximate).

  AF(A->B)  = L(A,B) / |A|   alignment fraction (query coverage)
  AF(B->A)  ≈ L(B,A->B) / |B| (target coverage, approximated from
              target coordinates of the A->B alignment)
"""


def _apply_preset(
    ctx: click.Context,
    preset_name: str | None,
    params: dict[str, object],
) -> dict[str, object]:
    """Apply preset defaults without overriding explicitly-set options.

    An option is considered explicitly set if the user provided it on
    the command line (i.e. the parameter source is COMMANDLINE).  All
    other options are replaced with the preset value if the preset
    defines one.
    """
    if not preset_name:
        return params
    preset = PRESETS.get(preset_name)
    if not preset:
        raise click.BadParameter(
            f"Unknown preset '{preset_name}'. "
            f"Choose from: {', '.join(PRESET_NAMES)}",
            param_hint="--preset",
        )
    # Determine which options the user set explicitly on the CLI
    explicit: set[str] = set()
    if ctx.params:
        for param in ctx.command.params:
            source = ctx.get_parameter_source(param.name)
            if source == click.core.ParameterSource.COMMANDLINE:
                explicit.add(param.name)
    for key, value in preset.items():
        if key not in explicit:
            params[key] = value
    return params


def load_edges_from_input(
    input_path: Path,
    input_type: str,
    ani_backend: str,
    threads: int,
    min_alignment_length: int,
    min_evalue: float,
    mmseqs_sensitivity: float,
    kmer_k: int,
    logger,
) -> pl.DataFrame:
    """Load or compute pairwise edges based on input type and backend.

    Args:
        input_path: Path to input FASTA or pre-computed table.
        input_type: One of 'fasta', 'blast6', 'ani-table', 'mmseqs'.
        ani_backend: Backend for on-the-fly computation ('pyskani',
            'blastn', 'mmseqs', 'kmer').
        threads: Number of threads.
        min_alignment_length: Min alignment length for blast6 parsing.
        min_evalue: Max evalue for blast6 parsing and blastn runs.
        mmseqs_sensitivity: Sensitivity for mmseqs searches.
        kmer_k: K-mer length for the kmer backend.
        logger: Logger instance.

    Returns:
        Polars DataFrame with standard edge columns.
    """
    if input_type == INPUT_TYPE_BLAST6:
        logger.info("Parsing BLAST outfmt 6 edges from %s", input_path)
        return parse_blast6_to_edges(
            input_path, min_alignment_length, min_evalue
        )
    elif input_type == INPUT_TYPE_ANI_TABLE:
        logger.info("Parsing CheckV-style ANI table from %s", input_path)
        return parse_ani_table(input_path)
    elif input_type == INPUT_TYPE_MMSEQS:
        logger.info("Parsing MMseqs2 tabular output from %s", input_path)
        return parse_mmseqs_table(input_path)
    elif input_type == INPUT_TYPE_FASTA:
        if ani_backend == "pyskani":
            logger.info(
                "Computing ANI with pyskani from %s", input_path
            )
            return compute_ani_pyskani(
                input_path,
                min_identity=0.0,
                threads=threads,
                logger=logger,
            )
        elif ani_backend == "pyfastani":
            logger.info(
                "Computing ANI with pyfastani from %s", input_path
            )
            return compute_ani_pyfastani(
                input_path,
                threads=threads,
                logger=logger,
            )
        elif ani_backend == "blastn":
            logger.info(
                "Computing ANI with blastn from %s", input_path
            )
            return compute_ani_blastn(
                input_path,
                threads=threads,
                min_alignment_length=min_alignment_length,
                min_evalue=min_evalue,
                logger=logger,
            )
        elif ani_backend == "mmseqs":
            logger.info(
                "Computing ANI with mmseqs from %s", input_path
            )
            return compute_ani_mmseqs(
                input_path,
                threads=threads,
                sensitivity=mmseqs_sensitivity,
                logger=logger,
            )
        elif ani_backend == "kmer":
            logger.info(
                "Estimating identity with k-mer overlap (k=%d) from %s",
                kmer_k,
                input_path,
            )
            kmer_df = compute_kmer_overlap_matrix(
                input_path,
                k=kmer_k,
                min_overlap=0.0,
                logger=logger,
            )
            return kmer_to_edge_table(kmer_df)
        else:
            raise click.ClickException(
                f"Unknown ANI backend: {ani_backend}"
            )
    else:
        raise click.ClickException(f"Unknown input type: {input_type}")


def load_seq_lengths(
    input_path: Path,
    input_type: str,
    fasta_path: Path | None,
    logger,
) -> dict[str, int]:
    """Load sequence lengths from a FASTA file.

    When the input type is 'fasta' the lengths come directly from the
    input. Otherwise a separate --fasta-lengths file must be supplied
    for length-sorted centroid clustering.
    """
    target_path: Path | None = None
    if input_type == INPUT_TYPE_FASTA:
        target_path = input_path
    elif fasta_path and fasta_path.exists():
        target_path = fasta_path

    if target_path is None:
        return {}

    logger.info("Loading sequence lengths from %s", target_path)
    seq_df = load_sequences(str(target_path))
    if seq_df.is_empty():
        return {}
    lengths: dict[str, int] = {}
    for row in seq_df.select(["contig_id", "seq_length"]).iter_rows():
        lengths[str(row[0])] = int(row[1])
    return lengths


def write_output(
    df: pl.DataFrame,
    output_path: Path,
    output_format: str,
    logger,
    label: str = "table",
) -> None:
    """Write a polars DataFrame to the specified output format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "tsv":
        df.write_csv(output_path, separator="\t")
    elif output_format == "csv":
        df.write_csv(output_path)
    elif output_format == "parquet":
        df.write_parquet(output_path)
    elif output_format == "jsonl":
        df.write_ndjson(output_path)
    else:
        df.write_csv(output_path, separator="\t")
    logger.info("Wrote %s (%s rows) to %s", label, df.height, output_path)


#  CLI command 


@click.command(
    short_help="Cluster sequences by ANI/AAI (centroid, connected-components, leiden)",
    epilog=COMMAND_EPILOG,
)
@click.option(
    "--preset",
    "preset_name",
    type=click.Choice(PRESET_NAMES, case_sensitive=False),
    default=None,
    help=(
        "Apply a named preset that configures multiple options at once. "
        "Explicit CLI flags always override the preset. "
        "See the epilog below for details on each preset."
    ),
)
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help=(
        "Input file: FASTA/FASTQ for on-the-fly ANI computation, "
        "or a pre-computed pairwise table (BLAST outfmt 6, "
        "CheckV ANI table, MMseqs2 easy-search output)"
    ),
)
@click.option(
    "--input-type",
    type=click.Choice(
        [INPUT_TYPE_FASTA, INPUT_TYPE_BLAST6, INPUT_TYPE_ANI_TABLE, INPUT_TYPE_MMSEQS],
        case_sensitive=False,
    ),
    default=INPUT_TYPE_FASTA,
    show_default=True,
    help=(
        "Type of input file. 'fasta' triggers on-the-fly ANI computation "
        "using the --ani-backend. The table formats expect pre-computed "
        "pairwise results."
    ),
)
@click.option(
    "--ani-backend",
    type=click.Choice(ANI_BACKENDS, case_sensitive=False),
    default="pyskani",
    show_default=True,
    help=(
        "Backend for computing pairwise ANI when --input-type is fasta. "
        "'pyskani' is fast and suitable for most use cases. "
        "'blastn' uses NCBI BLAST (requires blastn on PATH). "
        "'mmseqs' uses MMseqs2 easy-search (requires mmseqs on PATH). "
        "'kmer' uses k-mer overlap coefficient (fast, approximate)."
    ),
)
@click.option(
    "--clustering-method",
    type=click.Choice(CLUSTERING_METHODS, case_sensitive=False),
    default="centroid",
    show_default=True,
    help=(
        "Clustering algorithm. "
        "'centroid': greedy length-sorted (CD-HIT/CheckV style). "
        "'connected-components': union-find transitive closure. "
        "'leiden': Leiden community detection (requires igraph+leidenalg)."
    ),
)
@click.option(
    "--min-identity",
    type=click.FloatRange(0.0, 100.0),
    default=95.0,
    show_default=True,
    help="Minimum pairwise identity threshold (0-100 scale)",
)
@click.option(
    "--min-target-coverage",
    type=click.FloatRange(0.0, 100.0),
    default=85.0,
    show_default=True,
    help="Minimum target (shorter sequence) coverage threshold (0-100)",
)
@click.option(
    "--min-query-coverage",
    type=click.FloatRange(0.0, 100.0),
    default=0.0,
    show_default=True,
    help="Minimum query (longer sequence) coverage threshold (0-100)",
)
@click.option(
    "--min-alignment-fraction",
    type=click.FloatRange(0.0, 100.0),
    default=0.0,
    show_default=True,
    help=(
        "Minimum alignment fraction (min(qcov, tcov), 0-100). "
        "When > 0, overrides individual qcov/tcov thresholds."
    ),
)
@click.option(
    "--min-alignment-length",
    type=click.IntRange(0),
    default=0,
    show_default=True,
    help="Minimum individual alignment length (for blast6 parsing)",
)
@click.option(
    "--min-evalue",
    type=float,
    default=1e-3,
    show_default=True,
    help="Maximum evalue for individual alignments (blast6 parsing / blastn)",
)
@click.option(
    "--mmseqs-sensitivity",
    type=click.FloatRange(1.0, 9.0),
    default=7.5,
    show_default=True,
    help="MMseqs2 sensitivity parameter (-s) when --ani-backend is mmseqs",
)
@click.option(
    "--leiden-resolution",
    type=click.FloatRange(0.0),
    default=1.0,
    show_default=True,
    help="Resolution parameter for Leiden clustering (higher = more clusters)",
)
@click.option(
    "--fasta-lengths",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    default=None,
    help=(
        "FASTA file for reading sequence lengths (used by centroid "
        "clustering for length-sorted ordering). Only needed when "
        "--input-type is not fasta."
    ),
)
@click.option(
    "-o",
    "--output",
    "output_path",
    default="cluster_assignments.tsv",
    show_default=True,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Output path for per-sequence cluster assignments",
)
@click.option(
    "--summary-output",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help=(
        "Output path for cluster summary table "
        "(default: <output>.summary.<ext>)"
    ),
)
@click.option(
    "--edges-output",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help=(
        "Output path for the filtered edge table "
        "(default: not written; useful for inspection)"
    ),
)
@click.option(
    "--representatives-fasta",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help=(
        "Output FASTA of cluster representative sequences. "
        "Only available when --input-type is fasta or --fasta-lengths points "
        "to a FASTA file."
    ),
)
@click.option(
    "--output-format",
    type=click.Choice(OUTPUT_FORMATS, case_sensitive=False),
    default="tsv",
    show_default=True,
    help="Tabular output format for assignments and summary tables",
)
@click.option(
    "-t",
    "--threads",
    default=4,
    show_default=True,
    type=click.IntRange(1, 1000),
    help="Number of threads for ANI computation backends",
)
@click.option(
    "--similarity-measure",
    type=click.Choice(SIMILARITY_MEASURES, case_sensitive=False),
    default="identity",
    show_default=True,
    help=(
        "Which similarity column to threshold with --min-identity. "
        "'identity' (=ANI): identity over the aligned region. "
        "'tani': total ANI, bidirectional length-weighted. "
        "'global_ani'/'global_ani_query': identity over full query length. "
        "Derived columns are computed automatically when chosen."
    ),
)
@click.option(
    "--kmer-prefilter/--no-kmer-prefilter",
    default=False,
    show_default=True,
    help=(
        "Run a k-mer overlap prefilter before alignment-based ANI. "
        "Only sequence pairs passing the k-mer threshold are sent to "
        "the alignment backend. Ignored when --ani-backend is 'kmer'."
    ),
)
@click.option(
    "--kmer-k",
    type=click.IntRange(5, 31),
    default=15,
    show_default=True,
    help="K-mer length for the kmer backend or kmer prefilter",
)
@click.option(
    "--kmer-prefilter-threshold",
    type=click.FloatRange(0.0, 1.0),
    default=0.5,
    show_default=True,
    help=(
        "Minimum k-mer overlap coefficient (0-1) for prefilter pairs. "
        "Lower values retain more pairs (higher recall, slower). "
        "Only used when --kmer-prefilter is set."
    ),
)
@click.option(
    "--flag-repeats/--no-flag-repeats",
    default=True,
    show_default=True,
    help=(
        "Run a self-dotplot repeat check on every input sequence. "
        "Sequences whose longest internal repeat track spans more than "
        "--repeat-max-fraction of their length are flagged as potential "
        "assembly artefacts and excluded from representative selection "
        "(they are still clustered normally). Only available when the "
        "input is FASTA."
    ),
)
@click.option(
    "--repeat-k",
    type=click.IntRange(5, 31),
    default=15,
    show_default=True,
    help="K-mer size for the repeat-flag dotplot analysis",
)
@click.option(
    "--repeat-max-fraction",
    type=click.FloatRange(0.0, 1.0),
    default=0.40,
    show_default=True,
    help=(
        "Maximum fraction of sequence length covered by the longest "
        "repeat track before the sequence is flagged. "
        "Lower values flag more aggressively."
    ),
)
@click.option(
    "--log-file",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Optional log file path",
)
@click.option(
    "-ll",
    "--log-level",
    default="INFO",
    show_default=True,
    hidden=True,
)
def cluster(
    input_path,
    preset_name,
    input_type,
    ani_backend,
    clustering_method,
    min_identity,
    min_target_coverage,
    min_query_coverage,
    min_alignment_fraction,
    min_alignment_length,
    min_evalue,
    mmseqs_sensitivity,
    leiden_resolution,
    fasta_lengths,
    output_path,
    summary_output,
    edges_output,
    representatives_fasta,
    output_format,
    threads,
    similarity_measure,
    kmer_prefilter,
    kmer_k,
    kmer_prefilter_threshold,
    flag_repeats,
    repeat_k,
    repeat_max_fraction,
    log_file,
    log_level,
):
    """Cluster sequences by pairwise ANI/AAI identity and coverage.

    Supports on-the-fly ANI computation from FASTA files (pyskani, blastn,
    mmseqs2, k-mer overlap) and import of pre-computed alignment tables
    (BLAST outfmt 6, CheckV ANI table, MMseqs2 output). Multiple
    clustering algorithms are available: centroid/greedy (default,
    CD-HIT/CheckV style), connected-components (union-find), and Leiden
    community detection.

    Similarity measures (per Vclust DOI:10.1038/s41592-025-02701-7):
      - identity (ANI): nucleotide identity over the aligned region
      - tani: total ANI, bidirectional length-weighted average
      - global_ani: identity over the full query length (ANI * AF)
    Use --similarity-measure to select which one --min-identity applies to.

    Default thresholds (95% identity, 85% target coverage) follow the
    MIUViG species-level vOTU standards.

    \b
    Examples:
      # Cluster a FASTA file with default pyskani + centroid
      rolypoly cluster -i contigs.fasta -o clusters.tsv

      # Use pre-computed BLAST edges with connected-components
      rolypoly cluster -i blast.out --input-type blast6 \\
          --clustering-method connected-components -o clusters.tsv

      # Leiden clustering at genus level (70% identity, 0% AF)
      rolypoly cluster -i contigs.fasta --min-identity 70 \\
          --min-target-coverage 0 --clustering-method leiden -o clusters.tsv

      # Fast k-mer-only clustering (no alignment)
      rolypoly cluster -i contigs.fasta --ani-backend kmer -o clusters.tsv

      # pyskani with k-mer prefilter (fewer pairwise comparisons)
      rolypoly cluster -i contigs.fasta --kmer-prefilter -o clusters.tsv

      # Cluster using tANI instead of ANI for thresholding
      rolypoly cluster -i contigs.fasta --similarity-measure tani -o clusters.tsv

      # Use presets to mimic other tools
      rolypoly cluster -i contigs.fasta --preset checkv -o clusters.tsv
      rolypoly cluster -i contigs.fasta --preset fast-ani -o clusters.tsv
      rolypoly cluster -i contigs.fasta --preset kmer-fast -o clusters.tsv

      # Override a single preset option (e.g. lower identity threshold)
      rolypoly cluster -i contigs.fasta --preset checkv --min-identity 90 -o clusters.tsv
    """
    # Apply preset (overrides defaults but not explicit CLI flags)
    ctx = click.get_current_context()
    params = _apply_preset(ctx, preset_name, {
        "input_path": input_path,
        "input_type": input_type,
        "ani_backend": ani_backend,
        "clustering_method": clustering_method,
        "min_identity": min_identity,
        "min_target_coverage": min_target_coverage,
        "min_query_coverage": min_query_coverage,
        "min_alignment_fraction": min_alignment_fraction,
        "min_alignment_length": min_alignment_length,
        "min_evalue": min_evalue,
        "mmseqs_sensitivity": mmseqs_sensitivity,
        "leiden_resolution": leiden_resolution,
        "similarity_measure": similarity_measure,
        "kmer_prefilter": kmer_prefilter,
        "kmer_k": kmer_k,
        "kmer_prefilter_threshold": kmer_prefilter_threshold,
    })
    # Unpack (potentially overridden) values back into local scope
    input_type = params["input_type"]
    ani_backend = params["ani_backend"]
    clustering_method = params["clustering_method"]
    min_identity = params["min_identity"]
    min_target_coverage = params["min_target_coverage"]
    min_query_coverage = params["min_query_coverage"]
    min_alignment_fraction = params["min_alignment_fraction"]
    min_alignment_length = params["min_alignment_length"]
    min_evalue = params["min_evalue"]
    mmseqs_sensitivity = params["mmseqs_sensitivity"]
    leiden_resolution = params["leiden_resolution"]
    similarity_measure = params["similarity_measure"]
    kmer_prefilter = params["kmer_prefilter"]
    kmer_k = params["kmer_k"]
    kmer_prefilter_threshold = params["kmer_prefilter_threshold"]

    t0 = perf_counter()
    logger = setup_logging(log_file, log_level)
    if preset_name:
        logger.info(
            "Using preset '%s': %s",
            preset_name,
            _PRESET_DESCRIPTIONS.get(preset_name, ""),
        )
    log_start_info(logger, locals())

    # Resolve output paths
    output_path = Path(output_path)
    ext_map = {"tsv": ".tsv", "csv": ".csv", "parquet": ".parquet", "jsonl": ".jsonl"}
    ext = ext_map.get(output_format, ".tsv")
    if summary_output is None:
        summary_output = output_path.with_suffix(f".summary{ext}")

    # Step 1: Load or compute pairwise edges
    logger.info("Step 1: Loading/computing pairwise edges")
    edges = load_edges_from_input(
        input_path=input_path,
        input_type=input_type,
        ani_backend=ani_backend,
        threads=threads,
        min_alignment_length=min_alignment_length,
        min_evalue=min_evalue,
        mmseqs_sensitivity=mmseqs_sensitivity,
        kmer_k=kmer_k,
        logger=logger,
    )
    logger.info("Loaded %s raw pairwise edges", edges.height)

    # Step 1b (optional): k-mer prefilter — discard edges between pairs
    # whose k-mer overlap coefficient is below the threshold. Only makes
    # sense for alignment-based backends, not kmer backend.
    if kmer_prefilter and ani_backend != "kmer" and input_type == INPUT_TYPE_FASTA:
        logger.info(
            "Step 1b: Applying k-mer prefilter (k=%d, threshold=%.2f)",
            kmer_k,
            kmer_prefilter_threshold,
        )
        allowed_pairs = kmer_prefilter_pairs(
            input_path,
            k=kmer_k,
            min_overlap=kmer_prefilter_threshold,
            logger=logger,
        )
        before = edges.height
        edges = edges.filter(
            pl.struct(["query_id", "target_id"]).map_elements(
                lambda row: (row["query_id"], row["target_id"]) in allowed_pairs,
                return_dtype=pl.Boolean,
            )
        )
        logger.info(
            "K-mer prefilter: %d -> %d edges (%.1f%% removed)",
            before,
            edges.height,
            100.0 * (before - edges.height) / max(before, 1),
        )

    # Step 1c: Load sequence lengths early (needed for tANI enrichment
    # and centroid clustering)
    seq_lengths = load_seq_lengths(
        input_path, input_type, fasta_lengths, logger
    )

    # Step 2a: Enrich edges with derived metrics (global ANI, tANI)
    needs_derived = similarity_measure not in ("identity", "ani")
    if needs_derived or edges_output:
        logger.info("Step 2a: Computing derived metrics (global ANI, tANI)")
        edges = enrich_edges_with_derived_metrics(
            edges, seq_lengths=seq_lengths if seq_lengths else None
        )

    # Step 2b: Filter edges by identity and coverage thresholds
    logger.info(
        "Step 2b: Filtering edges (%s >= %.1f, qcov >= %.1f, tcov >= %.1f, AF >= %.1f)",
        similarity_measure,
        min_identity,
        min_query_coverage,
        min_target_coverage,
        min_alignment_fraction,
    )
    filtered_edges = filter_edges(
        edges,
        min_identity=min_identity,
        min_query_coverage=min_query_coverage,
        min_target_coverage=min_target_coverage,
        min_alignment_fraction=min_alignment_fraction,
        similarity_column=similarity_measure,
    )
    logger.info(
        "Retained %s edges after filtering (from %s)",
        filtered_edges.height,
        edges.height,
    )

    if edges_output:
        write_output(
            filtered_edges,
            Path(edges_output),
            output_format,
            logger,
            label="filtered edges",
        )

    # Collect all sequence IDs
    all_seq_ids: list[str] = []
    if seq_lengths:
        all_seq_ids = sorted(seq_lengths.keys())
    elif not filtered_edges.is_empty():
        id_set: set[str] = set()
        id_set.update(filtered_edges["query_id"].to_list())
        id_set.update(filtered_edges["target_id"].to_list())
        all_seq_ids = sorted(id_set)

    # Step 2c (optional): Flag repetitive sequences via self-dotplot
    repetitive_ids: set[str] = set()
    if flag_repeats and input_type == INPUT_TYPE_FASTA:
        logger.info(
            "Step 2c: Flagging repetitive sequences "
            "(k=%d, max_repeat_fraction=%.2f)",
            repeat_k,
            repeat_max_fraction,
        )
        seq_df = load_sequences(str(input_path))
        if not seq_df.is_empty():
            seq_map: dict[str, str] = {
                row[0]: row[1]
                for row in seq_df.select("contig_id", "sequence").iter_rows()
            }
            repetitive_ids = flag_repetitive_sequences(
                seq_map,
                k=repeat_k,
                max_repeat_fraction=repeat_max_fraction,
                logger=logger,
            )
    elif flag_repeats and input_type != INPUT_TYPE_FASTA:
        logger.debug(
            "Repeat-flag check skipped (input type '%s' is not FASTA)",
            input_type,
        )

    # Step 3: Cluster
    logger.info(
        "Step 3: Clustering %s sequences with method '%s'",
        len(all_seq_ids),
        clustering_method,
    )
    if clustering_method == "centroid":
        assignments = cluster_centroid_greedy(
            filtered_edges,
            seq_lengths=seq_lengths if seq_lengths else None,
            seq_ids=all_seq_ids if all_seq_ids else None,
            exclude_as_representatives=repetitive_ids or None,
        )
    elif clustering_method == "connected-components":
        assignments = cluster_connected_components(
            filtered_edges,
            seq_ids=all_seq_ids if all_seq_ids else None,
            exclude_as_representatives=repetitive_ids or None,
        )
    elif clustering_method == "leiden":
        assignments = cluster_leiden(
            filtered_edges,
            seq_ids=all_seq_ids if all_seq_ids else None,
            resolution=leiden_resolution,
            exclude_as_representatives=repetitive_ids or None,
        )
    else:
        raise click.ClickException(
            f"Unknown clustering method: {clustering_method}"
        )

    if assignments.is_empty():
        logger.warning("No clusters produced")
        assignments = empty_cluster_frame()

    # Cluster statistics
    n_clusters = assignments["cluster_id"].n_unique()
    n_singletons = (
        assignments.group_by("cluster_id")
        .agg(pl.col("seq_id").count().alias("n"))
        .filter(pl.col("n") == 1)
        .height
    )
    n_multi = n_clusters - n_singletons
    largest_cluster_size = (
        assignments.group_by("cluster_id")
        .agg(pl.col("seq_id").count().alias("n"))
        .select(pl.col("n").max())
        .item()
    ) if not assignments.is_empty() else 0
    logger.info(
        "Clustering produced %s clusters (%s multi-member, %s singletons, "
        "largest: %s members)",
        n_clusters,
        n_multi,
        n_singletons,
        largest_cluster_size,
    )

    # Step 5: Write outputs
    write_output(
        assignments, output_path, output_format, logger, label="cluster assignments"
    )

    summary = summarise_clusters(assignments)
    write_output(
        summary, summary_output, output_format, logger, label="cluster summary"
    )

    # Optional: write representative sequences FASTA
    if representatives_fasta:
        write_representative_fasta(
            assignments=assignments,
            input_path=input_path,
            input_type=input_type,
            fasta_path=fasta_lengths,
            output_fasta=representatives_fasta,
            logger=logger,
        )

    elapsed = perf_counter() - t0
    logger.info("Cluster command completed in %.1f seconds", elapsed)


#  Representative FASTA writer 
def write_representative_fasta(
    assignments: pl.DataFrame,
    input_path: Path,
    input_type: str,
    fasta_path: Path | None,
    output_fasta: Path,
    logger,
) -> None:
    """Write a FASTA file containing only representative sequences.

    Args:
        assignments: Cluster assignment DataFrame.
        input_path: Original input path (used if input_type is fasta).
        input_type: Type of the original input.
        fasta_path: Separate FASTA file for sequence data.
        output_fasta: Output FASTA path.
        logger: Logger instance.
    """
    from rolypoly.utils.bio.polars_fastx import frame_to_fastx

    source_fasta: Path | None = None
    if input_type == INPUT_TYPE_FASTA:
        source_fasta = input_path
    elif fasta_path and fasta_path.exists():
        source_fasta = fasta_path

    if source_fasta is None:
        logger.warning(
            "Cannot write representative FASTA: no FASTA source available. "
            "Provide --fasta-lengths when using pre-computed edge tables."
        )
        return

    seq_df = load_sequences(str(source_fasta))
    if seq_df.is_empty():
        logger.warning("Source FASTA is empty, skipping representative output")
        return

    rep_ids = set(
        assignments.filter(pl.col("is_representative"))["seq_id"].to_list()
    )
    rep_df = seq_df.filter(pl.col("contig_id").is_in(rep_ids))

    if rep_df.is_empty():
        logger.warning("No representative sequences found in FASTA")
        return

    # Rename columns for frame_to_fastx compatibility
    write_df = rep_df.rename({"contig_id": "header"})
    frame_to_fastx(write_df, str(output_fasta))
    logger.info(
        "Wrote %s representative sequences to %s",
        rep_df.height,
        output_fasta,
    )


if __name__ == "__main__":
    cluster()
