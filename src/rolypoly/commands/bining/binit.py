from __future__ import annotations

from pathlib import Path

import polars as pl
import rich_click as click

from rolypoly.utils.bio.polars_fastx import frame_to_fastx, load_sequences
from rolypoly.utils.logging.loggit import log_start_info, setup_logging


def run_subcommand(command: click.Command, args: list[str], label: str, logger) -> None:
	"""Run a click subcommand in-process with explicit args."""
	logger.info("Running %s: %s", label, " ".join(args))
	try:
		command.main(args=args, standalone_mode=False)
	except SystemExit as exc:
		if exc.code not in (0, None):
			raise click.ClickException(f"{label} failed with exit code {exc.code}")


def prepare_combined_inputs(
	rdrp_fasta: Path,
	cp_fasta: Path,
	combined_fasta: Path,
	metadata_path: Path,
) -> pl.DataFrame:
	"""Load RdRp and CP FASTA files and write a combined FASTA plus metadata."""
	rdrp_df = load_sequences(rdrp_fasta).with_columns(
		pl.lit("RdRp").alias("marker_type")
	)
	cp_df = load_sequences(cp_fasta).with_columns(
		pl.lit("CP").alias("marker_type")
	)

	if rdrp_df.is_empty() and cp_df.is_empty():
		raise click.ClickException("Both input FASTA files are empty")

	combined = pl.concat([rdrp_df, cp_df], how="vertical_relaxed").with_columns(
		pl.col("contig_id")
		.str.extract(r"^(.*?)(_.*)?$", group_index=1)
		.fill_null(pl.col("contig_id"))
		.alias("sample_id")
	)

	frame_to_fastx(
		combined.select(
			pl.col("contig_id").alias("header"),
			pl.col("sequence"),
		),
		combined_fasta,
	)

	combined.select(
		"contig_id", "marker_type", "sample_id", "seq_length"
	).write_csv(metadata_path, separator="\t")
	return combined


def build_presence_and_cluster_summary(
	cluster_assignments_path: Path,
	combined_df: pl.DataFrame,
	correlate_input_path: Path,
	cluster_summary_path: Path,
) -> pl.DataFrame:
	"""Build cluster summary and cluster x sample presence matrix for correlate."""
	cluster_assignments = pl.read_csv(cluster_assignments_path, separator="\t")
	contig_meta = combined_df.select(
		pl.col("contig_id"),
		pl.col("marker_type"),
		pl.col("sample_id"),
		pl.col("seq_length"),
	)

	cluster_with_meta = cluster_assignments.join(
		contig_meta,
		left_on="seq_id",
		right_on="contig_id",
		how="left",
	)

	cluster_summary = (
		cluster_with_meta.group_by("cluster_id")
		.agg(
			pl.len().alias("cluster_size"),
			(pl.col("marker_type") == "RdRp").sum().alias("n_rdrp_members"),
			(pl.col("marker_type") == "CP").sum().alias("n_cp_members"),
			pl.col("sample_id").n_unique().alias("n_samples"),
			pl.col("seq_id")
			.sort_by("seq_length", descending=True)
			.first()
			.alias("representative_contig"),
			((pl.col("marker_type") == "RdRp").sum() > 0).alias("has_rdrp"),
			((pl.col("marker_type") == "CP").sum() > 0).alias("has_cp"),
		)
		.sort("cluster_size", descending=True)
	)

	cluster_sample_long = (
		cluster_with_meta.select("cluster_id", "sample_id")
		.drop_nulls("sample_id")
		.unique()
		.with_columns(pl.lit(1).alias("present"))
	)

	if cluster_sample_long.is_empty():
		presence_wide = cluster_summary.select("cluster_id")
	else:
		all_samples = sorted(cluster_sample_long["sample_id"].unique().to_list())
		presence_wide = (
			cluster_sample_long.pivot(on="sample_id", index="cluster_id", values="present")
			.fill_null(0)
			.select(["cluster_id", *all_samples])
		)

	presence_wide.write_csv(correlate_input_path, separator="\t")
	cluster_summary.write_csv(cluster_summary_path, separator="\t")
	return cluster_summary


def build_candidate_pairs(
	outdir: Path,
	cluster_summary: pl.DataFrame,
	write_single_rdrp_strict: bool,
) -> pl.DataFrame:
	"""Join correlate, extend, and termini outputs into putative segment pairs."""
	correlation_pairs = pl.read_csv(
		outdir / "correlate/partiti.correlation_pairs.tsv", separator="\t"
	)
	cooccurrence_pairs = pl.read_csv(
		outdir / "correlate/partiti.cooccurrence_pairs.tsv", separator="\t"
	)
	extend_clusters = pl.read_csv(
		outdir / "extend/partiti_extend.clusters.tsv", separator="\t"
	)
	termini_assignments = pl.read_csv(
		outdir / "termini/partiti_termini.tsv", separator="\t"
	)

	cluster_reps = (
		extend_clusters.filter(pl.col("is_representative"))
		.select(
			pl.col("cluster_id"),
			pl.col("contig_id").alias("representative_contig"),
		)
	)

	rep_termini = cluster_reps.join(
		termini_assignments.select(
			"contig_id",
			"termini_group_1",
			"termini_group_1_motif",
			"termini_group_2",
			"termini_group_2_motif",
		),
		left_on="representative_contig",
		right_on="contig_id",
		how="left",
	)

	base_pairs = cooccurrence_pairs.rename(
		{
			"contig_1": "cluster_a_id",
			"contig_2": "cluster_b_id",
			"shared_samples": "co_present_sample_count",
		}
	)

	if "correlation" not in base_pairs.columns:
		base_pairs = base_pairs.join(
			correlation_pairs.rename(
				{
					"contig_1": "cluster_a_id",
					"contig_2": "cluster_b_id",
				}
			),
			on=["cluster_a_id", "cluster_b_id"],
			how="left",
		)

	base_pairs = base_pairs.join(
		cluster_summary.select(
			pl.col("cluster_id").alias("cluster_a_id"),
			pl.col("n_rdrp_members").alias("cluster_a_rdrp_count"),
			pl.col("n_cp_members").alias("cluster_a_cp_count"),
			pl.col("has_rdrp").alias("cluster_a_has_rdrp"),
			pl.col("has_cp").alias("cluster_a_has_cp"),
		),
		on="cluster_a_id",
		how="left",
	)

	base_pairs = base_pairs.join(
		cluster_summary.select(
			pl.col("cluster_id").alias("cluster_b_id"),
			pl.col("n_rdrp_members").alias("cluster_b_rdrp_count"),
			pl.col("n_cp_members").alias("cluster_b_cp_count"),
			pl.col("has_rdrp").alias("cluster_b_has_rdrp"),
			pl.col("has_cp").alias("cluster_b_has_cp"),
		),
		on="cluster_b_id",
		how="left",
	)

	base_pairs = base_pairs.with_columns(
		(
			pl.col("cluster_a_has_rdrp")
			& ~pl.col("cluster_a_has_cp")
			& pl.col("cluster_b_has_rdrp")
			& ~pl.col("cluster_b_has_cp")
		).alias("is_both_rdrp_only"),
		(
			pl.col("cluster_a_has_cp")
			& ~pl.col("cluster_a_has_rdrp")
			& pl.col("cluster_b_has_cp")
			& ~pl.col("cluster_b_has_rdrp")
		).alias("is_both_cp_only"),
		(
			(
				pl.col("cluster_a_has_rdrp")
				& ~pl.col("cluster_a_has_cp")
				& pl.col("cluster_b_has_cp")
				& ~pl.col("cluster_b_has_rdrp")
			)
			| (
				pl.col("cluster_a_has_cp")
				& ~pl.col("cluster_a_has_rdrp")
				& pl.col("cluster_b_has_rdrp")
				& ~pl.col("cluster_b_has_cp")
			)
		).alias("passes_function_complementarity_check"),
	)

	base_pairs = (
		base_pairs.join(
			cluster_reps.rename(
				{
					"cluster_id": "cluster_a_id",
					"representative_contig": "cluster_a_representative_contig",
				}
			),
			on="cluster_a_id",
			how="left",
		).join(
			cluster_reps.rename(
				{
					"cluster_id": "cluster_b_id",
					"representative_contig": "cluster_b_representative_contig",
				}
			),
			on="cluster_b_id",
			how="left",
		)
	)

	base_pairs = (
		base_pairs.join(
			rep_termini.select(
				pl.col("cluster_id").alias("cluster_a_id"),
				pl.col("termini_group_1").alias("cluster_a_termini_group_1"),
				pl.col("termini_group_2").alias("cluster_a_termini_group_2"),
				pl.col("termini_group_1_motif").alias("cluster_a_termini_seq_1"),
				pl.col("termini_group_2_motif").alias("cluster_a_termini_seq_2"),
			),
			on="cluster_a_id",
			how="left",
		).join(
			rep_termini.select(
				pl.col("cluster_id").alias("cluster_b_id"),
				pl.col("termini_group_1").alias("cluster_b_termini_group_1"),
				pl.col("termini_group_2").alias("cluster_b_termini_group_2"),
				pl.col("termini_group_1_motif").alias("cluster_b_termini_seq_1"),
				pl.col("termini_group_2_motif").alias("cluster_b_termini_seq_2"),
			),
			on="cluster_b_id",
			how="left",
		)
	)

	putative_candidates = base_pairs.select(
		"cluster_a_id",
		"cluster_b_id",
		"cluster_a_rdrp_count",
		"cluster_a_cp_count",
		"cluster_a_has_rdrp",
		"cluster_a_has_cp",
		"cluster_b_rdrp_count",
		"cluster_b_cp_count",
		"cluster_b_has_rdrp",
		"cluster_b_has_cp",
		"correlation",
		"co_present_sample_count",
		"is_both_rdrp_only",
		"is_both_cp_only",
		"passes_function_complementarity_check",
		"cluster_a_representative_contig",
		"cluster_b_representative_contig",
		"cluster_a_termini_group_1",
		"cluster_a_termini_group_2",
		"cluster_b_termini_group_1",
		"cluster_b_termini_group_2",
		"cluster_a_termini_seq_1",
		"cluster_a_termini_seq_2",
		"cluster_b_termini_seq_1",
		"cluster_b_termini_seq_2",
	)

	putative_candidates.write_csv(
		outdir / "candidate_pairs_rdrp_cp.tsv", separator="\t"
	)

	if write_single_rdrp_strict:
		strict = putative_candidates.filter(
			pl.col("passes_function_complementarity_check")
		)
		strict.write_csv(
			outdir / "candidate_pairs_rdrp_cp.strict.tsv", separator="\t"
		)
	return putative_candidates


@click.command(short_help="Run integrated segment binning workflow")
@click.option(
	"--rdrp-fasta",
	required=True,
	type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
	help="Input FASTA containing RdRp candidate contigs",
)
@click.option(
	"--cp-fasta",
	required=True,
	type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
	help="Input FASTA containing CP candidate contigs",
)
@click.option(
	"-o",
	"--outdir",
	required=True,
	type=click.Path(file_okay=False, writable=True, path_type=Path),
	help="Output directory for all intermediate and final workflow artifacts",
)
@click.option(
	"-t",
	"--threads",
	default=8,
	show_default=True,
	help="Threads used by cluster/extend/termini stages",
)
@click.option(
	"--ani-min-identity",
	type=click.FloatRange(0.0, 1.0),
	default=0.90,
	show_default=True,
	help="ANI identity threshold (0-1) used for cluster and extend",
)
@click.option(
	"--ani-min-af",
	type=click.FloatRange(0.0, 1.0),
	default=0.80,
	show_default=True,
	help="ANI aligned fraction threshold (0-1) used for cluster and extend",
)
@click.option(
	"--min-correlation",
	type=float,
	default=0.01,
	show_default=True,
	help="Minimum correlation for the correlate stage",
)
@click.option(
	"--min-prevalence",
	type=click.FloatRange(0.0, 1.0),
	default=0.05,
	show_default=True,
	help="Minimum sample prevalence used in the correlate stage",
)
@click.option(
	"--min-shared-samples",
	type=click.IntRange(1),
	default=1,
	show_default=True,
	help="Minimum shared sample count used in the correlate stage",
)
@click.option(
	"--termini-length",
	default="4-1145",
	show_default=True,
	help="Termini length or range passed to rolypoly termini",
)
@click.option(
	"--termini-distance",
	default=0,
	show_default=True,
	type=click.IntRange(0),
	help="Maximum Hamming mismatch distance for termini seed grouping",
)
@click.option(
	"--reuse-extend/--no-reuse-extend",
	default=False,
	show_default=True,
	help="Reuse existing extend outputs if present instead of rerunning extend",
)
@click.option(
	"--write-single-rdrp-strict/--no-write-single-rdrp-strict",
	default=False,
	show_default=True,
	help="Write strict candidate pairs that pass the complementarity check",
)
@click.option(
	"--log-file",
	type=click.Path(dir_okay=False, writable=True, path_type=Path),
	default=None,
	help="Optional workflow-level log file path",
)
@click.option(
	"-ll", "--log-level", default="INFO", show_default=True, hidden=True
)
def binit(
	rdrp_fasta: Path,
	cp_fasta: Path,
	outdir: Path,
	threads: int,
	ani_min_identity: float,
	ani_min_af: float,
	min_correlation: float,
	min_prevalence: float,
	min_shared_samples: int,
	termini_length: str,
	termini_distance: int,
	reuse_extend: bool,
	write_single_rdrp_strict: bool,
	log_file: Path | None,
	log_level: str,
) -> None:
	"""Run a first integrated segment-binning workflow over RdRp and CP contigs."""
	logger = setup_logging(log_file, log_level)
	log_start_info(logger, locals())

	# Import stage commands lazily so `rolypoly binit --help` can load even when
	# optional dependencies for other subcommands are unavailable.
	from rolypoly.commands.assembly.extend import extend as extend_command
	from rolypoly.commands.bining.cluster import cluster as cluster_command
	from rolypoly.commands.bining.correlate import correlate as correlate_command
	from rolypoly.commands.bining.termini import termini as termini_command

	outdir.mkdir(parents=True, exist_ok=True)
	(outdir / "correlate").mkdir(parents=True, exist_ok=True)
	(outdir / "extend").mkdir(parents=True, exist_ok=True)
	(outdir / "termini").mkdir(parents=True, exist_ok=True)
	(outdir / "logs").mkdir(parents=True, exist_ok=True)

	combined_fasta = outdir / "combined_rdrp_cp.fasta"
	metadata_path = outdir / "contig_metadata.tsv"
	cluster_assignments_path = outdir / "cluster_assignments.tsv"
	cluster_summary_new_path = outdir / "cluster_summary_new.tsv"
	cluster_summary_path = outdir / "cluster_summary.tsv"
	cluster_representatives_path = outdir / "cluster_representatives.fasta"
	correlate_input_path = outdir / "cluster_presence_for_correlate.tsv"

	extend_fasta_path = outdir / "extend/partiti_extend.fasta"
	extend_clusters_path = outdir / "extend/partiti_extend.clusters.tsv"
	termini_assignments_path = outdir / "termini/partiti_termini.tsv"
	termini_groups_path = outdir / "termini/partiti_termini.groups.tsv"
	termini_motifs_path = outdir / "termini/partiti_termini.motifs.fasta"

	combined_df = prepare_combined_inputs(
		rdrp_fasta=rdrp_fasta,
		cp_fasta=cp_fasta,
		combined_fasta=combined_fasta,
		metadata_path=metadata_path,
	)

	run_subcommand(
		cluster_command,
		args=[
			"-i",
			str(combined_fasta),
			"--ani-backend",
			"blastn",
			"--clustering-method",
			"centroid",
			"--min-identity",
			str(ani_min_identity * 100.0),
			"--min-target-coverage",
			str(ani_min_af * 100.0),
			"-o",
			str(cluster_assignments_path),
			"--summary-output",
			str(cluster_summary_new_path),
			"--representatives-fasta",
			str(cluster_representatives_path),
			"-t",
			str(threads),
			"--log-file",
			str(outdir / "logs/cluster.log"),
			"-ll",
			log_level,
		],
		label="cluster",
		logger=logger,
	)

	cluster_summary = build_presence_and_cluster_summary(
		cluster_assignments_path=cluster_assignments_path,
		combined_df=combined_df,
		correlate_input_path=correlate_input_path,
		cluster_summary_path=cluster_summary_path,
	)

	run_subcommand(
		correlate_command,
		args=[
			"-i",
			str(correlate_input_path),
			"-o",
			str(outdir / "correlate/partiti"),
			"-m",
			"both",
			"--method",
			"spearman",
			"--min-prevalence",
			str(min_prevalence),
			"--min-correlation",
			str(min_correlation),
			"--min-shared-samples",
			str(min_shared_samples),
			"--log-file",
			str(outdir / "logs/correlate.log"),
			"-ll",
			log_level,
		],
		label="correlate",
		logger=logger,
	)

	if reuse_extend and extend_fasta_path.exists() and extend_clusters_path.exists():
		logger.info("Reusing extend outputs at %s and %s", extend_fasta_path, extend_clusters_path)
	else:
		run_subcommand(
			extend_command,
			args=[
				"-i",
				str(combined_fasta),
				"-o",
				str(extend_fasta_path),
				"--clusters-output",
				str(extend_clusters_path),
				"--ani-min-identity",
				str(ani_min_identity),
				"--ani-min-af",
				str(ani_min_af),
				"--pileup-min-overlap",
				"50",
				"--pileup-min-identity",
				"0.98",
				"-t",
				str(threads),
				"--log-file",
				str(outdir / "logs/extend.log"),
				"-ll",
				log_level,
			],
			label="extend",
			logger=logger,
		)

	run_subcommand(
		termini_command,
		args=[
			"-i",
			str(extend_fasta_path),
			"-n",
			str(termini_length),
			"-d",
			str(termini_distance),
			"--max-clipped",
			"4",
			"--strand",
			"both",
			"--ani-prefilter",
			"--ani-min-identity",
			"0.95",
			"--ani-min-af",
			"0.80",
			"-o",
			str(termini_assignments_path),
			"--groups-output",
			str(termini_groups_path),
			"--motifs-fasta",
			str(termini_motifs_path),
			"-t",
			str(threads),
			"--log-file",
			str(outdir / "logs/termini.log"),
			"-ll",
			log_level,
		],
		label="termini",
		logger=logger,
	)

	putative_candidates = build_candidate_pairs(
		outdir=outdir,
		cluster_summary=cluster_summary,
		write_single_rdrp_strict=write_single_rdrp_strict,
	)

	logger.info("Workflow complete. Candidate pairs: %s", putative_candidates.height)


if __name__ == "__main__":
	binit()
