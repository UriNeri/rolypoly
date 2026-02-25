from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import polars as pl
import rich_click as click

from rolypoly.utils.logging.loggit import log_start_info, setup_logging
from rolypoly.utils.bio.sequences import read_fasta_df, revcomp

# Ensure the FASTX plugins are registered
from rolypoly.utils.bio import polars_fastx as _polars_fastx  # noqa: F401


def _parse_length_window(value: str) -> Tuple[int, int]:
	cleaned = value.replace(" ", "")
	if "-" in cleaned:
		start_str, end_str = cleaned.split("-", 1)
		start = int(start_str)
		end = int(end_str)
	else:
		start = end = int(cleaned)
	if start <= 0 or end <= 0:
		raise click.BadParameter("Length must be positive")
	if end < start:
		raise click.BadParameter("Length range end must be >= start")
	return start, end




def _load_sequences(fasta_path: Path) -> pl.DataFrame:
	df = read_fasta_df(str(fasta_path))
	if df.is_empty():
		return df
	column_map = {"header": "contig_id"}
	df = df.rename({k: v for k, v in column_map.items() if k in df.columns})
	if "contig_id" not in df.columns:
		df = df.with_row_index("contig_id", offset=1).with_columns(
			pl.col("contig_id").cast(pl.Utf8)
		)
	df = df.with_columns(pl.col("sequence").seq.length().alias("seq_length"))
	return _orientate_all(df)


def _ensure_rc_column(df: pl.DataFrame) -> pl.DataFrame:
	if "sequence_rc" in df.columns:
		return df
	return df.with_columns(
		pl.col("sequence")
		.map_elements(revcomp, return_dtype=pl.String)
		.alias("sequence_rc")
	)


def _orientate_all(df: pl.DataFrame) -> pl.DataFrame:
	"""Placeholder for future orientation normalization logic."""  # noqa: D401
	# TODO: implement restranding/orientate_all logic that favors the strand
	# containing the highest density of forward-encoded coding genes.
	return df


def _create_signature_frame(
	df: pl.DataFrame,
	sequence_col: str,
	min_len: int,
	max_len: int,
	label: str,
	is_plus: bool,
) -> pl.DataFrame:
	filtered = df.filter(pl.col("seq_length") >= min_len)
	if filtered.is_empty():
		return pl.DataFrame()

	def _build_end(end_name: str) -> pl.DataFrame:
		if end_name == "5prime":
			full_expr = pl.col(sequence_col).str.slice(0, max_len)
			min_expr = pl.col(sequence_col).str.slice(0, min_len)
		else:
			full_expr = pl.col(sequence_col).str.slice(-max_len, max_len)
			min_expr = pl.col(sequence_col).str.slice(-min_len, min_len)

		if not is_plus:
			full_expr = full_expr.map_elements(revcomp, return_dtype=pl.String)
			min_expr = min_expr.map_elements(revcomp, return_dtype=pl.String)

		genome_end = end_name if is_plus else ("3prime" if end_name == "5prime" else "5prime")
		start_expr = (
			pl.lit(1)
			if genome_end == "5prime"
			else pl.col("seq_length") - pl.lit(min_len) + pl.lit(1)
		)
		stop_expr = start_expr + pl.lit(min_len - 1)

		return (
			filtered.select(
				pl.col("contig_id"),
				pl.lit(f"{label}_{end_name}").alias("terminus"),
				pl.lit(genome_end).alias("end_label"),
				pl.lit("+" if is_plus else "-").alias("strand_label"),
				min_expr.alias("motif_min"),
				full_expr.alias("motif_full"),
				start_expr.alias("motif_start"),
				stop_expr.alias("motif_stop"),
				pl.lit(min_len).alias("window_min"),
				pl.lit(max_len).alias("window_max"),
			)
			.drop_nulls("motif_min")
			.filter(pl.col("motif_min") != "")
		)

	frames = [_build_end("5prime"), _build_end("3prime")]
	frames = [frame for frame in frames if not frame.is_empty()]
	if not frames:
		return pl.DataFrame()
	return pl.concat(frames, how="vertical_relaxed")


def _build_signature_table(
	df: pl.DataFrame, min_len: int, max_len: int, strand: str
) -> pl.DataFrame:
	frames: list[pl.DataFrame] = []
	if strand in {"plus", "both"}:
		frames.append(
			_create_signature_frame(df, "sequence", min_len, max_len, "+", True)
		)
	if strand in {"minus", "both"}:
		df = _ensure_rc_column(df)
		frames.append(
			_create_signature_frame(df, "sequence_rc", min_len, max_len, "-", False)
		)
	frames = [frame for frame in frames if not frame.is_empty()]
	if not frames:
		return pl.DataFrame(
			{
				"contig_id": pl.Series([], dtype=pl.Utf8),
				"terminus": pl.Series([], dtype=pl.Utf8),
				"end_label": pl.Series([], dtype=pl.Utf8),
				"strand_label": pl.Series([], dtype=pl.Utf8),
				"motif_full": pl.Series([], dtype=pl.Utf8),
				"motif_min": pl.Series([], dtype=pl.Utf8),
				"motif_start": pl.Series([], dtype=pl.Int64),
				"motif_stop": pl.Series([], dtype=pl.Int64),
				"window_min": pl.Series([], dtype=pl.Int64),
				"window_max": pl.Series([], dtype=pl.Int64),
			}
		)
	return pl.concat(frames, how="vertical_relaxed")


def _deduplicate_signatures(signatures: pl.DataFrame) -> pl.DataFrame:
	if signatures.is_empty():
		return signatures
	return (
		signatures.with_columns(
			pl.when(pl.col("strand_label") == "+")
			.then(0)
			.otherwise(1)
			.alias("strand_priority")
		)
		.sort(["contig_id", "end_label", "strand_priority"])
		.unique(subset=["contig_id", "end_label"], keep="first")
		.drop("strand_priority")
	)


def _aggregate_min_range_groups(
	signatures: pl.DataFrame,
	min_len: int,
	max_len: int,
	distance: int,
	metric: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
	if signatures.is_empty():
		return _empty_result_frame(), _empty_assignment_frame()
	if distance == 0:
		grouped = (
			signatures.group_by(["motif_min"], maintain_order=True)
			.agg(
				pl.len().alias("member_count"),
				pl.col("contig_id").alias("members"),
				pl.col("end_label").alias("member_ends"),
			)
			.filter(pl.col("member_count") > 1)
			.with_columns(
				pl.col("motif_min").alias("motif"),
				pl.col("member_ends").list.unique().alias("ends_present"),
				pl.lit(min_len).alias("window_min"),
				pl.lit(max_len).alias("window_max"),
				pl.lit(distance).alias("max_distance"),
				pl.lit(metric).alias("distance_metric"),
			)
			.with_columns(
				pl.when(pl.col("ends_present").list.len() == 1)
				.then(pl.col("ends_present").list.first())
				.otherwise(pl.lit("mixed"))
				.alias("end_label")
			)
			.drop("motif_min", "member_ends")
			.sort(["end_label", "member_count"], descending=[False, True])
		)
		if grouped.height == 0:
			return _empty_result_frame(), _empty_assignment_frame()
		grouped = grouped.with_row_index("group_id", offset=1)
		assignment = (
			signatures.join(
				grouped.select("group_id", "motif"),
				left_on="motif_min",
				right_on="motif",
				how="inner",
			)
			.select(pl.col("contig_id"), pl.col("end_label"), pl.col("group_id"))
		)
		return grouped, assignment

	clusters, assignment_rows = _cluster_distance_groups(
		signatures, min_len, max_len, distance, metric
	)
	if not clusters:
		return _empty_result_frame(), _empty_assignment_frame()
	group_df = pl.DataFrame(clusters).select(
		"group_id",
		"window_min",
		"window_max",
		"end_label",
		"ends_present",
		"motif",
		"member_count",
		"members",
		"max_distance",
		"distance_metric",
	)
	assignment_df = (
		pl.DataFrame(assignment_rows)
		if assignment_rows
		else _empty_assignment_frame()
	)
	return group_df, assignment_df


def _empty_result_frame() -> pl.DataFrame:
	return pl.DataFrame(
		{
			"group_id": pl.Series([], dtype=pl.Int64),
			"window_min": pl.Series([], dtype=pl.Int64),
			"window_max": pl.Series([], dtype=pl.Int64),
			"end_label": pl.Series([], dtype=pl.Utf8),
			"ends_present": pl.Series([], dtype=pl.List(pl.Utf8)),
			"motif": pl.Series([], dtype=pl.Utf8),
			"member_count": pl.Series([], dtype=pl.Int64),
			"members": pl.Series([], dtype=pl.List(pl.Utf8)),
			"max_distance": pl.Series([], dtype=pl.Int64),
			"distance_metric": pl.Series([], dtype=pl.Utf8),
		}
	)


def _empty_assignment_frame() -> pl.DataFrame:
	return pl.DataFrame(
		{
			"contig_id": pl.Series([], dtype=pl.Utf8),
			"end_label": pl.Series([], dtype=pl.Utf8),
			"group_id": pl.Series([], dtype=pl.Int64),
		}
	)


def _extend_group_motifs(
	group_df: pl.DataFrame,
	assignment_df: pl.DataFrame,
	signatures: pl.DataFrame,
) -> pl.DataFrame:
	if group_df.is_empty() or assignment_df.is_empty():
		return group_df
	available = signatures.select("contig_id", "end_label", "motif_full")
	join_df = (
		assignment_df.join(
			available, on=["contig_id", "end_label"], how="left"
		)
		.group_by("group_id", maintain_order=True)
		.agg(pl.col("motif_full").alias("motifs_full"))
	)
	if join_df.is_empty():
		return group_df

	def _extend_row(payload: dict[str, Any]) -> str:
		sequences = [seq for seq in payload["motifs_full"] or [] if seq]
		if not sequences:
			return payload["motif"]
		window_min = payload["window_min"]
		window_max = payload["window_max"]
		limit = min(window_max, min(len(seq) for seq in sequences))
		base = payload["motif"] or ""
		start_pos = max(len(base), window_min)
		if limit <= start_pos:
			return base
		for pos in range(start_pos, limit):
			candidate = sequences[0][pos]
			if all(seq[pos] == candidate for seq in sequences):
				base += candidate
			else:
				break
		return base

	enriched = group_df.join(join_df, on="group_id", how="left")
	if "motifs_full" not in enriched.columns:
		return group_df
	return (
		enriched.with_columns(
			pl.struct("motif", "window_min", "window_max", "motifs_full")
			.map_elements(_extend_row, return_dtype=pl.String)
			.alias("motif")
		)
		.drop("motifs_full")
	)


def _finalize_group_output(group_df: pl.DataFrame) -> pl.DataFrame:
	if group_df.is_empty():
		return pl.DataFrame(
			{
				"group_id": pl.Series([], dtype=pl.Int64),
				"member_count": pl.Series([], dtype=pl.Int64),
				"members": pl.Series([], dtype=pl.List(pl.Utf8)),
				"motif": pl.Series([], dtype=pl.Utf8),
				"motif_length": pl.Series([], dtype=pl.Int64),
				"found_in": pl.Series([], dtype=pl.List(pl.Utf8)),
				"max_distance": pl.Series([], dtype=pl.Int64),
				"distance_metric": pl.Series([], dtype=pl.Utf8),
			}
		)
	columns_to_drop = [col for col in ["window_min", "window_max"] if col in group_df.columns]
	result = (
		group_df.with_columns(
			pl.col("motif").str.len_chars().alias("motif_length")
		)
		.drop(columns_to_drop)
	)
	if "ends_present" in result.columns:
		result = result.rename({"ends_present": "found_in"})
	if "end_label" in result.columns:
		result = result.drop("end_label")
	if "member_count" in result.columns:
		result = result.sort("member_count", descending=True)
	preferred = [
		"group_id",
		"member_count",
		"motif",
		"motif_length",
		"found_in",
		"max_distance",
		"distance_metric",
		"members",
	]
	reordered = [col for col in preferred if col in result.columns]
	remainder = [col for col in result.columns if col not in reordered]
	return result.select(reordered + remainder)


def _hamming_distance(seq_a: str, seq_b: str) -> int:
	if len(seq_a) != len(seq_b):
		raise click.ClickException("Cannot compare motifs of different lengths with the hamming metric")
	return sum(ch_a != ch_b for ch_a, ch_b in zip(seq_a, seq_b))


def _consensus_motif(motifs: list[str]) -> str:
	if not motifs:
		return ""
	length = len(motifs[0])
	consensus_chars: list[str] = []
	for idx in range(length):
		symbol_counts: dict[str, int] = {}
		for motif in motifs:
			symbol = motif[idx]
			symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
		sorted_counts = sorted(symbol_counts.items(), key=lambda item: (-item[1], item[0]))
		consensus_chars.append(sorted_counts[0][0])
	return "".join(consensus_chars)


def _cluster_distance_groups(
	signatures: pl.DataFrame,
	min_len: int,
	max_len: int,
	max_distance: int,
	metric: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
	if metric != "hamming":
		raise click.ClickException(
			"Only the 'hamming' distance metric is supported when allowing mismatches"
		)
	rows = signatures.select("contig_id", "end_label", "motif_min").to_dicts()
	n = len(rows)
	if n < 2:
		return [], []

	parent = list(range(n))

	def _find(idx: int) -> int:
		while parent[idx] != idx:
			parent[idx] = parent[parent[idx]]
			idx = parent[idx]
		return idx

	def _union(a: int, b: int) -> None:
		root_a = _find(a)
		root_b = _find(b)
		if root_a != root_b:
			parent[root_b] = root_a

	for i in range(n):
		motif_i = rows[i]["motif_min"]
		for j in range(i + 1, n):
			motif_j = rows[j]["motif_min"]
			if _hamming_distance(motif_i, motif_j) <= max_distance:
				_union(i, j)

	clusters: dict[int, dict[str, Any]] = {}
	for idx, row in enumerate(rows):
		root = _find(idx)
		cluster = clusters.setdefault(
			root,
			{
				"records": [],
				"motifs": [],
			},
		)
		cluster["records"].append((row["contig_id"], row["end_label"]))
		cluster["motifs"].append(row["motif_min"])

	results: list[dict[str, object]] = []
	assignments: list[dict[str, object]] = []
	for cluster in clusters.values():
		records = list(cluster["records"])
		if len(records) < 2:
			continue
		motif_list = list(cluster["motifs"])
		consensus = _consensus_motif(motif_list)
		ends_present = sorted({end for _, end in records})
		results.append(
			{
				"motif": consensus,
				"member_count": len(records),
				"members": [rec[0] for rec in records],
				"ends_present": ends_present,
				"end_label": ends_present[0] if len(ends_present) == 1 else "mixed",
				"window_min": min_len,
				"window_max": max_len,
				"max_distance": max_distance,
				"distance_metric": metric,
				"_records": records,
			},
		)
	if not results:
		return [], []
	results.sort(key=lambda row: (row["end_label"], -row["member_count"], row["motif"]))
	for idx, cluster in enumerate(results, start=1):
		cluster["group_id"] = idx
		for contig_id, end_label in cluster["_records"]:
			assignments.append(
				{
					"contig_id": contig_id,
					"end_label": end_label,
					"group_id": idx,
				}
			)
	for cluster in results:
		cluster.pop("_records", None)
	return results, assignments


def _build_membership_table(
	seq_df: pl.DataFrame,
	signatures: pl.DataFrame,
	assignments: pl.DataFrame,
	group_df: pl.DataFrame,
) -> pl.DataFrame:
	base = seq_df.select(pl.col("contig_id"), pl.col("seq_length"))
	if signatures.is_empty():
		return base.with_columns(
			pl.lit(None, dtype=pl.Int64).alias("termini_group_1"),
			pl.lit(None, dtype=pl.Utf8).alias("termini_group_1_end"),
			pl.lit(None, dtype=pl.Utf8).alias("termini_group_1_motif"),
			pl.lit(None, dtype=pl.Int64).alias("termini_group_1_start"),
			pl.lit(None, dtype=pl.Int64).alias("termini_group_1_stop"),
			pl.lit(None, dtype=pl.Utf8).alias("termini_group_1_strand"),
			pl.lit(None, dtype=pl.Utf8).alias("termini_group_1_source"),
			pl.lit(None, dtype=pl.Int64).alias("termini_group_2"),
			pl.lit(None, dtype=pl.Utf8).alias("termini_group_2_end"),
			pl.lit(None, dtype=pl.Utf8).alias("termini_group_2_motif"),
			pl.lit(None, dtype=pl.Int64).alias("termini_group_2_start"),
			pl.lit(None, dtype=pl.Int64).alias("termini_group_2_stop"),
			pl.lit(None, dtype=pl.Utf8).alias("termini_group_2_strand"),
			pl.lit(None, dtype=pl.Utf8).alias("termini_group_2_source"),
		)

	annotated = signatures.join(
		assignments,
		on=["contig_id", "end_label"],
		how="left",
	)
	if not group_df.is_empty():
		annotated = annotated.join(
			group_df.select("group_id", pl.col("motif").alias("group_motif")),
			on="group_id",
			how="left",
		)
	else:
		annotated = annotated.with_columns(pl.lit(None).alias("group_motif"))

	annotated = annotated.with_columns(
		pl.when(pl.col("group_motif").is_not_null())
		.then(pl.col("group_motif"))
		.otherwise(pl.col("motif_min"))
		.alias("motif_assigned")
	)

	five_prime = annotated.filter(pl.col("end_label") == "5prime").select(
		pl.col("contig_id"),
		pl.col("group_id").alias("termini_group_1"),
		pl.lit("5prime").alias("termini_group_1_end"),
		pl.col("motif_assigned").alias("termini_group_1_motif"),
		pl.col("motif_start").alias("termini_group_1_start"),
		pl.col("motif_stop").alias("termini_group_1_stop"),
		pl.col("strand_label").alias("termini_group_1_strand"),
		pl.col("terminus").alias("termini_group_1_source"),
	)

	three_prime = annotated.filter(pl.col("end_label") == "3prime").select(
		pl.col("contig_id"),
		pl.col("group_id").alias("termini_group_2"),
		pl.lit("3prime").alias("termini_group_2_end"),
		pl.col("motif_assigned").alias("termini_group_2_motif"),
		pl.col("motif_start").alias("termini_group_2_start"),
		pl.col("motif_stop").alias("termini_group_2_stop"),
		pl.col("strand_label").alias("termini_group_2_strand"),
		pl.col("terminus").alias("termini_group_2_source"),
	)

	membership = base.join(five_prime, on="contig_id", how="left")
	membership = membership.join(three_prime, on="contig_id", how="left")
	return membership


def _write_table(
	df: pl.DataFrame,
	output_path: Path,
	output_format: str,
	logger,
	list_columns: list[str] | None = None,
	label: str = "table",
) -> None:
	fmt = output_format.lower()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	if fmt in {"tsv", "csv"} and list_columns:
		for column in list_columns:
			if column in df.columns:
				df = df.with_columns(pl.col(column).list.join(",").alias(column))
	if fmt == "tsv":
		df.write_csv(output_path, separator="\t")
	elif fmt == "csv":
		df.write_csv(output_path)
	elif fmt == "jsonl":
		df.write_json(output_path, row_oriented=True)
	else:
		raise click.ClickException(f"Unsupported output format: {output_format}")
	logger.info(f"Termini {label} written to {output_path}")


@click.command()
@click.option(
	"-i",
	"--input",
	required=True,
	type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
	help="Input contig FASTA/FASTQ file",
)
@click.option(
	"-n",
	"--length",
	"length_spec",
	default="40",
	show_default=True,
	help="Terminus length or range (e.g., 30 or 25-40)",
)
@click.option(
	"-d",
	"--distance",
	default=0,
	show_default=True,
	type=click.IntRange(0, 1000),
	help="Maximum allowed mismatches between termini",
)
@click.option(
	"--metric",
	default="hamming",
	show_default=True,
	type=click.Choice(["hamming", "edit"], case_sensitive=False),
	help="Distance metric to use for termini comparison",
)
@click.option(
	"--strand",
	default="both",
	show_default=True,
	type=click.Choice(["plus", "minus", "both"], case_sensitive=False),
	help="Strand orientation to consider",
)
@click.option(
	"-o",
	"--output",
	default="termini_assignments.tsv",
	show_default=True,
	type=click.Path(dir_okay=False, writable=True, path_type=Path),
	help="Output path for per-contig termini assignments",
)
@click.option(
	"--groups-output",
	type=click.Path(dir_okay=False, writable=True, path_type=Path),
	default=None,
	help="Optional output path for the aggregated motif groups",
)
@click.option(
	"--output-format",
	default="tsv",
	show_default=True,
	type=click.Choice(["tsv", "csv", "jsonl"], case_sensitive=False),
	help="Format for the grouped output",
)
@click.option(
	"--log-file",
	type=click.Path(dir_okay=False, writable=True, path_type=Path),
	default=None,
	help="Optional log file path",
)
@click.option("-ll", "--log-level", default="INFO", show_default=True, hidden=True)
def termini(
	input: Path,
	length_spec: str,
	distance: int,
	metric: str,
	strand: str,
	output: Path,
	groups_output: Path | None,
	output_format: str,
	log_file: Path | None,
	log_level: str,
) -> None:
	"""Group contigs that share termini of length *n* (or a range)."""

	logger = setup_logging(log_file, log_level)
	if output.suffix == "":
		logger.info("Output path missing suffix; defaulting to '.tsv' and TSV format")
		output = output.with_suffix(".tsv")
		if output_format.lower() != "tsv":
			output_format = "tsv"
	log_start_info(logger, locals())

	min_len, max_len = _parse_length_window(length_spec)
	strand_mode = strand.lower()
	metric = metric.lower()

	seq_df = _load_sequences(input)
	if seq_df.is_empty():
		logger.warning("No sequences found in input file")
		return

	if metric == "edit" and distance > 0:
		raise click.ClickException(
			"Edit-distance tolerances are not implemented yet; please use --metric hamming or set --distance 0"
		)

	signatures = _build_signature_table(seq_df, min_len, max_len, strand_mode)
	unique_signatures = _deduplicate_signatures(signatures)
	if unique_signatures.is_empty():
		logger.warning(
			"No sequences were long enough to satisfy the minimum window length"
		)
		group_df = _empty_result_frame()
		assignment_df = _empty_assignment_frame()
	else:
		group_df, assignment_df = _aggregate_min_range_groups(
			unique_signatures, min_len, max_len, distance, metric
		)
		if group_df.is_empty():
			logger.warning(
				"No shared termini detected with the current min/max window parameters"
			)
		else:
			logger.info(
				f"Identified {group_df.height} termini-sharing groups with min window {min_len} and max window {max_len}"
			)

	if not group_df.is_empty():
		group_df = _extend_group_motifs(group_df, assignment_df, unique_signatures)

	group_df = _finalize_group_output(group_df)

	membership_df = _build_membership_table(
		seq_df, unique_signatures, assignment_df, group_df
	)
	_write_table(
		membership_df,
		output,
		output_format,
		logger,
		label="assignments",
	)

	if groups_output is None:
		groups_output = output.with_name(
			f"{output.stem}.groups{output.suffix or ''}"
		)
	_write_table(
		group_df,
		groups_output,
		output_format,
		logger,
		list_columns=["members", "found_in"],
		label="groups",
	)

