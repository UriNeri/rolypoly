from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import polars as pl
import rich_click as click

from rolypoly.utils.logging.loggit import log_start_info, setup_logging
from rolypoly.utils.bio.polars_fastx import frame_to_fastx, load_sequences
from rolypoly.utils.bio.alignments import hamming_distance
from rolypoly.utils.bio.sequences import revcomp

# Ensure the FASTX plugins are registered
from rolypoly.utils.bio import polars_fastx as _polars_fastx  # noqa: F401

# ANI clustering is shared with the assembly extend command
from rolypoly.commands.assembly.extend import cluster_contigs_by_ani


def parse_length_window(value: str) -> Tuple[int, int]:
	"""Parse a fixed length or inclusive length range specification.

	Args:
		value: Length string such as ``40`` or ``25-40``.

	Returns:
		Tuple[int, int]: Minimum and maximum motif lengths.

	Raises:
		click.BadParameter: If values are non-positive or range is invalid.
	"""
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


def found_label_expr() -> pl.Expr:
	"""Create orientation-aware provenance labels for group summaries."""
	end_token = pl.col("end_label").str.replace("prime", "")
	return (
	 pl.when(pl.col("strand_label") == "+")
	 .then(pl.concat_str([pl.lit("fwd_on_"), end_token, pl.lit("_end")]))
	 .otherwise(pl.concat_str([pl.lit("rev_on_"), end_token, pl.lit("_end")]))
	 .alias("found_label")
	)


def load_and_orientate_sequences(fasta_path: Path) -> pl.DataFrame:
	"""Load sequences and add reverse complements for termini analysis."""
	df = load_sequences(fasta_path)
	if df.is_empty():
		return df
	return orientate_all(df)


def ensure_rc_column(df: pl.DataFrame) -> pl.DataFrame:
	"""Ensure a reverse-complement sequence column exists.

	Args:
		df: Input sequence table with at least a ``sequence`` column.

	Returns:
		pl.DataFrame: Original table with ``sequence_rc`` present.
	"""
	if "sequence_rc" in df.columns:
		return df
	return df.with_columns(
	 pl.col("sequence")
	 .map_elements(revcomp, return_dtype=pl.String)
	 .alias("sequence_rc")
	)


def orientate_all(df: pl.DataFrame) -> pl.DataFrame:
	"""Placeholder for future orientation normalization logic."""  # noqa: D401
	# TODO: implement restranding/orientate_all logic that favors the strand
	# containing the highest density of forward-encoded coding genes.
	return df



def apply_ani_prefilter(
	seq_df: pl.DataFrame,
	enabled: bool,
	min_identity: float,
	min_af: float,
	logger,
) -> pl.DataFrame:
	"""Cluster contigs by ANI and keep the longest representative per cluster.

	For overlap-based pileup extension of ANI clusters, use the extend
	command instead (under the assembly group).
	"""
	if not enabled or seq_df.is_empty() or seq_df.height < 2:
		return seq_df

	seq_rows = seq_df.select("contig_id", "sequence", "seq_length").to_dicts()
	for idx, row in enumerate(seq_rows):
		row["_order"] = idx

	clusters = cluster_contigs_by_ani(seq_rows, min_identity, min_af, logger)
	if not clusters:
		return seq_df

	representatives: list[dict[str, Any]] = []
	for cluster in clusters:
		representative = max(
			cluster,
			key=lambda row: (int(row["seq_length"]), -int(row["_order"])),
		)
		representatives.append({
			"contig_id": representative["contig_id"],
			"sequence": representative["sequence"],
			"seq_length": representative["seq_length"],
			"_order": representative["_order"],
		})

	filtered = pl.DataFrame(representatives).sort("_order").drop("_order")
	removed = seq_df.height - filtered.height
	if removed > 0:
		logger.info(
			"ANI prefilter retained %s/%s representative contigs (removed %s)",
			filtered.height,
			seq_df.height,
			removed,
		)

	return filtered


def create_signature_frame(
 df: pl.DataFrame,
 sequence_col: str,
 min_len: int,
 max_len: int,
 label: str,
 is_plus: bool,
) -> pl.DataFrame:
	"""Build termini signature rows for one strand orientation.

	Args:
		df: Sequence table with contig IDs and lengths.
		sequence_col: Column to slice for motif extraction.
		min_len: Seed motif length used for grouping.
		max_len: Maximum motif length retained for later extension.
		label: Strand label prefix used in ``terminus`` naming.
		is_plus: Whether current orientation is plus strand.

	Returns:
		pl.DataFrame: Signature rows for 5' and 3' termini in this orientation.
	"""
	filtered = df.filter(pl.col("seq_length") >= min_len)
	if filtered.is_empty():
		return pl.DataFrame()

	def build_end(end_name: str) -> pl.DataFrame:
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
		 .with_columns(found_label_expr())
		 .drop_nulls("motif_min")
		 .filter(pl.col("motif_min") != "")
		)

	frames = [build_end("5prime"), build_end("3prime")]
	frames = [frame for frame in frames if not frame.is_empty()]
	if not frames:
		return pl.DataFrame()
	return pl.concat(frames, how="vertical_relaxed")


def build_signature_table(
 df: pl.DataFrame, min_len: int, max_len: int, strand: str
) -> pl.DataFrame:
	"""Build the complete termini signature table for selected strands.

	Args:
		df: Input sequence table.
		min_len: Minimum motif length for grouping seeds.
		max_len: Maximum motif length for extension candidates.
		strand: Strand mode (plus, minus, or both).

	Returns:
		pl.DataFrame: Combined signature rows across requested orientations.
	"""
	frames: list[pl.DataFrame] = []
	if strand in {"plus", "both"}:
		frames.append(
		 create_signature_frame(df, "sequence", min_len, max_len, "+", True)
		)
	if strand in {"minus", "both"}:
		df = ensure_rc_column(df)
		frames.append(
		 create_signature_frame(df, "sequence_rc", min_len, max_len, "-", False)
		)
	frames = [frame for frame in frames if not frame.is_empty()]
	if not frames:
		return pl.DataFrame(
		 {
		  "contig_id": pl.Series([], dtype=pl.Utf8),
		  "terminus": pl.Series([], dtype=pl.Utf8),
		  "end_label": pl.Series([], dtype=pl.Utf8),
		  "strand_label": pl.Series([], dtype=pl.Utf8),
		  "found_label": pl.Series([], dtype=pl.Utf8),
		  "motif_full": pl.Series([], dtype=pl.Utf8),
		  "motif_min": pl.Series([], dtype=pl.Utf8),
		  "motif_start": pl.Series([], dtype=pl.Int64),
		  "motif_stop": pl.Series([], dtype=pl.Int64),
		  "window_min": pl.Series([], dtype=pl.Int64),
		  "window_max": pl.Series([], dtype=pl.Int64),
		 }
		)
	return pl.concat(frames, how="vertical_relaxed")


def deduplicate_signatures(signatures: pl.DataFrame) -> pl.DataFrame:
	"""Keep one signature per contig/end, preferring plus-strand records.

	Args:
		signatures: Raw signature table that may contain strand duplicates.

	Returns:
		pl.DataFrame: Deduplicated signatures keyed by contig and end label.
	"""
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


def aggregate_min_range_groups(
 signatures: pl.DataFrame,
 min_len: int,
 max_len: int,
 distance: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
	"""Group signatures by seed motif identity or Hamming distance.

	Args:
		signatures: Deduplicated termini signature table.
		min_len: Seed motif length used for grouping.
		max_len: Maximum motif length configured by user.
		distance: Allowed Hamming distance for seed clustering.

	Returns:
		Tuple of group summary table and per-contig assignment table.
	"""
	if signatures.is_empty():
		return empty_result_frame(), empty_assignment_frame()
	if distance == 0:
		grouped = (
		 signatures.group_by(["motif_min"], maintain_order=True)
		 .agg(
		  pl.len().alias("member_count"),
		  pl.col("contig_id").alias("members"),
		  pl.col("found_label").alias("member_found"),
		 )
		 .filter(pl.col("member_count") > 1)
		 .with_columns(
		  pl.col("motif_min").alias("motif"),
		  pl.col("member_found").list.unique().alias("ends_present"),
		  pl.lit(min_len).alias("window_min"),
		  pl.lit(max_len).alias("window_max"),
		  pl.lit(distance).alias("max_distance"),
		 )
		 .with_columns(
		  pl.when(pl.col("ends_present").list.len() == 1)
		  .then(pl.col("ends_present").list.first())
		  .otherwise(pl.lit("mixed"))
		  .alias("end_label")
		 )
		 .drop("motif_min", "member_found")
		 .sort(["end_label", "member_count"], descending=[False, True])
		)
		if grouped.height == 0:
			return empty_result_frame(), empty_assignment_frame()
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

	clusters, assignment_rows = cluster_distance_groups(
	 signatures, min_len, max_len, distance
	)
	if not clusters:
		return empty_result_frame(), empty_assignment_frame()
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
	)
	assignment_df = (
	 pl.DataFrame(assignment_rows)
	 if assignment_rows
	 else empty_assignment_frame()
	)
	return group_df, assignment_df


def empty_result_frame() -> pl.DataFrame:
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
	 }
	)


def empty_assignment_frame() -> pl.DataFrame:
	return pl.DataFrame(
	 {
	  "contig_id": pl.Series([], dtype=pl.Utf8),
	  "end_label": pl.Series([], dtype=pl.Utf8),
	  "group_id": pl.Series([], dtype=pl.Int64),
	 }
	)


def extend_group_motifs(
 group_df: pl.DataFrame,
 assignment_df: pl.DataFrame,
 signatures: pl.DataFrame,
) -> pl.DataFrame:
	"""Extend each group motif while all assigned members agree by position.

	Args:
		group_df: Group summary table from first-pass clustering.
		assignment_df: Per-signature mapping from contigs to groups.
		signatures: Signature table including full-window motif strings.

	Returns:
		pl.DataFrame: Group table with possibly extended motif sequences.
	"""
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

	def extend_row(payload: dict[str, Any]) -> str:
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
	  .map_elements(extend_row, return_dtype=pl.String)
	  .alias("motif")
	 )
	 .drop("motifs_full")
	)


def annotate_found_in_from_all_signatures(
 group_df: pl.DataFrame,
 assignment_df: pl.DataFrame,
 all_signatures: pl.DataFrame,
) -> pl.DataFrame:
	"""Restore full orientation provenance in ``found_in`` using raw signatures."""
	if group_df.is_empty() or assignment_df.is_empty() or all_signatures.is_empty():
		return group_df

	found_df = (
	 assignment_df.join(
	  all_signatures.select("contig_id", "end_label", "found_label"),
	  on=["contig_id", "end_label"],
	  how="inner",
	 )
	 .group_by("group_id", maintain_order=True)
	 .agg(pl.col("found_label").unique().sort().alias("found_from_all"))
	)

	if found_df.is_empty():
		return group_df

	annotated = group_df.join(found_df, on="group_id", how="left")
	if "found_from_all" not in annotated.columns:
		return group_df

	return (
	 annotated.with_columns(
	  pl.when(pl.col("found_from_all").is_not_null())
	  .then(pl.col("found_from_all"))
	  .otherwise(pl.col("ends_present"))
	  .alias("ends_present")
	 )
	 .drop("found_from_all")
	)


def finalize_group_output(group_df: pl.DataFrame) -> pl.DataFrame:
	"""Normalize and reorder group summary columns for final export.

	Args:
		group_df: Group table from extension/collapse steps.

	Returns:
		pl.DataFrame: Finalized group summary with stable output columns.
	"""
	if group_df.is_empty():
		return pl.DataFrame(
		 {
		  "group_id": pl.Series([], dtype=pl.Int64),
		  "member_count": pl.Series([], dtype=pl.Int64),
		  "motif": pl.Series([], dtype=pl.Utf8),
		  "motif_length": pl.Series([], dtype=pl.Int64),
		  "found_in": pl.Series([], dtype=pl.List(pl.Utf8)),
		  "max_distance": pl.Series([], dtype=pl.Int64),
		  "source_group_ids": pl.Series([], dtype=pl.List(pl.Int64)),
		  "clip_contains_source_ids": pl.Series([], dtype=pl.List(pl.Int64)),
		  "clip_contained_by_source_ids": pl.Series([], dtype=pl.List(pl.Int64)),
		  "members": pl.Series([], dtype=pl.List(pl.Utf8)),
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
	 "source_group_ids",
	 "clip_contains_source_ids",
	 "clip_contained_by_source_ids",
	 "members",
	]
	reordered = [col for col in preferred if col in result.columns]
	remainder = [col for col in result.columns if col not in reordered]
	return result.select(reordered + remainder)


def is_one_edge_clipped_containment(shorter: str, longer: str, max_clipped: int) -> bool:
	"""Check if ``shorter`` is contained in ``longer`` by single-edge clipping."""
	delta = len(longer) - len(shorter)
	if delta <= 0 or delta > max_clipped:
		return False
	return shorter == longer[delta:] or shorter == longer[:-delta]


def is_both_edge_clipped_containment(shorter: str, longer: str, max_clipped: int) -> bool:
	"""Check if ``shorter`` is contained in ``longer`` with clipping on both edges."""
	delta = len(longer) - len(shorter)
	if delta <= 0 or delta > max_clipped:
		return False
	for start in range(delta + 1):
		if longer[start : start + len(shorter)] == shorter:
			return True
	return False


def collapse_groups_by_clipping(
 group_df: pl.DataFrame,
 assignment_df: pl.DataFrame,
 max_clipped: int,
 clip_mode: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
	"""Collapse connected groups linked by clipped motif containment."""
	if group_df.is_empty() or max_clipped <= 0:
		return group_df, assignment_df
	if clip_mode not in {"one-edge", "both"}:
		raise click.ClickException(f"Unsupported --clip-mode: {clip_mode}")
	rows = group_df.select("group_id", "motif").to_dicts()
	if len(rows) < 2:
		group_df = group_df.with_columns(
		 pl.col("group_id").map_elements(lambda gid: [gid], return_dtype=pl.List(pl.Int64)).alias("source_group_ids"),
		 pl.lit([], dtype=pl.List(pl.Int64)).alias("clip_contains_source_ids"),
		 pl.lit([], dtype=pl.List(pl.Int64)).alias("clip_contained_by_source_ids"),
		)
		return group_df, assignment_df

	group_ids = [row["group_id"] for row in rows]
	parent: dict[int, int] = {gid: gid for gid in group_ids}
	contains_map: dict[int, set[int]] = {gid: set() for gid in group_ids}
	contained_by_map: dict[int, set[int]] = {gid: set() for gid in group_ids}

	def find(gid: int) -> int:
		while parent[gid] != gid:
			parent[gid] = parent[parent[gid]]
			gid = parent[gid]
		return gid

	def union(a: int, b: int) -> None:
		root_a = find(a)
		root_b = find(b)
		if root_a != root_b:
			parent[root_b] = root_a

	for idx, row_a in enumerate(rows):
		gid_a = row_a["group_id"]
		motif_a = row_a["motif"]
		for row_b in rows[idx + 1 :]:
			gid_b = row_b["group_id"]
			motif_b = row_b["motif"]
			if len(motif_a) <= len(motif_b):
				short_gid, short_motif = gid_a, motif_a
				long_gid, long_motif = gid_b, motif_b
			else:
				short_gid, short_motif = gid_b, motif_b
				long_gid, long_motif = gid_a, motif_a
			if clip_mode == "both":
				is_contained = is_both_edge_clipped_containment(
				 short_motif, long_motif, max_clipped
				)
			else:
				is_contained = is_one_edge_clipped_containment(
				 short_motif, long_motif, max_clipped
				)
			if is_contained:
				contains_map[long_gid].add(short_gid)
				contained_by_map[short_gid].add(long_gid)
				union(long_gid, short_gid)

	components: dict[int, list[int]] = {}
	for gid in group_ids:
		root = find(gid)
		components.setdefault(root, []).append(gid)

	membership_map: dict[int, int] = {}
	collapsed_rows: list[dict[str, Any]] = []
	for source_ids in components.values():
		source_ids = sorted(source_ids)
		component_df = group_df.filter(pl.col("group_id").is_in(source_ids))
		component_rows = component_df.to_dicts()
		representative = max(
		 component_rows,
		 key=lambda row: (len(row["motif"]), row["member_count"], -row["group_id"]),
		)
		new_group_id = min(source_ids)
		for source_group_id in source_ids:
			membership_map[source_group_id] = new_group_id

		members: list[str] = []
		seen_members: set[str] = set()
		for row in component_rows:
			for member in row.get("members", []) or []:
				if member not in seen_members:
					seen_members.add(member)
					members.append(member)

		found_in = sorted(
		 {
		  entry
		  for row in component_rows
		  for entry in (row.get("ends_present", []) or row.get("found_in", []) or [])
		 }
		)
		contains_ids = sorted({nested for gid in source_ids for nested in contains_map.get(gid, set())})
		contained_by_ids = sorted({nested for gid in source_ids for nested in contained_by_map.get(gid, set())})

		collapsed_rows.append(
		 {
		  "group_id": new_group_id,
		  "motif": representative["motif"],
		  "member_count": len(members),
		  "members": members,
		  "ends_present": found_in,
		  "max_distance": representative.get("max_distance", 0),
		  "window_min": representative.get("window_min"),
		  "window_max": representative.get("window_max"),
		  "source_group_ids": source_ids,
		  "clip_contains_source_ids": contains_ids,
		  "clip_contained_by_source_ids": contained_by_ids,
		 }
		)

	collapsed_df = pl.DataFrame(collapsed_rows)
	if assignment_df.is_empty():
		return collapsed_df, assignment_df
	remapped_assignments = assignment_df.with_columns(
	 pl.col("group_id")
	 .map_elements(lambda gid: membership_map.get(gid, gid), return_dtype=pl.Int64)
	 .alias("group_id")
	)
	return collapsed_df, remapped_assignments


def consensus_motif(motifs: list[str]) -> str:
	"""Build a position-wise consensus motif using majority vote ties by symbol."""
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


def cluster_distance_groups(
 signatures: pl.DataFrame,
 min_len: int,
 max_len: int,
 max_distance: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
	"""Cluster signatures by Hamming distance and emit group/assignment records.

	Args:
		signatures: Signature table with minimum-window motifs.
		min_len: Minimum motif window length.
		max_len: Maximum motif window length.
		max_distance: Maximum pairwise Hamming distance for connectivity.

	Returns:
		Tuple of Python records for grouped motifs and contig assignments.
	"""
	rows = signatures.select(
	 "contig_id", "end_label", "strand_label", "found_label", "motif_min"
	).to_dicts()
	n = len(rows)
	if n < 2:
		return [], []

	parent = list(range(n))

	def find(idx: int) -> int:
		while parent[idx] != idx:
			parent[idx] = parent[parent[idx]]
			idx = parent[idx]
		return idx

	def union(a: int, b: int) -> None:
		root_a = find(a)
		root_b = find(b)
		if root_a != root_b:
			parent[root_b] = root_a

	for i in range(n):
		motif_i = rows[i]["motif_min"]
		for j in range(i + 1, n):
			motif_j = rows[j]["motif_min"]
			if hamming_distance(motif_i, motif_j) <= max_distance:
				union(i, j)

	clusters: dict[int, dict[str, Any]] = {}
	for idx, row in enumerate(rows):
		root = find(idx)
		cluster = clusters.setdefault(
		 root,
		 {
		  "records": [],
		  "motifs": [],
		 },
		)
		cluster["records"].append(
		 (row["contig_id"], row["end_label"], row["found_label"])
		)
		cluster["motifs"].append(row["motif_min"])

	results: list[dict[str, object]] = []
	assignments: list[dict[str, object]] = []
	for cluster in clusters.values():
		records = list(cluster["records"])
		if len(records) < 2:
			continue
		motif_list = list(cluster["motifs"])
		consensus = consensus_motif(motif_list)
		ends_present = sorted({found for _, _, found in records})
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
		  "_records": records,
		 },
		)
	if not results:
		return [], []
	results.sort(key=lambda row: (row["end_label"], -row["member_count"], row["motif"]))
	for idx, cluster in enumerate(results, start=1):
		cluster["group_id"] = idx
		for contig_id, end_label, _ in cluster["_records"]:
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


def build_membership_table(
 seq_df: pl.DataFrame,
 signatures: pl.DataFrame,
 assignments: pl.DataFrame,
 group_df: pl.DataFrame,
) -> pl.DataFrame:
	"""Build per-contig termini membership columns for output.

	Args:
		seq_df: Input contig table.
		signatures: Signature rows for contigs and ends.
		assignments: Mapping of signatures to group IDs.
		group_df: Final group summary table with consensus motifs.

	Returns:
		pl.DataFrame: One row per contig with 5' and 3' group annotations.
	"""
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


def write_group_motifs_fasta(
 group_df: pl.DataFrame,
 assignments: pl.DataFrame,
 signatures: pl.DataFrame,
 fasta_path: Path,
 logger,
) -> None:
	"""Write one representative motif sequence per final group to FASTA."""
	if group_df.is_empty():
		fasta_path.parent.mkdir(parents=True, exist_ok=True)
		fasta_path.write_text("")
		logger.info(f"Termini motifs FASTA written to {fasta_path} (no groups)")
		return

	lookup = signatures.select(
	 "contig_id", "end_label", "motif_start", "motif_stop"
	)
	exemplar = (
	 assignments.join(lookup, on=["contig_id", "end_label"], how="left")
	 .group_by("group_id", maintain_order=True)
	 .agg(
	  pl.col("contig_id").first().alias("example_contig"),
	  pl.col("end_label").first().alias("example_end"),
	  pl.col("motif_start").first().alias("example_seed_start"),
	  pl.col("motif_stop").first().alias("example_seed_stop"),
	 )
	)

	to_write = group_df.join(exemplar, on="group_id", how="left").with_columns(
	 pl.col("motif").str.len_chars().alias("motif_length_calc")
	)
	to_write = to_write.with_columns(
	 pl.when(pl.col("example_end") == "5prime")
	 .then(pl.col("example_seed_start"))
	 .otherwise(pl.col("example_seed_stop") - pl.col("motif_length_calc") + 1)
	 .alias("example_start"),
	 pl.when(pl.col("example_end") == "5prime")
	 .then(pl.col("example_seed_start") + pl.col("motif_length_calc") - 1)
	 .otherwise(pl.col("example_seed_stop"))
	 .alias("example_stop"),
	)

	def build_header(row: dict[str, Any]) -> str:
		group_id = row.get("group_id")
		member_count = row.get("member_count")
		contig = row.get("example_contig") or "na"
		start = row.get("example_start")
		stop = row.get("example_stop")
		end_label = row.get("example_end") or "na"
		coord = (
		 f"{contig}:{start}-{stop}:{end_label}"
		 if start is not None and stop is not None
		 else f"{contig}:na"
		)
		return f"group_id_{group_id}_mems_{member_count}_exmp_{coord}"

	rows = to_write.select(
	 "group_id",
	 "member_count",
	 "motif",
	 "example_contig",
	 "example_end",
	 "example_start",
	 "example_stop",
	).to_dicts()

	headers = [build_header(row) for row in rows]
	seqs = [row.get("motif") or "" for row in rows]
	fasta_path.parent.mkdir(parents=True, exist_ok=True)
	fasta_df = pl.DataFrame({"header": headers, "sequence": seqs})
	frame_to_fastx(fasta_df, fasta_path, seq_col="sequence", header_col="header")
	logger.info(f"Termini motifs FASTA written to {fasta_path}")


def write_table(
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
				df = df.with_columns(
				 pl.col(column)
				 .list.eval(pl.element().cast(pl.String))
				 .list.join(",")
				 .alias(column)
				)
	if fmt == "tsv":
		df.write_csv(output_path, separator="\t")
	elif fmt == "csv":
		df.write_csv(output_path)
	elif fmt == "parquet":
		df.write_parquet(output_path)
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
	help="Maximum Hamming mismatches allowed in the first-pass grouping seed",
)
@click.option(
	"--max-clipped",
	default=4,
	show_default=True,
	type=click.IntRange(0, 1000),
	help="Second-pass collapse: maximum total clipped bases allowed when one motif is contained in another",
)
@click.option(
	"--max-clipped-collapse/--no-max-clipped-collapse",
	default=True,
	show_default=True,
	help="Enable/disable second-pass collapse of groups linked by clipped motif containment",
)
@click.option(
	"--clip-mode",
	default="both",
	show_default=True,
	type=click.Choice(["one-edge", "both"], case_sensitive=False),
	help="Containment mode for second pass: one-edge clipping only, or clipping distributed across both edges",
)
@click.option(
	"--strand",
	default="both",
	show_default=True,
	type=click.Choice(["plus", "minus", "both"], case_sensitive=False),
	help="Strand orientation(s) used when building termini signatures",
)
@click.option(
	"--ani-prefilter/--no-ani-prefilter",
	default=True,
	show_default=True,
	help="Before termini grouping, collapse highly similar contigs so each ANI cluster contributes one representative (longest). For overlap-based extension, use the extend command.",
)
@click.option(
	"--ani-min-identity",
	default=0.95,
	show_default=True,
	type=click.FloatRange(0.0, 1.0),
	help="Minimum ANI identity (0-1) for contigs to be considered in the same prefilter cluster",
)
@click.option(
	"--ani-min-af",
	default=0.80,
	show_default=True,
	type=click.FloatRange(0.0, 1.0),
	help="Minimum aligned fraction (min(query_fraction, reference_fraction), 0-1) for ANI prefilter clustering",
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
	help="Optional output path for grouped motif summary (default: <output>.groups.<ext>)",
)
@click.option(
	"--motifs-fasta",
	type=click.Path(dir_okay=False, writable=True, path_type=Path),
	default=None,
	help="Output path for group motif FASTA entries (defaults to <output>.motifs.fasta)",
)
@click.option(
	"--output-format",
	default="tsv",
	show_default=True,
	type=click.Choice(["tsv", "csv", "parquet", "jsonl"], case_sensitive=False),
	help="Tabular output format for assignments and groups tables",
)
@click.option(
	"--log-file",
	type=click.Path(dir_okay=False, writable=True, path_type=Path),
	default=None,
	help="Optional log file path",
)
@click.option("-ll", "--log-level", default="INFO", show_default=True, hidden=True)
@click.option(
	"-t",
	"--threads",
	default=4,
	show_default=True,
	type=click.IntRange(1, 1000),
	help="Number of threads to use for parallel processing IF applicable",
)
def termini(
	input: Path,
	length_spec: str,
	distance: int,
	max_clipped: int,
	max_clipped_collapse: bool,
	clip_mode: str,
	strand: str,
	ani_prefilter: bool,
	ani_min_identity: float,
	ani_min_af: float,
	output: Path,
	groups_output: Path | None,
	motifs_fasta: Path | None,
	output_format: str,
	log_file: Path | None,
	log_level: str,
	threads: int,
) -> None:
	"""Group contigs that share termini of length *n* (or a range).

	Workflow:
	1) Optional ANI pre-pass (on by default) clusters highly similar contigs and keeps
	   the longest representative per cluster. For overlap pileup extension use the
	   extend command (under the assembly group).
	2) First pass groups contigs by the minimum-window motif seed (exact when --distance 0,
	   Hamming-tolerant otherwise).
	3) Motifs are then extended up to --length max while all members share the same added bases.
	4) Optional second pass (on by default) collapses groups when one motif is contained in another
	   by clipped containment (--clip-mode), up to --max-clipped bases.

	Outputs:
	- Assignments table at --output.
	- Group summary table at --groups-output (or <output>.groups.<ext> by default).
	- Group motifs FASTA at --motifs-fasta (or <output>.motifs.fasta by default).

	Group-output columns:
	- found_in: orientation-aware labels for member placement (e.g., fwd_on_5_end, rev_on_3_end).
	- source_group_ids: first-pass group IDs represented in the final row.
	- clip_contains_source_ids: source groups whose motifs are clipped-contained by this group.
	- clip_contained_by_source_ids: source groups that clipped-contain this group.
	"""

	logger = setup_logging(log_file, log_level)
	if output.suffix == "":
		logger.debug("Output path missing suffix; defaulting to '.tsv' and TSV format")
		output = output.with_suffix(".tsv")
		if output_format.lower() != "tsv":
			output_format = "tsv"
	log_start_info(logger, locals())

	min_len, max_len = parse_length_window(length_spec)
	strand_mode = strand.lower()
	clip_mode = clip_mode.lower()

	seq_df = load_and_orientate_sequences(input)
	if seq_df.is_empty():
		logger.warning("No sequences found in input file")
		return
	try:
		seq_df = apply_ani_prefilter(
			seq_df=seq_df,
			enabled=ani_prefilter,
			min_identity=ani_min_identity,
			min_af=ani_min_af,
			logger=logger,
		)
	except click.ClickException:
		raise
	except Exception as exc:
		logger.exception("ANI prefilter failed with an unexpected error")
		raise click.ClickException(f"ANI prefilter failed: {exc}") from exc

	if seq_df.is_empty():
		logger.warning("No sequences remain after ANI prefilter")
		return

	raw_signatures = build_signature_table(seq_df, min_len, max_len, strand_mode)
	unique_signatures = deduplicate_signatures(raw_signatures)
	if unique_signatures.is_empty():
		logger.warning(
			"No sequences were long enough to satisfy the minimum window length"
		)
		group_df = empty_result_frame()
		assignment_df = empty_assignment_frame()
	else:
		group_df, assignment_df = aggregate_min_range_groups(
			unique_signatures, min_len, max_len, distance
		)
		if group_df.is_empty():
			logger.warning(
				"No shared termini detected with the current min/max window parameters"
			)
		else:
			singleton_groups = (
				group_df.filter(pl.col("member_count") == 1).height
				if "member_count" in group_df.columns
				else 0
			)
			logger.info(
				f"Identified {group_df.height} termini-sharing groups with min window {min_len} and max window {max_len} (singletons: {singleton_groups})"
			)

	if not group_df.is_empty():
		group_df = extend_group_motifs(group_df, assignment_df, unique_signatures)
		if max_clipped_collapse:
			pre_collapse_groups = group_df.height
			group_df, assignment_df = collapse_groups_by_clipping(
				group_df, assignment_df, max_clipped, clip_mode
			)
			post_collapse_groups = group_df.height
			if "source_group_ids" in group_df.columns:
				source_counts = group_df.select(
					pl.col("source_group_ids").list.len().alias("n")
				).get_column("n")
				merged_groups = sum(1 for n in source_counts if n > 1)
				source_total = int(sum(source_counts))
				collapsed_away = source_total - post_collapse_groups
			else:
				merged_groups = 0
				collapsed_away = pre_collapse_groups - post_collapse_groups
			logger.info(
				"Second-pass clipped collapse (--clip-mode %s, --max-clipped %s): %s -> %s groups (%s collapsed across %s merged groups; singletons: %s)",
				clip_mode,
				max_clipped,
				pre_collapse_groups,
				post_collapse_groups,
				collapsed_away,
				merged_groups,
				group_df.filter(pl.col("member_count") == 1).height,
			)
		else:
			logger.info("Second-pass clipped collapse disabled (--no-max-clipped-collapse)")
		group_df = annotate_found_in_from_all_signatures(
			group_df, assignment_df, raw_signatures
		)

	group_df = finalize_group_output(group_df)

	membership_df = build_membership_table(
		seq_df, unique_signatures, assignment_df, group_df
	)
	write_table(
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
	write_table(
		group_df,
		groups_output,
		output_format,
		logger,
		list_columns=[
			"members",
			"found_in",
			"source_group_ids",
			"clip_contains_source_ids",
			"clip_contained_by_source_ids",
		],
		label="groups",
	)

	if motifs_fasta is None:
		motifs_fasta = output.with_name(f"{output.stem}.motifs.fasta")
	write_group_motifs_fasta(
		group_df,
		assignment_df,
		unique_signatures,
		motifs_fasta,
		logger,
	)
