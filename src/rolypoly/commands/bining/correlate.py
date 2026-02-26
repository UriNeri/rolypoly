from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import rich_click as click

from rolypoly.utils.logging.loggit import log_start_info, setup_logging


def infer_separator(input_path: Path, separator: str) -> str:
    """Infer separator from file extension when requested."""
    if separator != "auto":
        return separator
    suffix = input_path.suffix.lower()
    if suffix in {".tsv", ".tab", ".txt"}:
        return "\t"
    return ","


def parse_abundance_table(
    input_path: Path,
    separator: str,
    logger,
) -> tuple[pl.DataFrame, str, list[str]]:
    """Read a contig x sample abundance/presence table."""
    if not input_path.exists():
        raise click.ClickException(f"Input file does not exist: {input_path}")

    sep = infer_separator(input_path, separator)
    df = pl.read_csv(input_path, separator=sep)

    if df.width < 2:
        raise click.ClickException(
            "Input table must contain one contig id column and at least one sample column"
        )

    id_column = df.columns[0]
    sample_columns = df.columns[1:]

    logger.info(
        "Loaded table with %s contigs and %s samples",
        df.height,
        len(sample_columns),
    )

    casted = df.with_columns(
        [pl.col(column).cast(pl.Float64, strict=False) for column in sample_columns]
    )

    failed_cast_count = casted.select(
        [
            pl.col(column)
            .is_null()
            .sum()
            .alias(column)
            for column in sample_columns
        ]
    ).row(0)

    if any(count == casted.height for count in failed_cast_count):
        bad_cols = [
            sample_columns[idx]
            for idx, count in enumerate(failed_cast_count)
            if count == casted.height
        ]
        raise click.ClickException(
            "Sample columns must be numeric. Failed to parse columns: "
            + ", ".join(bad_cols)
        )

    numeric_df = casted.with_columns(
        [pl.col(column).fill_null(0.0) for column in sample_columns]
    )
    numeric_df = numeric_df.with_columns(pl.col(id_column).cast(pl.String))
    return numeric_df, id_column, sample_columns


def infer_table_type(values: np.ndarray, table_type: str) -> str:
    """Infer whether values are binary or continuous."""
    if table_type != "auto":
        return table_type
    unique_values = np.unique(values)
    if np.isin(unique_values, np.array([0.0, 1.0])).all():
        return "presence-absence"
    return "abundance"


def rank_vector(values: np.ndarray) -> np.ndarray:
    """Compute average-tie ranks for one vector."""
    n_values = values.size
    sorted_order = np.argsort(values, kind="mergesort")
    sorted_values = values[sorted_order]
    ranks = np.empty(n_values, dtype=np.float64)

    start = 0
    while start < n_values:
        stop = start + 1
        while stop < n_values and sorted_values[stop] == sorted_values[start]:
            stop += 1
        rank_value = (start + stop - 1) / 2.0 + 1.0
        ranks[sorted_order[start:stop]] = rank_value
        start = stop
    return ranks


def rank_matrix_rows(values: np.ndarray) -> np.ndarray:
    """Rank transform each row of a 2D matrix."""
    ranked = np.empty_like(values, dtype=np.float64)
    for idx in range(values.shape[0]):
        ranked[idx, :] = rank_vector(values[idx, :])
    return ranked


def build_pair_frame(
    contig_ids: np.ndarray,
    matrix_values: np.ndarray,
    threshold: float,
    metric_column: str,
) -> pl.DataFrame:
    """Build long-form pair table from a symmetric matrix."""
    row_idx, col_idx = np.where(np.tril(matrix_values, k=-1) >= threshold)
    if row_idx.size == 0:
        return pl.DataFrame(
            {
                "contig_1": pl.Series([], dtype=pl.String),
                "contig_2": pl.Series([], dtype=pl.String),
                metric_column: pl.Series([], dtype=pl.Float64),
            }
        )

    return pl.DataFrame(
        {
            "contig_1": contig_ids[row_idx],
            "contig_2": contig_ids[col_idx],
            metric_column: matrix_values[row_idx, col_idx],
        }
    ).sort(["contig_1", "contig_2"])


def build_groups(
    contig_ids: np.ndarray,
    edge_frame: pl.DataFrame,
) -> pl.DataFrame:
    """Create connected components from undirected edges."""
    if edge_frame.is_empty():
        return pl.DataFrame(
            {
                "group_id": pl.Series([], dtype=pl.String),
                "member_count": pl.Series([], dtype=pl.Int64),
                "members": pl.Series([], dtype=pl.String),
            }
        )

    parent: dict[str, str] = {contig_id: contig_id for contig_id in contig_ids}

    def find_root(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union_nodes(node_a: str, node_b: str) -> None:
        root_a = find_root(node_a)
        root_b = find_root(node_b)
        if root_a != root_b:
            parent[root_b] = root_a

    for edge in edge_frame.select(["contig_1", "contig_2"]).iter_rows(named=True):
        union_nodes(edge["contig_1"], edge["contig_2"])

    component_map: dict[str, list[str]] = {}
    for contig_id in contig_ids:
        root = find_root(contig_id)
        component_map.setdefault(root, []).append(contig_id)

    members = [sorted(component) for component in component_map.values()]
    members = [component for component in members if len(component) > 1]
    members.sort(key=lambda x: (-len(x), x[0]))

    return pl.DataFrame(
        {
            "group_id": [f"group_{idx + 1}" for idx in range(len(members))],
            "member_count": [len(component) for component in members],
            "members": [";".join(component) for component in members],
        }
    )


@click.command(short_help="Find co-occurring contigs across samples")
@click.option(
    "-i",
    "--input",
    required=True,
    type=click.Path(path_type=Path),
    help="Input table (contig IDs x sample IDs) with presence/absence or abundance values",
)
@click.option(
    "-o",
    "--output-prefix",
    default=lambda: f"{Path.cwd()}/correlate",
    show_default=True,
    help="Output file prefix",
)
@click.option(
    "-m",
    "--mode",
    type=click.Choice(["correlation", "cooccurrence", "both"]),
    default="both",
    show_default=True,
    help="Analysis mode",
)
@click.option(
    "--method",
    type=click.Choice(["spearman", "pearson"]),
    default="spearman",
    show_default=True,
    help="Correlation method used in correlation mode",
)
@click.option(
    "--table-type",
    type=click.Choice(["auto", "presence-absence", "abundance"]),
    default="auto",
    show_default=True,
    help="Input value type",
)
@click.option(
    "--min-prevalence",
    type=float,
    default=0.1,
    show_default=True,
    help="Minimum fraction of samples where contig must be present",
)
@click.option(
    "--min-correlation",
    type=float,
    default=0.5,
    show_default=True,
    help="Minimum correlation threshold for keeping contig pairs",
)
@click.option(
    "--min-shared-samples",
    type=int,
    default=1,
    show_default=True,
    help="Minimum number of shared-present samples for permissive co-occurrence",
)
@click.option(
    "--presence-threshold",
    type=float,
    default=0.0,
    show_default=True,
    help="Values greater than this threshold count as present",
)
@click.option(
    "--separator",
    type=click.Choice(["auto", ",", "\t"]),
    default="auto",
    show_default=True,
    help="Input delimiter",
)
@click.option("-t", "--threads", default=1, show_default=True, help="Number of threads")
@click.option(
    "-g",
    "--log-file",
    default=lambda: f"{Path.cwd()}/correlate_logfile.txt",
    help="Path to log file",
)
@click.option(
    "-ll", "--log-level", hidden=True, default="INFO", help="Log level"
)
def correlate(
    input,
    output_prefix,
    mode,
    method,
    table_type,
    min_prevalence,
    min_correlation,
    min_shared_samples,
    presence_threshold,
    separator,
    threads,
    log_file,
    log_level,
):
    """Correlate and cluster contigs by cross-sample co-occurrence patterns."""
    logger = setup_logging(log_file, log_level)
    log_start_info(logger, locals())

    if not (0.0 <= min_prevalence <= 1.0):
        raise click.ClickException("--min-prevalence must be between 0 and 1")
    if not (-1.0 <= min_correlation <= 1.0):
        raise click.ClickException("--min-correlation must be between -1 and 1")
    if min_shared_samples < 1:
        raise click.ClickException("--min-shared-samples must be >= 1")
    if threads < 1:
        raise click.ClickException("--threads must be >= 1")

    if threads > 1:
        logger.info(
            "threads=%s provided; current implementation uses vectorized single-process operations",
            threads,
        )

    input_path = Path(input)
    output_base = Path(output_prefix)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    df, id_column, sample_columns = parse_abundance_table(
        input_path,
        separator,
        logger,
    )

    contig_ids = df[id_column].to_numpy()
    values = df.select(sample_columns).to_numpy().astype(np.float64)

    inferred_table_type = infer_table_type(values, table_type)
    logger.info("Using table type: %s", inferred_table_type)

    presence_matrix = (values > presence_threshold).astype(np.float64)
    prevalence = presence_matrix.mean(axis=1)
    keep_mask = prevalence >= min_prevalence

    if keep_mask.sum() < 2:
        logger.warning(
            "Not enough contigs passed prevalence filtering (%s). Writing empty outputs.",
            min_prevalence,
        )
        empty_edges = pl.DataFrame(
            {
                "contig_1": pl.Series([], dtype=pl.String),
                "contig_2": pl.Series([], dtype=pl.String),
            }
        )
        empty_groups = pl.DataFrame(
            {
                "group_id": pl.Series([], dtype=pl.String),
                "member_count": pl.Series([], dtype=pl.Int64),
                "members": pl.Series([], dtype=pl.String),
            }
        )
        empty_edges.write_csv(
            output_base.with_suffix(".selected_pairs.tsv"),
            separator="\t",
        )
        empty_groups.write_csv(
            output_base.with_suffix(".groups.tsv"),
            separator="\t",
        )
        return

    contig_ids = contig_ids[keep_mask]
    values = values[keep_mask, :]
    presence_matrix = presence_matrix[keep_mask, :]

    logger.info(
        "Prevalence filter retained %s contigs out of %s",
        values.shape[0],
        df.height,
    )

    correlation_pairs = pl.DataFrame()
    if mode in {"correlation", "both"}:
        corr_input = values
        if inferred_table_type == "presence-absence":
            corr_input = presence_matrix

        if method == "spearman":
            corr_input = rank_matrix_rows(corr_input)

        correlation_matrix = np.corrcoef(corr_input)
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=-1.0)
        np.fill_diagonal(correlation_matrix, 1.0)

        correlation_pairs = build_pair_frame(
            contig_ids,
            correlation_matrix,
            min_correlation,
            "correlation",
        )
        correlation_pairs.write_csv(
            output_base.with_suffix(".correlation_pairs.tsv"),
            separator="\t",
        )
        logger.info(
            "Wrote %s correlation-supported pairs",
            correlation_pairs.height,
        )

    cooccurrence_pairs = pl.DataFrame()
    if mode in {"cooccurrence", "both"}:
        shared_matrix = presence_matrix.astype(np.int32) @ presence_matrix.astype(np.int32).T
        cooccurrence_pairs = build_pair_frame(
            contig_ids,
            shared_matrix.astype(np.float64),
            float(min_shared_samples),
            "shared_samples",
        ).with_columns(pl.col("shared_samples").cast(pl.Int64))
        cooccurrence_pairs.write_csv(
            output_base.with_suffix(".cooccurrence_pairs.tsv"),
            separator="\t",
        )
        logger.info(
            "Wrote %s permissive co-occurrence pairs",
            cooccurrence_pairs.height,
        )

    if mode == "correlation":
        selected_pairs = correlation_pairs
    elif mode == "cooccurrence":
        selected_pairs = cooccurrence_pairs
    else:
        selected_pairs = pl.concat(
            [
                correlation_pairs.select(["contig_1", "contig_2"]),
                cooccurrence_pairs.select(["contig_1", "contig_2"]),
            ],
            how="vertical_relaxed",
        ).unique()

    selected_pairs.write_csv(
        output_base.with_suffix(".selected_pairs.tsv"),
        separator="\t",
    )

    groups = build_groups(contig_ids, selected_pairs)
    groups.write_csv(output_base.with_suffix(".groups.tsv"), separator="\t")

    logger.info("Wrote %s groups", groups.height)
    logger.info("Finished correlate")


if __name__ == "__main__":
    correlate()
