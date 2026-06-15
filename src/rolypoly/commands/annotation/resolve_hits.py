"""Optional Click wrapper around consolidate_hits for manual/dev use.

Not registered in the public rolypoly CLI. Overlap resolution is exposed via
--resolve-mode on marker-search, annotate-prot, and annotate-rna instead.
"""

from typing import Optional, Union

import polars as pl
import rich_click as click

from rolypoly.utils.bio.interval_ops import consolidate_hits
from rolypoly.utils.logging.loggit import setup_logging


def infer_column_specs(input_df: pl.DataFrame, column_specs: str) -> str:
    """Infer query/target ID columns when legacy defaults do not fit the input schema."""
    if column_specs != "qseqid,sseqid":
        return column_specs

    cols = set(input_df.columns)
    if {"qseqid", "sseqid"}.issubset(cols):
        return column_specs
    if {"query_full_name", "hmm_full_name"}.issubset(cols):
        return "query_full_name,hmm_full_name"
    if {"query", "target"}.issubset(cols):
        return "query,target"
    return column_specs


def rank_columns_exist(input_df: pl.DataFrame, rank_columns: str) -> bool:
    """Validate that all rank columns are present in the input dataframe."""
    cols = set(input_df.columns)
    requested_cols = {
        token.strip().lstrip("+-")
        for token in rank_columns.split(",")
        if token.strip()
    }
    return requested_cols.issubset(cols)


def infer_rank_columns(input_df: pl.DataFrame, rank_columns: str) -> str:
    """Pick robust ranking defaults when legacy '-score' is not available."""
    if rank_columns != "-score":
        return rank_columns

    if rank_columns_exist(input_df, rank_columns):
        return rank_columns

    rank_fallbacks = [
        "-full_hmm_score,+full_hmm_evalue,-this_dom_score",
        "-this_dom_score,+full_hmm_evalue",
        "-bitscore,+evalue",
        "-full_hmm_score",
        "-this_dom_score",
        "+full_hmm_evalue",
    ]

    for candidate in rank_fallbacks:
        if rank_columns_exist(input_df, candidate):
            return candidate

    return rank_columns


@click.command(name="resolve-overlaps")
@click.option("-i", "--input", required=True, help="Input hit table file")
@click.option(
    "-o",
    "--output",
    default=None,
    help="Output consolidated hit table file (if no output file is will be piped to std.out",
)
@click.option(
    "-rc",
    "--rank-columns",
    type=str,
    default="-score",
    help="Column to use for ranking hits. Can be multiple columns separated by commas, each prefixed with - or + to sort in descending or ascending order, respectively. e.g. '-score,+pident'",
)
@click.option(
    "-cs",
    "--column-specs",
    type=str,
    default="qseqid,sseqid",
    help="Comma separated list of the field names from the input to use for the 'query_id,target_id'",
)
@click.option(
    "-n",
    "--min-overlap-positions",
    default=3,
    type=int,
    help="Minimum number of positions two hits must share to be considered overlapping.",
)
@click.option(
    "-opq",
    "--one-per-query",
    default=False,
    is_flag=True,
    help="Keep one hit per query",
)
@click.option(
    "-opr",
    "--one-per-range",
    default=False,
    is_flag=True,
    help="Keep one hit per range (start-end)",
)
@click.option(
    "-d",
    "-drop",
    "--drop-contained",
    default=False,
    is_flag=True,
    help="Drop hits that are contained within other hits",
)
@click.option(
    "-s",
    "--split",
    default=False,
    is_flag=True,
    help="Split every pair of overlapping hit",
)
@click.option(
    "-m",
    "--merge",
    default=False,
    is_flag=True,
    help="Merge overlapping domains/profiles hits into one - not recommended unless the profiles are from the same functional family",
)
@click.option(
    "-ll", "--log-level", hidden=True, default="INFO", help="Log level"
)
def consolidate_hits_rich(
    input: Union[str, pl.DataFrame],
    output: Optional[str],
    rank_columns: str,
    one_per_query: bool,
    one_per_range: bool,
    min_overlap_positions: int,
    merge: bool,
    column_specs: str,
    drop_contained: bool,
    split: bool,
    log_level: str,
):
    """Resolve overlaps in a hit table using various strategies.

    This command provides multiple strategies for resolving overlapping hits in
    a tabular hit file, such as keeping one hit per query, merging overlaps,
    or splitting overlapping regions.

    Args:
        input (Union[str, pl.DataFrame]): Input hit table file or Polars DataFrame
        output (str, optional): Output file path. If None, returns DataFrame.
        rank_columns (str): Columns for ranking hits with sort direction prefix
        one_per_query (bool): Keep only the best hit per query
        one_per_range (bool): Keep only the best hit per range
        min_overlap_positions (int): Minimum overlap to consider
        merge (bool): Merge overlapping hits
        column_specs (str): Column names for query and target IDs
        drop_contained (bool): Remove hits contained within others
        split (bool): Split overlapping hits

    Returns:
        Optional[pl.DataFrame]: Processed DataFrame if no output file specified

    Example:
             consolidate_hits_rich(
                 "hits.tsv",
                 "resolved.tsv",
                 rank_columns="-score,+evalue",
                 one_per_query=True
             )
    """
    from sys import stdout

    setup_logging(None, log_level)

    input_df = pl.read_csv(input, separator="\t") if isinstance(input, str) else input

    inferred_column_specs = infer_column_specs(input_df, column_specs)
    inferred_rank_columns = infer_rank_columns(input_df, rank_columns)

    if inferred_column_specs != column_specs:
        click.echo(
            f"[resolve-overlaps] Auto-detected column specs: '{inferred_column_specs}'",
            err=True,
        )
    if inferred_rank_columns != rank_columns:
        click.echo(
            f"[resolve-overlaps] Auto-detected rank columns: '{inferred_rank_columns}'",
            err=True,
        )

    if output is None:
        output = stdout
    tmpdf = consolidate_hits(
        input=input_df,
        # output=output,
        rank_columns=inferred_rank_columns,
        one_per_query=one_per_query,
        one_per_range=one_per_range,
        min_overlap_positions=min_overlap_positions,
        merge=merge,
        column_specs=inferred_column_specs,
        drop_contained=drop_contained,
        split=split,
    )

    tmpdf.write_csv(output, separator="\t")


# TODO: add tests to src/../tests/
