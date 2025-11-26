from typing import Optional, Union

import polars as pl
import rich_click as click

from rolypoly.utils.bio.interval_ops import consolidate_hits


@click.command(name="resolve_overlaps")
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

    if output is None:
        output = stdout
    tmpdf = consolidate_hits(
        input=input,
        # output=output,
        rank_columns=rank_columns,
        one_per_query=one_per_query,
        one_per_range=one_per_range,
        min_overlap_positions=min_overlap_positions,
        merge=merge,
        column_specs=column_specs,
        drop_contained=drop_contained,
        split=split,
    )

    tmpdf.write_csv(output, separator="\t")


# TODO: add tests to src/../tests/
