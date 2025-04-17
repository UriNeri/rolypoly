import os
import warnings

import polars as pl
import rich_click as click

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="numpy",
)  # see https://moyix.blogspot.com/2022/09/someones-been-messing-with-my-subnormals.html
from typing import List, Optional, Tuple, Union

import intervaltree as itree
from genomicranges import GenomicRanges
from iranges import IRanges
from rolypoly.utils.various import order_columns_to_match, vstack_easy, cast_cols_to_match, order_columns_to_match

# TODO: make this more robust and less dependent on external libraries.


@click.command(name="resolve_overlaps")
@click.option(
    "-i", "--input", required=True, help="Input hit table file or polars dataframe"
)
@click.option(
    "-o",
    "--output",
    default=None,
    help="Output consolidated hit table file (if no output file is specified, the hits will be returned as a polars dataframe)",
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
    tmpdf = consolidate_hits(
        input=input,
        output=output,
        rank_columns=rank_columns,
        one_per_query=one_per_query,
        one_per_range=one_per_range,
        min_overlap_positions=min_overlap_positions,
        merge=merge,
        column_specs=column_specs,
        drop_contained=drop_contained,
        split=split,
    )
    if output is not None:
        tmpdf.write_csv(output, separator="\t")
    else:
        return tmpdf


def consolidate_hits(
    input: Union[str, pl.DataFrame],
    rank_columns: str = "-score",
    one_per_query: bool = False,
    one_per_range: bool = False,
    min_overlap_positions: int = 1,
    merge: bool = False,
    column_specs: str = "qseqid,sseqid",
    drop_contained: bool = False,
    split: bool = False,
):
    """Resolves overlaps in a tabular hit table file or polars dataframe.
    Notes: some flags are mutually exclusive, e.g. you cannot set both split and merge, or rather - if you do that, you'll get unexpected results."""

    # Read the input hit table
    hit_table = pl.read_csv(input, separator="\t") if isinstance(input, str) else input
    og_cols = hit_table.columns

    work_table = hit_table.clone().unique()

    # Parse column specs and rank columns
    query_id_col, target_id_col = column_specs.split(",")
    rank_list, rank_order = parse_rank_columns(rank_columns)

    # Rename rank columns for internal use
    settetet = set(rank_list).difference(set(work_table.columns))
    if len(settetet) > 0:
        print(
            f"Warning: the following rank columns were not found in the input dataframe and will be ignored: {settetet}"
        )
        # breakpoint()
    work_table, rank_list_renamed = rename_rank_columns(work_table, rank_list)

    # Get column names for overlap resolution
    q1_col, q2_col = get_column_names(
        work_table
    )  # p1_col, p2_col, qlen_col, tlen_col etc are not needed for overlap resolution

    # Sort the dataframe
    work_table = sort_hit_table(work_table, query_id_col, rank_list_renamed, rank_order)

    # First is the easiest implementation, culling by one per query - doing this here as it relies on the sort.
    if one_per_query:
        work_table_culled = work_table.group_by(query_id_col).first()
        work_table_culled = work_table_culled.rename(
            {rank_list_renamed[i]: rank_list[i] for i in range(len(rank_list))}
        )
        return work_table_culled.select(og_cols)

    # cast coordinates to int64
    work_table = work_table.with_columns(
        pl.col(q1_col).cast(pl.Int64), pl.col(q2_col).cast(pl.Int64)
    )

    # Add width
    work_table = work_table.with_columns(
        (pl.col(q2_col).cast(pl.Int64) - pl.col(q1_col).cast(pl.Int64)).alias("width")
    )

    # Add unique identifier for each range
    work_table = work_table.with_row_index(name="uid")

    if split:
        work_table = work_table.with_columns(
            pl.col(q1_col).alias("start"), pl.col(q2_col).alias("end")
        )
        print("Splitting overlapping hits")
        work_table = clip_overlapping_ranges_pl(
            input_df=work_table, min_overlap=min_overlap_positions, id_col="uid"
        )
        work_table = work_table.rename(
            {rank_list_renamed[i]: rank_list[i] for i in range(len(rank_list))}
        )
        # breakpoint()
        return work_table.select(og_cols)

    # drop contained hits
    if drop_contained:
        print("Dropping contained hits")
        grouped_by_query = work_table.group_by(query_id_col)
        subdfs = []
        for _, subdf in grouped_by_query:
            subdf = subdf.select(query_id_col, q1_col, q2_col, "uid").rename(
                {q1_col: "start", q2_col: "end"}
            )
            subdf = drop_all_contained_intervals_pl(subdf)
            subdfs.append(subdf)
        tmp_concat = pl.concat(subdfs)
        work_table_culled = work_table.filter(
            pl.col("uid").is_in(tmp_concat.get_column("uid"))
        )
        work_table_culled = work_table_culled.rename(
            {rank_list_renamed[i]: rank_list[i] for i in range(len(rank_list))}
        )
        return work_table_culled.select(og_cols)

    # one-per-range
    if one_per_range:
        print("Dropping to best hit per range")
        # Converted to GenomicRanges
        gr_hits = GenomicRanges(
            seqnames=work_table.get_column(query_id_col),
            names=work_table.get_column("uid"),
            ranges=IRanges(
                start=work_table.get_column(q1_col).cast(pl.Int32),
                width=work_table.get_column("width"),
            ),
        )

        # Find overlaps
        overlapping_hits = gr_hits.find_overlaps(
            gr_hits, min_overlap=min_overlap_positions, select="first", query_type="any"
        )

        # Get unique intervals
        unique_hits = list(set(overlapping_hits))
        work_table_culled = work_table.filter(
            pl.col("uid").cast(pl.Utf8).is_in(unique_hits)
        ).unique(subset="uid")

        work_table_culled = work_table_culled.rename(
            {rank_list_renamed[i]: rank_list[i] for i in range(len(rank_list))}
        )
        return work_table_culled.select(og_cols)

    # merge overlapping hits into one
    if merge:
        # negate the values of a rank column who's ordered for descending columns
        for col_indx, is_descending in enumerate(rank_order):
            if is_descending:
                work_table = work_table.with_columns(
                    (pl.col(rank_list_renamed[col_indx]) * -1).alias(
                        rank_list_renamed[col_indx]
                    )
                )

        # Sort the dataframe by query, position, and rank columns
        sort_columns = [query_id_col, q1_col, q2_col] + rank_list_renamed
        sort_descending = [False, False, False] + [
            False for _ in range(len(rank_order))
        ]
        work_table = work_table.sort(sort_columns, descending=sort_descending)

        work_table = work_table.select(
            pl.col(query_id_col).cast(pl.Utf8).alias("seqnames"),
            pl.col(target_id_col).cast(pl.Utf8),
            pl.col(q1_col).cast(pl.Int64).alias("start"),
            pl.col(q2_col).cast(pl.Int64).alias("end"),
            *[pl.col(rank_col) for rank_col in rank_list_renamed],
        )

        # Convert to GenomicRanges for merging
        gr_hits = GenomicRanges(
            seqnames=work_table.get_column("seqnames"),
            ranges=IRanges(
                start=work_table.get_column("start"),
                width=work_table.get_column("end") - work_table.get_column("start") + 1,
            ),
        )

        # Merge overlapping intervals
        merged_ranges = gr_hits.merge(min_overlap=min_overlap_positions)

        # Process merged intervals
        results = []
        for row in merged_ranges.iter_rows():
            hits_in_cluster = work_table.filter(
                pl.col("seqnames") == row[0],
                pl.col("start") >= row[1],
                pl.col("end") <= row[2],
            )
            merged_hit = hits_in_cluster.group_by(["seqnames"]).agg(
                pl.col(target_id_col).first().alias(target_id_col),
                pl.col("start").cast(pl.Int64).min().alias("start"),
                pl.col("end").cast(pl.Int64).max().alias("end"),
                *[pl.col(rank_col).first() for rank_col in rank_list_renamed],
            )
            merged_hit = merged_hit.select(pl.col(col) for col in merged_hit.columns)
            results.append(merged_hit)

        resolved_hits = pl.concat(results)
        resolved_hits = convert_back_columns(
            resolved_hits,
            rank_list,
            rank_list_renamed,
            rank_order,
            query_id_col,
            q1_col,
            q2_col,
        )
        resolved_hits = resolved_hits.join(
            hit_table.drop(rank_list),
            on=[query_id_col, target_id_col, q1_col, q2_col],
            how="left",
        )
        return resolved_hits.select(og_cols)


# TODO: finish implementing functionaliy, write tests and examples.


def interval_tree_from_df(df: pl.DataFrame, data_col: str = "id") -> itree.IntervalTree:
    """Create an interval tree from a Polars DataFrame.

    Args:
        df (pl.DataFrame): DataFrame with 'start' and 'end' columns
        data_col (str, optional): Column to use as interval data.

    Returns:
        itree.IntervalTree: Interval tree containing the intervals

    """
    tree = itree.IntervalTree()
    for row in df.iter_rows(named=True):
        tree.addi(begin=row["start"], end=row["end"], data=row[data_col])
    return tree


def interval_tree_to_df(tree: itree.IntervalTree) -> pl.DataFrame:
    """Convert an interval tree to a Polars DataFrame.

    Args:
        tree (itree.IntervalTree): Interval tree to convert

    Returns:
        pl.DataFrame: DataFrame with 'start', 'end', and 'id' columns

    """
    return pl.DataFrame(
        {
            "start": [interval.begin for interval in tree],
            "end": [interval.end for interval in tree],
            "id": [interval.data for interval in tree],
        }
    )


def clip_intersecting_ranges(
    range1: list[int, int], range2: list[int, int]
) -> list[int, int]:
    """
    Clip the overlap of two ranges

    :param range1: First range as a tuple (start, end)
    :param range2: Second range as a tuple (start, end)
    :return: updated a tuple of format (start,end) with range1 clipped to remove overlaps with range2.
    Note that if range1 is nested or identical to range2, range1 is returned empty.
    If range2 is nested in range1, up to two empty ranges are returned.
    """
    print(type(range1))
    tree = itree.IntervalTree()
    tree.addi(begin=range1[0], end=range1[1], data="range1")
    tree.chop(range2[0], range2[1])
    return [(start, end) for start, end, data in tree]


def clip_overlapping_ranges_pl(
    input_df: pl.DataFrame, min_overlap: int = 0, id_col: Optional[str] = None
) -> pl.DataFrame:
    """
    Clip overlapping ranges in a polars dataframe.

    :param df: A polars DataFrame with 'start' and 'end' columns. Asummes the df is sorted by some rank columns.
    :param min_overlap: Minimum overlap to consider for clipping
    :return: A DataFrame with clipped ranges. The start and end of the ranges are updated to remove the overlap, so that the first range (i.e. index of it is lower) is the one that is the one not getting clipped, and other are trimmed to not overlap with it.
    """
    
    
    
    df = get_all_overlaps_pl(input_df, min_overlap=min_overlap, id_col=id_col)  # type: ignore
    df = df.with_columns(
        pl.col("overlapping_intervals").list.len().alias("n_overlapping")
    )
    subset_df = df.filter(pl.col("n_overlapping") == 1)
    rest_df = df.filter(pl.col("n_overlapping") > 1)
    tree = itree.IntervalTree()

    for i in range(1, 10):  # (that's the max number of iterations we'll allow
        for row in rest_df.iter_rows(named=True):
            if len(row["overlapping_intervals"]) > 1:
                # get intervals that overlaps this row and isn't self
                all_ovl = [
                    ovl
                    for ovl in row["overlapping_intervals"]
                    if ovl != row[id_col] and ovl not in subset_df[id_col].to_list()
                ]
                all_ovl_df = df.filter(pl.col(id_col).is_in(all_ovl))
                tree = interval_tree_from_df(all_ovl_df, data_col=id_col)
                tree.chop(row["start"], row["end"])
                tree_df = interval_tree_to_df(tree)
                tree_df = tree_df.rename({"id": id_col})
                tree_df = tree_df.with_columns(
                    pl.col(id_col).cast(subset_df.get_column(id_col).dtype)
                )
                tree_df = tree_df.join(
                    rest_df.drop(["start", "end"]), on=id_col, how="left"
                )
                # breakpoint()
                subset_df = vstack_easy(subset_df, tree_df)
    return vstack_easy(subset_df, rest_df)

def get_all_envelopes_pl(
    input_df: pl.DataFrame, id_col: Optional[str] = None
) -> pl.DataFrame:
    """
    Given a polars dataframe with start and end columns, return a dataframe with an additional column
    containing lists of indices of the entries that envelope each row.

    :param df: A polars DataFrame with 'start' and 'end' columns
    :param id_col: The column name to use for the interval IDs (if none, will use row index). Should not have duplicates.
    :return: A DataFrame with an additional 'enveloping_intervals' column, containing lists of ids (if id_col is provided) or row indices of the entries that envelope this row's start and end.
    """
    b = False
    if id_col is None:
        input_df = input_df.with_row_index(name="intops_id")
        id_col = "intops_id"
        b = True

    typeof_id_col = type(input_df.select(pl.col(id_col)).to_series().to_list()[0])

    df = input_df.select(
        [
            pl.all(),
            pl.struct(["start", "end"])
            .map_elements(
                lambda x: input_df.filter(
                    (pl.col("start") <= x["start"]) & (pl.col("end") >= x["end"])
                )
                .select(pl.col(id_col))
                .to_series()
                .to_list(),
                return_dtype=pl.List(typeof_id_col),
            )
            .alias("enveloping_intervals"),
        ]
    )
    out_df = input_df.join(df[[id_col, "enveloping_intervals"]], on=id_col, how="left")
    if b:
        out_df = out_df.drop(id_col)
    return out_df


def drop_all_contained_intervals_pl(input_df: pl.DataFrame) -> pl.DataFrame:
    """Remove intervals that are completely contained within other intervals.

    Args:
        input_df (pl.DataFrame): DataFrame with 'start' and 'end' columns

    Returns:
        pl.DataFrame: DataFrame with contained intervals removed
    """
    id_col = "daci_pl"
    input_df = input_df.with_row_index(name=id_col)
    df = input_df.filter(pl.col("start") != pl.col("end"))  # points throw errors
    tree = interval_tree_from_df(df, data_col=id_col)
    bla = tree.find_nested()
    containedd = [interval.data for intervals in bla.values() for interval in intervals]
    df = input_df.filter(~pl.col(id_col).is_in(containedd)).drop(id_col)
    return df


def get_all_overlaps_pl(
    input_df: pl.DataFrame, min_overlap: int = 0, id_col: Optional[str] = None
) -> pl.DataFrame:
    """Find all overlapping intervals in a DataFrame.

    Args:
        input_df (pl.DataFrame): DataFrame with 'start' and 'end' columns
        min_overlap (int, optional): Minimum overlap required.
        id_col (str, optional): Column to use as interval ID.

    Returns:
        pl.DataFrame: Input DataFrame with added 'overlapping_intervals' column

    Example:
        ```python
        df = pl.DataFrame({
            "start": [1, 5, 10],
            "end": [6, 8, 15],
            "id": ["a", "b", "c"]
        })
        result = get_all_overlaps_pl(df, min_overlap=2)
        ```
    """
    if id_col is None:
        input_df = input_df.with_row_index(name="intops_id")
        id_col = "intops_id"

    typeof_id_col = type(input_df.select(pl.col(id_col)).to_series().to_list()[0])

    tree = interval_tree_from_df(input_df, data_col=id_col)
    ovl_intervals = [[] for _ in range(len(input_df))]
    for indx, row in enumerate(input_df.iter_rows(named=True)):
        overlapping = tree.overlap(begin=row["start"], end=row["end"])
        for ovl in overlapping:
            if ovl.overlap_size(begin=row["start"], end=row["end"]) >= min_overlap:
                ovl_intervals[indx].append(ovl.data)

    return input_df.with_columns(
        pl.Series(name="overlapping_intervals", values=ovl_intervals, strict=False)
    )


# def merge_all_overlapping_intervals(input_df: pl.DataFrame, min_overlap: int = 0) -> pl.DataFrame:
#     """
#     Merge overlapping regions using Polars.

#     :param input_df: A polars DataFrame with 'start' and 'end' columns
#     :return: A DataFrame with merged ranges
#     """
#     df = get_all_envelopes_pl(input_df)
#     df = df.with_columns(
#         pl.col("enveloped_intervals").list.len().alias("n_enveloped")
#     )
#     df = df.filter(
#         pl.col("n_enveloped") < 2
#     )

#     tree.merge_overlaps()
#     merged_df = interval_tree_to_df(tree)


def return_or_write(df: pl.DataFrame, output: Optional[str]):
    """Write output or return dataframe."""
    if isinstance(output, str):
        df.write_csv(output, separator="\t")
        exit()
    else:
        return df  # pragma: no cover


def parse_rank_columns(rank_columns: str) -> Tuple[List[str], List[bool]]:
    """Parse rank columns string into list of column names and sort orders."""
    rank_list = [col.strip()[1:] for col in rank_columns.split(",")]
    rank_order = [False if col[0] == "+" else True for col in rank_columns.split(",")]
    return rank_list, rank_order


def rename_rank_columns(
    df: pl.DataFrame, rank_list: List[str]
) -> Tuple[pl.DataFrame, List[str]]:
    """Rename rank columns for internal use."""
    rank_list_renamed = [f"ranker_{i + 1}" for i in range(len(rank_list))]
    rename_dict = {old: new for old, new in zip(rank_list, rank_list_renamed)}
    return df.rename(rename_dict), rank_list_renamed


def convert_back_columns(
    df: pl.DataFrame,
    rank_list: List[str],
    rank_list_renamed: List[str],
    rank_order: List[bool],
    query_id_col: str,
    q1_col: str,
    q2_col: str,
) -> pl.DataFrame:
    """Rename the rank columns back to the original names."""
    rename_dict = {old: new for old, new in zip(rank_list_renamed, rank_list)}
    rename_dict["Chromosome"] = query_id_col
    rename_dict["Start"] = q1_col
    rename_dict["End"] = q2_col
    df = df.rename(rename_dict)
    for col_indx, is_descending in enumerate(rank_order):
        if not is_descending:
            df = df.with_columns(
                (pl.col(rank_list[col_indx]) * -1).alias(rank_list[col_indx])
            )
    return df


def sort_hit_table(
    input_df: pl.DataFrame,
    query_id_col: str,
    rank_list_renamed: List[str],
    rank_order: List[bool],
) -> pl.DataFrame:
    """Sort the hit table by query, position, and rank columns."""
    sort_columns = [query_id_col] + rank_list_renamed  # q1_col, q2_col
    sort_descending = [False] + rank_order
    return input_df.sort(by=sort_columns, descending=sort_descending)


def name_cols_for_gr(
    df: pl.DataFrame, q1_col: str, q2_col: str, query_id_col: str
) -> pl.DataFrame:
    """Name columns for use with genomicranges."""
    rename_dict = {q1_col: "start", q2_col: "end", query_id_col: "seqnames"}
    df = df.with_columns(
        pl.col(q2_col) - pl.col(q1_col).alias("width"),
    )
    return df.rename(rename_dict)


def revert_names_from_gr(
    df: pl.DataFrame, q1_col: str, q2_col: str, query_id_col: str
) -> pl.DataFrame:
    """Revert columns to original names."""
    rename_dict = {"starts": q1_col, "ends": q2_col, "seqnames": query_id_col}
    return df.rename(rename_dict)


def get_column_names(df: pl.DataFrame) -> Tuple[str, str]:
    """Get the column names for various attributes from the dataframe."""
    column_mapping = {
        "q1_col": ["q1", "start", "qstart", "start_pos", "ali_from"],
        "q2_col": ["q2", "end", "qend", "end_pos", "ali_to"],
        # 'p1_col': ['p1', 'sstart', 'tstart', 'env_from'],
        # 'p2_col': ['p2', 'send', 'tend', 'env_to'],
        # 'ali_len_col': ['ali_len', 'alilen', 'len'],
        # 'qlen_col': ['qlen', 'query_length', 'q_len'],
        # 'tlen_col': ['tlen', 'target_length', 't_len', 's_len', 'slen', 'subject_length']
    }

    result = {}
    for col_type, possible_names in column_mapping.items():
        for name in possible_names:
            if name in df.columns:
                result[col_type] = name
                break
        if col_type not in result:
            raise ValueError(f"Could not find a column for {col_type}")

    return tuple(result.values())


if __name__ == "__main__":
    consolidate_hits()
    # Debug arguments
    debug = False
    if debug:
        input = pl.read_csv(
            "/clusterfs/jgi/scratch/science/metagen/neri/tests/rp_tests/inputs/tables/hit_table.tsv",
            separator="\t",
        )
        output = "/clusterfs/jgi/scratch/science/metagen/neri/tests/rp_tests/inputs/tables/consolidated_output.tsv"
        best = False
        rank_columns = "-score,+evalue"
        column_specs = "qseqid,sseqid"
        culling_mode = "one_per_range"
        env_mode = "envelope"
        # max_overlap_fraction=0.1
        min_overlap_positions = 10
        clip = True
        drop_contained = True
        one_per_query = False
        one_per_range = False
        merge = False
        consolidate_hits(
            input,
            None,
            best,
            rank_columns,
            culling_mode,
            min_overlap_positions,
            clip,
            drop_contained,
            merge,
            column_specs,
        )

    # # Create a sample polars DataFrame
    # data = {
    #     'start': [1, 3, 5, 7, 9, 11],
    #     'end': [4, 6, 8, 10, 14, 13],
    #     'id': ["a", "b", "c", "d", "e", "f"]
    # }
    # input_df = pl.DataFrame(data)

    # df = get_all_overlaps_pl(input_df)
    # print("Overlapping regions:")
    # print(df)

    # print("\nNested regions:")
    # print(get_all_envelopes_pl(df))

    # # Example for clip_overlap
    # value_data = {
    #     'position': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #     'value': [1, 3, 2, 5, 4, 6, 2, 1, 3, 4]
    # }
    # value_df = pl.DataFrame(value_data)
    # print("\nClipped overlap:")
    # print(clip_overlapping_ranges_pl(value_df, min_overlap=2))

    # print("\nMerged regions:")
    # print(merge_all_overlapping_intervals(df))

    # print("\nOverlapping regions with overlap:")
    # print(get_all_overlaps_pl(df, 2))

    # test_df = generate_dummy_hit_table(1011) # type: ignore
    # df1 = sort_hit_table(test_df, "qseqid", ["score"], [True])


# def generate_dummy_hit_table(n: int) -> pl.DataFrame:
#     """
#     Generate a dummy hit table for debugging purposes.

#     Args:
#     n (int): Number of rows in the hit table.

#     Returns:
#     pl.DataFrame: A dummy hit table with n rows.
#     """
#     np.random.seed(42)  # For reproducibility

#     # Generate random data
#     qseqid = [f"query_{i//3}" for i in range(n)]
#     sseqid = [f"target_{i}" for i in range(n)]
#     qstart = np.random.randint(1, 1000, n)
#     qend = qstart + np.random.randint(10, 200, n)
#     score = np.random.randint(50, 1000, n)
#     evalue = np.random.uniform(1e-10, 1e-2, n)

#     # Create the DataFrame
#     df = pl.DataFrame({
#         "qseqid": qseqid,
#         "sseqid": sseqid,
#         "qstart": qstart,
#         "qend": qend,
#         "score": score,
#         "evalue": evalue
#     })

#     # Sort by qseqid, qstart
#     df = df.sort(["qseqid", "qstart"])

#     return df


# def resolve_overlaps(hit_table: pl.DataFrame, query_id_col: str, q1_col: str, q2_col: str, rank_col: str,
#                      max_overlap_fraction: float, max_overlap_positions: int, clip: bool,
#                      drop_unresolved: bool, merge: bool, culling_mode: str, env_mode: str) -> pl.DataFrame:
#     """Resolve overlaps in the hit table using NumPy."""
#     result = []
#     for query in hit_table[query_id_col].unique():
#         query_df = hit_table.filter(pl.col(query_id_col) == query)

#         # Convert to numpy arrays for efficient operations
#         starts = query_df[q1_col].to_numpy()
#         ends = query_df[q2_col].to_numpy()
#         ranks = query_df[rank_col].to_numpy()

#         # Sort by start position
#         sort_idx = np.argsort(starts)
#         starts = starts[sort_idx]
#         ends = ends[sort_idx]
#         ranks = ranks[sort_idx]

#         # Resolve overlaps
#         resolved_idx = resolve_overlap_numpy(starts, ends, ranks, max_overlap_fraction, max_overlap_positions,
#                                              clip, drop_unresolved, merge, env_mode)

#         # Apply culling mode
#         if culling_mode == "one_per_query":
#             best_hit = np.argmax(ranks[resolved_idx])
#             resolved_idx = resolved_idx[best_hit:best_hit+1]
#         elif culling_mode == "one_per_range":
#             resolved_idx = remove_overlapping_intervals_numpy(starts[resolved_idx], ends[resolved_idx], ranks[resolved_idx])

#         # Add resolved hits to result
#         result.extend(query_df.row(idx) for idx in sort_idx[resolved_idx])

#     return pl.DataFrame(result)

# def resolve_overlap_numpy(starts: np.ndarray, ends: np.ndarray, ranks: np.ndarray,
#                           max_overlap_fraction: float, max_overlap_positions: int,
#                           clip: bool, drop_unresolved: bool, merge: bool, env_mode: str) -> np.ndarray:
#     """Resolve overlaps using NumPy operations."""
#     n = len(starts)
#     keep = np.ones(n, dtype=bool)

#     for i in range(n):
#         if not keep[i]:
#             continue
#         for j in range(i+1, n):
#             if not keep[j]:
#                 continue

#             overlap = min(ends[i], ends[j]) - max(starts[i], starts[j])
#             if overlap <= 0:
#                 break  # No more overlaps possible

#             min_len = min(ends[i] - starts[i], ends[j] - starts[j])
#             if overlap > max_overlap_positions and overlap / min_len > max_overlap_fraction:
#                 if clip:
#                     mid = (max(starts[i], starts[j]) + min(ends[i], ends[j])) // 2
#                     if starts[i] < starts[j]:
#                         ends[i] = mid
#                         starts[j] = mid + 1
#                     else:
#                         ends[j] = mid
#                         starts[i] = mid + 1
#                 elif merge:
#                     starts[i] = min(starts[i], starts[j])
#                     ends[i] = max(ends[i], ends[j])
#                     ranks[i] = max(ranks[i], ranks[j])
#                     keep[j] = False
#                 elif drop_unresolved:
#                     keep[i] = keep[j] = False
#                 elif env_mode == "envelope":
#                     if ends[i] - starts[i] >= ends[j] - starts[j]:
#                         keep[j] = False
#                     else:
#                         keep[i] = False
#                         break
#                 elif env_mode == "letter":
#                     if ends[i] - starts[i] <= ends[j] - starts[j]:
#                         keep[j] = False
#                     else:
#                         keep[i] = False
#                         break

#     return np.where(keep)[0]

# def remove_overlapping_intervals_numpy(starts: np.ndarray, ends: np.ndarray, ranks: np.ndarray) -> np.ndarray:
#     """Remove overlapping intervals, keeping the highest ranking one in each overlap."""
#     sorted_idx = np.argsort(-ranks)  # Sort by descending rank
#     starts = starts[sorted_idx]
#     ends = ends[sorted_idx]

#     keep = np.ones(len(starts), dtype=bool)
#     last_end = 0

#     for i, (start, end) in enumerate(zip(starts, ends)):
#         if start > last_end:
#             last_end = end
#         else:
#             keep[i] = False

#     return sorted_idx[keep]
