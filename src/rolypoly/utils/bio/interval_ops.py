import warnings

import polars as pl

warnings.filterwarnings(
    "ignore", category=UserWarning, module="numpy"
)  # see https://moyix.blogspot.com/2022/09/someones-been-messing-with-my-subnormals.html
from typing import List, Optional, Tuple, Union

import intervaltree as itree
from genomicranges import GenomicRanges
from iranges import IRanges

from rolypoly.utils.various import vstack_easy

# TODO: make this more robust and less dependent on external libraries. Candidate destination library is polars-bio.


def calculate_adaptive_overlap_threshold(
    ali_len: int, is_polyprotein_like: bool = False
) -> int:
    """Calculate adaptive overlap threshold based on alignment length.

    Args:
        ali_len: Alignment length in amino acids
        is_polyprotein_like: Whether the sequence appears to be a polyprotein

    Returns:
        Overlap threshold in amino acids
    """
    if is_polyprotein_like:
        # Use stricter thresholds for polyproteins to preserve domain boundaries
        if ali_len < 100:
            return 5
        elif ali_len < 200:
            return 8
        elif ali_len < 400:
            return 12
        else:
            return 15
    else:
        # Use more relaxed thresholds for fuzzy domain boundaries
        if ali_len < 100:
            return 10
        elif ali_len < 200:
            return 15
        elif ali_len < 400:
            return 25
        else:
            return 40


def detect_polyprotein_pattern(
    hit_df: pl.DataFrame,
    query_id: str,
    query_id_col: str = "query_full_name",
    target_id_col: str = "hmm_full_name",
    q1_col: str = "ali_from",
    q2_col: str = "ali_to",
    bin_size: int = 50,
) -> bool:
    """Detect if a query sequence has a polyprotein-like pattern.

    Looks for multiple distinct regions with many different profile hits,
    suggesting a polyprotein with multiple domains.

    Args:
        hit_df: DataFrame with hits for a single query
        query_id: The query sequence ID
        query_id_col: Column name for query IDs
        target_id_col: Column name for target profile IDs
        q1_col: Column name for alignment start
        q2_col: Column name for alignment end
        bin_size: Size of bins in amino acids

    Returns:
        True if polyprotein pattern detected
    """
    query_hits = hit_df.filter(pl.col(query_id_col) == query_id)

    if len(query_hits) < 10:  # Need sufficient hits to detect pattern
        return False

    # Get query length
    qlen = query_hits.select(pl.col("qlen").max()).item()
    if qlen < 200:  # Short sequences unlikely to be polyproteins
        return False

    # Create bins
    n_bins = (qlen // bin_size) + 1
    bin_profile_counts = [set() for _ in range(n_bins)]

    # Assign hits to bins based on their midpoint
    for row in query_hits.iter_rows(named=True):
        start = row[q1_col]
        end = row[q2_col]
        midpoint = (start + end) // 2
        bin_idx = min(midpoint // bin_size, n_bins - 1)
        bin_profile_counts[bin_idx].add(row[target_id_col])

    # Count bins with significant profile diversity
    bins_with_hits = [
        len(profiles) for profiles in bin_profile_counts if len(profiles) > 0
    ]

    if len(bins_with_hits) < 2:
        return False

    # Check if there are at least 2 bins with multiple unique profiles
    diverse_bins = sum(1 for count in bins_with_hits if count >= 3)

    # Polyprotein pattern: multiple regions with diverse hits
    return diverse_bins >= 2


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
    alphabet: str = "aa",
    adaptive_overlap: bool = False,
) -> pl.DataFrame:
    """Resolves overlaps in a tabular hit table file or polars dataframe.

    Args:
        input: Input hit table (file path or DataFrame)
        rank_columns: Columns to rank by (prefix with - for descending, + for ascending)
        one_per_query: Keep only one hit per query sequence
        one_per_range: Keep only one hit per overlapping range
        min_overlap_positions: Minimum overlap to consider (used if adaptive_overlap=False)
        merge: Merge overlapping hits
        column_specs: Query and target column names (comma-separated)
        drop_contained: Drop hits contained within other hits
        split: Split overlapping ranges
        alphabet: Sequence alphabet ('aa' for amino acid, 'nucl' for nucleotide)
        adaptive_overlap: Use adaptive overlap thresholds based on alignment length and polyprotein detection

    Returns:
        DataFrame with resolved overlaps

    Notes:
        Some flags are mutually exclusive, e.g. you cannot set both split and merge.
        Adaptive overlap is only applied for amino acid sequences (alphabet='aa').
    """

    # Read the input hit table
    hit_table = (
        pl.read_csv(input, separator="\t") if isinstance(input, str) else input
    )
    og_cols = hit_table.columns

    work_table = hit_table.clone().unique()

    # Apply adaptive overlap thresholds if requested (only for amino acid sequences)
    if adaptive_overlap and alphabet == "aa":
        # Parse column specs to get query and target ID columns
        query_id_col, target_id_col = column_specs.split(",")

        # Detect column names for positions
        q1_col, q2_col = get_column_names(work_table)

        # Detect polyprotein patterns for each query
        unique_queries = (
            work_table.select(query_id_col).unique().to_series().to_list()
        )
        polyprotein_queries = set()

        for query_id in unique_queries:
            if detect_polyprotein_pattern(
                work_table,
                query_id,
                query_id_col=query_id_col,
                target_id_col=target_id_col,
                q1_col=q1_col,
                q2_col=q2_col,
            ):
                polyprotein_queries.add(query_id)

        # Calculate adaptive thresholds for each hit
        def calc_overlap_threshold(row):
            ali_len = row[q2_col] - row[q1_col]
            is_polyprotein = row[query_id_col] in polyprotein_queries
            return calculate_adaptive_overlap_threshold(ali_len, is_polyprotein)

        # Add adaptive threshold column
        work_table = work_table.with_columns(
            pl.struct([query_id_col, q1_col, q2_col])
            .map_elements(calc_overlap_threshold, return_dtype=pl.Int64)
            .alias("adaptive_threshold")
        )

        # Use minimum threshold as conservative baseline for this batch
        min_overlap_positions = work_table.select(
            pl.col("adaptive_threshold").min()
        ).item()

        # Drop the adaptive threshold column before processing
        work_table = work_table.drop("adaptive_threshold")

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
    work_table = sort_hit_table(
        work_table, query_id_col, rank_list_renamed, rank_order
    )

    # First is the easiest implementation, culling by one per query - doing this here as it relies on the sort.
    if one_per_query:
        work_table_culled = work_table.group_by(query_id_col).first()
        work_table_culled = work_table_culled.rename(
            {rank_list_renamed[i]: rank_list[i] for i in range(len(rank_list))}
        )
        return work_table_culled.select(og_cols).unique()

    # cast coordinates to int64
    work_table = work_table.with_columns(
        pl.col(q1_col).cast(pl.Int64), pl.col(q2_col).cast(pl.Int64)
    )

    # Add width
    work_table = work_table.with_columns(
        (pl.col(q2_col).cast(pl.Int64) - pl.col(q1_col).cast(pl.Int64)).alias(
            "width"
        )
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
        return work_table.select(og_cols).unique()

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
        return work_table_culled.select(og_cols).unique()

    # one-per-range
    if one_per_range:
        print("Dropping to best hit per range")

        # Group by query and use interval tree to find non-overlapping hits
        grouped_by_query = work_table.group_by(query_id_col)
        subdfs = []

        for _, subdf in grouped_by_query:
            if len(subdf) == 0:
                continue

            # Create interval tree for this query
            tree = itree.IntervalTree()
            kept_uids = []

            # Sort by rank (already sorted in work_table)
            for row in subdf.iter_rows(named=True):
                start = row[q1_col]
                end = row[q2_col]
                uid = row["uid"]

                # Check if this interval significantly overlaps with any kept interval
                overlaps = tree.overlap(start, end)
                has_significant_overlap = False

                for ovl in overlaps:
                    overlap_size = min(end, ovl.end) - max(start, ovl.begin)
                    if overlap_size >= min_overlap_positions:
                        has_significant_overlap = True
                        break

                # If no significant overlap, keep this hit
                if not has_significant_overlap:
                    tree.addi(start, end, uid)
                    kept_uids.append(uid)

            # Filter to kept UIDs
            if kept_uids:
                subdfs.append(subdf.filter(pl.col("uid").is_in(kept_uids)))

        if subdfs:
            work_table_culled = pl.concat(subdfs)
        else:
            work_table_culled = work_table.head(
                0
            )  # Empty dataframe with same schema

        work_table_culled = work_table_culled.rename(
            {rank_list_renamed[i]: rank_list[i] for i in range(len(rank_list))}
        )
        return work_table_culled.select(og_cols).unique()

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
            seqnames=work_table.get_column("seqnames").to_list(),
            ranges=IRanges(
                start=work_table.get_column("start").to_list(),
                width=(
                    work_table.get_column("end")
                    - work_table.get_column("start")
                    + 1
                ).to_list(),
            ),
        )

        # Merge overlapping intervals
        merged_ranges = gr_hits.find_overlaps(
            query=gr_hits, min_overlap=min_overlap_positions
        ).to_polars()

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
            merged_hit = merged_hit.select(
                pl.col(col) for col in merged_hit.columns
            )
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
        return resolved_hits.select(og_cols).unique()


# TODO: finish implementing functionaliy, write tests and examples.


def interval_tree_from_df(
    df: pl.DataFrame, data_col: str = "id"
) -> itree.IntervalTree:
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
                    if ovl != row[id_col]
                    and ovl not in subset_df[id_col].to_list()  # type: ignore
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

    typeof_id_col = type(
        input_df.select(pl.col(id_col)).to_series().to_list()[0]
    )

    df = input_df.select(
        [
            pl.all(),
            pl.struct(["start", "end"])
            .map_elements(
                lambda x: input_df.filter(
                    (pl.col("start") <= x["start"])
                    & (pl.col("end") >= x["end"])
                )
                .select(pl.col(id_col))
                .to_series()
                .to_list(),
                return_dtype=pl.List(typeof_id_col),
            )
            .alias("enveloping_intervals"),
        ]
    )
    out_df = input_df.join(
        df[[id_col, "enveloping_intervals"]], on=id_col, how="left"
    )
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
    df = input_df.filter(
        pl.col("start") != pl.col("end")
    )  # points throw errors
    tree = interval_tree_from_df(df, data_col=id_col)
    bla = tree.find_nested()
    containedd = [
        interval.data for intervals in bla.values() for interval in intervals
    ]
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

    # typeof_id_col = type(input_df.select(pl.col(id_col)).to_series().to_list()[0])

    tree = interval_tree_from_df(input_df, data_col=id_col)
    ovl_intervals = [[] for _ in range(len(input_df))]
    for indx, row in enumerate(input_df.iter_rows(named=True)):
        overlapping = tree.overlap(begin=row["start"], end=row["end"])
        for ovl in overlapping:
            if (
                ovl.overlap_size(begin=row["start"], end=row["end"])
                >= min_overlap
            ):
                ovl_intervals[indx].append(ovl.data)

    return input_df.with_columns(
        pl.Series(
            name="overlapping_intervals", values=ovl_intervals, strict=False
        )
    )


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
    rank_order = [
        False if col[0] == "+" else True for col in rank_columns.split(",")
    ]
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
    df = df.with_columns(pl.col(q2_col) - pl.col(q1_col).alias("width"))
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


def mask_sequence_mp(seq: str, start: int, end: int, is_reverse: bool) -> str:
    """Mask a portion of a mappy (minimap2) aligned sequence with N's.

    Args:
        seq (str): Input sequence to mask
        start (int): Start position of the region to mask (0-based)
        end (int): End position of the region to mask (exclusive)
        is_reverse (bool): Whether the sequence is reverse complemented

    Returns:
        str: Sequence with the specified region masked with N's

    Note:
        Handles reverse complement if needed by using mappy's revcomp function.
    """
    import mappy as mp

    is_reverse = is_reverse == -1
    if is_reverse:
        seq = str(mp.revcomp(seq))
    masked_seq = seq[:start] + "N" * (end - start) + seq[end:]
    return str(mp.revcomp(masked_seq)) if is_reverse else masked_seq


def mask_nuc_range(
    input_fasta: str, input_table: str, output_fasta: str
) -> None:
    """Mask nucleotide sequences in a FASTA file based on provided range table.

    Args:
        input_fasta (str): Path to the input FASTA file
        input_table (str): Path to the table file with the ranges to mask
            (tab-delimited with columns: seq_id, start, stop, strand)
        output_fasta (str): Path to the output FASTA file

    Note:
        The ranges in the table should be 1-based coordinates.
        Handles both forward and reverse strand masking.
    """
    from .sequences import revcomp

    # Read ranges
    ranges = {}
    with open(input_table, "r") as f:
        for line in f:
            seq_id, start, stop, strand = line.strip().split("\t")
            if seq_id not in ranges:
                ranges[seq_id] = []
            ranges[seq_id].append((int(start), int(stop), strand))

    # Process FASTA file
    with open(input_fasta, "r") as in_f, open(output_fasta, "w") as out_f:
        current_id = ""
        current_seq = ""
        for line in in_f:
            if line.startswith(">"):
                if current_id:
                    if current_id in ranges:
                        for start, stop, strand in ranges[current_id]:
                            if start > stop:
                                start, stop = stop, start
                            if strand == "-":
                                current_seq = revcomp(current_seq)
                            current_seq = (
                                current_seq[: start - 1]
                                + "N" * (stop - start + 1)
                                + current_seq[stop:]
                            )
                            if strand == "-":
                                current_seq = revcomp(current_seq)
                    out_f.write(f">{current_id}\n{current_seq}\n")
                current_id = line[1:].strip()
                current_seq = ""
            else:
                current_seq += line.strip()

        if current_id:
            if current_id in ranges:
                for start, stop, strand in ranges[current_id]:
                    if start > stop:
                        start, stop = stop, start
                    if strand == "-":
                        current_seq = revcomp(current_seq)
                    current_seq = (
                        current_seq[: start - 1]
                        + "N" * (stop - start + 1)
                        + current_seq[stop:]
                    )
                    if strand == "-":
                        current_seq = revcomp(current_seq)
            out_f.write(f">{current_id}\n{current_seq}\n")


def mask_nuc_range_from_sam(
    input_fasta: str, input_sam: str, output_fasta: str
) -> None:
    """Mask nucleotide sequences in a FASTA file based on provided SAM.
        All sequences in ranges from the input fasta that are aligned in the SAM will be replaced by N's.

    Args:
        input_fasta (str): Path to the input FASTA file
        input_sam (str): Path to the SAM file with the alignments to mask
        output_fasta (str): Path to the output FASTA file

    Note:
        Handles both forward and reverse strand masking.
    """
    import re

    from intervaltree import (
        IntervalTree as itree,  # assumed available as in other functions
    )
    from rich.progress import track

    from rolypoly.utils.logging.loggit import get_logger

    from .sequences import revcomp

    logger = get_logger()

    # Parse SAM and collect reference intervals (1-based inclusive, with strand)
    ranges = {}
    with open(input_sam, "r") as f:
        for line in f:
            if line.startswith("@"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            try:
                flag = int(parts[1])
            except Exception:
                continue
            rname = parts[2]
            if rname == "*" or rname == "":
                continue
            try:
                pos = int(parts[3])
            except Exception:
                continue
            cigar = parts[5]
            if cigar == "*":
                continue

            # Compute reference-consuming length from CIGAR
            ref_consumed = 0
            for m in re.finditer(r"(\d+)([MIDNSHP=XB])", cigar):
                clen = int(m.group(1))
                op = m.group(2)
                if op in ("M", "D", "N", "=", "X"):
                    ref_consumed += clen

            if ref_consumed <= 0:
                continue

            start = pos
            stop = pos + ref_consumed - 1
            strand = "-" if (flag & 16) else "+"
            ranges.setdefault(rname, []).append((start, stop, strand))

    # Merge overlapping intervals per reference and strand using intervaltree
    merged = {}
    for rname, lst in ranges.items():
        # Separate by strand
        by_strand = {"+": [], "-": []}
        for s, e, strand in lst:
            by_strand[strand].append((s, e))
        merged[rname] = {}
        for strand, intervals in by_strand.items():
            if not intervals:
                continue
            tree = itree()
            for s, e in intervals:
                tree.addi(s, e + 1)
            tree.merge_overlaps()
            merged_intervals = [(iv.begin, iv.end - 1) for iv in sorted(tree)]
            merged[rname][strand] = merged_intervals

    # Count total records with ranges for progress
    records_with_ranges = [
        rid for rid in merged if merged[rid].get("+") or merged[rid].get("-")
    ]
    total_records = len(records_with_ranges)

    # Read input FASTA and write masked sequences
    with open(input_fasta, "r") as in_f, open(output_fasta, "w") as out_f:
        current_id = ""
        current_seq_parts = []
        processed = 0
        for line in in_f:
            if line.startswith(">"):
                if current_id:
                    seq_str = "".join(current_seq_parts)
                    if current_id in merged:
                        for strand in ("+", "-"):
                            intervals = merged[current_id].get(strand, [])
                            if not intervals:
                                continue
                            # If strand is '-', mask on revcomp, then revcomp back
                            if strand == "+":
                                for s, e in intervals:
                                    if s > e:
                                        s, e = e, s
                                    seq_str = (
                                        seq_str[: s - 1]
                                        + "N" * (e - s + 1)
                                        + seq_str[e:]
                                    )
                            else:
                                # Mask on revcomp
                                rc_seq = revcomp(seq_str)
                                seqlen = len(seq_str)
                                for s, e in intervals:
                                    if s > e:
                                        s, e = e, s
                                    # Convert 1-based coordinates to revcomp
                                    rc_s = seqlen - e + 1
                                    rc_e = seqlen - s + 1
                                    if rc_s > rc_e:
                                        rc_s, rc_e = rc_e, rc_s
                                    rc_seq = (
                                        rc_seq[: rc_s - 1]
                                        + "N" * (rc_e - rc_s + 1)
                                        + rc_seq[rc_e:]
                                    )
                                seq_str = revcomp(rc_seq)
                        processed += (
                            1
                            if (
                                merged[current_id].get("+")
                                or merged[current_id].get("-")
                            )
                            else 0
                        )
                        if processed % 10 == 0 or processed == total_records:
                            logger.info(
                                f"[mask_nuc_range_from_sam] Processed {processed}/{total_records} records with ranges."
                            )
                    out_f.write(f">{current_id}\n{seq_str}\n")
                current_id = line[1:].strip()
                current_seq_parts = []
            else:
                current_seq_parts.append(line.strip())

        # last record
        if current_id:
            seq_str = "".join(current_seq_parts)
            if current_id in merged:
                for strand in ("+", "-"):
                    intervals = merged[current_id].get(strand, [])
                    if not intervals:
                        continue
                    if strand == "+":
                        for s, e in intervals:
                            if s > e:
                                s, e = e, s
                            seq_str = (
                                seq_str[: s - 1]
                                + "N" * (e - s + 1)
                                + seq_str[e:]
                            )
                    else:
                        rc_seq = revcomp(seq_str)
                        seqlen = len(seq_str)
                        for s, e in intervals:
                            if s > e:
                                s, e = e, s
                            rc_s = seqlen - e + 1
                            rc_e = seqlen - s + 1
                            if rc_s > rc_e:
                                rc_s, rc_e = rc_e, rc_s
                            rc_seq = (
                                rc_seq[: rc_s - 1]
                                + "N" * (rc_e - rc_s + 1)
                                + rc_seq[rc_e:]
                            )
                        seq_str = revcomp(rc_seq)
                processed += (
                    1
                    if (
                        merged[current_id].get("+")
                        or merged[current_id].get("-")
                    )
                    else 0
                )
                logger.info(
                    f"[mask_nuc_range_from_sam] Processed {processed}/{total_records} records with ranges."
                )
            out_f.write(f">{current_id}\n{seq_str}\n")


def main(**kwargs):
    pass
    consolidate_hits(**kwargs)


if __name__ == "__main__":
    main()
    # Debug arguments
    # debug = False
    # if debug:
    #     input = pl.read_csv(
    #         "/clusterfs/jgi/scratch/science/metagen/neri/tests/rp_tests/inputs/tables/hit_table.tsv",
    #         separator="\t",
    #     )
    #     output = "/clusterfs/jgi/scratch/science/metagen/neri/tests/rp_tests/inputs/tables/consolidated_output.tsv"
    #     best = False
    #     rank_columns = "-score,+evalue"
    #     column_specs = "qseqid,sseqid"
    #     culling_mode = "one_per_range"
    #     env_mode = "envelope"
    #     # max_overlap_fraction=0.1
    #     min_overlap_positions = 10
    #     clip = True
    #     drop_contained = True
    #     one_per_query = False
    #     one_per_range = False
    #     merge = False
    #     consolidate_hits(
    #         input,
    #         None,
    #         best,
    #         rank_columns,
    #         culling_mode,
    #         min_overlap_positions,
    #         clip,
    #         drop_contained,
    #         merge,
    #         column_specs,
    #     )
