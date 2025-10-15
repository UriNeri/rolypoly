"""
polars plugins and functions for GFF3 and FASTA/FASTQ files.
The schema utilities are msotly for annotation data.
TODO: check the coordinate system for GFF3 is 1-based (if not then shift all coordinates by 1)
TODO: Add write functions for GFF3 and FASTA/FASTQ files (partially done see frame_to_fastx).

""" 

from pathlib import Path
from typing import  Optional, Union, Iterator
from collections import defaultdict

import polars as pl
from polars.io.plugins import register_io_source
from needletail import parse_fastx_file


#TODO: drop all map_elements and use polars native fucntions.
# Register custom expressions for sequence analysis
@pl.api.register_expr_namespace("seq")
class SequenceExpr:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def gc_content(self) -> pl.Expr:
        """Calculate GC content of sequence"""
        return (
            self._expr.str.count_matches("G") + self._expr.str.count_matches("C")
        ) / self._expr.str.len_chars()

    def n_count(self) -> pl.Expr:
        """Count N's in sequence"""
        return self._expr.str.count_matches("N")

    def length(self) -> pl.Expr:
        """Get sequence length"""
        return self._expr.str.len_chars()

    def codon_usage(self) -> pl.Expr:
        """Calculate codon usage frequencies"""
        def _calc_codons(seq: str) -> dict:
            codons = defaultdict(int)
            for i in range(0, len(seq) - 2, 3):
                codon = seq[i : i + 3].upper()
                if "N" not in codon:
                    codons[codon] += 1
            total = sum(codons.values())
            return {k: v / total for k, v in codons.items()} if total > 0 else {}

        return self._expr.map_elements(_calc_codons, return_dtype=pl.Struct)

    def generate_hash(self, length: int = 32) -> pl.Expr:
        """Generate a hash for a sequence"""
        import hashlib
        def _hash(seq: str) -> str:
            return hashlib.md5(seq.encode()).hexdigest()[:length]

        return self._expr.map_elements(_hash, return_dtype=pl.String)



@pl.api.register_lazyframe_namespace("from_fastx")
def from_fastx_lazy(input_file: Union[str, Path]) -> pl.LazyFrame:
    """Scan a FASTA/FASTQ file into a lazy polars DataFrame.

    This function extends polars with the ability to lazily read FASTA/FASTQ files.
    It can be used directly as pl.LazyFrame.fastx.scan("sequences.fasta").

    Args:
        path (Union[str, Path]): Path to the FASTA/FASTQ file
        batch_size (int, optional): Number of records to read per batch. Defaults to 512.

    Returns:
        pl.LazyFrame: Lazy DataFrame with columns:
            - header: Sequence headers (str)
            - sequence: Sequences (str) # TODO: maybe somehow move to largeutf8?
            - quality: Quality scores (only for FASTQ)
    """
    reader = parse_fastx_file(input_file)
    has_quality = True if next(reader).is_fastq() else False

    if has_quality:
        schema = pl.Schema({"header": pl.String, "sequence": pl.String, "quality": pl.String})
    else:
        schema = pl.Schema({"header": pl.String, "sequence": pl.String})

    def source_generator(
        with_columns: Optional[list] ,
        predicate: Optional[pl.Expr] ,
        n_rows: Optional[int] ,
        batch_size: Optional[int] ,
    ) -> Iterator[pl.LazyFrame]:
        if batch_size is None:
            batch_size = 512

        reader = parse_fastx_file(input_file)
        # print (n_rows)
        # print ("here")
        while n_rows is None or n_rows > 0:
            if n_rows is not None:
                batch_size = min(batch_size, n_rows)
            rows = []
            for _ in range(batch_size):
                try:
                    record = next(reader)
                    row = [record.id, record.seq]#, record.qual]
                    if has_quality:
                        row.append(record.qual)
                except StopIteration:
                    n_rows = 0
                    break
                rows.append(row)
            df = pl.from_records(rows, schema=schema, orient="row")
            # print (df.shape)
            if n_rows:
                n_rows -= df.height
            if with_columns is not None:
                df = df.select(with_columns)
            if predicate is not None:
                df = df.filter(predicate)
            yield df

    return register_io_source(io_source=source_generator,schema=schema)


@pl.api.register_dataframe_namespace("from_fastx")
def from_fastx_eager(file: Union[str, Path]) -> pl.DataFrame:
    return pl.LazyFrame.from_fastx(file).collect()



@pl.api.register_lazyframe_namespace("from_gff")
def from_gff_lazy(input_file: Union[str, Path]) -> pl.LazyFrame:
    """Scan a gff(3) file into a lazy polars DataFrame.

    Args:
        path (Union[str, Path]): Path to the FASTA/FASTQ file

    Returns:
        pl.LazyFrame: Lazy DataFrame with columns as gff3 specs.
    """

    schema = pl.Schema([('seqid', pl.String),
        ('source', pl.String),
        ('type', pl.String),
        ('start', pl.UInt32),
        ('end', pl.UInt32),
        ('score', pl.Float32),
        ('strand', pl.String),
        ('phase', pl.UInt32),
        ('attributes', pl.String)])
    
    spattern = r'(?P<key>\w+)=(?P<value>[^;]+)'
    reader = pl.scan_csv(input_file, has_header=False, separator="\t", comment_prefix="#", schema=schema,null_values=["."])
    reader = reader.with_columns(pl.col("attributes").str.extract_all(spattern))
    return reader

@pl.api.register_dataframe_namespace("from_gff")
def from_gff_eager(gff_file: Union[str, Path],unnest_attributes: bool = False) -> pl.DataFrame:
    lf = pl.LazyFrame.from_gff(gff_file)
    df = lf.collect()
    if unnest_attributes:
        df = df.with_columns(
        pl.col("attributes").list.eval(pl.element().str.split("=").list.to_struct()
                                       )
        # .list.to_struct()
        )
    return df


def count_kmers_df(df: pl.DataFrame, seq_col: str = "seq", id_col: str = "seqid", k: int = 3, relative: bool = False) -> pl.DataFrame:
    """Calculate k-mer counts for all sequences in a DataFrame"""
    # Split sequences into characters
    split_chars_expr = pl.col(seq_col).str.split('').alias('chars')
    
    # Create k-mers by shifting and concatenating
    create_kmers_expr = pl.concat_str(
        [pl.col('chars').shift(-i).over(id_col) for i in range(k)]
    ).alias('substrings')
    
    # Filter for complete k-mers only
    filter_complete_kmers_expr = pl.col('substrings').str.len_chars() == k
    
    # Aggregate expressions
    agg_exprs = [
        pl.first(seq_col),  # Keep the original sequence
        pl.col('substrings').value_counts(normalize=relative).alias('kmer_counts'),
        pl.exclude(seq_col, 'chars', 'substrings').first()  # Keep all other original columns
    ]
    
    return (
        df
        .with_columns(split_chars_expr)
        .explode('chars')
        .with_columns(create_kmers_expr)
        .filter(filter_complete_kmers_expr)
        .group_by(id_col, maintain_order=True)
        .agg(*agg_exprs)
    )

def filter_repetitive_kmers(df: pl.DataFrame, seq_col: str = "seq", id_col: str = "header", k: int = 3, max_count: int = 10, relative: bool = False) -> pl.DataFrame:
    """Filter sequences that have any k-mer appearing more than max_count times"""
    # First get k-mer counts
    df_with_kmers = count_kmers_df(df, seq_col, k, relative=False)
    
    # Filter for sequences without highly repetitive k-mers
    filter_repetitive_expr = ~pl.col('kmer_counts').list.eval(
        pl.element().struct.field('count') > max_count
    ).list.any()
    
    return df_with_kmers.filter(filter_repetitive_expr)

#  sequence statistics calculator with filtering 
def fasta_stats(
    input_file: str,
    output_file: Optional[str] = None, # 
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    fields: str = "header,sequence,length,gc_content,n_count,hash",
    # kmer_length: int = 3,
    circular: bool = False,
) -> pl.DataFrame:
    """Calculate sequence statistics using Polars expressions
    
    Args:
        input: Input file or directory
        output: Output path
        min_length: Minimum sequence length to consider
        max_length: Maximum sequence length to consider
        fields: Comma-separated list of fields to include (available: header,sequence,length,gc_content,n_count,hash,)
        circular: indicate if the sequences are circular (in which case, they will be rotated to their minimal lexicographical option, before the other stuff).
    Returns:
        pl.DataFrame: DataFrame with sequence statistics
    Note:
        - No support yet for reverse complement (not in circular or hash). TODO: <--
    """
    # import sys
    # output_path = Path(output_file) if output_file else sys.stdout

    # Read sequences into DataFrame
    df = pl.DataFrame.from_fastx(input_file) # type: ignore
    
    # init_height = df.height
    # Apply length filters
    if min_length:
        df = df.filter(pl.  col("sequence").seq.length() >= min_length)
    if max_length:
        df = df.filter(pl.col("sequence").seq.length() <= max_length)
    # print(f"Filtered {init_height - df.height} sequences out of {init_height}")
    if circular:
        # Rotate sequences to minimal lexicographical option
        # Apply Booth's algorithm using map_batches - # Note: this is using map_batches and a python function, 
        # so probably not nearly as fast as Antonio's seq-hash
        # Also this doesn't consider reverse complement.
        df = df.with_columns(
            pl.col("sequence")
            .map_batches(
                lambda s: pl.Series([least_rotation(val) for val in s]),
                return_dtype=pl.String
            )
            .alias("sequence")
        )
    # Define available fields and their dependencies
    field_options = {
        "length": {"desc": "Sequence length"},
        "gc_content": {"desc": "GC content percentage"},
        "n_count": {"desc": "Count of Ns in sequence"},
        "hash": {"desc": "Sequence hash (MD5)"},
        # "codon_usage": {"desc": "Codon usage frequencies"},
        # "kmer_freq": {"desc": "K-mer frequencies"},
        "header": {"desc": "Sequence header"},
        "sequence": {"desc": "DNA/RNA sequence"}
    }

    # Parse fields
    selected_fields = ["header"]
    if fields:
        selected_fields = [f.strip().lower() for f in fields.split(",")]
        # Validate fields
        valid_fields = list(field_options.keys())
        invalid_fields = [f for f in selected_fields if f not in valid_fields]
        if invalid_fields:
            print(f"Unknown field(s): {', '.join(invalid_fields)}")
            print(f"Available fields are: {', '.join(valid_fields)}")
        selected_fields = [f for f in selected_fields if f in valid_fields]

    # Build the stats expressions
    stats_expr = []
    # for field in selected_fields: # this doesn't work  :(
    #     stats_expr.append(pl.col("sequence").seq.field(field).alias(field))
    
    if "length" in selected_fields:
        stats_expr.append(pl.col("sequence").seq.length().alias("length"))
    if "gc_content" in selected_fields:
        stats_expr.append(pl.col("sequence").seq.gc_content().alias("gc_content"))
    if "n_count" in selected_fields:
        stats_expr.append(pl.col("sequence").seq.n_count().alias("n_count"))
    if "hash" in selected_fields:
        stats_expr.append(pl.col("sequence").seq.generate_hash().alias("hash")) 
    # if "codon_usage" in selected_fields:
    #     stats_expr.append(pl.col("sequence").seq.codon_usage().alias("codon_usage"))
    # if "kmer_freq" in selected_fields:
        # stats_expr.append(pl.col("sequence").seq.calculate_kmer_frequencies(kmer_length).alias("kmer_freq"))

    # Apply all the stats expressions
    df = df.with_columns(stats_expr)
    df = df.select(selected_fields)

    # Convert all nested columns to strings
    for col in df.columns:
        if col != "header":  # Keep header as is
            if isinstance(df[col].dtype, pl.Struct) or isinstance(df[col].dtype, pl.List):
                df = df.with_columns(
                    [pl.col(col).cast(pl.Utf8).alias(f"{col}")]
                )
    if output_file:
        df.write_csv(output_file, separator="\t")
    # print("Successfully wrote file after converting data types")
    return df


def least_rotation(s: str) -> str:
    """Finds the lexicographically minimal cyclic shift of a string using Booth's algorithm."""
    n = len(s)
    if n == 0:
        return ""
    
    s_double = s + s
    
    # KMP preprocessing failure function
    f = [-1] * (2 * n)
    k = 0 # Least starting index
    
    for j in range(1, 2 * n):
        i = f[j - k - 1]
        while i != -1 and s_double[j] != s_double[k + i + 1]:
            if s_double[j] < s_double[k + i + 1]:
                k = j - i - 1
            i = f[i]
        
        if i == -1 and s_double[j] != s_double[k + i + 1]:
            if s_double[j] < s_double[k + i + 1]:
                k = j
            f[j - k] = -1
        else:
            f[j - k] = i + 1
            
    return s_double[k:k+n]


# this one is complex but works...
# def parse_fasta_lazy(file_path: str) -> pl.LazyFrame:
#     # Read the whole file as a single column of lines
#     txt = pl.scan_csv(
#         file_path,
#         # treat every line as a string column called "line"
#         has_header=False,
#         separator="\n",
#                # read line‑by‑line
#         comment_prefix=None,   # we don't want Polars to skip anything
#         infer_schema=False,
#         schema={"line": pl.Utf8}
#     )#.select(pl.col("column_1").alias("line"))

#     # Group lines into blocks that start with '>'
#     # The trick: compute a cumulative sum that increments on header lines.
#     # All lines belonging to the same record get the same group id.
#     block_id = (pl.col("line").str.starts_with(">")).cum_sum()
#     block = txt.with_columns(block_id.alias("block"))

#     # Aggregate each block into a struct of (header, seq)
#     # `agg_list` collects all lines in the same block into a list.
#     agg = (
#         block
#         .group_by("block")
#         .agg(pl.concat_list(pl.col("line")).alias("lines"))
#         ) #.collect()
#     agg = (
#         block
#         .group_by("block")
#         .agg(pl.col("line"))
#         )# .collect()
#     agg = agg.select(
#             # first line (item) is the header, the rest SHOULD be (seq)
#             pl.col("line").list.first().str.slice(1).alias("header"),
#             pl.col("line").list.gather_every(1,1).list.join("").alias("seq")
#         )# ["seq"][1].to_list()
#     return agg

# lf = parse_fasta_lazy("/home/neri/Documents/projects/YNP/reps_side2_mot1.fas")
# df = lf.collect()
# print(df.head())

####################################################################################
#### output fastx/q files from polars dataframes/lazyframes
def frame_to_fastx(frame: Union[pl.LazyFrame,pl.DataFrame], output_file: Union[str, Path], seq_col:str = "sequence", header_col: str = "header", qual_col: Optional[str] = None):
    """Write a fastx file from a polars dataframe/lazyframe.    
    Args:
        frame: polars dataframe/lazyframe
        output_file: path to output file
        seq_col: name of the column containing the sequence
        header_col: name of the column containing the header
        qual_col: name of the column containing the quality scores (optional)
    Returns:
        None
    Note:
        if the input is a lazyframe, the output will be written in streaming mode using sink_csv - I'm not sure this is very robust in case the file exists already / has other stuff.
    """
    if qual_col:
        prefix = "@"
    else:
        prefix = ">"

    add_prefix = (pl.lit(prefix) + pl.col(header_col)).alias(header_col)
    if qual_col:
        add_sep=pl.lit("+").alias("sep")
        add_prefix = [add_prefix ,add_sep]
        expr2_select =(pl.col([header_col,seq_col,"sep",qual_col]))
    else:
        expr2_select =(pl.col([header_col,seq_col]))
    
    frame = frame.with_columns(add_prefix).select(expr2_select)
    # write out- for lazy will try to use sink 
    if isinstance(frame, pl.LazyFrame):
        frame.sink_csv(output_file,
        include_header=False,
        separator="\n",
        quote_style="never",
        engine="streaming"
        )
    else:
        frame.write_csv(output_file,
        include_header=False,
        separator="\n",
        quote_style="never",
        # engine="streaming"
        )
    

####################################################################################
#### Schema utilities for annotation data (mostly gff).
def normalize_column_names(df):
    """Normalize common column name variations to standard names.
    
    Maps various column names to standard annotation schema:
    - begin/from/seq_from -> start
    - to/seq_to -> end  
    - qseqid/sequence_ID/contig_id -> sequence_id
    - etc.
    """
    
    # Define column name mappings
    column_mappings = {
        # Start position variations
        'begin': 'start',
        'from': 'start', 
        'seq_from': 'start',
        'query_start': 'start',
        'qstart': 'start',
        
        # End position variations
        'to': 'end',
        'seq_to': 'end',
        'query_end': 'end',
        'qend': 'end',
        
        # Sequence ID variations
        'qseqid': 'sequence_id',
        'sequence_ID': 'sequence_id',
        'contig_id': 'sequence_id',
        'contig': 'sequence_id',
        'query': 'sequence_id',
        'id': 'sequence_id',
        'name': 'sequence_id',
        
        # Score variations
        'bitscore': 'score',
        'bit_score': 'score',
        'bits': 'score',
        'evalue': 'evalue',
        'e_value': 'evalue',
        
        # Source variations
        'tool': 'source',
        'method': 'source',
        'db': 'source',
        'database': 'source',
        
        # Type variations
        'feature': 'type',
        'annotation': 'type',
        'category': 'type',
    }
    
    # Rename columns if they exist
    rename_dict = {}
    for old_name, new_name in column_mappings.items():
        if old_name in df.columns:
            rename_dict[old_name] = new_name
    
    if rename_dict:
        df = df.rename(rename_dict)
    
    return df


def create_minimal_annotation_schema(df, annotation_type, source, tool_specific_cols=None):
    """Create a minimal standardized annotation schema.
    
    Args:
        df: Input DataFrame
        annotation_type: Type of annotation (e.g., 'ribozyme', 'tRNA', 'IRES')
        source: Source tool name
        tool_specific_cols: List of tool-specific columns to preserve
        
    Returns:
        DataFrame with standardized minimal schema. SHOULD be similar to gff3
    """
    
    # First normalize column names
    df = normalize_column_names(df)
    
    # Define minimal required columns with defaults
    minimal_schema = {
        'sequence_id': pl.Utf8, # TODO: check if "seqid" is preferred over "chrom" or "sequence_id"
        'type': pl.Utf8, 
        'start': pl.Int64,
        'end': pl.Int64,
        'score': pl.Float64,
        'source': pl.Utf8,
        'strand': pl.Utf8,
        'phase': pl.Utf8
    }
    
    # Add missing columns with appropriate defaults (null is .)
    for col, dtype in minimal_schema.items():
        if col not in df.columns:
            if col == 'type':
                default_val = annotation_type
            elif col == 'source':
                default_val = source
            elif col in ['start', 'end']:
                default_val = 0
            elif col == 'score':
                default_val = 0.0
            elif col == 'strand':
                default_val = '+'
            elif col == 'phase':
                default_val = '.'
            else:
                default_val = ''
            
            df = df.with_columns(pl.lit(default_val).alias(col).cast(dtype))
    
    # Select minimal columns plus any tool-specific ones
    columns_to_keep = list(minimal_schema.keys())
    if tool_specific_cols:
        for col in tool_specific_cols:
            if col in df.columns and col not in columns_to_keep:
                columns_to_keep.append(col)
    
    # Only select columns that actually exist
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    df = df.select(existing_columns)
    
    # Ensure all minimal schema columns exist even if not in original data
    for col, dtype in minimal_schema.items():
        if col not in df.columns:
            if col == 'type':
                default_val = annotation_type
            elif col == 'source':
                default_val = source
            elif col in ['start', 'end']:
                default_val = 0
            elif col == 'score':
                default_val = 0.0
            elif col == 'strand':
                default_val = '+'
            elif col == 'phase':
                default_val = '.'
            else:
                default_val = ''
            
            df = df.with_columns(pl.lit(default_val).alias(col).cast(dtype))
    
    return df


def ensure_unified_schema(dataframes):
    """Ensure all DataFrames have the same unified schema.
    
    Args:
        dataframes: List of (name, dataframe) tuples
        
    Returns:
        List of DataFrames with unified schema
    """
    
    if not dataframes:
        return []
    
    # Define the unified schema
    unified_schema = {
        'sequence_id': pl.Utf8,
        'type': pl.Utf8, 
        'start': pl.Int64,
        'end': pl.Int64,
        'score': pl.Float64,
        'source': pl.Utf8,
        'strand': pl.Utf8,
        'phase': pl.Utf8
    }
    
    # Add common tool-specific columns
    tool_specific_columns = {
        'profile_name': pl.Utf8,
        'evalue': pl.Float64,
        'ribozyme_description': pl.Utf8,
        'tRNA_type': pl.Utf8,
        'anticodon': pl.Utf8,
        'motif_type': pl.Utf8,
        'structure': pl.Utf8,
        'sequence': pl.Utf8
    }
    
    # Combine schemas
    full_schema = {**unified_schema, **tool_specific_columns}
    
    unified_dataframes = []
    
    for name, df in dataframes:
        # Add missing columns with appropriate defaults
        for col, dtype in full_schema.items():
            if col not in df.columns:
                if col == 'type':
                    default_val = name
                elif col == 'source':
                    default_val = 'unknown'
                elif col in ['start', 'end']:
                    default_val = 0
                elif col == 'score':
                    default_val = 0.0
                elif col == 'evalue':
                    default_val = 1.0
                elif col == 'strand':
                    default_val = '+'
                elif col == 'phase':
                    default_val = '.'
                else:
                    default_val = ''
                
                df = df.with_columns(pl.lit(default_val).alias(col).cast(dtype))
        
        # Ensure column order is consistent
        ordered_columns = list(full_schema.keys())
        df = df.select([col for col in ordered_columns if col in df.columns])
        
        unified_dataframes.append(df)
    
    return unified_dataframes 