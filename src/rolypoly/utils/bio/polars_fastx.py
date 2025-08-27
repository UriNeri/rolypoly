from pathlib import Path
import sys
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

    def calculate_kmer_frequencies(self, k: int = 3) -> pl.Expr:
        """Calculate k-mer frequencies in the sequence"""
        def _calc_kmers(seq: str, k: int) -> dict:
            if not seq or len(seq) < k:
                return {}
            kmers = defaultdict(int)
            for i in range(len(seq) - k + 1):
                kmer = seq[i : i + k].upper()
                if "N" not in kmer:
                    kmers[kmer] += 1
            total = sum(kmers.values())
            return {k: v / total for k, v in kmers.items()} if total > 0 else {}

        return self._expr.map_elements(
            lambda x: _calc_kmers(x, k), return_dtype=pl.Struct
        )


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
                    row = [record.id, record.seq, record.qual]
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


### example usage
def fasta_stats(
    input_file: str,
    output_file: Optional[str] = None, # if not provided, will print to stdout
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    fields: str = "header,length,gc_content,n_count,hash,codon_usage,kmer_freq",
    kmer_length: int = 3,
) -> None:
    """Calculate sequence statistics using Polars expressions
    
    Args:
        input: Input file or directory
        output: Output path
        min_length: Minimum sequence length to consider
        max_length: Maximum sequence length to consider
        fields: Comma-separated list of fields to include (available: header,sequence,length,gc_content,n_count,hash,codon_usage,kmer_freq)
    """
    output_path = Path(output_file) if output_file else sys.stdout

    # Read sequences into DataFrame
    df = pl.DataFrame.from_fastx(input_file)
    
    # init_height = df.height
    # Apply length filters
    if min_length:
        df = df.filter(pl.col("sequence").seq.length() >= min_length)
    if max_length:
        df = df.filter(pl.col("sequence").seq.length() <= max_length)
    # print(f"Filtered {init_height - df.height} sequences out of {init_height}")

    # Define available fields and their dependencies
    field_options = {
        "length": {"desc": "Sequence length"},
        "gc_content": {"desc": "GC content percentage"},
        "n_count": {"desc": "Count of Ns in sequence"},
        "hash": {"desc": "Sequence hash (MD5)"},
        "codon_usage": {"desc": "Codon usage frequencies"},
        "kmer_freq": {"desc": "K-mer frequencies"},
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
    if "codon_usage" in selected_fields:
        stats_expr.append(pl.col("sequence").seq.codon_usage().alias("codon_usage"))
    if "kmer_freq" in selected_fields:
        stats_expr.append(pl.col("sequence").seq.calculate_kmer_frequencies(kmer_length).alias("kmer_freq"))

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

    df.write_csv(output_path, separator="\t")
    print("Successfully wrote file after converting data types")


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
