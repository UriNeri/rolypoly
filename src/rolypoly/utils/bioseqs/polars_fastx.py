from pathlib import Path
import sys
from typing import  Optional, Union, TYPE_CHECKING, Callable
from collections import defaultdict

import polars as pl
from needletail import parse_fastx_file, reverse_complement


# Custom polars expression namespace that provides sequence analysis methods
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

    # Count the number of ambiguous nucleotides (N) in sequences
    def n_count(self) -> pl.Expr:
        """Count N's in sequence"""
        return self._expr.str.count_matches("N")

    # Get the total length of sequences in characters
    def length(self) -> pl.Expr:
        """Get sequence length"""
        return self._expr.str.len_chars()

    # Calculate relative frequencies of all codons (3-nucleotide combinations) in sequences
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

    # Generate MD5 hash identifiers for sequences (useful for deduplication)
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
    
    def translate(self, genetic_code: int = 11) -> pl.Expr:
        """Translate the sequence to amino acids"""
        from rolypoly.utils.bioseqs.translation import translate
        return self._expr.map_elements(
            lambda x: translate(x, genetic_code), return_dtype=pl.String
        )
    
    def reverse_complement(self) -> pl.Expr:
        """Reverse complement the sequence"""
        return self._expr.map_elements(reverse_complement, return_dtype=pl.String)
    

# LazyFrame namespace extension to enable lazy reading of FASTA/FASTQ files
@pl.api.register_lazyframe_namespace("from_fastx")
def from_fastx_lazy(input_file: Union[str, Path], batch_size: int = 512) -> pl.LazyFrame:
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
    def file_has_quality(file: Union[str, Path]) -> bool:
        first_record = next(parse_fastx_file(file))
        return first_record.qual is not None # type: ignore

    has_quality = file_has_quality(input_file)
    if has_quality:
        schema = pl.Schema({"header": pl.String, "sequence": pl.String, "quality": pl.String})
    else:
        schema = pl.Schema({"header": pl.String, "sequence": pl.String})

    def read_chunks():
        reader = parse_fastx_file(input_file)
        while True:
            chunk = []
            for _ in range(batch_size):
                try:
                    record = next(reader)
                    row = [record.id, record.seq] # type: ignore
                    if has_quality:
                        row.append(record.qual) # type: ignore
                    chunk.append(row)
                except StopIteration:
                    if chunk:
                        yield pl.LazyFrame(chunk, schema=schema, orient="row")
                    return
            yield pl.LazyFrame(chunk, schema=schema, orient="row")

    return pl.concat(read_chunks(), how="vertical")


# DataFrame namespace extension to enable eager reading of FASTA/FASTQ files 
@pl.api.register_dataframe_namespace("from_fastx")
def from_fastx_eager(file: Union[str, Path], batch_size: int = 512) -> pl.DataFrame:
    return pl.LazyFrame.from_fastx(file, batch_size).collect() # type: ignore

# # Initialization function for users to call in other modules
# def init_polars_extensions() -> None:
#     """Initialize custom Polars extensions for global use.
    
#     Call this function in modules where you want to use the custom
#     namespaces (seq, from_fastx) to ensure they are registered.
    
#     Example:
#         from rolypoly.utils.bioseqs.polars_fastx import init_polars_extensions
#         init_polars_extensions()  # Now you can use pl.col('seq').seq.length() etc.
#     """
#     # The decorators already register the namespaces when this module is imported
#     pass

