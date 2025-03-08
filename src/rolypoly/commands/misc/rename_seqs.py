"""Rename sequences in a FASTA file with consistent IDs.

This script provides functionality to rename sequences in FASTA files with consistent
IDs, either using sequential numbers or hashes. It also generates a lookup table
mapping old IDs to new IDs and optionally includes sequence statistics.

Note on naming: to be consiset, the default for the raw assembly outputs sets new names as "CID_####" (CID is contig ID, and the ### is a padded running number).  
Subsequent steps in the pipeline will prepend additional identifiers to the contig IDs - after binning we add Bin_####, and after marker search we add vid_####.  
ORFs/CDS, get a unique name regardless of their position in the genome or the contig ID - they can be traced back to the CIDs in the GFF or some table.
"""

import os
from pathlib import Path
import rich_click as click
from rich.console import Console
import hashlib
from typing import Dict, Tuple
import polars as pl

from rolypoly.utils.fax import read_fasta2polars_df

console = Console()

def generate_hash(sequence: str, length: int = 32) -> str:
    """Generate a hash for a sequence.
    
    Args:
        sequence (str): Input sequence
        length (int, optional): Length of hash to return. Defaults to all (32).
    
    Returns:
        str: Hash string of specified length
    """
    return hashlib.md5(sequence.encode()).hexdigest()[:length]

def rename_sequences(df: pl.DataFrame, 
                    prefix: str = "CID", 
                    use_hash: bool = False) -> Tuple[pl.DataFrame, Dict[str, str]]:
    """Rename sequences with consistent IDs.
    
    Args:
        df (pl.DataFrame): DataFrame with 'header' and 'sequence' columns
        prefix (str, optional): Prefix for new IDs. Defaults to "CID".
        use_hash (bool, optional): Use hash instead of numbers. Defaults to False.
    
    Returns:
        Tuple[pl.DataFrame, Dict[str, str]]: 
            - DataFrame with renamed sequences
            - Dictionary mapping old IDs to new IDs
    """
    id_map = {}
    new_headers = []
    
    # Calculate padding based on total number of sequences
    padding = len(str(len(df)))
    
    for i, (header, seq) in enumerate(zip(df['header'], df['sequence'])):
        if use_hash:
            new_id = f"{prefix}_{generate_hash(seq)}"
        else:
            new_id = f"{prefix}_{str(i+1).zfill(padding)}"
        id_map[header] = new_id
        new_headers.append(new_id)
    
    return df.with_columns(pl.Series("header", new_headers)), id_map

def calculate_sequence_stats(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate basic sequence statistics.
    
    Args:
        df (pl.DataFrame): DataFrame with 'header' and 'sequence' columns
    
    Returns:
        pl.DataFrame: DataFrame with added statistics columns
    """
    return df.with_columns([
        pl.col("sequence").str.lengths().alias("length"),
        (pl.col("sequence").str.count_matches("G|C").cast(pl.Float64) / 
        pl.col("sequence").str.lengths() * 100.0).alias("gc_content")
    ])

@click.command()
@click.option('-i', '--input', required=True, help='Input FASTA file')
@click.option('-o', '--output', required=True, help='Output FASTA file')
@click.option('-m', '--mapping', required=True, help='Output mapping file (TSV)')
@click.option('-p', '--prefix', default='CID', help='Prefix for new sequence IDs')
@click.option('--hash/--no-hash', default=False, help='Use hash instead of a padded running number for IDs')
@click.option('--stats/--no-stats', default=True, help='Include sequence statistics in mapping file (length, GC content)')
def main(input: str, output: str, mapping: str, prefix: str, 
         hash: bool, stats: bool):
    """Rename sequences in a FASTA file with consistent IDs.
    
    This tool renames sequences in a FASTA file using either sequential numbers
    or hashes, and generates a lookup table mapping old IDs to new IDs.
    Optionally includes sequence statistics (length, GC content).
    """
    # Read input FASTA
    console.print(f"Reading sequences from {input}")
    df = read_fasta2polars_df(input)
    
    # Rename sequences
    console.print(f"Renaming sequences with prefix '{prefix}'")
    df_renamed, id_map = rename_sequences(df, prefix, hash)
    
    # Calculate stats if requested
    if stats:
        console.print("Calculating sequence statistics")
        df_renamed = calculate_sequence_stats(df_renamed)
        
        # Prepare mapping DataFrame with stats
        mapping_df = pl.DataFrame({
            "old_id": list(id_map.keys()),
            "new_id": list(id_map.values()),
            "length": df_renamed["length"],
            "gc_content": df_renamed["gc_content"].round(2)
        })
    else:
        # Mapping DataFrame without stats
        mapping_df = pl.DataFrame({
            "old_id": list(id_map.keys()),
            "new_id": list(id_map.values())
        })
    
    # Write output files
    console.print(f"Writing renamed sequences to {output}")
    with open(output, 'w') as f:
        for header, seq in zip(df_renamed['header'], df_renamed['sequence']):
            f.write(f">{header}\n{seq}\n")
    
    console.print(f"Writing ID mapping to {mapping}")
    mapping_df.write_csv(mapping, separator='\t')
    
    console.print("[green]Done![/green]")

if __name__ == "__main__":
    main()
