"""
Schema utilities for annotation data (mostly gff).
TODO: check the coordinate system for GFF3 is 1-based (if not then shift all coordinates by 1)
""" 

import polars as pl

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
        'sequence_id': pl.Utf8,
        'type': pl.Utf8, 
        'start': pl.Int64,
        'end': pl.Int64,
        'score': pl.Float64,
        'source': pl.Utf8,
        'strand': pl.Utf8,
        'phase': pl.Utf8
    }
    
    # Add missing columns with appropriate defaults
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