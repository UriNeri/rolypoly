import rich_click as click
from rolypoly.utils.loggit import setup_logging, log_start_info
from rolypoly.utils.citation_reminder import remind_citations
from rolypoly.utils.various import ensure_memory
from rolypoly.utils.fax import is_nucl_string
from pathlib import Path
from rich.console import Console

console = Console()

def calculate_sequence_stats(sequence, include_structure=False):
    """Calculate basic statistics for a sequence.
    
    Args:
        sequence (Bio.Seq.Seq): Input sequence
        include_structure (bool): Whether to calculate RNA structure potential
        
    Returns:
        dict: Dictionary containing sequence statistics
    """
    stats = {
        'length': len(sequence),
        'gc_content': (sequence.count('G') + sequence.count('C')) / len(sequence),
        'n_count': sequence.count('N'),
    }
    
    if include_structure and len(sequence) <= 10000:  # Only calculate for sequences under 10kb
        try:
            import RNA
            # Calculate minimum free energy and ensemble diversity
            fc = RNA.fold_compound(sequence)
            mfe_struct, mfe = fc.mfe()
            stats['mfe'] = mfe
            stats['mfe_per_nt'] = mfe / len(sequence) if len(sequence) > 0 else 0
            
            # Calculate ensemble diversity
            (_, mfe), p = fc.pf()
            stats['ensemble_diversity'] = fc.mean_bp_distance()
        except Exception as e:
            console.print(f"[yellow]Warning: RNA structure calculation failed: {e}[/yellow]")
    
    return stats

def calculate_codon_usage(sequence):
    """Calculate codon usage frequencies for a sequence.
    
    Args:
        sequence (Bio.Seq.Seq): Input sequence
        
    Returns:
        dict: Dictionary containing codon frequencies
    """
    from collections import defaultdict
    codons = defaultdict(int)
    for i in range(0, len(sequence) - 2, 3):
        codon = sequence[i:i+3].upper()
        if 'N' not in codon:
            codons[codon] += 1
    
    total = sum(codons.values())
    return {k: v/total for k, v in codons.items()} if total > 0 else {}

@click.command()
@click.option('-i', '--input', required=True, help='Input file or directory')
@click.option('-o', '--output', default='output', help='Output directory')
@click.option('-t', '--threads', default=1, help='Number of threads')
@click.option('-M', '--memory', default='6g', help='Memory allocation')
@click.option('--log-file', default='command.log', help='Path to log file')
@click.option('--log-level', default='INFO', help='Log level')
@click.option('--min_length', default=None, type=int, help='min_length')
@click.option('--max_length', default=None, type=int, help='max_length')
@click.option('--format', default='text', type=click.Choice(['text', 'json', 'tsv']), help='format')
@click.option('--include_structure', default=False, type=bool, help='include_structure')
def sequence_stats(input, output, threads, memory, log_file, log_level, min_length, max_length, format, include_structure):
    """
    Calculate and display basic sequence statistics for viral sequences including length distribution, GC content, codon usage, and RNA structure potential
    """
    import numpy as np
    import json
    from needletail import parse_fastx_file
    from rich.table import Table
    import polars as pl

    logger = setup_logging(log_file, log_level)
    log_start_info(logger, locals())
    ensure_memory(memory)

    # Create output directory
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process sequences in batches to reduce memory usage
    sequences = []
    total_seqs = 0
    filtered_seqs = 0
    batch_size = 1000  # Process 1000 sequences at a time
    
    logger.info("Reading sequences and calculating statistics...")
    current_batch = []
    
    for record in parse_fastx_file(input):
        total_seqs += 1
        seq_len = len(record.seq)
        
        # Apply length filters if specified
        if min_length and seq_len < min_length:
            continue
        if max_length and seq_len > max_length:
            continue
            
        filtered_seqs += 1
        
        # Calculate statistics
        stats = calculate_sequence_stats(record.seq, include_structure)
        stats['id'] = record.id
        stats['description'] = record.description
        
        # Calculate codon usage if sequence length is multiple of 3 and is nucleotide
        if len(record.seq) % 3 == 0 and is_nucl_string(record.seq):
            stats['codon_usage'] = calculate_codon_usage(record.seq)
        
        current_batch.append(stats)
        
        # Process batch if it reaches the size limit
        if len(current_batch) >= batch_size:
            sequences.extend(current_batch)
            current_batch = []
    
    # Add remaining sequences
    if current_batch:
        sequences.extend(current_batch)

    # Calculate summary statistics
    if sequences:
        lengths = [s['length'] for s in sequences]
        gc_contents = [s['gc_content'] for s in sequences]
        
        summary_stats = {
            'total_sequences': total_seqs,
            'filtered_sequences': filtered_seqs,
            'min_length': min(lengths),
            'max_length': max(lengths),
            'mean_length': np.mean(lengths),
            'median_length': np.median(lengths),
            'mean_gc': np.mean(gc_contents),
            'median_gc': np.median(gc_contents),
        }
        
        if include_structure:
            mfe_values = [s.get('mfe_per_nt', 0) for s in sequences if 'mfe_per_nt' in s]
            if mfe_values:
                summary_stats['mean_mfe_per_nt'] = np.mean(mfe_values)
                summary_stats['median_mfe_per_nt'] = np.median(mfe_values)

        # Output results based on format
        if format == 'json':
            with open(output_path / 'sequence_stats.json', 'w') as f:
                json.dump({'summary': summary_stats, 'sequences': sequences}, f, indent=2)
        
        elif format == 'tsv':
            # Flatten sequence stats for TSV output
            flat_stats = []
            for seq in sequences:
                flat_seq = {k: v for k, v in seq.items() if k != 'codon_usage'}
                if 'codon_usage' in seq:
                    for codon, freq in seq['codon_usage'].items():
                        flat_seq[f'codon_{codon}'] = freq
                flat_stats.append(flat_seq)
            
            # Use polars for efficient TSV writing
            df = pl.DataFrame(flat_stats)
            df.write_csv(output_path / 'sequence_stats.tsv', separator='\t')
            
            # Write summary stats
            with open(output_path / 'summary_stats.tsv', 'w') as f:
                for k, v in summary_stats.items():
                    f.write(f"{k}\t{v}\n")
        
        else:  # text format
            # Create rich table for summary stats
            summary_table = Table(title="Sequence Summary Statistics")
            summary_table.add_column("Metric")
            summary_table.add_column("Value")
            
            for k, v in summary_stats.items():
                summary_table.add_row(k, f"{v:.2f}" if isinstance(v, float) else str(v))
            
            console.print(summary_table)
            
            # Create sequence details table
            seq_table = Table(title="Individual Sequence Statistics")
            seq_table.add_column("ID")
            seq_table.add_column("Length")
            seq_table.add_column("GC%")
            if include_structure:
                seq_table.add_column("MFE/nt")
            
            for seq in sequences[:10]:  # Show first 10 sequences
                row = [
                    seq['id'],
                    str(seq['length']),
                    f"{seq['gc_content']:.1f}",
                ]
                if include_structure and 'mfe_per_nt' in seq:
                    row.append(f"{seq['mfe_per_nt']:.3f}")
                seq_table.add_row(*row)
            
            if len(sequences) > 10:
                console.print(f"\n[italic]Showing first 10 of {len(sequences)} sequences[/italic]")
            
            console.print(seq_table)
            
            # Save tables to file
            with open(output_path / 'sequence_stats.txt', 'w') as f:
                console = Console(file=f)
                console.print(summary_table)
                console.print("\n")
                console.print(seq_table)

    else:
        logger.warning("No sequences found matching the specified criteria")

    logger.info("sequence-stats completed successfully!")
    tools = ["ViennaRNA"]
    remind_citations(tools)

if __name__ == "__main__":
    sequence_stats() 