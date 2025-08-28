

import rich_click as click
from rich.console import Console
from pathlib import Path
import polars as pl
from rolypoly.utils.bio.polars_fastx import from_fastx_eager as read_fasta_df
# from rolypoly.utils.bio.sequences import rename_sequences, process_sequences
console = Console()

@click.command(
        name="fastx_stats"
)
@click.option(
    "-i",
    "--input",
    required=True,
    type=click.Path(exists=True),
    help="Input file (fasta, fa, fna, faa)",
)
@click.option(
    "-agg",
    "--aggregate",
    default=False,
    type=bool,
    help="aggregate statistics across all sequences",
)
@click.option(
    "-o",
    "--output",
    default="rp_sequence_stats.txt",
    type=click.Path(exists=False),
    help="Output path",
)
@click.option(
    "--log-file",
    # default="command.log",
    type=click.Path(exists=False),
    help="Path to log file",
    hidden=True,
)
@click.option(
    "-ll",
    "--log-level",
    hidden=True,
    default="INFO",
    help="Log level",
)
@click.option(
    "--min_length",
    hidden=True,
    default=None,
    type=int,
    help="minimum sequence length to consider",
)
@click.option(
    "--max_length",
    hidden=True,
    default=None,
    type=int,
    help="maximum sequence length to consider",
)
@click.option(
    "--format",
    default="tsv",
    type=click.Choice(case_sensitive=False, choices=["csv", "tsv", "md", "parquet"]),
    help="output format, either a parquet/csv/tsv file with the data or a markdown file with summary statistics",
)
@click.option(
    "-f",
    "--fields",
    type=click.Choice(case_sensitive=False, choices=["length", "gc_content", "n_count", "hash"]),
    multiple=True,
    default=["length", "gc_content", "n_count", "hash"],
    help="""
              comma-separated list of fields to include.  
              Available:
              length - mandatory
              gc_content - percentage of GC nucleotides
              n_count - total number of Ns 
              hash - md5 hash of the sequence
              """,
)
@click.option(
    "-c",
    "--circular",
    is_flag=True,
    help="indicate if the sequences are circular (in which case, they will be rotated to their minimal lexicographical option, before the other stuff).",
)
def fastx_stats(
    input,
    aggregate,
    output,
    log_file,
    log_level,
    min_length,
    max_length,
    format,
    fields,
    circular,
):
    """Calculate sequence statistics using Polars expressions.
        Note:
        - No support yet for reverse complement (not in circular or hash). TODO: <--
    """
    from rolypoly.utils.logging.loggit import log_start_info, setup_logging
    from rolypoly.utils.bio.polars_fastx import fasta_stats

    logger = setup_logging(log_file, log_level)
    log_start_info(logger, locals())

    output_path = Path(output)

    # just using the fasta_stats function here, with output as none 
    df = fasta_stats(input, output_file=None, min_length=min_length, max_length=max_length, fields=",".join(fields), circular=circular)
    total_seqs = df.height
    logger.info(f"got {total_seqs} sequences from {input}")

    if aggregate:
        # Create aggregated statistics
        agg_stats = {}
        
        if "length" in fields:
            length_stats = df.select([
                pl.col("length").min().alias("min_length"),
                pl.col("length").max().alias("max_length"),
                pl.col("length").mean().alias("mean_length"),
                pl.col("length").median().alias("median_length"),
                pl.col("length").std().alias("std_length"),
                pl.col("length").sum().alias("total_length")
            ]).to_dicts()[0]
            agg_stats.update(length_stats)
        
        if "gc_content" in fields:
            gc_stats = df.select([
                pl.col("gc_content").min().alias("min_gc"),
                pl.col("gc_content").max().alias("max_gc"),
                pl.col("gc_content").mean().alias("mean_gc"),
                pl.col("gc_content").median().alias("median_gc"),
                pl.col("gc_content").std().alias("std_gc")
            ]).to_dicts()[0]
            agg_stats.update(gc_stats)
        
        if "n_count" in fields:
            n_stats = df.select([
                pl.col("n_count").min().alias("min_n_count"),
                pl.col("n_count").max().alias("max_n_count"),
                pl.col("n_count").mean().alias("mean_n_count"),
                pl.col("n_count").sum().alias("total_n_count")
            ]).to_dicts()[0]
            agg_stats.update(n_stats)
        
        agg_stats["total_sequences"] = total_seqs
        agg_stats["sequences_before_filter"] = total_seqs
        
        # Convert to DataFrame for consistent output
        df_output = pl.DataFrame([agg_stats])

    # Output results
    if format.lower() == "parquet":
        df_output.write_parquet(output_path)
        logger.info(f"Results written to {output_path} (parquet format)")
    
    elif format.lower() == "csv":
        df_output.write_csv(output_path)
        logger.info(f"Results written to {output_path} (CSV format)")
    
    elif format.lower() == "tsv":
        df_output.write_csv(output_path, separator="\t")
        logger.info(f"Results written to {output_path} (TSV format)")
    
    elif format.lower() == "md":
        # Create markdown summary - TODO: use rich/logger?
        md_content = "# Sequence Statistics Report\n\n"
        md_content += f"**Input file:** {input}\n"
        md_content += f"**Total sequences:** {total_seqs}\n"
        
        if aggregate:
            md_content += "## Aggregate Statistics\n\n"
            for key, value in agg_stats.items():
                if isinstance(value, float):
                    md_content += f"- **{key}:** {value:.2f}\n"
                else:
                    md_content += f"- **{key}:** {value}\n"
        else:
            md_content += "## Summary Statistics\n\n"
            # Add basic summary stats even when not aggregating
            if "length" in fields:
                length_summary = df.select([
                    pl.col("length").min().alias("min"),
                    pl.col("length").max().alias("max"),
                    pl.col("length").mean().alias("mean"),
                    pl.col("length").median().alias("median")
                ]).to_dicts()[0]
                
                md_content += "### Length Statistics\n"
                for stat, value in length_summary.items():
                    md_content += f"- **{stat}:** {value:.2f}\n"
                md_content += "\n"
            
            md_content += "\n## First 10 sequences\n\n"
            md_content += df_output.head(10).to_pandas().to_markdown(index=False)
        
        with open(output_path, 'w') as f:
            f.write(md_content)
        logger.info(f"Markdown report written to {output_path}")

    # Display summary to console
    logger.info(f"✓ Processed {total_seqs} sequences")
    logger.info(f" Output: {output_path}")
    
    if not aggregate and format.lower() != "md":
        logger.info("First 5 rows:")
        logger.info(df_output.head(5))

    # # Remind about citations
    # tools = ["polars"]  # Tools used in this analysis
    # remind_citations(tools)
    
    logger.info("Sequence statistics calculation completed successfully")

