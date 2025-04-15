from pathlib import Path

import polars as pl
import rich_click as click
from rich.console import Console

from rolypoly.utils.citation_reminder import remind_citations
from rolypoly.utils.fax import (
    #     RNAStructureExpr,
    #     SequenceExpr,
    is_nucl_string,
    read_fasta_df,
)

# show all columns
pl.Config().set_tbl_cols(-1)

# global console
console = Console()


@click.command()
@click.option("-i", "--input", required=True, help="Input file or directory")
@click.option(
    "-agg",
    "--aggregate",
    default=False,
    type=bool,
    help="aggregate statistics across all sequences",
)
@click.option("-o", "--output", default="rp_sequence_stats.txt", help="Output path")
@click.option("-t", "--threads", hidden=True, default=1, help="Number of threads")
@click.option("-M", "--memory", hidden=True, default="6g", help="Memory allocation")
@click.option("--log-file", default="command.log", help="Path to log file")
@click.option("--log-level", hidden=True, default="INFO", help="Log level")
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
    type=click.Choice(case_sensitive=False, choices=["text", "json", "tsv"]),
    help="output format",
)
@click.option(
    "-rt",
    "--rna_tool",
    default="ViennaRNA",
    type=click.Choice(case_sensitive=False, choices=["ViennaRNA", "LinearFold"]),
    help="RNA secondary structure prediction tool",
)
@click.option(
    "-f",
    "--fields",
    default="length,gc_content,n_count,hash,structure,mfe,mfe_per_nt,codon_usage,kmer_freq",
    help="""
              comma-separated list of fields to include.  
              Available:
              length - mandatory
              gc_content - percentage of GC nucleotides
              n_count - total number of Ns 
              hash - md5 hash of the sequence
              structure - Dot bracket notation of the RNA secondary structure (requires rna_tool)
              mfe - minimum free energy of the RNA secondary structure (requires rna_tool)
              mfe_per_nt - minimum free energy of the RNA secondary structure per nucleotide (requires rna_tool)
              codon_usage - codon usage frequencies (only works if length % 3 == 0 cause I'm lazy)
              kmer_freq - k-mer frequencies (k=3 by default)
              
              """,
)
def sequence_stats(
    input,
    aggregate,
    output,
    threads,
    memory,
    log_file,
    log_level,
    min_length,
    max_length,
    format,
    rna_tool,
    fields,
):
    """Calculate sequence statistics using Polars expressions"""
    import json

    import polars as pl

    from rolypoly.utils.loggit import log_start_info, setup_logging

    logger = setup_logging(log_file, log_level)
    log_start_info(logger, locals())

    output_path = Path(output)

    # Read sequences into DataFrame
    df = read_fasta_df(input)
    total_seqs = len(df)

    # Apply length filters
    if min_length:
        df = df.filter(pl.col("sequence").seq.length() >= min_length)
    if max_length:
        df = df.filter(pl.col("sequence").seq.length() <= max_length)

    filtered_seqs = len(df)

    # Define available fields and their dependencies
    field_options = {
        "length": {"desc": "Sequence length"},
        "gc_content": {"desc": "GC content percentage"},
        "n_count": {"desc": "Count of Ns in sequence"},
        "hash": {"desc": "Sequence hash (MD5)"},
        "structure": {"desc": "RNA secondary structure", "needs_tool": True},
        "mfe": {"desc": "Minimum free energy", "needs_structure": True},
        "mfe_per_nt": {
            "desc": "Minimum free energy per nucleotide",
            "needs_structure": True,
        },
        "codon_usage": {"desc": "Codon usage frequencies", "complex": True},
        "kmer_freq": {"desc": "K-mer frequencies", "complex": True},
    }

    # Parse fields
    selected_fields = []
    if fields:
        selected_fields = [f.strip().lower() for f in fields.split(",")]
        # Validate fields
        valid_fields = list(field_options.keys())
        invalid_fields = [f for f in selected_fields if f not in valid_fields]
        if invalid_fields:
            logger.warning(f"Unknown field(s): {', '.join(invalid_fields)}")
            logger.warning(f"Available fields are: {', '.join(valid_fields)}")
        selected_fields = [f for f in selected_fields if f in valid_fields]

    # Always include length for summaries
    if "length" not in selected_fields:
        selected_fields.append("length")

    # Define which stats to include based on selected fields
    include_length = "length" in selected_fields
    include_gc = "gc_content" in selected_fields
    include_n = "n_count" in selected_fields
    include_hash = "hash" in selected_fields
    include_structure = any(
        f in selected_fields for f in ["structure", "mfe", "mfe_per_nt"]
    )
    include_codon = "codon_usage" in selected_fields
    include_kmer = "kmer_freq" in selected_fields
    # Build the stats expressions
    stats_expr = []

    # Always include basic length stats for filtering/summaries
    stats_expr.append(pl.col("sequence").seq.length().alias("length"))

    if include_gc:
        stats_expr.append(pl.col("sequence").seq.gc_content().alias("gc_content"))

    if include_n:
        stats_expr.append(pl.col("sequence").seq.n_count().alias("n_count"))

    if include_hash:
        stats_expr.append(pl.col("sequence").seq.generate_hash().alias("hash"))

    if include_kmer:
        stats_expr.append(
            pl.col("sequence")
            .seq.calculate_kmer_frequencies(k=3)
            .alias("kmer_frequencies")
        )

    # Add structure prediction if requested
    if include_structure:
        if rna_tool.lower() == "linearfold":
            stats_expr.append(
                pl.col("sequence")
                .rna.predict_structure_with_tool("LinearFold")
                .alias("rna_struct")
            )
        else:
            stats_expr.append(
                pl.col("sequence")
                .rna.predict_structure_with_tool("ViennaRNA")
                .alias("rna_struct")
            )

    # Apply all the stats expressions
    df = df.with_columns(stats_expr)

    # Extract structure fields if needed
    if include_structure:
        df = df.with_columns(
            [
                pl.col("rna_struct").struct.field("structure").alias("structure"),
                pl.col("rna_struct").struct.field("mfe").alias("mfe"),
                (pl.col("rna_struct").struct.field("mfe") / pl.col("length")).alias(
                    "mfe_per_nt"
                ),
            ]
        ).drop("rna_struct")

    # Process codon usage separately since it's more complex
    if include_codon:
        codon_df = df.filter(pl.col("sequence").seq.is_valid_codon()).with_columns(
            [pl.col("sequence").seq.codon_usage().alias("codons")]
        )

        if len(codon_df) > 0:
            # Unnest the codon usage struct and join back
            codon_data = codon_df.select([pl.col("header"), pl.col("codons")]).unnest(
                "codons"
            )

            df = df.join(
                codon_data.select(pl.exclude("sequence")), on="header", how="left"
            )

    if include_kmer:
        try:
            # Don't try to unnest kmer frequencies in the main display_df selection
            # Instead, handle it separately similar to codon_usage
            kmer_df = df.select([pl.col("header"), pl.col("kmer_frequencies")])

            # For CSV/TSV output, we need to convert the struct to strings
            if format.lower() in ["tsv", "text"]:
                kmer_df = kmer_df.with_columns(
                    [pl.col("kmer_frequencies").cast(pl.Utf8).alias("kmer_freq_str")]
                )
                df = df.join(
                    kmer_df.select(["header", "kmer_freq_str"]), on="header", how="left"
                )
            else:
                # For JSON output, we can keep the struct
                kmer_df = kmer_df.unnest("kmer_frequencies")
                df = df.join(
                    kmer_df.select(pl.exclude("sequence")), on="header", how="left"
                )
        except Exception as e:
            logger.warning(f"Error processing k-mer frequencies: {str(e)}")
            logger.warning("Skipping k-mer frequencies output")

    if len(df) > 0:
        # Prepare display columns for per-contig stats
        display_columns = ["header"]

        # Add requested columns if they're in the dataframe
        for field in selected_fields:
            if field != "codon_usage" and field != "kmer_freq" and field in df.columns:
                display_columns.append(field)

        # Add codon columns if requested
        if include_codon:
            codon_cols = [
                col
                for col in df.columns
                if col
                not in [
                    "header",
                    "sequence",
                    "length",
                    "gc_content",
                    "n_count",
                    "hash",
                    "mfe",
                    "mfe_per_nt",
                    "structure",
                ]
            ]
            display_columns.extend(codon_cols)

        # Add kmer columns if requested
        if include_kmer:
            kmer_cols = [
                col
                for col in df.columns
                if col.startswith("kmer_") and col != "kmer_frequencies"
            ]
            display_columns.extend(kmer_cols)

        # Make sure we don't have duplicate columns
        display_columns = list(dict.fromkeys(display_columns))
        display_df = df.select(display_columns)

        # Check for nested columns that could cause CSV write problems
        has_nested_cols = any(
            isinstance(dtype, pl.Struct) or isinstance(dtype, pl.List)
            for dtype in display_df.dtypes
        )

        if has_nested_cols and format.lower() in ["tsv", "text"]:
            # Find nested columns
            nested_cols = [
                col
                for col, dtype in zip(display_df.columns, display_df.dtypes)
                if isinstance(dtype, pl.Struct) or isinstance(dtype, pl.List)
            ]

            logger.warning(
                f"Found nested columns which can't be directly written to CSV/TSV: {nested_cols}"
            )

            # Convert nested columns to string representation
            for col in nested_cols:
                display_df = display_df.with_columns(
                    [pl.col(col).cast(pl.Utf8).alias(col)]
                )

            logger.info(
                "Converted nested columns to string representation for CSV/TSV output"
            )

        # Calculate summary statistics if aggregation is requested
        if aggregate:
            summary_exprs = [
                pl.lit(total_seqs).alias("total_sequences"),
                pl.lit(filtered_seqs).alias("filtered_sequences"),
                pl.col("length").min().alias("min_length"),
                pl.col("length").max().alias("max_length"),
                pl.col("length").mean().alias("mean_length"),
                pl.col("length").median().alias("median_length"),
            ]

            if include_gc:
                summary_exprs.extend(
                    [
                        pl.col("gc_content").mean().alias("mean_gc"),
                        pl.col("gc_content").median().alias("median_gc"),
                    ]
                )

            if include_structure and "mfe_per_nt" in df.columns:
                summary_exprs.extend(
                    [
                        pl.col("mfe_per_nt").mean().alias("mean_mfe_per_nt"),
                        pl.col("mfe_per_nt").median().alias("median_mfe_per_nt"),
                    ]
                )

            summary_stats = df.select(summary_exprs)

        # Output results
        if format.lower() == "json":
            output_data = {"sequences": display_df.to_dict(as_series=False)}
            if aggregate:
                output_data["summary"] = summary_stats.to_dict(as_series=False)

            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)

        elif format.lower() == "tsv":
            try:
                display_df.write_csv(output_path, separator="\t")
                if aggregate:
                    summary_output = output_path.with_suffix(".summary.tsv")
                    summary_stats.write_csv(summary_output, separator="\t")
                    logger.info(f"Summary statistics written to {summary_output}")
            except Exception as e:
                logger.error(f"Error writing TSV file: {str(e)}")
                logger.info("Trying to convert problematic data types to strings...")

                # Convert all columns to strings as a last resort
                for col in display_df.columns:
                    if col != "header":  # Keep header as is
                        try:
                            display_df = display_df.with_columns(
                                [pl.col(col).cast(pl.Utf8).alias(f"{col}")]
                            )
                        except:
                            logger.warning(
                                f"Couldn't convert column {col} to string, dropping it"
                            )
                            display_df = display_df.drop(col)

                # Try writing again
                display_df.write_csv(output_path, separator="\t")
                logger.info("Successfully wrote file after converting data types")

        else:  # text format
            if aggregate:
                console.print("\nSequence Summary Statistics:")
                console.print(summary_stats)

            console.print("\nIndividual Sequence Statistics:")
            if len(df) > 20 and not aggregate:
                console.print(display_df.head(20))
                console.print(
                    f"\n[italic]Showing first 20 of {len(df)} sequences. Use -agg flag for summary statistics or export to file for full data.[/italic]"
                )
            else:
                console.print(display_df)

            with open(output_path, "w") as f:
                if aggregate:
                    f.write("Sequence Summary Statistics:\n")
                    summary_stats.write_csv(f, separator="\t")
                    f.write("\n\n")
                f.write("Individual Sequence Statistics:\n")
                display_df.write_csv(f, separator="\t")

    else:
        logger.warning("No sequences found matching the specified criteria")

    logger.info("sequence-stats completed successfully!")
    logger.info(f"Output written to {output_path}")
    tools = []
    if include_structure:
        if rna_tool.lower() == "linearfold":
            tools.append("LinearFold")
        else:
            tools.append("ViennaRNA")
    remind_citations(tools)


if __name__ == "__main__":
    sequence_stats()
