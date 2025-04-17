from pathlib import Path

import rich_click as click
from rich.console import Console

from rolypoly.utils.citation_reminder import remind_citations

# console = Console()


def parse_marker_results(marker_file):
    """Parse marker search results file.

    Args:
        marker_file (str): Path to marker search results

    Returns:
        pl.DataFrame: DataFrame with marker results
    """
    import json

    import polars as pl

    if marker_file.endswith(".tsv"):
        return pl.read_csv(marker_file, separator="\t")
    elif marker_file.endswith(".json"):
        with open(marker_file) as f:
            data = json.load(f)
        return pl.DataFrame(data)
    else:
        raise ValueError(f"Unsupported file format for {marker_file}")


def run_genomad_hmm_search(input_fasta, output_dir, threads, logger):
    """Run genomad RNA viral HMM search.

    Args:
        input_fasta (str): Path to input fasta file
        output_dir (Path): Output directory
        threads (int): Number of threads to use
        logger (Logger): Logger object

    Returns:
        pl.DataFrame: DataFrame with HMM search results
    """
    import os

    import polars as pl

    from rolypoly.utils.fax import search_hmmdb

    hmm_db = (
        Path(os.environ["ROLYPOLY_DATA"]) / "hmmdbs" / "genomad_rna_viral_markers.hmm"
    )
    output_file = output_dir / "genomad_hmm_results.txt"

    # Use existing HMM search functionality
    try:
        search_hmmdb(
            amino_file=input_fasta,
            db_path=str(hmm_db),
            output=str(output_file),
            threads=threads,
            logger=logger,
            output_format="tblout",
        )

        # Parse results
        results = []
        with open(output_file) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    results.append(
                        {
                            "query": parts[0],
                            "target": parts[2],
                            "evalue": float(parts[4]),
                            "score": float(parts[5]),
                        }
                    )

        return pl.DataFrame(results)
    except Exception as e:
        logger.error(f"HMM search failed: {e}")
        return pl.DataFrame()


def assign_taxonomy(sequence_id, markers, hmm_results, min_score=50):
    """Assign taxonomy based on marker and HMM results.

    Args:
        sequence_id (str): Sequence identifier
        markers (pl.DataFrame): Marker search results
        hmm_results (pl.DataFrame): HMM search results
        min_score (float): Minimum score threshold

    Returns:
        dict: Taxonomy assignment
    """
    import polars as pl

    # Filter results by score
    seq_markers = markers.filter(pl.col("query") == sequence_id)
    seq_hmms = hmm_results.filter(
        (pl.col("query") == sequence_id) & (pl.col("score") >= min_score)
    )

    # Initialize taxonomy assignment
    tax = {
        "sequence_id": sequence_id,
        "confidence": 0.0,
        "taxonomy": "Unknown",
        "evidence": [],
    }

    # Look for RdRP markers first (highest confidence)
    rdrp_markers = seq_markers.filter(pl.col("target").str.contains("RdRP|RdRp|rdrp"))
    if not rdrp_markers.is_empty():
        best_hit = rdrp_markers.sort("score", descending=True).row(0)
        tax["taxonomy"] = (
            best_hit["taxonomy"] if "taxonomy" in best_hit else "RNA virus"
        )
        tax["confidence"] = min(1.0, best_hit["score"] / 100)
        tax["evidence"].append(f"RdRP marker: {best_hit['target']}")

    # Check genomad HMM results
    if not seq_hmms.is_empty():
        hmm_hits = seq_hmms.sort("score", descending=True)
        for hit in hmm_hits.iter_rows():
            tax["evidence"].append(f"RNA viral marker: {hit['target']}")
            if tax["confidence"] < 0.8:  # Only update if current confidence is low
                tax["confidence"] = min(0.8, hit["score"] / 100)

    # Check other viral markers
    other_markers = seq_markers.filter(~pl.col("target").str.contains("RdRP|RdRp|rdrp"))
    if not other_markers.is_empty():
        for hit in other_markers.iter_rows():
            tax["evidence"].append(f"Other marker: {hit['target']}")
            if tax["confidence"] < 0.6:  # Only update if current confidence is low
                tax["confidence"] = min(0.6, hit["score"] / 100)

    return tax


def summarize_taxonomy(assignments):
    """Generate taxonomy summary statistics.

    Args:
        assignments (list): List of taxonomy assignments

    Returns:
        dict: Summary statistics
    """
    from collections import defaultdict

    total = len(assignments)
    assigned = sum(1 for a in assignments if a["taxonomy"] != "Unknown")

    # Count taxonomy levels
    tax_counts = defaultdict(int)
    confidence_sum = 0

    for assign in assignments:
        if assign["taxonomy"] != "Unknown":
            tax_counts[assign["taxonomy"]] += 1
            confidence_sum += assign["confidence"]

    return {
        "total_sequences": total,
        "assigned_sequences": assigned,
        "percent_assigned": (assigned / total * 100) if total > 0 else 0,
        "mean_confidence": confidence_sum / assigned if assigned > 0 else 0,
        "taxonomy_distribution": dict(tax_counts),
    }


@click.command()
@click.option("-i", "--input", required=True, help="Input file or directory")
@click.option("-o", "--output", default="output", help="Output directory")
@click.option("-t", "--threads", default=1, help="Number of threads")
@click.option("--log-file", default="command.log", help="Path to log file")
@click.option("-ll", "--log-level", hidden=True, default="INFO", help="Log level")
@click.option("--marker_results", default=None, type=str, help="marker_results")
@click.option(
    "--format",
    default="text",
    type=click.Choice(["text", "json", "tsv"]),
    help="format",
)
@click.option("--min_score", default=50.0, type=float, help="min_score")
@click.option("--summarize", default=True, type=bool, help="summarize")
def quick_taxonomy(
    input,
    output,
    threads,
    log_file,
    log_level,
    marker_results,
    format,
    min_score,
    summarize,
):
    """
    Rapid taxonomy assignment for viral sequences using marker search results and genomad RNA viral HMMs
    """
    import json

    import polars as pl
    from needletail import parse_fastx_file
    from rich.table import Table

    from rolypoly.utils.loggit import log_start_info, setup_logging

    logger = setup_logging(log_file, log_level)
    log_start_info(logger, locals())

    # Create output directory
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    sequences = []
    for record in parse_fastx_file(input):
        sequences.append({"id": record.id, "seq": record.seq})

    if not sequences:
        logger.error(f"No sequences found in {input}")
        return

    # Parse marker results if provided
    markers_df = None
    if marker_results:
        try:
            markers_df = parse_marker_results(marker_results)
        except Exception as e:
            logger.warning(f"Error parsing marker results: {e}")

    # Run genomad HMM search
    logger.info("Running genomad RNA viral HMM search...")
    hmm_results = run_genomad_hmm_search(input, output_path, threads, logger)

    # Assign taxonomy
    logger.info("Assigning taxonomy...")
    assignments = []
    for seq in sequences:
        tax = assign_taxonomy(
            seq["id"],
            markers_df if markers_df is not None else pl.DataFrame(),
            hmm_results,
            min_score,
        )
        assignments.append(tax)

    # Generate summary if requested
    if summarize:
        summary = summarize_taxonomy(assignments)

    # Output results based on format
    if format == "json":
        results = {
            "assignments": assignments,
            "summary": summary if summarize else None,
        }
        with open(output_path / "taxonomy_assignments.json", "w") as f:
            json.dump(results, f, indent=2)

    elif format == "tsv":
        # Flatten assignments for TSV
        flat_assignments = []
        for assign in assignments:
            flat_assign = {
                "sequence_id": assign["sequence_id"],
                "taxonomy": assign["taxonomy"],
                "confidence": assign["confidence"],
                "evidence": ";".join(assign["evidence"]),
            }
            flat_assignments.append(flat_assign)

        pl.DataFrame(flat_assignments).write_csv(
            output_path / "taxonomy_assignments.tsv", separator="\t"
        )

        if summarize:
            with open(output_path / "taxonomy_summary.tsv", "w") as f:
                for k, v in summary.items():
                    if k != "taxonomy_distribution":
                        f.write(f"{k}\t{v}\n")
                f.write("\nTaxonomy distribution:\n")
                for tax, count in summary["taxonomy_distribution"].items():
                    f.write(f"{tax}\t{count}\n")

    else:  # text format
        # Create assignments table
        assign_table = Table(title="Taxonomy Assignments")
        assign_table.add_column("Sequence ID")
        assign_table.add_column("Taxonomy")
        assign_table.add_column("Confidence")
        assign_table.add_column("Evidence")

        for assign in assignments:
            assign_table.add_row(
                assign["sequence_id"],
                assign["taxonomy"],
                f"{assign['confidence']:.2f}",
                "\n".join(assign["evidence"]),
            )

        console.print(assign_table)

        if summarize:
            # Create summary table
            summary_table = Table(title="Taxonomy Summary")
            summary_table.add_column("Metric")
            summary_table.add_column("Value")

            for k, v in summary.items():
                if k != "taxonomy_distribution":
                    summary_table.add_row(
                        k, f"{v:.1f}" if isinstance(v, float) else str(v)
                    )

            console.print("\n")
            console.print(summary_table)

            # Create taxonomy distribution table
            dist_table = Table(title="Taxonomy Distribution")
            dist_table.add_column("Taxonomy")
            dist_table.add_column("Count")
            dist_table.add_column("Percentage")

            total = sum(summary["taxonomy_distribution"].values())
            for tax, count in summary["taxonomy_distribution"].items():
                dist_table.add_row(tax, str(count), f"{count / total * 100:.1f}%")

            console.print("\n")
            console.print(dist_table)

            # Save to file
            with open(output_path / "taxonomy_assignments.txt", "w") as f:
                console = Console(file=f)
                console.print(assign_table)
                if summarize:
                    console.print("\n")
                    console.print(summary_table)
                    console.print("\n")
                    console.print(dist_table)

    logger.info("quick-taxonomy completed successfully!")
    tools = ["HMMER", "geNomad"]
    remind_citations(tools)


if __name__ == "__main__":
    quick_taxonomy()
