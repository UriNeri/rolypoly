"""Interactive genome/marker maps report (`rolypoly report`).

Thin CLI wrapper around rolypoly.utils.viz.genome_maps. Writes a single
self-contained interactive HTML report (no internet needed to view) with tabs that
appear only when the corresponding data is available: Table / Genome maps
(annotate-protein + annotate-rna), Nucleic hits (nucleic-search), Run stats (reads
filtering + assembly), and any number of user-supplied extra tabs (e.g. predicted
taxonomy).

``--input`` may be either a single protein/marker hit table (auto-detecting the
hmmsearch vs mmseqs2/diamond schema) or a roll/annotate output *directory*, in
which case the RNA, nucleic-search and run-stats data are discovered automatically.
"""

from pathlib import Path

import rich_click as click

from rolypoly.utils.viz.genome_maps import (
    BEST_CRITERIA,
    MarkerTableSpec,
    table_to_tab,
    write_genome_maps,
    write_report_for_dir,
)


def build_extra_tabs(extra_tab_specs):
    """Turn ``label=path`` CLI strings into extra-tab payloads via table_to_tab."""
    tabs = []
    for spec in extra_tab_specs or ():
        if "=" in spec:
            label, path = spec.split("=", 1)
        else:
            label, path = Path(spec).stem, spec
        tabs.append(table_to_tab(path.strip(), label.strip()))
    return tabs or None


@click.command(name="report")
@click.option("-i", "--input", "input_path", required=True,
              help="A protein/marker hit table (combined_annotations.tsv, or any "
                   "TSV/CSV/Parquet), OR a roll/annotate output directory (RNA, "
                   "nucleic-search and run-stats are then discovered automatically).")
@click.option("-o", "--output", "output_path", required=True, help="Output HTML file.")
@click.option("-r", "--rna", "rna_path", default=None,
              help="Optional annotate-rna table (ignored in directory mode, where it is discovered).")
@click.option("-nu", "--nucleic", "nucleic_paths", multiple=True,
              help="Optional nucleic-search table(s) (repeatable; ignored in directory mode).")
@click.option("-x", "--extra-tab", "extra_tab_specs", multiple=True,
              help="Add a generic table tab as 'Label=path.tsv' (repeatable), e.g. for "
                   "predicted taxonomy or host prediction.")
@click.option("--rrna-mapping", default=None,
              help="Path to rrna_to_genome_mapping.parquet to enrich the rRNA stats with "
                   "reference organism names (default: $ROLYPOLY_DATA/contam/rrna/...).")
@click.option("-T", "--title", default="RolyPoly — Genome / marker maps", show_default=True,
              help="Title shown in the report header.")
@click.option("-ms", "--min-score", type=float, default=None,
              help="Drop protein hits with bit score below this value.")
@click.option("-me", "--max-evalue", type=float, default=None,
              help="Drop protein hits with E-value above this value.")
@click.option("-b/-a", "--best-only/--all-hits", "best_only", default=False, show_default=True,
              help="Initial view mode (toggleable in the viewer).")
@click.option("-bb", "--best-by", type=click.Choice(list(BEST_CRITERIA)), default="score",
              show_default=True, help="Initial 'best' criterion: score | evalue | longest | source.")
@click.option("-n", "--min-overlap", "min_overlap", type=int, default=1, show_default=True,
              help="Min overlapping positions to collapse hits during best-hit resolution "
                   "(1 = also collapse partial/nested overlaps).")
@click.option("-sp", "--source-priority", default=None,
              help="Comma-separated precedence order for the 'source' criterion "
                   "(default: rvmt,nvpc,pfam,genomad,vfam). Lower-priority sources still "
                   "win any locus no higher-priority hit overlaps.")
@click.option("-st", "--start-tab", default="table", show_default=True,
              help="Which tab to open on load (table | maps | nucleic | stats | <extra id>); "
                   "falls back to the first available tab.")
@click.option("-rb", "--rna-bins", type=int, default=150, show_default=True,
              help="Number of windows for the RNA base-pairing-density strip.")
@click.option("--no-stats", is_flag=True, default=False,
              help="Do not collect reads/assembly run statistics (directory mode).")
@click.option("--col-query", default=None,
              help="Override the ORF/query id column (default: auto-detect the schema).")
@click.option("--col-profile", default=None,
              help="Override the profile/marker name column (default: auto-detect).")
@click.option("--col-source", default=None,
              help="Override the source/database column that drives colour (default: auto-detect).")
@click.option("--col-aligned", default=None,
              help="Override the aligned-region / consensus column shown on hover. "
                   "'' disables it. Default: auto-detect (identity_str for hmmsearch).")
@click.option("-lf", "--log-file", default=None, help="Path to log file.")
@click.option("-ll", "--log-level", hidden=True, default="INFO", help="Log level")
def report(input_path, output_path, rna_path, nucleic_paths, extra_tab_specs, rrna_mapping,
           title, min_score, max_evalue, best_only, best_by, min_overlap, source_priority,
           start_tab, rna_bins, no_stats, col_query, col_profile, col_source, col_aligned,
           log_file, log_level):
    """Render an interactive per-contig genome-map report from RolyPoly outputs.

    Tabs appear only when data is present: Table / Genome maps (protein domains +
    RNA + nucleic tracks), Nucleic hits, Run stats, and any --extra-tab layers.
    Toggle all-hits vs best-only and pick the best-by criterion in the toolbar;
    protein and RNA hits are resolved separately, sourcing rolypoly's
    consolidate_hits. RNA discrete features are classified (rRNA / tRNA / IRES /
    ribozyme / riboswitch / frameshift / UTR / CRE / motif). The "source" criterion
    applies a precedence order (RVMT > NVPC > Pfam > genomad > VFAM by default)
    without excluding lower-priority sources.

    Args:
        input_path: protein/marker hit table, or a roll/annotate output directory.
        output_path: output HTML path.

    Returns:
        None. Writes a standalone HTML report to output_path.
    """
    from rolypoly.utils.logging.loggit import setup_logging

    logger = setup_logging(log_file, log_level.upper())

    priority = ([s.strip() for s in source_priority.split(",") if s.strip()]
                if source_priority else None)
    extra_tabs = build_extra_tabs(extra_tab_specs)
    common = dict(
        title=title, min_score=min_score, max_evalue=max_evalue,
        mark_best=True, min_overlap_positions=min_overlap, source_priority=priority,
        rna_bins=rna_bins, extra_tabs=extra_tabs,
        initial_mode=("best" if best_only else "all"),
        initial_criterion=best_by, initial_tab=start_tab,
    )

    if Path(input_path).is_dir():
        output = write_report_for_dir(
            input_path, output_path, with_stats=not no_stats,
            rrna_mapping_path=rrna_mapping, **common,
        )
        if output is None:
            raise click.ClickException(f"No annotation tables found under {input_path}")
    else:
        overrides = [col_query, col_profile, col_source, col_aligned]
        if any(o is not None for o in overrides):
            base = MarkerTableSpec()
            spec = MarkerTableSpec(
                query=col_query or base.query,
                profile=col_profile or base.profile,
                source=col_source or base.source,
                aligned_seq=(None if col_aligned == "" else (col_aligned or base.aligned_seq)),
            )
        else:
            spec = None
        nucleic_tables = None
        if nucleic_paths:
            nucleic_tables = []
            for path in nucleic_paths:
                stem = Path(path).stem
                label = stem.split("_vs_")[-1] if "_vs_" in stem else stem
                nucleic_tables.append((label, path))
        output = write_genome_maps(
            input_path, output_path, spec=spec,
            rna=rna_path, nucleic=nucleic_tables, **common,
        )
    logger.info("Wrote interactive report to %s", output)
    click.echo(f"✓ report written to {Path(output).resolve()}")


if __name__ == "__main__":
    report()
