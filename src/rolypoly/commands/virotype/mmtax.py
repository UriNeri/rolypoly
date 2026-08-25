"""Assign ICTV taxonomy to contigs from protein similarity searches."""

from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
from pathlib import Path

import polars as pl
import rich_click as click

from rolypoly.utils.bio.polars_fastx import (
    read_protein_gff_map,
    validate_protein_to_contig_map,
)

DEFAULT_DATABASE_ALIASES = {"ncbi_virus", "ncbi_nr_ictv", "ncbi-nr-ictv"} # all these are the same
PLACEHOLDER_DATABASES = {"rvmt", "nvpc"}
ICTV_RANKS = (
    "realm",
    "subrealm",
    "kingdom",
    "subkingdom",
    "phylum",
    "subphylum",
    "class",
    "subclass",
    "order",
    "suborder",
    "family",
    "subfamily",
    "genus",
    "subgenus",
    "species",
)

SENSITIVITY_TO_MMSEQS = {
    "faster": 1.0,
    "fast": 2.0,
    "mid-sensitive": 3.0,
    "normal": 4.0,
    "sensitive": 5.0,
    "more-sensitive": 6.0,
    "very-sensitive": 7.0,
    "ultra-sensitive": 8.0,
}
SENSITIVITY_TO_DIAMOND = {
    "faster": "faster",
    "fast": "fast",
    "mid-sensitive": "mid-sensitive",
    "normal": None,
    "sensitive": "sensitive",
    "more-sensitive": "more-sensitive",
    "very-sensitive": "very-sensitive",
    "ultra-sensitive": "ultra-sensitive",
}


def normalize_sensitivity(value: str) -> str:
    """Normalize a named or numeric sensitivity to a shared preset name."""
    value = str(value).strip().lower().replace("_", "-")
    if value in SENSITIVITY_TO_MMSEQS:
        return value
    try:
        numeric = float(value)
    except ValueError as error:
        choices = ", ".join(SENSITIVITY_TO_MMSEQS)
        raise click.BadParameter(
            f"Sensitivity must be 1-8 or one of: {choices}"
        ) from error
    if not 1 <= numeric <= 8:
        raise click.BadParameter("Sensitivity must be between 1 and 8")
    index = min(max(round(numeric), 1), 8) - 1
    return tuple(SENSITIVITY_TO_MMSEQS)[index]


def default_database_path(backend: str) -> Path:
    """Return the default all-virus NR/ICTV database path for a backend."""
    base = Path(os.environ["ROLYPOLY_DATA"]) / "reference_seqs" / "ncbi_virus"
    if backend == "diamond":
        return base / "diamond" / "ncbi_virus.dmnd"
    return base / "mmseqs" / "ncbi_virus"


def default_taxdump_path() -> Path:
    """Return the (modified) taxdump paired with the default ncbi_virus databases."""
    return (
        Path(os.environ["ROLYPOLY_DATA"])
        / "reference_seqs"
        / "ncbi_virus"
        / "taxonomy"
    )


def resolve_database(
    database: str, backend: str, taxdump: Path | None
) -> tuple[Path, Path]:
    """Resolve database/taxdump paths and enforce custom database metadata."""
    normalized = database.strip().lower()
    if normalized in PLACEHOLDER_DATABASES:
        raise click.ClickException(
            f"The {normalized} taxonomy source is reserved but not built yet. "
            "It first needs taxonomy enrichment from ncbi_virus or its source framework."
        )

    is_default = normalized in DEFAULT_DATABASE_ALIASES
    if is_default:
        if taxdump is not None:
            raise click.UsageError(
                "--taxdump is only accepted with a custom --database; the "
                "ncbi_virus taxdump is resolved automatically."
            )
        database_path = default_database_path(backend)
        taxdump_path = default_taxdump_path()
    else:
        if taxdump is None:
            raise click.UsageError(
                "--taxdump is required when --database is a custom path."
            )
        database_path = Path(database).expanduser().resolve()
        taxdump_path = taxdump.expanduser().resolve()
        if backend == "diamond" and not database_path.exists():
            suffixed = Path(f"{database_path}.dmnd")
            if suffixed.exists():
                database_path = suffixed

    required = [
        database_path,
        taxdump_path / "nodes.dmp",
        taxdump_path / "names.dmp",
    ]
    if backend == "mmseqs":
        required.extend(
            [
                Path(f"{database_path}_mapping"),
                Path(f"{database_path}_taxonomy"),
            ]
        )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise click.ClickException(
            "Taxonomy database is incomplete; missing: "
            + ", ".join(str(path) for path in missing)
        )
    return database_path, taxdump_path


def query_alphabet(input_path: Path, requested_type: str) -> str:
    """Return ``nucl`` or ``protein`` for a FASTA/FASTQ query file."""
    if requested_type != "auto":
        return requested_type
    from needletail import parse_fastx_file

    from rolypoly.utils.bio.sequences import is_aa_string, is_nucl_string

    try:
        sequence = next(parse_fastx_file(input_path)).seq
    except StopIteration as error:
        raise click.ClickException(
            f"Input sequence file is empty: {input_path}"
        ) from error
    if isinstance(sequence, bytes):
        sequence = sequence.decode()
    if is_nucl_string(sequence):
        return "nucl"
    if is_aa_string(sequence):
        return "protein"
    raise click.ClickException(
        "Could not detect the query alphabet; pass --query-type explicitly."
    )


def protein_ids(protein_fasta: Path) -> pl.DataFrame:
    """Read the identifier token from each protein FASTA record."""
    from rolypoly.utils.bio.polars_fastx import from_fastx_eager

    return from_fastx_eager(protein_fasta).select(
        pl.col("header")
        .cast(pl.String)
        .str.split(" ")
        .list.first()
        .alias("protein")
    )


def sequence_lengths(
    fasta_path: Path, id_column: str, length_column: str
) -> pl.DataFrame:
    """Read first-token FASTA identifiers and sequence lengths."""
    from rolypoly.utils.bio.polars_fastx import from_fastx_eager

    return from_fastx_eager(fasta_path).select(
        pl.col("header")
        .cast(pl.String)
        .str.split(" ")
        .list.first()
        .alias(id_column),
        pl.col("sequence").str.len_bytes().cast(pl.Int64).alias(length_column),
    )


def read_single_cds_coordinates(path: Path) -> pl.DataFrame:
    """Read coordinates for proteins represented by one CDS feature."""
    columns = [
        "seqid",
        "source",
        "type",
        "start",
        "end",
        "score",
        "strand",
        "phase",
        "attributes",
    ]
    protein_id = pl.coalesce(
        [
            pl.col("attributes").str.extract(r"(?:^|;)\s*ID=([^;]+)", 1),
            pl.col("attributes").str.extract(
                r"(?:^|;)\s*protein_id=([^;]+)", 1
            ),
            pl.col("attributes").str.extract(
                r'(?:^|;)\s*protein_id "([^"]+)"', 1
            ),
            pl.col("attributes").str.extract(
                r'(?:^|;)\s*transcript_id "([^"]+)"', 1
            ),
        ]
    )
    return (
        pl.scan_csv(
            path,
            has_header=False,
            separator="\t",
            comment_prefix="#",
            new_columns=columns,
            infer_schema=False,
            truncate_ragged_lines=True,
        )
        .filter(pl.col("attributes").is_not_null())
        .with_columns(
            protein_id.str.split(" ").list.first().alias("protein"),
            pl.col("start").cast(pl.Int64, strict=False).alias("cds_start"),
            pl.col("end").cast(pl.Int64, strict=False).alias("cds_end"),
            pl.col("strand").alias("cds_strand"),
        )
        .filter(pl.col("protein").is_not_null())
        .group_by("protein")
        .agg(
            pl.len().alias("cds_segment_count"),
            pl.col("cds_start").first(),
            pl.col("cds_end").first(),
            pl.col("cds_strand").first(),
        )
        .filter(pl.col("cds_segment_count") == 1)
        .drop("cds_segment_count")
        .collect()
    )


def read_gff_contig_lengths(path: Path) -> pl.DataFrame:
    """Read contig lengths from standard or pyrodigal GFF comments."""
    lengths: dict[str, int] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            sequence_region = re.match(
                r"##sequence-region\s+(\S+)\s+\d+\s+(\d+)", line
            )
            pyrodigal_region = re.match(
                r'# Sequence Data:.*?seqlen=(\d+);seqhdr="([^"]+)"', line
            )
            if sequence_region:
                lengths[sequence_region.group(1).split()[0]] = int(
                    sequence_region.group(2)
                )
            elif pyrodigal_region:
                lengths[pyrodigal_region.group(2).split()[0]] = int(
                    pyrodigal_region.group(1)
                )
    return pl.DataFrame(
        {
            "contig": list(lengths),
            "gff_contig_length": list(lengths.values()),
        },
        schema={"contig": pl.String, "gff_contig_length": pl.Int64},
    )
def infer_contig_id(protein_id: str) -> str:
    """Infer a contig ID from common RolyPoly-generated protein headers."""
    identifier = str(protein_id).split()[0]
    identifier = re.sub(r"(?:_frame[=:]|\|frame=)[+-]?[1-3]$", "", identifier)
    return re.sub(r"_\d+$", "", identifier)


def read_protein_map(path: Path) -> pl.DataFrame:
    """Read a two-column protein-to-contig TSV with or without a header."""
    frame = pl.read_csv(path, separator="\t", has_header=False)
    if frame.width < 2:
        raise ValueError(
            "Protein map must contain at least two tab-separated columns"
        )
    frame = frame.select(frame.columns[:2]).rename(
        {frame.columns[0]: "protein", frame.columns[1]: "contig"}
    )
    if frame.height and [str(value).lower() for value in frame.row(0)] in (
        ["protein", "contig"],
        ["protein_id", "contig_id"],
    ):
        frame = frame.slice(1)
    frame = frame.with_columns(pl.all().cast(pl.String))
    return validate_protein_to_contig_map(frame)


def filter_top_hits(hits: pl.DataFrame, top: float) -> pl.DataFrame:
    """Keep hits within ``top`` percent of each protein's best bitscore."""
    if hits.is_empty():
        return hits
    return hits.filter(
        pl.col("bitscore")
        >= pl.col("bitscore").max().over("protein") * (1.0 - top / 100.0)
    )


def hit_weight(bitscore: float, evalue: float, method: str) -> float:
    """Convert an alignment score into a positive ORF-vote weight."""
    if method == "bitscore":
        return max(float(bitscore), 0.0)
    if float(evalue) <= 0:
        return 1000.0
    return max(-math.log(float(evalue)), 0.0)


def lca_taxid(taxids: list[str], taxdb) -> str:
    """Return the LCA taxid for one or more taxids."""
    import taxopy

    taxa = [taxopy.Taxon(str(taxid), taxdb) for taxid in dict.fromkeys(taxids)]
    if len(taxa) == 1:
        return taxa[0].taxid
    return taxopy.find_lca(taxa, taxdb).taxid


def weighted_majority_taxid(
    taxids: list[str], weights: list[float], taxdb, majority: float
) -> tuple[str, float]:
    """Choose a weighted taxid by walking down compatible ranked lineages.

    At each rank, only assignments resolved to that rank or below contribute to
    the denominator. Shallower assignments remain useful at their own ranks but
    are neutral at deeper ranks. Every selected child must descend from the
    previously selected node, so the result is always one real Taxopy lineage.
    """
    import taxopy

    if not taxids or sum(weights) <= 0:
        return "0", 0.0
    taxa = [
        (taxopy.Taxon(str(taxid), taxdb), float(weight))
        for taxid, weight in zip(taxids, weights, strict=True)
        if float(weight) > 0
    ]
    assignment = "1"
    assignment_support = 1.0
    for rank in ICTV_RANKS:
        rank_support: dict[str, float] = {}
        informative_weight = 0.0
        for taxon, weight in taxa:
            if assignment not in taxon.taxid_lineage:
                continue
            rank_taxid = next(
                (
                    node
                    for node in taxon.taxid_lineage
                    if taxdb.taxid2rank.get(node) == rank
                ),
                None,
            )
            if rank_taxid is None:
                continue
            informative_weight += weight
            rank_support[rank_taxid] = (
                rank_support.get(rank_taxid, 0.0) + weight
            )
        if not rank_support or informative_weight <= 0:
            continue
        best_weight = max(rank_support.values())
        winners = [
            taxid
            for taxid, weight in rank_support.items()
            if weight == best_weight
        ]
        if len(winners) != 1 or best_weight / informative_weight < majority:
            break
        candidate = winners[0]
        if assignment not in taxopy.Taxon(candidate, taxdb).taxid_lineage:
            break
        assignment = candidate
        assignment_support = rank_support[candidate] / informative_weight
    return assignment, assignment_support


def rank_vote_metrics(
    taxids: list[str],
    weights: list[float],
    assignment_taxid: str,
    rank: str,
    taxdb,
) -> tuple[float | None, float | None]:
    """Return conditional support and informative-weight fraction for a rank."""
    import taxopy

    assignment = taxopy.Taxon(str(assignment_taxid), taxdb)
    target = next(
        (
            node
            for node in assignment.taxid_lineage
            if taxdb.taxid2rank.get(node) == rank
        ),
        None,
    )
    if target is None:
        return None, None
    rank_index = ICTV_RANKS.index(rank)
    parent = "1"
    for parent_rank in reversed(ICTV_RANKS[:rank_index]):
        parent_node = next(
            (
                node
                for node in assignment.taxid_lineage
                if taxdb.taxid2rank.get(node) == parent_rank
            ),
            None,
        )
        if parent_node is not None:
            parent = parent_node
            break

    compatible_weight = informative_weight = target_weight = 0.0
    parent_lineage = set(taxopy.Taxon(parent, taxdb).taxid_lineage)
    for taxid, weight in zip(taxids, weights, strict=True):
        taxon = taxopy.Taxon(str(taxid), taxdb)
        weight = float(weight)
        is_descendant = parent in taxon.taxid_lineage
        is_unresolved_ancestor = taxon.taxid in parent_lineage
        if not (is_descendant or is_unresolved_ancestor):
            continue
        compatible_weight += weight
        rank_taxid = next(
            (
                node
                for node in taxon.taxid_lineage
                if taxdb.taxid2rank.get(node) == rank
            ),
            None,
        )
        if rank_taxid is None or not is_descendant:
            continue
        informative_weight += weight
        if rank_taxid == target:
            target_weight += weight
    support = target_weight / informative_weight if informative_weight else None
    informative_fraction = (
        informative_weight / compatible_weight if compatible_weight else None
    )
    return support, informative_fraction


def merged_interval_length(intervals: list[tuple[int, int]]) -> int:
    """Return the inclusive length of the union of integer intervals."""
    if not intervals:
        return 0
    merged_length = 0
    current_start, current_end = sorted(intervals)[0]
    for start, end in sorted(intervals)[1:]:
        if start <= current_end + 1:
            current_end = max(current_end, end)
        else:
            merged_length += current_end - current_start + 1
            current_start, current_end = start, end
    return merged_length + current_end - current_start + 1


def alignment_breadth(hits: pl.DataFrame) -> tuple[int, int | None]:
    """Return aligned amino-acid and projected genomic-nucleotide breadth."""
    aligned_residues = 0
    genomic_intervals: list[tuple[int, int]] = []
    for protein_hits in hits.partition_by("protein", maintain_order=True):
        residue_intervals = (
            [
                (min(int(start), int(end)), max(int(start), int(end)))
                for start, end in protein_hits.select(
                    "query_start", "query_end"
                )
                .drop_nulls()
                .iter_rows()
            ]
            if {"query_start", "query_end"}.issubset(protein_hits.columns)
            else []
        )
        if residue_intervals:
            aligned_residues += merged_interval_length(residue_intervals)
        else:
            aligned_residues += int(protein_hits["alignment_length"].max())

        if not residue_intervals or "cds_start" not in protein_hits.columns:
            continue
        cds_start = protein_hits["cds_start"][0]
        cds_end = protein_hits["cds_end"][0]
        strand = protein_hits["cds_strand"][0]
        if cds_start is None or cds_end is None or strand not in {"+", "-"}:
            continue
        cds_start, cds_end = int(cds_start), int(cds_end)
        for residue_start, residue_end in residue_intervals:
            if strand == "+":
                start = cds_start + (residue_start - 1) * 3
                end = min(cds_start + residue_end * 3 - 1, cds_end)
            else:
                start = max(cds_end - residue_end * 3 + 1, cds_start)
                end = cds_end - (residue_start - 1) * 3
            if start <= end:
                genomic_intervals.append((start, end))
    projected_nt = (
        merged_interval_length(genomic_intervals) if genomic_intervals else None
    )
    return aligned_residues, projected_nt


def assign_contig_taxonomy(
    hits: pl.DataFrame,
    protein_map: pl.DataFrame,
    taxdb,
    top: float = 10.0,
    weight_method: str = "bitscore",
    majority: float = 0.5,
    lineage_mode: int = 2,
    backend: str = "",
) -> pl.DataFrame:
    """Aggregate taxonomic protein hits into weighted contig assignments."""
    import taxopy

    output_columns = {
        "query": pl.String,
        "taxid": pl.String,
        "rank": pl.String,
        "taxon_name": pl.String,
        "support": pl.Float64,
        "informative_fraction": pl.Float64,
        "best_match_target": pl.String,
        "best_match_taxid": pl.String,
        "best_match_taxon": pl.String,
        "best_match_rank": pl.String,
        "best_match_bitscore": pl.Float64,
        "best_match_evalue": pl.Float64,
        "best_match_identity": pl.Float64,
        "best_match_alignment_length": pl.Int64,
        "proteins_assigned": pl.Int64,
        "total_proteins": pl.Int64,
        "protein_hit_fraction": pl.Float64,
        "aligned_residues": pl.Int64,
        "total_protein_residues": pl.Int64,
        "residue_alignment_fraction": pl.Float64,
        "projected_aligned_nt": pl.Int64,
        "genome_length": pl.Int64,
        "projected_alignment_genome_fraction": pl.Float64,
        "hits_retained": pl.Int64,
        "lineage": pl.String,
        "backend": pl.String,
        "method": pl.String,
    }
    for rank in ICTV_RANKS:
        output_columns[rank] = pl.String
        output_columns[f"{rank}_support"] = pl.Float64
    if hits.is_empty():
        return pl.DataFrame(schema=output_columns)

    retained = filter_top_hits(hits, top).join(
        protein_map, on="protein", how="inner", validate="m:1"
    )
    protein_assignments = []
    for group in retained.partition_by("protein", maintain_order=True):
        taxon_hits = group.group_by("taxid").agg(
            pl.col("bitscore").max(),
            pl.col("evalue").min(),
        )
        protein_taxids = taxon_hits["taxid"].cast(pl.String).to_list()
        protein_weights = [
            hit_weight(bitscore, evalue, weight_method)
            for bitscore, evalue in taxon_hits.select(
                "bitscore", "evalue"
            ).iter_rows()
        ]
        protein_taxid, _ = weighted_majority_taxid(
            protein_taxids, protein_weights, taxdb, majority
        )
        protein_taxon = taxopy.Taxon(protein_taxid, taxdb)
        protein_assignment = {
            "protein": str(group["protein"][0]),
            "contig": str(group["contig"][0]),
            "taxid": protein_taxid,
            "weight": hit_weight(
                float(group["bitscore"].max()),
                float(group["evalue"].min()),
                weight_method,
            ),
            "hit_count": group.height,
        }
        for rank in ICTV_RANKS:
            protein_assignment[f"{rank}_informative_fraction"] = (
                rank_vote_metrics(
                    protein_taxids,
                    protein_weights,
                    protein_taxid,
                    rank,
                    taxdb,
                )[1]
                if any(
                    taxdb.taxid2rank.get(node) == rank
                    for node in protein_taxon.taxid_lineage
                )
                else None
            )
        protein_assignments.append(protein_assignment)

    rows = []
    for group in pl.DataFrame(protein_assignments).partition_by(
        "contig", maintain_order=True
    ):
        contig_id = str(group["contig"][0])
        contig_hits = retained.filter(pl.col("contig") == contig_id)
        best_hit = contig_hits.sort(
            ["bitscore", "evalue", "alignment_length", "target"],
            descending=[True, False, True, False],
        ).row(0, named=True)
        best_match_taxon = taxopy.Taxon(str(best_hit["taxid"]), taxdb)
        contig_mapping = protein_map.filter(pl.col("contig") == contig_id)
        total_proteins = contig_mapping["protein"].n_unique()
        total_protein_residues = (
            int(
                contig_mapping.unique(subset=["protein"])[
                    "protein_length"
                ].sum()
            )
            if "protein_length" in contig_mapping.columns
            else None
        )
        genome_length = (
            int(contig_mapping["contig_length"].drop_nulls()[0])
            if "contig_length" in contig_mapping.columns
            and len(contig_mapping["contig_length"].drop_nulls()) > 0
            else None
        )
        aligned_residues, projected_aligned_nt = alignment_breadth(contig_hits)
        taxids = group["taxid"].cast(pl.String).to_list()
        weights = group["weight"].cast(pl.Float64).to_list()
        assignment_taxid, support = weighted_majority_taxid(
            taxids, weights, taxdb, majority
        )
        taxon = taxopy.Taxon(assignment_taxid, taxdb)
        lineage_taxids = list(reversed(taxon.taxid_lineage))
        lineage_names = list(reversed(taxon.name_lineage))
        contig_informative_fraction = (
            rank_vote_metrics(
                taxids, weights, assignment_taxid, taxon.rank, taxdb
            )[1]
            if taxon.rank in ICTV_RANKS
            else 1.0
        )
        resolved_protein_rows = [
            row
            for row in group.iter_rows(named=True)
            if taxon.rank in ICTV_RANKS
            and any(
                taxdb.taxid2rank.get(node) == taxon.rank
                for node in taxopy.Taxon(row["taxid"], taxdb).taxid_lineage
            )
        ]
        resolved_protein_weight = sum(
            float(row["weight"]) for row in resolved_protein_rows
        )
        protein_informative_fraction = (
            sum(
                float(row["weight"])
                * float(
                    row[f"{taxon.rank}_informative_fraction"] or 0.0
                )
                for row in resolved_protein_rows
            )
            / resolved_protein_weight
            if resolved_protein_weight
            else 1.0
        )
        row = {
            "query": contig_id,
            "taxid": assignment_taxid,
            "rank": taxon.rank,
            "taxon_name": taxon.name,
            "support": support,
            "informative_fraction": (
                float(contig_informative_fraction or 0.0)
                * protein_informative_fraction
            ),
            "best_match_target": str(best_hit["target"]),
            "best_match_taxid": str(best_hit["taxid"]),
            "best_match_taxon": best_match_taxon.name,
            "best_match_rank": best_match_taxon.rank,
            "best_match_bitscore": float(best_hit["bitscore"]),
            "best_match_evalue": float(best_hit["evalue"]),
            "best_match_identity": float(best_hit["identity"]),
            "best_match_alignment_length": int(
                best_hit["alignment_length"]
            ),
            "proteins_assigned": group.height,
            "total_proteins": total_proteins,
            "protein_hit_fraction": (
                group.height / total_proteins if total_proteins else None
            ),
            "aligned_residues": aligned_residues,
            "total_protein_residues": total_protein_residues,
            "residue_alignment_fraction": (
                aligned_residues / total_protein_residues
                if total_protein_residues
                else None
            ),
            "projected_aligned_nt": projected_aligned_nt,
            "genome_length": genome_length,
            "projected_alignment_genome_fraction": (
                projected_aligned_nt / genome_length
                if projected_aligned_nt is not None and genome_length
                else None
            ),
            "hits_retained": int(group["hit_count"].sum()),
            "lineage": ";".join(
                lineage_names if lineage_mode == 1 else lineage_taxids
            )
            if lineage_mode
            else "",
            "backend": backend,
            "method": f"rank-aware-weighted:{weight_method}",
        }
        for rank in ICTV_RANKS:
            rank_taxid = next(
                (
                    node
                    for node in lineage_taxids
                    if taxdb.taxid2rank.get(node) == rank
                ),
                None,
            )
            row[rank] = taxdb.taxid2name.get(rank_taxid) if rank_taxid else None
            row[f"{rank}_support"] = rank_vote_metrics(
                taxids, weights, assignment_taxid, rank, taxdb
            )[0]
        rows.append(row)
    return pl.DataFrame(rows, schema=output_columns)


def prepare_proteins(
    input_path: Path,
    alphabet: str,
    proteins: Path | None,
    protein_map_path: Path | None,
    protein_gff_path: Path | None,
    infer_protein_map: bool,
    work_dir: Path,
    threads: int,
) -> tuple[Path, pl.DataFrame, list[str]]:
    """Resolve or predict proteins and their parent-contig mapping."""
    used_tools = []
    if protein_map_path is not None and protein_gff_path is not None:
        raise click.UsageError(
            "Use either --protein-map or --protein-gff, not both."
        )
    if proteins is not None:
        protein_fasta = proteins
        if protein_map_path is None and not infer_protein_map:
            if protein_gff_path is not None:
                infer_protein_map = False
            else:
                raise click.UsageError(
                    "--proteins with contig input requires --protein-map, "
                    "--protein-gff, or --infer-protein-map for "
                    "RolyPoly-style ORF headers."
                )
    elif alphabet == "protein":
        protein_fasta = input_path
    else:
        if protein_gff_path is not None:
            raise click.UsageError(
                "--protein-gff is only used with existing protein input; "
                "nucleotide input predicted by mmtax uses its generated GFF."
            )
        from rolypoly.utils.bio.translation import pyro_predict_orfs

        protein_fasta = work_dir / "predicted_orfs.faa"
        pyro_predict_orfs(input_path, protein_fasta, threads=threads)
        protein_gff_path = protein_fasta.with_suffix(".gff")
        used_tools.append("pyrodigal-rv")

    identifiers = protein_ids(protein_fasta)
    duplicate_identifiers = (
        identifiers.group_by("protein")
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") > 1)
    )
    if duplicate_identifiers.height:
        raise click.ClickException(
            "Protein FASTA has duplicate search identifiers after whitespace "
            f"parsing; first duplicate: {duplicate_identifiers['protein'][0]}"
        )
    if protein_map_path is not None:
        mapping = read_protein_map(protein_map_path)
    elif protein_gff_path is not None:
        mapping = read_protein_gff_map(protein_gff_path)
    elif infer_protein_map:
        mapping = identifiers.with_columns(
            pl.col("protein")
            .map_elements(infer_contig_id, return_dtype=pl.String)
            .alias("contig")
        )
    else:
        mapping = identifiers.with_columns(pl.col("protein").alias("contig"))

    missing = identifiers.join(mapping, on="protein", how="anti")
    if missing.height:
        raise click.ClickException(
            f"Protein map is missing {missing.height} protein IDs; first missing ID: "
            f"{missing['protein'][0]}"
        )
    mapping = mapping.join(
        sequence_lengths(protein_fasta, "protein", "protein_length"),
        on="protein",
        how="left",
        validate="1:1",
    )
    if protein_gff_path is not None:
        mapping = mapping.join(
            read_single_cds_coordinates(protein_gff_path),
            on="protein",
            how="left",
            validate="1:1",
        )
    if alphabet == "nucl":
        mapping = mapping.join(
            sequence_lengths(input_path, "contig", "contig_length"),
            on="contig",
            how="left",
            validate="m:1",
        )
    if protein_gff_path is not None:
        mapping = (
            mapping.join(
                read_gff_contig_lengths(protein_gff_path),
                on="contig",
                how="left",
                validate="m:1",
            )
            .with_columns(
                pl.coalesce("contig_length", "gff_contig_length").alias(
                    "contig_length"
                )
                if "contig_length" in mapping.columns
                else pl.col("gff_contig_length").alias("contig_length")
            )
            .drop("gff_contig_length")
        )
    return protein_fasta, mapping, used_tools


def search_mmseqs(
    proteins: Path,
    database: Path,
    output: Path,
    tmp_dir: Path,
    threads: int,
    memory: str,
    evalue: float,
    identity: float,
    min_aln_len: int,
    sensitivity: str,
    logger,
) -> None:
    """Run MMseqs2 easy-search and write headered taxonomic protein hits."""
    from rolypoly.utils.various import run_command_comp

    success = run_command_comp(
        base_cmd="mmseqs easy-search",
        positional_args=[
            str(proteins),
            str(database),
            str(output),
            str(tmp_dir),
        ],
        positional_args_location="start",
        params={
            "threads": threads,
            "split-memory-limit": memory.upper(),
            "e": evalue,
            "min-seq-id": identity,
            "min-aln-len": min_aln_len,
            "s": SENSITIVITY_TO_MMSEQS[sensitivity],
            "max-seqs": 1000,
            "format-mode": 4,
            "format-output": (
                "query,target,fident,alnlen,bits,evalue,taxid,"
                "qstart,qend,qlen"
            ),
        },
        check_output=False,
        logger=logger,
    )
    if not success:
        raise click.ClickException("MMseqs2 search failed; see the log above.")


def search_diamond(
    proteins: Path,
    database: Path,
    output: Path,
    tmp_dir: Path,
    threads: int,
    evalue: float,
    identity: float,
    sensitivity: str,
    logger,
) -> None:
    """Run DIAMOND blastp and write taxonomic protein hits."""
    from rolypoly.utils.various import run_command_comp

    params = {
        "query": proteins,
        "db": database,
        "out": output,
        "outfmt": (
            "6 qseqid sseqid pident length bitscore evalue staxids "
            "qstart qend qlen"
        ),
        "threads": threads,
        "tmpdir": tmp_dir,
        "evalue": evalue,
        "id": identity * 100,
        "min-orf": 20,
        "min-query-len": 20,
        "header": "simple",
    }
    preset = SENSITIVITY_TO_DIAMOND[sensitivity]
    if preset:
        params[preset] = True
    success = run_command_comp(
        base_cmd="diamond blastp",
        params=params,
        check_output=False,
        logger=logger,
    )
    if not success:
        raise click.ClickException("DIAMOND search failed; see the log above.")


def read_search_hits(
    path: Path, backend: str, min_aln_len: int
) -> pl.DataFrame:
    """Read and normalize MMseqs2 or DIAMOND taxonomic protein hits."""
    if not path.exists() or path.stat().st_size == 0:
        return pl.DataFrame(
            schema={
                "protein": pl.String,
                "target": pl.String,
                "identity": pl.Float64,
                "alignment_length": pl.Int64,
                "bitscore": pl.Float64,
                "evalue": pl.Float64,
                "taxid": pl.String,
                "query_start": pl.Int64,
                "query_end": pl.Int64,
                "query_length": pl.Int64,
            }
        )
    if backend == "mmseqs":
        frame = pl.read_csv(path, separator="\t")
        frame = frame.rename(
            {
                "query": "protein",
                "alnlen": "alignment_length",
                "bits": "bitscore",
                "qstart": "query_start",
                "qend": "query_end",
                "qlen": "query_length",
            }
        ).with_columns(pl.col("fident").cast(pl.Float64).alias("identity"))
    elif backend == "diamond":
        diamond_columns = [
            "protein",
            "target",
            "identity",
            "alignment_length",
            "bitscore",
            "evalue",
            "taxid",
            "query_start",
            "query_end",
            "query_length",
        ]
        with path.open() as handle:
            first_line = handle.readline()
            has_header = first_line.startswith(("qseqid\t", "qtitle\t"))
        if has_header:
            frame = pl.read_csv(path, separator="\t")
            rename_columns = {
                "qseqid": "protein",
                "qtitle": "protein",
                "sseqid": "target",
                "salltitles": "target",
                "pident": "identity",
                "length": "alignment_length",
                "staxids": "taxid",
                "qstart": "query_start",
                "qend": "query_end",
                "qlen": "query_length",
            }
            frame = frame.rename(
                {
                    old: new
                    for old, new in rename_columns.items()
                    if old in frame.columns
                }
            )
        else:
            frame = pl.read_csv(
                path,
                separator="\t",
                has_header=False,
                new_columns=diamond_columns,
            )
        frame = frame.with_columns(
            pl.col("protein")
            .cast(pl.String)
            .str.split(" ")
            .list.first()
            .alias("protein"),
            pl.col("target")
            .cast(pl.String)
            .str.split(" ")
            .list.first()
            .alias("target"),
            (pl.col("identity").cast(pl.Float64) / 100.0).alias("identity"),
            pl.col("alignment_length").cast(pl.Int64),
            pl.col("bitscore").cast(pl.Float64),
            pl.col("evalue").cast(pl.Float64),
        )
    for column in ("query_start", "query_end", "query_length"):
        if column not in frame.columns:
            frame = frame.with_columns(
                pl.lit(None, dtype=pl.Int64).alias(column)
            )
    return (
        frame.with_columns(pl.col("taxid").cast(pl.String).str.split(";"))
        .explode("taxid", empty_as_null=True)
        .filter(
            pl.col("taxid").str.contains(r"^\d+$")
            & (pl.col("taxid") != "0")
            & (pl.col("alignment_length") >= min_aln_len)
        )
        .with_columns(
            pl.col("bitscore").cast(pl.Float64),
            pl.col("evalue").cast(pl.Float64),
            pl.col("query_start").cast(pl.Int64, strict=False),
            pl.col("query_end").cast(pl.Int64, strict=False),
            pl.col("query_length").cast(pl.Int64, strict=False),
        )
    )


@click.command(name="mmtax")
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(
        exists=True, dir_okay=False, path_type=Path, resolve_path=True
    ),
    help="Input contig or protein FASTA/FASTQ.",
)
@click.option(
    "-q",
    "--query-type",
    type=click.Choice(["auto", "nucl", "protein"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Input alphabet.",
)
@click.option(
    "-p",
    "--proteins",
    type=click.Path(
        exists=True, dir_okay=False, path_type=Path, resolve_path=True
    ),
    default=None,
    help="Existing ORF/protein FASTA for nucleotide contigs; skips ORF prediction.",
)
@click.option(
    "-pm",
    "--protein-map",
    type=click.Path(
        exists=True, dir_okay=False, path_type=Path, resolve_path=True
    ),
    default=None,
    help="Two-column protein-to-contig TSV for --proteins.",
)
@click.option(
    "-pg",
    "--protein-gff",
    type=click.Path(
        exists=True, dir_okay=False, path_type=Path, resolve_path=True
    ),
    default=None,
    help=(
        "GFF/GTF with CDS IDs matching --proteins; uses seqid as the parent "
        "contig."
    ),
)
@click.option(
    "-ipm",
    "--infer-protein-map",
    is_flag=True,
    help="Infer contigs from RolyPoly-style ORF header suffixes.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default="mmtax.tsv",
    show_default=True,
    help="Contig taxonomy assignments TSV.",
)
@click.option(
    "-d",
    "--db",
    "--database",
    "database",
    default="ncbi_virus",
    show_default=True,
    help="ncbi_virus, reserved rvmt/nvpc, or a custom database path.",
)
@click.option(
    "-td",
    "--taxdump",
    type=click.Path(
        exists=True, file_okay=False, path_type=Path, resolve_path=True
    ),
    default=None,
    help="Taxdump for a custom database (required only for custom databases).",
)
@click.option(
    "-b",
    "--backend",
    type=click.Choice(["mmseqs", "diamond"], case_sensitive=False),
    default="mmseqs",
    show_default=True,
)
@click.option(
    "-s",
    "--sens",
    "--sensitivity",
    "sensitivity",
    default="normal",
    show_default=True,
    help="Shared sensitivity preset or numeric level 1-8 (lower is less sensitive but faster)",
)
@click.option(
    "-tp",
    "--top",
    type=click.FloatRange(0.0, 100.0),
    default=10.0,
    show_default=True,
    help="Keep hits within this percent of each protein's best bitscore BEFORE LCA.",
)
@click.option(
    "-w",
    "--weight",
    type=click.Choice(["bitscore", "evalue"]),
    default="bitscore",
    show_default=True,
    help="ORF weight used for contig-level majority voting.",
)
@click.option(
    "-ma",
    "--majority",
    type=click.FloatRange(0.5, 1.0),
    default=0.5,
    show_default=True,
    help="Minimum weighted support for a contig assignment.",
)
@click.option(
    "-tl",
    "--tax-lineage",
    type=click.Choice([0, 1, 2]),
    default=2,
    show_default=True,
    help="Lineage output: 0 none, 1 taxon names, 2 taxids.",
)
@click.option(
    "-e",
    "--evalue",
    type=click.FloatRange(min=0, min_open=True),
    default=1e-5,
    show_default=True,
)
@click.option(
    "-id",
    "--identity",
    type=click.FloatRange(0, 1),
    default=0.1,
    show_default=True,
    help="Minimum percent identity for protein hits (0-1)."
)
@click.option(
    "-al",
    "--min-aln-len",
    type=click.IntRange(min=0),
    default=30,
    show_default=True,
    help="Minimum alignment length for protein hits (in AA residues)."
)
def mmtax(
    input_path: Path,
    query_type: str,
    proteins: Path | None,
    protein_map: Path | None,
    protein_gff: Path | None,
    infer_protein_map: bool,
    output: Path,
    database: str,
    taxdump: Path | None,
    backend: str,
    sensitivity: str,
    top: float,
    weight: str,
    majority: float,
    tax_lineage: int,
    evalue: float,
    identity: float,
    min_aln_len: int,
    threads: int,
    memory: str,
    temp_dir: Path | None,
    keep_tmp: bool,
    log_file: Path,
    log_level: str,
) -> None:
    """Assign ICTV taxonomy using weighted protein-to-contig LCA voting.

    Nucleotide contigs are translated with pyrodigal-rv unless --proteins and a
    protein-to-contig mapping are supplied. Both MMseqs2 and DIAMOND searches
    feed the same post-search top-score filter and taxopy assignment logic.
    """
    import taxopy

    from rolypoly.utils.logging.citation_reminder import remind_citations
    from rolypoly.utils.logging.loggit import log_start_info, setup_logging

    output = output.expanduser().resolve()
    log_file = log_file.expanduser().resolve()
    logger = setup_logging(log_file, log_level)
    sensitivity = normalize_sensitivity(sensitivity)
    database_path, taxdump_path = resolve_database(database, backend, taxdump)
    alphabet = query_alphabet(input_path, query_type)

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_parent = temp_dir.expanduser().resolve() if temp_dir else output.parent
    temp_parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=".mmtax-", dir=temp_parent))
    raw_hits = work_dir / f"{backend}_protein_hits.tsv"
    used_tools = ["mmseqs2" if backend == "mmseqs" else "diamond", "taxopy"]
    log_start_info(
        logger,
        {
            "input": input_path,
            "proteins": proteins,
            "protein_map": protein_map,
            "protein_gff": protein_gff,
            "output": output,
            "database": database_path,
            "taxdump": taxdump_path,
            "backend": backend,
            "sensitivity": sensitivity,
            "top": top,
            "weight": weight,
            "majority": majority,
            "tax_lineage": tax_lineage,
            "evalue": evalue,
            "identity": identity,
            "min_aln_len": min_aln_len,
            "threads": threads,
            "memory": memory,
        },
    )

    try:
        protein_fasta, mapping, protein_tools = prepare_proteins(
            input_path,
            alphabet,
            proteins,
            protein_map,
            protein_gff,
            infer_protein_map,
            work_dir,
            threads,
        )
        used_tools.extend(protein_tools)
        if backend == "mmseqs":
            search_mmseqs(
                protein_fasta,
                database_path,
                raw_hits,
                work_dir / "mmseqs_tmp",
                threads,
                memory,
                evalue,
                identity,
                min_aln_len,
                sensitivity,
                logger,
            )
        else:
            search_diamond(
                protein_fasta,
                database_path,
                raw_hits,
                work_dir,
                threads,
                evalue,
                identity,
                sensitivity,
                logger,
            )
        hits = read_search_hits(raw_hits, backend, min_aln_len)
        taxdb = taxopy.TaxDb(
            nodes_dmp=str(taxdump_path / "nodes.dmp"),
            names_dmp=str(taxdump_path / "names.dmp"),
            keep_files=True,
        )
        assignments = assign_contig_taxonomy(
            hits,
            mapping,
            taxdb,
            top=top,
            weight_method=weight,
            majority=majority,
            lineage_mode=tax_lineage,
            backend=backend,
        )
        assignments.write_csv(output, separator="\t")
    finally:
        if keep_tmp:
            logger.info("Kept temporary files at %s", work_dir)
        else:
            shutil.rmtree(work_dir, ignore_errors=True)

    logger.info(
        "Wrote %d contig taxonomy assignments to %s", assignments.height, output
    )
    if logger.level != 10:
        with log_file.open("a") as log_handle:
            log_handle.write(
                remind_citations(used_tools, return_bibtex=True) or ""
            )


if __name__ == "__main__":
    mmtax()
