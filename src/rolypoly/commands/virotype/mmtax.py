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
        / "protein_taxdb"
        / "ictv_taxdump"
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
    """Choose the deepest lineage node supported by a weighted majority."""
    import taxopy

    total_weight = sum(weights)
    if not taxids or total_weight <= 0:
        return "0", 0.0
    support: dict[str, float] = {}
    depth: dict[str, int] = {}
    for taxid, weight in zip(taxids, weights, strict=True):
        taxon = taxopy.Taxon(str(taxid), taxdb)
        for node_depth, lineage_taxid in enumerate(
            reversed(taxon.taxid_lineage)
        ):
            support[lineage_taxid] = support.get(lineage_taxid, 0.0) + weight
            depth[lineage_taxid] = max(depth.get(lineage_taxid, 0), node_depth)
    eligible = [
        taxid
        for taxid, value in support.items()
        if value / total_weight >= majority and taxid != "1"
    ]
    if not eligible:
        return "1", 1.0
    deepest = max(depth[taxid] for taxid in eligible)
    candidates = [taxid for taxid in eligible if depth[taxid] == deepest]
    assignment = lca_taxid(candidates, taxdb)
    return assignment, support.get(assignment, total_weight) / total_weight


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
        "proteins_assigned": pl.Int64,
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
        protein_assignments.append(
            {
                "protein": str(group["protein"][0]),
                "contig": str(group["contig"][0]),
                "taxid": lca_taxid(
                    group["taxid"].cast(pl.String).to_list(), taxdb
                ),
                "weight": hit_weight(
                    float(group["bitscore"].max()),
                    float(group["evalue"].min()),
                    weight_method,
                ),
                "hit_count": group.height,
            }
        )

    rows = []
    for group in pl.DataFrame(protein_assignments).partition_by(
        "contig", maintain_order=True
    ):
        taxids = group["taxid"].cast(pl.String).to_list()
        weights = group["weight"].cast(pl.Float64).to_list()
        assignment_taxid, support = weighted_majority_taxid(
            taxids, weights, taxdb, majority
        )
        taxon = taxopy.Taxon(assignment_taxid, taxdb)
        lineage_taxids = list(reversed(taxon.taxid_lineage))
        lineage_names = list(reversed(taxon.name_lineage))
        row = {
            "query": str(group["contig"][0]),
            "taxid": assignment_taxid,
            "rank": taxon.rank,
            "taxon_name": taxon.name,
            "support": support,
            "proteins_assigned": group.height,
            "hits_retained": int(group["hit_count"].sum()),
            "lineage": ";".join(
                lineage_names if lineage_mode == 1 else lineage_taxids
            )
            if lineage_mode
            else "",
            "backend": backend,
            "method": f"weighted-lca:{weight_method}",
        }
        total_weight = sum(weights)
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
            row[f"{rank}_support"] = (
                sum(
                    weight
                    for protein_taxid, weight in zip(
                        taxids, weights, strict=True
                    )
                    if rank_taxid
                    in taxopy.Taxon(protein_taxid, taxdb).taxid_lineage
                )
                / total_weight
                if rank_taxid and total_weight
                else None
            )
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
            "format-output": "query,target,fident,alnlen,bits,evalue,taxid",
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
        "outfmt": "6 qseqid sseqid pident length bitscore evalue staxids",
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
            }
        )
    if backend == "mmseqs":
        frame = pl.read_csv(path, separator="\t")
        frame = frame.rename(
            {
                "query": "protein",
                "alnlen": "alignment_length",
                "bits": "bitscore",
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
@click.option(
    "-t", "--threads", type=click.IntRange(min=1), default=1, show_default=True
)
@click.option("-M", "--memory", default="6G", show_default=True)
@click.option(
    "-tmp",
    "--temp-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
)
@click.option("-k", "--keep-tmp", is_flag=True)
@click.option(
    "-g",
    "--log-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default="mmtax.log",
    show_default=True,
)
@click.option("-ll", "--log-level", hidden=True, default="INFO")
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
