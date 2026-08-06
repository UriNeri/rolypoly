"""Interactive, self-contained genome/marker maps for RolyPoly.

Builds a single standalone HTML report (no CDN / internet needed) from RolyPoly
annotation outputs. Rendering uses vanilla JS + inline SVG embedded in the page;
data is embedded as JSON. No plotting library or browser bundler is used.

The report is organised into tabs, each shown only when the corresponding data is
available (and arbitrary extra tabs can be supplied, e.g. predicted taxonomy):

* **Table** / **Genome maps** -- per-contig protein-domain maps (annotate-protein
  ``combined_annotations.tsv``, hmmsearch or mmseqs2/diamond) with an all-hits vs
  best-only toggle and a selectable "best" criterion, plus an RNA track
  (annotate-rna: discrete features classified into rRNA / tRNA / IRES / ribozyme /
  riboswitch / frameshift / UTR / CRE / motif, and a base-pairing-density strip for
  full-length dot-bracket structures) and a nucleic track (per-contig alignments to
  reference viruses).
* **Nucleic hits** -- the nucleic-search (``results_vs_*.tab``) table.
* **Run stats** -- reads filtering and assembly statistics.
* **Extra tabs** -- any additional tabular layers passed by the caller.

Best-hit resolution collapses partial and nested overlaps and sources RolyPoly's
own ``consolidate_hits`` (rolypoly.utils.bio.interval_ops), with a native polars
fallback. Protein and RNA hits are resolved separately. The "source" best-criterion
uses a *precedence* order (RVMT > NVPC > Pfam > genomad > VFAM by default): a hit
from a lower-priority source is only superseded where a higher-priority hit
overlaps it -- lower-priority sources are never globally excluded.

Conventions: polars only (no pandas), native header parsing (no biopython),
generic column mapping via the *Spec dataclasses.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import polars as pl

try:
    from rolypoly.utils.logging.loggit import get_logger

    logger = get_logger()
except Exception:
    import logging

    logger = logging.getLogger("rolypoly.utils.viz.genome_maps")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)


__all__ = [
    "MarkerTableSpec", "load_marker_table", "tag_best_hits", "build_contig_models",
    "infer_marker_spec", "normalize_hit_columns",
    "RnaFeatureSpec", "load_rna_features", "build_rna_by_contig", "attach_rna",
    "classify_rna_feature",
    "NucleicSpec", "build_nucleic_by_contig", "attach_nucleic", "find_nucleic_tables",
    "MotifSpec", "build_motifs_by_contig", "attach_motifs", "find_motif_tables",
    "load_run_stats", "parse_rrna_reference",
    "table_to_tab", "render_html", "write_genome_maps", "build_palette",
    "find_annotation_tables", "write_report_for_dir",
    "DEFAULT_PALETTE", "BEST_CRITERIA", "DEFAULT_SOURCE_PRIORITY",
]

BASE_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]
DEFAULT_PALETTE: dict[str, str] = {
    "rvmt": "#C44E52", "nvpc": "#DD8452", "pfam": "#4C72B0",
    "genomad": "#55A868", "vfam": "#8172B3",
}
RNA_TYPE_COLORS: dict[str, str] = {
    "rRNA": "#2E8B57", "tRNA": "#1F77B4", "IRES": "#9467BD",
    "ribozyme": "#D62728", "riboswitch": "#E377C2", "frameshift": "#BCBD22",
    "UTR": "#17BECF", "CRE": "#FF7F0E", "motif": "#8C564B",
    "structure": "#9aa2b1", "other": "#7F7F7F",
}
NUCLEIC_COLORS: dict[str, str] = {
    "rvmt": "#C44E52", "ncbi_ribovirus": "#3d6fb4", "other": "#6b7280",
}
# RdRp catalytic motifs A-G (A/B/C are the canonical core; D-G seen in some clades).
MOTIF_COLORS: dict[str, str] = {
    "A": "#C44E52", "B": "#4C72B0", "C": "#55A868", "D": "#8172B3",
    "E": "#DD8452", "F": "#17BECF", "G": "#DA8BC3", "other": "#7F7F7F",
}
BEST_CRITERIA: dict[str, str] = {
    "score": "highest score",
    "evalue": "lowest E-value",
    "longest": "longest",
    "source": "source priority",
}
# Precedence order for the "source" criterion: RVMT first, then NVPC, Pfam,
# genomad, VFAM. Lower-priority sources are NOT excluded -- they win wherever no
# higher-priority hit overlaps them.
DEFAULT_SOURCE_PRIORITY = ["rvmt", "nvpc", "pfam", "genomad", "vfam"]
DEFAULT_RNA_SOURCE_PRIORITY = ["Infernal", "Rfam", "tRNAscan-SE", "Aragorn", "IRESfinder", "RNAMotif"]
PAIRED_SYMBOLS = set("()[]{}<>AaBbCcDd")


def build_palette(sources, overrides=None):
    """Return a {source: hex} palette honouring DEFAULT_PALETTE and user overrides,
    assigning distinct fallback colours to any unknown source."""
    palette: dict[str, str] = {}
    used = set(DEFAULT_PALETTE.values())
    fallback = iter(BASE_COLORS)
    for source in sources:
        if overrides and source in overrides:
            palette[source] = overrides[source]
        elif source in DEFAULT_PALETTE:
            palette[source] = DEFAULT_PALETTE[source]
        else:
            colour = next((c for c in fallback if c not in used), None)
            if colour is None:
                colour = BASE_COLORS[abs(hash(source)) % len(BASE_COLORS)]
            palette[source] = colour
            used.add(colour)
    return palette


def parse_orf_header(query):
    """Parse a pyrodigal/Prodigal-style ORF header into gene coordinates.
    Returns keys orf_id, contig, g_start, g_end, strand, contig_len."""
    parts = [p.strip() for p in str(query).split(" # ")]
    orf_id = parts[0]
    g_start = g_end = strand = None
    if len(parts) >= 4:
        try:
            g_start, g_end, strand = int(parts[1]), int(parts[2]), int(parts[3])
        except ValueError:
            pass
    contig = re.sub(r"_\d+$", "", orf_id) if re.search(r"_\d+$", orf_id) else orf_id
    length_match = re.search(r"length_(\d+)", orf_id)
    contig_len = int(length_match.group(1)) if length_match else None
    return {"orf_id": orf_id, "contig": contig, "g_start": g_start,
            "g_end": g_end, "strand": strand, "contig_len": contig_len}


def read_hit_table(path):
    """Read a TSV / CSV / Parquet table by extension into a polars DataFrame.
    Text formats are read as all-Utf8 so mixed/empty cells never break typing."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return pl.read_parquet(path)
    separator = "\t" if suffix in (".tsv", ".tab", ".txt") else ","
    return pl.read_csv(path, separator=separator, infer_schema_length=0,
                       null_values=["", "NA", "nan", "NaN"], quote_char=None)


def nonempty(value):
    """Return a stripped string if ``value`` carries real content, else None.
    Treats '', whitespace, and literal quote-only strings ('""', "''") as empty."""
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", '""', "''", "NA", "nan", "NaN", "None", "."):
        return None
    return text


# MARKERS (annotate-protein / marker-search)
@dataclass
class MarkerTableSpec:
    """Maps logical fields to columns in a hit table. Defaults = annotate-protein
    / marker-search hmmsearch schema."""

    query: str = "query_full_name"
    profile: str = "hmm_full_name"
    source: str = "source"

    env_from: str = "env_from"
    env_to: str = "env_to"
    qlen: str = "qlen"

    hmm_from: str = "hmm_from"
    hmm_to: str = "hmm_to"
    hmm_len: str = "hmm_len"

    evalue: str = "full_hmm_evalue"
    score: str = "full_hmm_score"
    coverage: str = "hmm_cov"
    ali_len: str = "ali_len"

    accession: str | None = "profile_accession"
    aligned_seq: str | None = "identity_str"
    description_cols: tuple = (
        "dom_desc", "nvpc_meta_Description",
        "genomad_meta_ANNOTATION_DESCRIPTION",
        "vfam_meta_ConsensusFunctionalDescription",
    )

    q1: str = "q1"
    q2: str = "q2"

    contig: str | None = None
    contig_len: str | None = None
    nt_from: str | None = None
    nt_to: str | None = None
    g_start: str | None = None
    g_end: str | None = None
    strand: str | None = None

    query_parser: Callable[[str], dict] = field(default=parse_orf_header)
    max_desc_len: int = 260


def normalize_hit_columns(df):
    """Canonicalise column-name variants by sourcing RolyPoly's
    ``normalize_column_names`` (rolypoly.utils.bio.polars_fastx); falls back to the
    same mapping if the import is unavailable. Safe/idempotent on hmmsearch tables."""
    try:
        from rolypoly.utils.bio.polars_fastx import normalize_column_names
        return normalize_column_names(df)
    except Exception:
        mapping = {
            "begin": "start", "from": "start", "seq_from": "start",
            "query_start": "start", "qstart": "start",
            "to": "end", "seq_to": "end", "query_end": "end", "qend": "end",
            "qseqid": "sequence_id", "sequence_ID": "sequence_id",
            "contig_id": "sequence_id", "contig": "sequence_id",
            "query": "sequence_id", "id": "sequence_id", "name": "sequence_id",
            "bitscore": "score", "bit_score": "score", "bits": "score",
            "e_value": "evalue", "tool": "source", "method": "source",
            "db": "source", "database": "source",
            "feature": "type", "annotation": "type", "category": "type",
        }
        rename = {k: v for k, v in mapping.items() if k in df.columns}
        return df.rename(rename) if rename else df


def infer_marker_spec(df):
    """Return a MarkerTableSpec auto-configured for the table's schema (hmmsearch
    vs normalised mmseqs2/diamond BLAST tab-6)."""
    cols = set(df.columns)
    if {"query_full_name", "hmm_full_name"}.issubset(cols):
        return MarkerTableSpec()

    def first(cands, default=None):
        return next((c for c in cands if c in cols), default)

    pos_from = first(["q1", "qstart", "start", "env_from"], "start")
    pos_to = first(["q2", "qend", "end", "env_to"], "end")
    return MarkerTableSpec(
        query=first(["query_full_name", "qseqid", "query", "sequence_id"], "sequence_id"),
        profile=first(["hmm_full_name", "sseqid", "target", "theader", "profile_name"], "sseqid"),
        source=first(["source", "database", "db"], "source"),
        env_from=pos_from, env_to=pos_to, q1=pos_from, q2=pos_to,
        qlen=first(["qlen", "query_length"], "qlen"),
        hmm_from=first(["hmm_from", "sstart"], "sstart"),
        hmm_to=first(["hmm_to", "send"], "send"),
        hmm_len=first(["hmm_len", "slen", "tlen"], "slen"),
        evalue=first(["full_hmm_evalue", "evalue", "e_value"], "evalue"),
        score=first(["full_hmm_score", "score", "bitscore", "bits"], "score"),
        coverage=first(["hmm_cov", "coverage"], "hmm_cov"),
        ali_len=first(["ali_len", "length", "alnlen"], "length"),
        accession=first(["profile_accession", "target_accession", "accession"], "profile_accession"),
        aligned_seq=first(["identity_str", "aligned_region"], None),
        description_cols=(
            "dom_desc", "nvpc_meta_Description",
            "genomad_meta_ANNOTATION_DESCRIPTION",
            "vfam_meta_ConsensusFunctionalDescription", "stitle", "salltitles",
        ),
    )


def load_marker_table(data, spec=None, min_score=None, max_evalue=None):
    """Load and normalise a protein/marker hit table into a canonical schema.
    If ``spec`` is None the schema is auto-detected. Adds a stable ``rp_row_uid``."""
    df = data.clone() if isinstance(data, pl.DataFrame) else read_hit_table(data)
    if spec is None:
        df = normalize_hit_columns(df)
        spec = infer_marker_spec(df)
    df = df.with_row_index("rp_row_uid")

    parsed = pl.DataFrame([spec.query_parser(q) for q in df[spec.query].to_list()])

    def take(colname, key, dtype):
        if colname and colname in df.columns:
            return pl.col(colname).cast(dtype, strict=False)
        return pl.Series(key, parsed[key]).cast(dtype, strict=False)

    df = df.with_columns(orf_id=parsed["orf_id"]).with_columns([
        take(spec.contig, "contig", pl.Utf8).alias("contig"),
        take(spec.contig_len, "contig_len", pl.Float64).alias("contig_len"),
        take(spec.g_start, "g_start", pl.Float64).alias("g_start"),
        take(spec.g_end, "g_end", pl.Float64).alias("g_end"),
        take(spec.strand, "strand", pl.Float64).alias("strand"),
    ])

    def numcol(name):
        return (pl.col(name).cast(pl.Float64, strict=False) if name in df.columns
                else pl.lit(None, dtype=pl.Float64))

    df = df.with_columns([
        pl.col(spec.env_from).cast(pl.Float64, strict=False).alias("aa_from"),
        pl.col(spec.env_to).cast(pl.Float64, strict=False).alias("aa_to"),
        numcol(spec.qlen).alias("qlen_n"),
        numcol(spec.hmm_from).alias("hmm_from_n"),
        numcol(spec.hmm_to).alias("hmm_to_n"),
        numcol(spec.hmm_len).alias("hmm_len_n"),
        numcol(spec.evalue).alias("evalue"),
        numcol(spec.score).alias("score"),
        numcol(spec.coverage).alias("coverage"),
        numcol(spec.ali_len).alias("ali_len_n"),
        pl.col(spec.profile).cast(pl.Utf8).alias("profile"),
        pl.col(spec.source).cast(pl.Utf8).alias("source_n"),
    ])

    df = df.with_columns(
        (pl.col(spec.accession).cast(pl.Utf8) if spec.accession in df.columns
         else pl.lit(None, dtype=pl.Utf8)).alias("accession"),
        (pl.col(spec.aligned_seq).cast(pl.Utf8)
         if (spec.aligned_seq and spec.aligned_seq in df.columns)
         else pl.lit(None, dtype=pl.Utf8)).alias("aligned_seq"),
    )

    desc_present = [c for c in spec.description_cols if c in df.columns]
    desc_expr = (pl.coalesce([
        pl.when(pl.col(c).cast(pl.Utf8).str.strip_chars().str.len_chars() > 0)
        .then(pl.col(c).cast(pl.Utf8)).otherwise(None) for c in desc_present
    ]) if desc_present else pl.lit(None, dtype=pl.Utf8))
    df = df.with_columns(desc_expr.alias("description"))

    df = df.with_columns([
        pl.col("strand").fill_null(1).alias("strand"),
        pl.col("g_start").fill_null(1).alias("g_start"),
        pl.when(pl.col("g_end").is_null())
        .then(pl.when(pl.col("qlen_n").is_not_null() & (pl.col("qlen_n") > 0))
              .then(pl.col("qlen_n") * 3).otherwise(pl.col("aa_to") * 3))
        .otherwise(pl.col("g_end")).alias("g_end"),
    ])

    if (spec.nt_from and spec.nt_from in df.columns
            and spec.nt_to and spec.nt_to in df.columns):
        df = df.with_columns([
            pl.col(spec.nt_from).cast(pl.Float64, strict=False).alias("nt_from"),
            pl.col(spec.nt_to).cast(pl.Float64, strict=False).alias("nt_to"),
        ])
    else:
        nt_a = (pl.when(pl.col("strand") == -1)
                .then(pl.col("g_end") - (pl.col("aa_from") - 1) * 3)
                .otherwise(pl.col("g_start") + (pl.col("aa_from") - 1) * 3))
        nt_b = (pl.when(pl.col("strand") == -1)
                .then(pl.col("g_end") - (pl.col("aa_to") - 1) * 3)
                .otherwise(pl.col("g_start") + (pl.col("aa_to") - 1) * 3))
        df = df.with_columns([pl.min_horizontal(nt_a, nt_b).alias("nt_from"),
                              pl.max_horizontal(nt_a, nt_b).alias("nt_to")])

    df = df.with_columns(
        pl.when(pl.col("contig_len").is_null() | (pl.col("contig_len") <= 0))
        .then(pl.max_horizontal(pl.col("nt_to"), pl.col("g_end")).max().over("contig"))
        .otherwise(pl.col("contig_len")).alias("contig_len")
    )

    if min_score is not None:
        df = df.filter(pl.col("score") >= min_score)
    if max_evalue is not None:
        df = df.filter(pl.col("evalue") <= max_evalue)

    logger.info("genome_maps: loaded %d protein hits / %d contigs / %d ORFs / %d source(s)",
                df.height, df["contig"].n_unique(), df["orf_id"].n_unique(),
                df["source_n"].n_unique())
    return df


def rank_columns_for_criterion(criterion, spec, columns):
    """Build a consolidate_hits rank string for a criterion from available columns."""
    have = columns.__contains__
    if criterion == "evalue":
        tokens = ([f"+{spec.evalue}"] if have(spec.evalue) else []) + \
                 ([f"-{spec.score}"] if have(spec.score) else [])
    elif criterion == "longest":
        tokens = ["-rp_width"] + ([f"-{spec.score}"] if have(spec.score) else [])
    elif criterion == "source":
        tokens = ["+rp_src_rank"] + ([f"-{spec.score}"] if have(spec.score) else [])
    else:
        tokens = ([f"-{spec.score}"] if have(spec.score) else []) + \
                 ([f"+{spec.evalue}"] if have(spec.evalue) else [])
    return ",".join(tokens) or "-rp_width"


def native_best_per_range(slim, spec, criterion, min_overlap_positions):
    """Dependency-free per-query greedy resolution mirroring one_per_range."""
    if criterion == "evalue":
        by, descending = ["rp_ev", "rp_sc"], [False, True]
    elif criterion == "longest":
        by, descending = ["rp_width", "rp_sc"], [True, True]
    elif criterion == "source":
        by, descending = ["rp_src_rank", "rp_sc"], [False, True]
    else:
        by, descending = ["rp_sc", "rp_ev"], [True, False]
    work = slim.sort(by, descending=descending, nulls_last=True)
    keep: set[int] = set()
    for _, sub in work.group_by(spec.query, maintain_order=True):
        placed: list[tuple[int, int]] = []
        for row in sub.iter_rows(named=True):
            start, end = int(min(row["rp_q1"], row["rp_q2"])), int(max(row["rp_q1"], row["rp_q2"]))
            if all(min(end, pe) - max(start, ps) < min_overlap_positions for ps, pe in placed):
                keep.add(int(row["rp_row_uid"]))
                placed.append((start, end))
    return keep


def tag_best_hits(df, spec=None, criteria=tuple(BEST_CRITERIA),
                  min_overlap_positions=1, source_priority=None):
    """Add one boolean ``is_best_<criterion>`` column per criterion (score / evalue
    / longest / source). Sources RolyPoly's ``consolidate_hits`` when importable,
    else a native fallback.

    The "source" criterion applies ``source_priority`` as a *precedence* order (via
    ``rp_src_rank``, tie-broken by score): a hit is kept unless a higher-priority
    source overlaps it. Unlisted sources sort last but are still eligible -- i.e. a
    source lower in the list (e.g. VFAM) still wins any locus that no
    higher-priority hit covers; it is never globally excluded.
    """
    spec = spec or MarkerTableSpec()
    source_priority = list(source_priority or DEFAULT_SOURCE_PRIORITY)
    columns = set(df.columns)
    have_pos = spec.q1 in columns and spec.q2 in columns
    priority_map = {s: i for i, s in enumerate(source_priority)}

    df = df.with_columns([
        (pl.col(spec.q2).cast(pl.Int64, strict=False)
         - pl.col(spec.q1).cast(pl.Int64, strict=False)).abs().alias("rp_width")
        if have_pos else (pl.col("aa_to") - pl.col("aa_from")).abs().alias("rp_width"),
        pl.col("source_n").replace_strict(priority_map, default=len(source_priority))
          .cast(pl.Int64).alias("rp_src_rank"),
    ])

    slim_native = df.select([
        pl.col("rp_row_uid"), pl.col(spec.query).alias(spec.query),
        (pl.col(spec.q1).cast(pl.Int64, strict=False) if have_pos
         else pl.col("aa_from").cast(pl.Int64)).alias("rp_q1"),
        (pl.col(spec.q2).cast(pl.Int64, strict=False) if have_pos
         else pl.col("aa_to").cast(pl.Int64)).alias("rp_q2"),
        pl.col("score").alias("rp_sc"), pl.col("evalue").alias("rp_ev"),
        pl.col("rp_width"), pl.col("rp_src_rank"),
    ])

    consolidate = None
    if have_pos:
        try:
            from rolypoly.utils.bio.interval_ops import consolidate_hits
            consolidate = consolidate_hits
        except Exception as exc:
            logger.info("genome_maps: consolidate_hits unavailable (%s); native fallback", exc)

    for criterion in criteria:
        if consolidate is not None:
            slim = df.select([
                c for c in {"rp_row_uid", spec.query, spec.profile, spec.q1, spec.q2,
                            spec.score, spec.evalue, "rp_width", "rp_src_rank"}
                if c in df.columns
            ]).with_columns([
                pl.col(spec.score).cast(pl.Float64, strict=False),
                pl.col(spec.evalue).cast(pl.Float64, strict=False),
            ])
            rank_columns = rank_columns_for_criterion(criterion, spec, set(slim.columns))
            try:
                resolved = consolidate(
                    input=slim, one_per_range=True,
                    column_specs=f"{spec.query},{spec.profile}",
                    rank_columns=rank_columns,
                    min_overlap_positions=min_overlap_positions,
                )
                best_uids = set(resolved["rp_row_uid"].to_list())
            except Exception as exc:
                logger.info("genome_maps: consolidate_hits failed for '%s' (%s); native", criterion, exc)
                best_uids = native_best_per_range(slim_native, spec, criterion, min_overlap_positions)
        else:
            best_uids = native_best_per_range(slim_native, spec, criterion, min_overlap_positions)

        df = df.with_columns(pl.col("rp_row_uid").is_in(list(best_uids)).alias(f"is_best_{criterion}"))
        logger.info("genome_maps: best[%s] kept %d/%d", criterion, len(best_uids), df.height)

    return df


def clean_value(value, kind="str"):
    """Coerce a polars cell to a JSON-safe python scalar (dropping NaN)."""
    if value is None:
        return None
    if kind == "int":
        return None if (isinstance(value, float) and math.isnan(value)) else int(value)
    if kind == "float":
        return None if (isinstance(value, float) and math.isnan(value)) else float(value)
    return value


def build_contig_models(df, spec=None, criteria=tuple(BEST_CRITERIA)):
    """Turn a normalised protein frame into JSON-ready per-contig dicts, sorted by
    best score. Each hit carries a ``best`` map {criterion: bool}."""
    spec = spec or MarkerTableSpec()
    for criterion in criteria:
        column = f"is_best_{criterion}"
        if column not in df.columns:
            df = df.with_columns(pl.lit(True).alias(column))
    contigs: list[dict] = []

    for contig, contig_df in df.group_by("contig", maintain_order=True):
        contig = contig[0] if isinstance(contig, tuple) else contig
        contig_len = int(contig_df["contig_len"].max() or contig_df["nt_to"].max() or 1)
        orfs = []
        for orf_id, orf_df in contig_df.group_by("orf_id", maintain_order=True):
            orf_id = orf_id[0] if isinstance(orf_id, tuple) else orf_id
            first = orf_df.row(0, named=True)
            hits = []
            for row in orf_df.iter_rows(named=True):
                desc = row["description"] or ""
                if len(desc) > spec.max_desc_len:
                    desc = desc[: spec.max_desc_len - 3] + "..."
                aligned = row.get("aligned_seq") or ""
                if len(aligned) > 400:
                    aligned = aligned[:397] + "..."
                hits.append({
                    "source": row["source_n"], "profile": str(row["profile"]),
                    "acc": row["accession"] or "",
                    "evalue": clean_value(row["evalue"], "float"),
                    "score": None if row["score"] is None else round(row["score"], 1),
                    "cov": None if row["coverage"] is None else round(row["coverage"], 3),
                    "ali_len": clean_value(row["ali_len_n"], "int"),
                    "aa_from": clean_value(row["aa_from"], "int"),
                    "aa_to": clean_value(row["aa_to"], "int"),
                    "hmm_from": clean_value(row["hmm_from_n"], "int"),
                    "hmm_to": clean_value(row["hmm_to_n"], "int"),
                    "hmm_len": clean_value(row["hmm_len_n"], "int"),
                    "nt_from": clean_value(row["nt_from"], "int"),
                    "nt_to": clean_value(row["nt_to"], "int"),
                    "aln": aligned, "desc": desc,
                    "best": {c: bool(row[f"is_best_{c}"]) for c in criteria},
                })
            hits.sort(key=lambda h: (h["nt_from"] or 0, h["nt_to"] or 0))
            orfs.append({
                "orf_id": orf_id, "start": clean_value(first["g_start"], "int"),
                "end": clean_value(first["g_end"], "int"),
                "strand": clean_value(first["strand"], "int") or 1,
                "qlen": clean_value(first["qlen_n"], "int"), "hits": hits,
            })

        all_hits = [h for orf in orfs for h in orf["hits"]]
        best_score = max((h["score"] or 0) for h in all_hits)
        sources = sorted({h["source"] for h in all_hits})
        top = max(all_hits, key=lambda h: (h["score"] or 0))
        name_match = re.match(r"(.+?)_length_(\d+)", contig)
        short = f"{name_match.group(1)} ({int(name_match.group(2))} bp)" if name_match else contig
        n_best = {c: sum(1 for h in all_hits if h["best"][c]) for c in criteria}
        contigs.append({
            "contig": contig, "short": short, "length": contig_len, "orfs": orfs,
            "n_orfs": len(orfs), "n_hits": len(all_hits), "n_best": n_best,
            "n_source": len(sources), "sources": sources,
            "best_score": round(best_score, 1), "top_profile": top["profile"],
            "rna": None, "nucleic": None, "motifs": None,
        })

    contigs.sort(key=lambda c: -c["best_score"])
    return contigs


# RNA (annotate-rna output)
@dataclass
class RnaFeatureSpec:
    """Column mapping for a RolyPoly annotate-rna GFF-like table."""

    seq_id: str = "sequence_id"
    type: str = "type"
    start: str = "start"
    end: str = "end"
    score: str = "score"
    source: str = "source"
    strand: str = "strand"
    profile: str = "profile_name"
    evalue: str = "evalue"
    ribozyme_desc: str = "ribozyme_description"
    trna_type: str = "tRNA_type"
    anticodon: str = "anticodon"
    motif_type: str = "motif_type"
    structure: str = "structure"
    sequence: str = "sequence"
    structure_type: str = "RNA_secondary_structure"
    max_seq_preview: int = 60


def classify_rna_feature(rtype, profile="", descr="", motif=""):
    """Classify an annotate-rna discrete feature into a display class.

    annotate-rna often labels cmsearch/Rfam hits generically (e.g. ``type ==
    'ribozyme'``) while the real class lives in ``profile_name`` /
    ``ribozyme_description`` (IRES, rRNA, 5'/3'UTR, frameshift element, ...). Maps
    to: rRNA / tRNA / IRES / ribozyme / riboswitch / frameshift / UTR / CRE / motif
    / structure / other. Do NOT assume tRNA.
    """
    text = f"{rtype} {profile} {descr} {motif}".lower()
    if "ires" in text:
        return "IRES"
    if any(k in text for k in ("rrna", "ribosomal", "ssu", "lsu",
                               "16s", "18s", "23s", "28s", "5.8s")):
        return "rRNA"
    if "trna" in text:
        return "tRNA"
    if "riboswitch" in text:
        return "riboswitch"
    if "frameshift" in text or "fse" in text:
        return "frameshift"
    if "utr" in text:
        return "UTR"
    if any(k in text for k in ("ribozyme", "hammerhead", "hdv", "hairpin",
                               "twister", "pistol", "hatchet", "glms")):
        return "ribozyme"
    if "cre" in text or "cis-acting" in text or "cis-regulatory" in text:
        return "CRE"
    if (rtype or "").strip() == "RNA_secondary_structure":
        return "structure"
    return (rtype or "other").strip() or "other"


def rna_feature_note(klass, spec, row, get):
    """Build a concise, informative note for a discrete RNA feature.

    Uses only meaningful fields: the human description (ribozyme_description) for
    all classes, tRNA type + anticodon *only* for tRNA features, and motif type
    *only* for motif features. Empty/placeholder values are dropped, so we never
    emit noise like ``tRNA: "" \xb7 anticodon: "" \xb7 motif: ""``.
    """
    notes = []
    descr = nonempty(get(row, spec.ribozyme_desc))
    if descr:
        notes.append(descr)
    if klass == "tRNA":
        trna = nonempty(get(row, spec.trna_type))
        anti = nonempty(get(row, spec.anticodon))
        if trna:
            notes.append(f"tRNA-{trna}")
        if anti:
            notes.append(f"anticodon {anti}")
    if klass == "motif":
        motif = nonempty(get(row, spec.motif_type))
        if motif:
            notes.append(f"motif {motif}")
    return " · ".join(notes)


def pairing_density_profile(dbn, nbins):
    """Return (windowed paired fraction, overall paired fraction) for a DBN string."""
    length = len(dbn)
    if length == 0:
        return [], 0.0
    bin_size = max(1, math.ceil(length / nbins))
    density = []
    for start in range(0, length, bin_size):
        segment = dbn[start:start + bin_size]
        paired = sum(1 for c in segment if c in PAIRED_SYMBOLS)
        density.append(round(paired / len(segment), 3))
    overall = round(sum(1 for c in dbn if c in PAIRED_SYMBOLS) / length, 3)
    return density, overall


def gc_fraction(seq):
    """Return GC fraction over ACGT/U, or None for empty/non-nucleotide input."""
    if not seq:
        return None
    seq = seq.upper()
    gc = sum(1 for c in seq if c in "GC")
    total = sum(1 for c in seq if c in "ACGTU")
    return round(gc / total, 3) if total else None


def to_float(value, default):
    """Best-effort float conversion with a fallback default."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def resolve_rna_feature_best(features, criteria, source_priority, min_overlap_positions=1):
    """Tag each discrete RNA feature in-place with best[criterion]=bool, resolved
    separately from protein hits (native greedy one-per-range, per contig)."""
    priority = {s: i for i, s in enumerate(source_priority)}

    def sort_key(criterion):
        if criterion == "evalue":
            return lambda f: (to_float(f["evalue"], math.inf), -(f["score"] or -math.inf))
        if criterion == "longest":
            return lambda f: (-(((f["end"] or 0) - (f["start"] or 0))), -(f["score"] or -math.inf))
        if criterion == "source":
            return lambda f: (priority.get(f["source"], len(priority)), -(f["score"] or -math.inf))
        return lambda f: (-(f["score"] or -math.inf), to_float(f["evalue"], math.inf))

    for criterion in criteria:
        placed: list[tuple[int, int]] = []
        best_ids = set()
        for feature in sorted(features, key=sort_key(criterion)):
            if feature["start"] is None or feature["end"] is None:
                continue
            start, end = min(feature["start"], feature["end"]), max(feature["start"], feature["end"])
            if all(min(end, pe) - max(start, ps) < min_overlap_positions for ps, pe in placed):
                best_ids.add(id(feature))
                placed.append((start, end))
        for feature in features:
            feature.setdefault("best", {})[criterion] = id(feature) in best_ids


def load_rna_features(data, spec=None):
    """Read an annotate-rna table (TSV/CSV/Parquet or DataFrame) as-is."""
    spec = spec or RnaFeatureSpec()
    return data.clone() if isinstance(data, pl.DataFrame) else read_hit_table(data)


def build_rna_by_contig(data, spec=None, nbins=150, criteria=tuple(BEST_CRITERIA),
                        source_priority=None):
    """Group annotate-rna rows by contig into
    {contig: {"features": [...], "structure": {...} | None}}. Discrete features are
    classified (``klass``) and carry their own best[criterion] flags; full-length
    DBN rows become a base-pairing-density profile."""
    spec = spec or RnaFeatureSpec()
    source_priority = list(source_priority or DEFAULT_RNA_SOURCE_PRIORITY)
    df = load_rna_features(data, spec)
    columns = set(df.columns)

    def get(row, key, default=None):
        return row.get(key, default) if key in columns else default

    out: dict[str, dict] = {}
    for row in df.iter_rows(named=True):
        contig = str(row[spec.seq_id])   # sequence_id IS the contig id (no ORF suffix)
        entry = out.setdefault(contig, {"features": [], "structure": None})
        rtype = (get(row, spec.type) or "").strip()
        structure = get(row, spec.structure) or ""
        sequence = get(row, spec.sequence) or ""
        try:
            start = int(float(get(row, spec.start))) if get(row, spec.start) not in (None, "") else None
            end = int(float(get(row, spec.end))) if get(row, spec.end) not in (None, "") else None
        except (TypeError, ValueError):
            start = end = None
        score = get(row, spec.score)
        try:
            score = float(score) if score not in (None, "") else None
        except (TypeError, ValueError):
            score = None

        is_structure = (rtype == spec.structure_type) or (structure and rtype in ("", spec.structure_type))
        if is_structure and structure:
            density, overall = pairing_density_profile(structure, nbins)
            length = len(structure)
            entry["structure"] = {
                "start": start or 1, "end": end or length, "len": length,
                "source": get(row, spec.source) or "", "mfe": score,
                "paired_fraction": overall, "gc": gc_fraction(sequence),
                "nbins": len(density), "density": density,
            }
        else:
            profile = nonempty(get(row, spec.profile)) or ""
            descr = nonempty(get(row, spec.ribozyme_desc)) or ""
            motif = nonempty(get(row, spec.motif_type)) or ""
            klass = classify_rna_feature(rtype, profile, descr, motif)
            preview = sequence[: spec.max_seq_preview] + ("…" if len(sequence) > spec.max_seq_preview else "")
            entry["features"].append({
                "type": rtype or "other", "klass": klass, "start": start, "end": end,
                "score": None if score is None else round(score, 1),
                "source": get(row, spec.source) or "", "strand": get(row, spec.strand) or "",
                "profile": profile, "evalue": get(row, spec.evalue) or "",
                "note": rna_feature_note(klass, spec, row, get),
                "dbn": structure[:120], "seq": preview,
            })

    for entry in out.values():
        if entry["features"]:
            resolve_rna_feature_best(entry["features"], criteria, source_priority)
    return out


def attach_rna(contigs, rna_by_contig, criteria=tuple(BEST_CRITERIA)):
    """Attach RNA data onto matching contig models and append structure-only
    contigs (no protein hits) so nothing is lost."""
    present = {c["contig"] for c in contigs}
    for contig in contigs:
        if contig["contig"] in rna_by_contig:
            contig["rna"] = rna_by_contig[contig["contig"]]
    for contig_id, rna in rna_by_contig.items():
        if contig_id in present:
            continue
        if not rna.get("structure") and not rna.get("features"):
            continue
        length = (rna["structure"]["len"] if rna.get("structure")
                  else max((f["end"] or 0) for f in rna["features"]) or 1)
        name_match = re.match(r"(.+?)_length_(\d+)", contig_id)
        short = f"{name_match.group(1)} ({int(name_match.group(2))} bp)" if name_match else contig_id
        contigs.append({
            "contig": contig_id, "short": short, "length": length, "orfs": [],
            "n_orfs": 0, "n_hits": 0, "n_best": {c: 0 for c in criteria},
            "n_source": 0, "sources": [], "best_score": 0.0,
            "top_profile": "(RNA only)", "rna": rna, "nucleic": None, "motifs": None,
        })
    return contigs


# NUCLEIC search (results_vs_*.tab)
@dataclass
class NucleicSpec:
    """Column mapping for nucleic-search (virus-mapping / mmseqs) output."""

    query: str = "qheader"
    target: str = "theader"
    qlen: str = "qlen"
    tlen: str = "tlen"
    qstart: str = "qstart"
    qend: str = "qend"
    tstart: str = "tstart"
    tend: str = "tend"
    alnlen: str = "alnlen"
    pident: str = "pident"
    score: str = "bits"
    evalue: str = "evalue"
    qcov: str = "qcov"
    tcov: str = "tcov"
    strand: str = "strand"


def find_nucleic_tables(output_dir):
    """Return a list of (source_label, path) for nucleic-search result tables
    (``*_vs_*.tab`` / ``results.tab``) found recursively under a directory."""
    output_dir = Path(output_dir)
    found = []
    for path in sorted(output_dir.glob("**/*_vs_*.tab")):
        label = path.stem.split("_vs_")[-1] or "nucleic"
        found.append((label, path))
    if not found:
        for path in sorted(output_dir.glob("**/nucleic_search*/**/*.tab")):
            found.append((path.stem, path))
    return found


def build_nucleic_by_contig(tables, spec=None):
    """Build {contig: [hit, ...]} (+ ``__all__`` flat list) from nucleic-search
    table(s). ``tables`` may be a path/DataFrame or a list of (label, path|df)."""
    spec = spec or NucleicSpec()
    if not isinstance(tables, list):
        tables = [(None, tables)]

    out: dict[str, list] = {}
    flat: list[dict] = []
    for label, data in tables:
        df = data if isinstance(data, pl.DataFrame) else read_hit_table(data)
        cols = set(df.columns)

        def g(row, key, default=None):
            return row.get(key, default) if key in cols else default

        for row in df.iter_rows(named=True):
            contig = str(g(row, spec.query, ""))
            if not contig:
                continue
            hit = {
                "source": label or (g(row, "source") or "nucleic"),
                "target": g(row, spec.target, ""),
                "qstart": int(to_float(g(row, spec.qstart), 0)),
                "qend": int(to_float(g(row, spec.qend), 0)),
                "tstart": int(to_float(g(row, spec.tstart), 0)),
                "tend": int(to_float(g(row, spec.tend), 0)),
                "qlen": int(to_float(g(row, spec.qlen), 0)),
                "tlen": int(to_float(g(row, spec.tlen), 0)),
                "pident": round(to_float(g(row, spec.pident), 0.0), 1),
                "score": round(to_float(g(row, spec.score), 0.0), 1),
                "evalue": g(row, spec.evalue, ""),
                "qcov": round(to_float(g(row, spec.qcov), 0.0), 3),
                "tcov": round(to_float(g(row, spec.tcov), 0.0), 3),
                "strand": g(row, spec.strand, ""),
                "contig": contig,
            }
            out.setdefault(contig, []).append(hit)
            flat.append(hit)
    for contig in out:
        out[contig].sort(key=lambda h: -(h["score"] or 0))
    out["__all__"] = sorted(flat, key=lambda h: -(h["score"] or 0))
    logger.info("genome_maps: loaded %d nucleic hits across %d contigs",
                len(flat), len([k for k in out if k != "__all__"]))
    return out


def attach_nucleic(contigs, nucleic_by_contig):
    """Attach nucleic hit lists onto matching contig models by contig id."""
    for contig in contigs:
        hits = nucleic_by_contig.get(contig["contig"])
        if hits:
            contig["nucleic"] = hits
    return contigs


# RdRp motifs (rdrp-motif-search output)
@dataclass
class MotifSpec:
    """Column mapping for a rdrp-motif-search results table."""

    seq_id: str = "sequence_id"
    seq_length: str = "sequence_length"
    conformation: str = "motif_conformation"
    total: str = "total_motifs"
    details: str = "motif_details"


def parse_frame_header(seq_id):
    """Split a rdrp-motif ``sequence_id`` like ``CID_001_frame=2`` into
    (contig_id, frame). Frame is +1..+3 (forward) or -1..-3 (reverse); returns
    (contig, 1) when no frame tag is present."""
    text = str(seq_id).strip()
    frame = 1
    match = re.search(r"_frame=(-?\d+)\s*$", text)
    if match:
        frame = int(match.group(1))
        text = text[: match.start()]
    return text, frame


def find_motif_tables(output_dir):
    """Return rdrp-motif-search result tables (``*rdrp_motif*results.tsv``) found
    recursively under a directory."""
    output_dir = Path(output_dir)
    return sorted(output_dir.glob("**/*rdrp_motif*results.tsv"))


def build_motifs_by_contig(tables, spec=None):
    """Parse rdrp-motif-search table(s) into {contig: {"conformation", "motifs"}}.

    Each motif carries its letter (A/B/C/D), amino-acid span within the frame, the
    frame, and profile / score / evalue / alignment. Nucleotide coordinates are
    left to :func:`attach_motifs`, which knows each contig's length.
    """
    spec = spec or MotifSpec()
    if not isinstance(tables, list):
        tables = [tables]

    out: dict[str, dict] = {}
    for data in tables:
        df = data if isinstance(data, pl.DataFrame) else read_hit_table(data)
        cols = set(df.columns)
        if spec.seq_id not in cols or spec.details not in cols:
            continue
        for row in df.iter_rows(named=True):
            contig, frame = parse_frame_header(row[spec.seq_id])
            raw = row[spec.details] or ""
            try:
                details = json.loads(raw) if raw else {}
            except (TypeError, ValueError):
                # The details column is CSV-quoted (wrapped in "..." with inner
                # quotes doubled as ""); our reader keeps quotes literal, so undo
                # that escaping before parsing.
                text = str(raw).strip()
                if text.startswith('"') and text.endswith('"'):
                    text = text[1:-1]
                text = text.replace('""', '"')
                try:
                    details = json.loads(text)
                except (TypeError, ValueError):
                    details = {}
            motifs = []
            for letter, hits in details.items():
                for hit in hits if isinstance(hits, list) else [hits]:
                    motifs.append({
                        "letter": letter,
                        "aa_from": int(to_float(hit.get("start"), 0)),
                        "aa_to": int(to_float(hit.get("end"), 0)),
                        "frame": frame,
                        "score": round(to_float(hit.get("score"), 0.0), 1),
                        "evalue": hit.get("evalue", ""),
                        "profile": hit.get("profile", ""),
                        "alignment": hit.get("alignment", ""),
                    })
            if not motifs:
                continue
            entry = out.setdefault(contig, {"conformation": "", "motifs": []})
            # Prefer the richest conformation string seen for a contig.
            conf = row.get(spec.conformation) or ""
            if len(str(conf)) > len(entry["conformation"]):
                entry["conformation"] = str(conf)
            entry["motifs"].extend(motifs)
    return out


def attach_motifs(contigs, motifs_by_contig):
    """Attach RdRp motifs to contig models, converting each motif's amino-acid
    span within its reading frame to nucleotide coordinates on the contig.

    Frame f>0 reads the forward strand at offset ``f-1``; f<0 reads the reverse
    strand at offset ``|f|-1``. Coordinates are clamped to the contig length.
    """
    for contig in contigs:
        entry = motifs_by_contig.get(contig["contig"])
        if not entry:
            continue
        length = contig.get("length") or 0
        placed = []
        for motif in entry["motifs"]:
            frame = motif["frame"]
            offset = abs(frame) - 1
            aa_from, aa_to = motif["aa_from"], motif["aa_to"]
            if frame >= 0:
                nt_from = offset + (aa_from - 1) * 3 + 1
                nt_to = offset + aa_to * 3
            else:
                nt_to = length - offset - (aa_from - 1) * 3
                nt_from = length - offset - aa_to * 3 + 1
            nt_from = max(1, min(nt_from, length or nt_from))
            nt_to = max(1, min(nt_to, length or nt_to))
            placed.append({**motif,
                           "nt_from": int(min(nt_from, nt_to)),
                           "nt_to": int(max(nt_from, nt_to))})
        placed.sort(key=lambda m: m["nt_from"])
        contig["motifs"] = {"conformation": entry["conformation"], "motifs": placed}
    return contigs


# RUN STATS (filter-reads + assembly)
def compute_n50(lengths):
    """Return the N50 of a list of contig lengths (0 if empty)."""
    lengths = sorted((int(x) for x in lengths if x), reverse=True)
    if not lengths:
        return 0
    half = sum(lengths) / 2
    cumulative = 0
    for length in lengths:
        cumulative += length
        if cumulative >= half:
            return length
    return lengths[-1]


def summarize_lengths(lengths):
    """Return a small stats dict (n_contigs, total_bp, max, min, mean, n50) for a
    list of contig lengths. Empty input yields zeros."""
    lengths = [int(x) for x in lengths if x]
    if not lengths:
        return {"n_contigs": 0, "total_bp": 0, "max": 0, "min": 0, "mean": 0, "n50": 0}
    return {
        "n_contigs": len(lengths),
        "total_bp": sum(lengths),
        "max": max(lengths),
        "min": min(lengths),
        "mean": round(sum(lengths) / len(lengths), 1),
        "n50": compute_n50(lengths),
    }


def fasta_lengths(path):
    """Stream a FASTA and return the list of sequence lengths, without loading the
    whole file (accumulates one record at a time). Returns [] on any read error."""
    lengths = []
    current = 0
    try:
        with open(path) as handle:
            for line in handle:
                if line.startswith(">"):
                    if current:
                        lengths.append(current)
                    current = 0
                else:
                    current += len(line.strip())
        if current:
            lengths.append(current)
    except Exception:
        return []
    return lengths


def parse_rrna_reference(name):
    """Parse a RolyPoly rRNA reference id into structured fields.

    RolyPoly's rRNA decontamination DB (a SILVA + NCBI merge) uses ``@``-delimited
    ids: ``<taxid>@<source>@<rrna_type>@<hash>`` (e.g.
    ``152268@NCBI@LSU_prokaryote_rRNA@a6685...``). Returns a dict with:
    taxid (int|None), source, rrna_type (raw token), subunit (SSU/LSU/5S/other) and
    domain (prokaryotic/eukaryotic/unknown), inferred from the token.
    """
    parts = str(name).split("@")
    taxid = None
    if parts and parts[0].isdigit():
        taxid = int(parts[0])
    source = parts[1] if len(parts) > 1 else ""
    rrna_type = parts[2] if len(parts) > 2 else (parts[0] if not taxid else "")
    token = rrna_type.lower()

    if any(k in token for k in ("ssu", "16s", "18s")):
        subunit = "SSU"
    elif any(k in token for k in ("lsu", "23s", "25s", "28s")):
        subunit = "LSU"
    elif "5.8s" in token or re.search(r"\b5s\b", token):
        subunit = "5S"
    else:
        subunit = "other"

    if any(k in token for k in ("prokaryot", "bacteria", "archaea",
                                "16s_ribosomal", "23s")):
        domain = "prokaryotic"
    elif any(k in token for k in ("eukaryot", "eukarya", "fungal",
                                  "18s", "28s", "5.8s")):
        domain = "eukaryotic"
    else:
        domain = "unknown"
    return {"taxid": taxid, "source": source, "rrna_type": rrna_type,
            "subunit": subunit, "domain": domain}


def enrich_rrna_with_mapping(rows, mapping_path):
    """Optionally join rRNA rows (each with a ``taxid``) against RolyPoly's
    ``rrna_to_genome_mapping.parquet`` to add reference organism info. No-op if the
    file is missing/unreadable. Adds keys: query_name, query_rank, relationship,
    assembly_level, genome_size."""
    if not mapping_path or not Path(mapping_path).exists():
        return rows
    try:
        mapping = pl.read_parquet(mapping_path)
    except Exception as exc:
        logger.info("genome_maps: could not read rRNA mapping (%s)", exc)
        return rows
    taxids = [r["taxid"] for r in rows if r.get("taxid") is not None]
    if not taxids:
        return rows
    sub = mapping.filter(pl.col("query_tax_id").is_in(taxids))
    by_taxid = {row["query_tax_id"]: row for row in sub.iter_rows(named=True)}
    for r in rows:
        m = by_taxid.get(r.get("taxid"))
        if m:
            r["query_name"] = m.get("query_name")
            r["query_rank"] = m.get("query_rank")
            r["relationship"] = m.get("relationship")
            r["assembly_level"] = m.get("assembly_level")
            r["genome_size"] = m.get("genome_size")
    return rows


def load_run_stats(output_dir, rrna_mapping_path=None):
    """Collect optional reads-filtering and assembly statistics from a roll/annotate
    output directory. Returns a dict with keys ``reads``, ``rrna_top``,
    ``rrna_domain`` (total euk/prok/unknown %), ``assembly``, ``files`` (any may be
    empty), or None if nothing is found.

    ``rrna_mapping_path`` (or the env var ``ROLYPOLY_DATA`` ->
    ``contam/rrna/rrna_to_genome_mapping.parquet``) optionally enriches the top
    rRNA matches with reference organism names.
    """
    output_dir = Path(output_dir)
    stats: dict = {"reads": [], "rrna_top": [], "rrna_domain": {},
                   "assembly": {}, "files": [], "falco": [], "merge": {},
                   "adapters": []}

    if rrna_mapping_path is None:
        data_dir = os.environ.get("ROLYPOLY_DATA", "")
        if data_dir:
            candidate = Path(data_dir) / "contam/rrna/rrna_to_genome_mapping.parquet"
            if candidate.exists():
                rrna_mapping_path = str(candidate)

    # --- reads: bbduk stats_*.txt (dedup overlapping globs by resolved path) ---
    seen_stats = set()
    for path in sorted(set(output_dir.glob("**/stats_*.txt")), key=str):
        resolved = Path(path).resolve()
        if resolved in seen_stats:
            continue
        seen_stats.add(resolved)
        total = matched = None
        pct = ""
        rows = []
        try:
            for line in Path(path).read_text().splitlines():
                if line.startswith("#Total"):
                    total = int(line.split("\t")[1])
                elif line.startswith("#Matched"):
                    cells = line.split("\t")
                    matched = int(cells[1])
                    pct = cells[2].strip() if len(cells) > 2 else ""
                elif line and not line.startswith("#") and "\t" in line:
                    cells = line.split("\t")
                    rows.append((cells[0], cells[1] if len(cells) > 1 else "",
                                 cells[2].strip() if len(cells) > 2 else ""))
        except Exception:
            continue
        step = Path(path).name.replace("stats_", "", 1)[:-4]
        # bbduk #Matched = reads matching the reference; for decontamination /
        # host / adapter filtering that is the *removed* count, so the reads kept
        # after the step is total - matched. Surface both for a clear progression.
        kept = (total - matched) if (total is not None and matched is not None) else None
        kept_pct = (round(100.0 * kept / total, 3) if kept is not None and total else None)
        stats["reads"].append({"step": step, "total": total, "matched": matched,
                               "pct": pct, "kept": kept, "kept_pct": kept_pct})
        # rRNA decontamination stats -> structured top table + domain totals.
        if "rrna" in Path(path).name.lower() and rows:
            top = []
            euk = prok = unk = 0.0
            for name, reads, pct_str in rows:
                info = parse_rrna_reference(name)
                pct_val = to_float(pct_str.replace("%", ""), 0.0)
                if info["domain"] == "eukaryotic":
                    euk += pct_val
                elif info["domain"] == "prokaryotic":
                    prok += pct_val
                else:
                    unk += pct_val
                if len(top) < 12:
                    top.append({"taxid": info["taxid"], "source": info["source"],
                                "subunit": info["subunit"], "domain": info["domain"],
                                "rrna_type": info["rrna_type"],
                                "reads": reads, "pct": pct_str})
            stats["rrna_top"] = enrich_rrna_with_mapping(top, rrna_mapping_path)
            stats["rrna_domain"] = {"eukaryotic": round(euk, 3),
                                    "prokaryotic": round(prok, 3),
                                    "unknown": round(unk, 3)}

    # --- files: output_tracker.csv ---
    for path in sorted(output_dir.glob("**/output_tracker.csv")):
        try:
            tracker = read_hit_table(path)
            for row in tracker.iter_rows(named=True):
                stats["files"].append({
                    "step": row.get("command_name", ""),
                    "type": row.get("file_type", ""),
                    "size": int(to_float(row.get("file_size"), 0)),
                    "merged": str(row.get("is_merged", "")).lower() in ("true", "1"),
                })
        except Exception:
            pass
        break

    # assembly stats. The headline numbers are the *final* assembly (after any
    # dereplication / clustering / length filtering); when the run used more than
    # one assembler we also report each assembler's own contigs (from
    # contigs_id_map.tsv, which lists every raw contig with its assembler).
    for path in sorted(output_dir.glob("**/contigs_id_map.tsv")):
        assembly_dir = path.parent
        assembly = {}
        # Per-assembler stats from the id map (raw, pre-dereplication contigs).
        try:
            amap = read_hit_table(path)
            if "length" in amap.columns and "assembler" in amap.columns:
                lengths = [int(to_float(x, 0)) for x in amap["length"].to_list()]
                assemblers = amap["assembler"].to_list()
                per = {}
                for asm, length in zip(assemblers, lengths):
                    per.setdefault(asm, []).append(length)
                if len(per) > 1:
                    assembly["assemblers"] = [
                        dict(name=asm, **summarize_lengths(vals))
                        for asm, vals in sorted(per.items())
                    ]
        except Exception:
            pass
        # Headline = the final assembly endpoint FASTA if we can find one, else the
        # raw id-map totals.
        final_fasta = next(
            (assembly_dir / name for name in (
                "length_filtered.fasta", "clustered_assembly.fasta",
                "final_assembly.fasta", "dereplicated_contigs.fasta")
             if (assembly_dir / name).exists()),
            None,
        )
        if final_fasta is not None:
            assembly.update(summarize_lengths(fasta_lengths(final_fasta)))
            assembly["source"] = final_fasta.name
        else:
            # No endpoint FASTA found: fall back to the raw id-map lengths.
            try:
                assembly.update(summarize_lengths(
                    [int(to_float(x, 0)) for x in amap["length"].to_list()]))
                assembly["source"] = "contigs_id_map.tsv"
            except Exception:
                pass
        if assembly.get("n_contigs") or assembly.get("assemblers"):
            stats["assembly"] = assembly
        break

    # --- falco / FastQC: <falco dir>/*_fastqc_data.txt + *_summary.txt ---
    falco_data = sorted(set(output_dir.glob("**/*_fastqc_data.txt")), key=str)
    for path in falco_data:
        basic = {}
        try:
            in_basic = False
            for line in Path(path).read_text().splitlines():
                if line.startswith(">>Basic Statistics"):
                    in_basic = True
                    continue
                if in_basic:
                    if line.startswith(">>END_MODULE"):
                        break
                    if line.startswith("#") or "\t" not in line:
                        continue
                    key, _, val = line.partition("\t")
                    basic[key.strip()] = val.strip()
        except Exception:
            continue
        if not basic:
            continue
        # matching PASS/WARN/FAIL module flags from the sibling *_summary.txt
        modules = {}
        summary = Path(str(path).replace("_fastqc_data.txt", "_summary.txt"))
        if summary.exists():
            try:
                for line in summary.read_text().splitlines():
                    cells = line.split("\t")
                    if len(cells) >= 2:
                        modules[cells[1]] = cells[0]  # module -> PASS/WARN/FAIL
            except Exception:
                pass
        # Embed the falco/FastQC HTML report itself (if present) so the user can
        # see the full per-module plots inside an iframe, not just a flag summary.
        report_html = ""
        report = Path(str(path).replace("_fastqc_data.txt", "_fastqc_report.html"))
        if report.exists():
            try:
                report_html = report.read_text()
            except Exception:
                report_html = ""
        stats["falco"].append({
            "file": basic.get("Filename", Path(path).name),
            "total_sequences": int(to_float(basic.get("Total Sequences"), 0)) or None,
            "total_bases": basic.get("Total Bases", ""),
            "gc": to_float(basic.get("%GC"), None),
            "length": basic.get("Sequence length", ""),
            "flags": modules,
            "report_html": report_html,
        })

    # Discovered adapter sequences from bbmerge (validated file preferred). A
    # single "N" means no adapter was found, so those are skipped.
    adapter_files = sorted(output_dir.glob("**/validated_bbmerge_discovered_*.fa")) or \
        sorted(output_dir.glob("**/bbmerge_discovered_*.fa"))
    for adapter_path in adapter_files[:1]:
        try:
            name = None
            seq_parts = []
            entries = []
            for line in Path(adapter_path).read_text().splitlines():
                if line.startswith(">"):
                    if name is not None:
                        entries.append((name, "".join(seq_parts)))
                    name = line[1:].strip()
                    seq_parts = []
                else:
                    seq_parts.append(line.strip())
            if name is not None:
                entries.append((name, "".join(seq_parts)))
            for entry_name, seq in entries:
                if seq and set(seq.upper()) - {"N"}:  # skip empty / all-N
                    stats["adapters"].append({"name": entry_name, "sequence": seq})
        except Exception:
            pass

    # --- bbmerge % merged: parse "Joined:\t<n>\t<pct>%" from the pipeline log ---
    # bbmerge prints the join rate to the log; it is the most reliable source of a
    # true "% merged" (the output_tracker only records an is_merged flag/size).
    for log_path in sorted(output_dir.glob("**/*_pipeline.log")) + \
            sorted(output_dir.glob("**/*.log")):
        try:
            text = Path(log_path).read_text()
        except Exception:
            continue
        pairs = joined = None
        pct = None
        for line in text.splitlines():
            m = re.match(r"\s*Pairs:\s*\t?\s*([\d,]+)", line)
            if m:
                pairs = int(m.group(1).replace(",", ""))
            m = re.match(r"\s*Joined:\s*\t?\s*([\d,]+)\s*\t?\s*([\d.]+)\s*%", line)
            if m:
                joined = int(m.group(1).replace(",", ""))
                pct = float(m.group(2))
        if joined is not None:
            stats["merge"] = {"pairs": pairs, "joined": joined, "pct": pct}
            break

    if not (stats["reads"] or stats["assembly"] or stats["files"] or stats["falco"]):
        return None
    return stats


# GENERIC EXTRA TABS
def table_to_tab(data, label, tab_id=None, columns=None, max_rows=5000):
    """Turn any tabular data into a generic extra-tab payload for the report.

    This is the extension point for future information layers (e.g. predicted
    taxonomy, host prediction, coverage): pass a path/DataFrame and it becomes a
    sortable/filterable table tab. Returns a dict {id, label, columns, rows}.
    """
    df = data if isinstance(data, pl.DataFrame) else read_hit_table(data)
    if columns:
        columns = [c for c in columns if c in df.columns]
        df = df.select(columns)
    else:
        columns = list(df.columns)
    if df.height > max_rows:
        df = df.head(max_rows)
    rows = [[("" if v is None else str(v)) for v in row] for row in df.iter_rows()]
    tab_id = tab_id or re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or "extra"
    return {"id": tab_id, "label": label, "columns": columns, "rows": rows}


def taxonomy_to_tab(data, label="Taxonomy", tab_id="taxonomy", max_rows=5000):
    """Build a taxonomy table tab with a family-composition chart payload."""
    df = data if isinstance(data, pl.DataFrame) else read_hit_table(data)
    payload = table_to_tab(df, label, tab_id=tab_id, max_rows=max_rows)
    composition_column = next(
        (column for column in ("family", "taxon_name", "realm") if column in df.columns),
        None,
    )
    if composition_column:
        composition = (
            df.with_columns(
                pl.col(composition_column).cast(pl.String).fill_null("Unclassified")
                .replace("", "Unclassified")
            )
            .group_by(composition_column)
            .len(name="count")
            .sort("count", descending=True)
        )
        payload["composition_label"] = composition_column
        payload["composition"] = [
            {"label": str(name), "count": int(count)}
            for name, count in composition.iter_rows()
        ]
    return payload


def load_original_contig_ids(path):
    """Return a ``new_id -> original assembler/input ID`` mapping."""
    mapping = read_hit_table(path)
    if not {"new_id", "old_id"}.issubset(mapping.columns):
        return {}
    original_ids: dict[str, list[str]] = {}
    for new_id, old_id in mapping.select("new_id", "old_id").iter_rows():
        new_key, old_value = str(new_id), str(old_id)
        if old_value not in original_ids.setdefault(new_key, []):
            original_ids[new_key].append(old_value)
    return {
        key: "; ".join(values)
        for key, values in original_ids.items()
        if any(value != key for value in values)
    }


def find_report_log(output_dir):
    """Find and read the primary pipeline log for embedding in a report."""
    output_dir = Path(output_dir)
    candidates = [output_dir / "rolypoly_pipeline.log"]
    candidates.extend(sorted(output_dir.glob("*_pipeline.log")))
    candidates.extend(sorted(output_dir.glob("*.log")))
    candidates.extend(sorted(output_dir.glob("**/*.log")))
    for path in candidates:
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
    return None


def build_report_file_catalog(
    output_dir,
    report_output,
    *,
    marker_table=None,
    rna_table=None,
    nucleic_tables=None,
    motif_tables=None,
    taxonomy_path=None,
):
    """Build relative links to original tables and external FASTA files.

    Sequences remain outside the HTML. Browsers can download these files
    directly, or load a FASTA on demand for individual sequence display/export.
    """
    output_dir, report_output = Path(output_dir), Path(report_output)
    report_parent = report_output.parent.resolve()

    def entry(path, label, kind):
        if path is None or not Path(path).is_file():
            return None
        relative = os.path.relpath(Path(path).resolve(), report_parent)
        return {"label": label, "kind": kind, "path": Path(relative).as_posix()}

    tables = []
    table_specs = [
        (marker_table, "Original protein-hit table", "protein"),
        (rna_table, "Original RNA table", "rna"),
        (taxonomy_path, "Original taxonomy table", "taxonomy"),
    ]
    for nucleic_table in nucleic_tables or []:
        if isinstance(nucleic_table, tuple):
            source_label, path = nucleic_table
        else:
            path = nucleic_table
            source_label = Path(path).stem
        table_specs.append(
            (path, f"Original nucleic-hit table — {source_label}", "nucleic")
        )
    table_specs.extend(
        (path, f"Original RdRp-motif table — {Path(path).stem}", "motif")
        for path in (motif_tables or [])
    )
    seen = set()
    for path, label, kind in table_specs:
        item = entry(path, label, kind)
        if item and item["path"] not in seen:
            seen.add(item["path"])
            tables.append(item)

    fasta_candidates = [
        ("all_matched_contigs.fasta", "Matched contigs", "contigs"),
        ("predicted_orfs.faa", "Predicted ORFs", "orfs"),
        ("marker_search_matched_regions.faa", "Marker-matched amino-acid regions", "regions"),
        ("marker_search_matched_input_seqs.fna", "Marker-matched nucleotide inputs", "contigs"),
        ("marker_search_matched_input_seqs.faa", "Marker-matched protein inputs", "orfs"),
    ]
    fastas = []
    seen.clear()
    for filename, label, kind in fasta_candidates:
        for path in sorted(output_dir.glob(f"**/{filename}")):
            item = entry(path, label, kind)
            if item and item["path"] not in seen:
                seen.add(item["path"])
                fastas.append(item)
    if not any(item["kind"] == "contigs" for item in fastas):
        for filename in ("length_filtered.fasta", "clustered_assembly.fasta", "final_assembly.fasta"):
            path = next(iter(sorted(output_dir.glob(f"**/{filename}"))), None)
            item = entry(path, f"Contigs — {filename}", "contigs")
            if item:
                fastas.append(item)
                break
    return {"tables": tables, "fastas": fastas}


# RENDER
def render_html(contigs, palette=None, title="RolyPoly — Genome / marker maps",
                subtitle=None, criteria=tuple(BEST_CRITERIA),
                initial_mode="all", initial_criterion=None, initial_tab="table",
                nucleic=None, run_stats=None, extra_tabs=None,
                command_line=None, log_text=None, source_files=None):
    """Return a full, standalone HTML document string for the given contig models
    and optional nucleic-hits / run-stats / extra-tab payloads."""
    all_sources = sorted({s for c in contigs for s in c.get("sources", [])})
    palette = dict(palette) if palette else build_palette(all_sources)
    for source in all_sources:
        palette.setdefault(source, build_palette([source])[source])

    n_hits = sum(c["n_hits"] for c in contigs)
    n_orfs = sum(c["n_orfs"] for c in contigs)
    n_rna = sum(1 for c in contigs if c.get("rna"))
    n_motif = sum(1 for c in contigs if c.get("motifs"))
    nucleic_flat = (nucleic or {}).get("__all__", []) if nucleic else []
    if subtitle is None:
        bits = [f"{n_hits:,} protein hits", f"{n_orfs} ORFs", f"{len(contigs)} contigs",
                f"{len(all_sources)} source(s)"]
        if n_rna:
            bits.append(f"RNA on {n_rna}")
        if n_motif:
            bits.append(f"RdRp motifs on {n_motif}")
        if nucleic_flat:
            bits.append(f"{len(nucleic_flat)} nucleic hits")
        subtitle = " · ".join(bits)

    payload = {
        "contigs": contigs, "colors": palette, "rna_colors": RNA_TYPE_COLORS,
        "nucleic_colors": NUCLEIC_COLORS, "motif_colors": MOTIF_COLORS,
        "criteria": {c: BEST_CRITERIA.get(c, c) for c in criteria},
        "summary": {"n_contigs": len(contigs), "n_orfs": n_orfs, "n_hits": n_hits},
        "title": title, "subtitle": subtitle,
        "initial_mode": ("best" if initial_mode == "best" else "all"),
        "initial_criterion": (initial_criterion if initial_criterion in criteria
                              else next(iter(criteria))),
        "initial_tab": initial_tab,
        "nucleic": nucleic_flat,
        "stats": run_stats or None,
        "extra_tabs": extra_tabs or [],
        "command_line": command_line or None,
        "log_text": log_text or None,
        "source_files": source_files or {"tables": [], "fastas": []},
    }
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("/*__DATA__*/", data_json)


def write_genome_maps(data, output, spec=None, palette=None,
                      title="RolyPoly — Genome / marker maps",
                      min_score=None, max_evalue=None, mark_best=True,
                      min_overlap_positions=1, source_priority=None,
                      rna=None, rna_spec=None, rna_bins=150,
                      nucleic=None, nucleic_spec=None, motifs=None, motif_spec=None,
                      run_stats=None, extra_tabs=None,
                      initial_mode="all", initial_criterion=None, initial_tab="table",
                      original_ids=None, command_line=None, log_text=None,
                      source_files=None):
    """Read a protein table (+ optional annotate-rna, nucleic-search, rdrp-motif
    tables, a run-stats dict, and extra tabs) and write a standalone interactive
    HTML report. Returns Path."""
    df = load_marker_table(data, spec, min_score=min_score, max_evalue=max_evalue)
    if spec is None:
        spec = infer_marker_spec(df)
    if mark_best:
        df = tag_best_hits(df, spec, min_overlap_positions=min_overlap_positions,
                           source_priority=source_priority)
    contigs = build_contig_models(df, spec)
    if rna is not None:
        contigs = attach_rna(contigs, build_rna_by_contig(rna, rna_spec, nbins=rna_bins))
    nucleic_map = None
    if nucleic is not None:
        nucleic_map = build_nucleic_by_contig(nucleic, nucleic_spec)
        contigs = attach_nucleic(contigs, nucleic_map)
    if motifs is not None:
        contigs = attach_motifs(contigs, build_motifs_by_contig(motifs, motif_spec))
    if original_ids:
        for contig in contigs:
            contig["raw_id"] = original_ids.get(contig["contig"])
    contigs.sort(key=lambda c: -c["best_score"])
    html = render_html(contigs, palette, title=title, initial_mode=initial_mode,
                       initial_criterion=initial_criterion, initial_tab=initial_tab,
                       nucleic=nucleic_map, run_stats=run_stats, extra_tabs=extra_tabs,
                       command_line=command_line, log_text=log_text,
                       source_files=source_files)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    logger.info("genome_maps: wrote %s (%.0f KB, %d contigs)",
                output, output.stat().st_size / 1024, len(contigs))
    return output


def find_annotation_tables(output_dir):
    """Locate the protein/marker table and the annotate-rna table under an output
    directory, by header. Returns (marker_table_path | None, rna_table_path | None)."""
    output_dir = Path(output_dir)

    def header_of(path):
        try:
            return set(read_hit_table(path).head(0).columns)
        except Exception:
            return set()

    def looks_like_markers(cols):
        if {"query_full_name", "hmm_full_name"}.issubset(cols):
            return True
        has_query = bool({"qseqid", "query", "sequence_id"} & cols)
        has_target = bool({"sseqid", "target", "theader", "profile_name"} & cols)
        return has_query and has_target and "structure" not in cols

    candidates, seen = [], set()
    for pattern in ("combined_annotations.tsv", "combined_annotations.csv",
                    "combined_annotations.parquet"):
        for path in sorted(output_dir.glob(f"**/{pattern}")):
            if path not in seen:
                seen.add(path)
                candidates.append(path)

    marker_table = next((p for p in candidates if looks_like_markers(header_of(p))), None)
    rna_table = next((p for p in candidates if "structure" in header_of(p)), None)
    return marker_table, rna_table


def write_report_for_dir(output_dir, output=None, *, title="RolyPoly — Genome / marker maps",
                         marker_table=None, rna_table=None, nucleic_tables=None,
                         motif_tables=None, with_stats=True, rrna_mapping_path=None,
                         extra_tabs=None, command_line=None, log_file=None,
                         contig_id_map=None, **kwargs):
    """Discover annotation / nucleic / motif / stats data under ``output_dir`` (by
    header / path) and write the HTML report. Shared by ``report``, the
    ``annotate*`` commands, and ``roll``. Explicit tables override discovery. Extra
    kwargs go to :func:`write_genome_maps`. Returns the report Path, or None."""
    output_dir = Path(output_dir)
    if marker_table is None or rna_table is None:
        found_marker, found_rna = find_annotation_tables(output_dir)
        marker_table = marker_table or found_marker
        rna_table = rna_table or found_rna
    if nucleic_tables is None:
        nucleic_tables = find_nucleic_tables(output_dir) or None
    if motif_tables is None:
        motif_tables = find_motif_tables(output_dir) or None
    extra_tabs = list(extra_tabs or [])
    taxonomy_path = next(iter(sorted(output_dir.glob("**/mmtax.tsv"))), None)
    if taxonomy_path is not None and not any(
        tab.get("id") == "taxonomy" for tab in extra_tabs
    ):
        extra_tabs.append(taxonomy_to_tab(taxonomy_path))
    run_stats = load_run_stats(output_dir, rrna_mapping_path) if with_stats else None
    id_map_path = Path(contig_id_map) if contig_id_map else next(
        iter(sorted(output_dir.glob("**/contigs_id_map.tsv"))), None
    )
    original_ids = load_original_contig_ids(id_map_path) if id_map_path else None
    if log_file:
        log_path = Path(log_file)
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else None
    else:
        log_text = find_report_log(output_dir)
    output = Path(output) if output else (output_dir / "genome_maps.html")
    source_files = build_report_file_catalog(
        output_dir,
        output,
        marker_table=marker_table,
        rna_table=rna_table,
        nucleic_tables=nucleic_tables,
        motif_tables=motif_tables,
        taxonomy_path=taxonomy_path,
    )

    if marker_table is None:
        if rna_table is None and not nucleic_tables and not extra_tabs:
            logger.warning("genome_maps: no annotation tables under %s; skipping report", output_dir)
            return None
        contigs = []
        if rna_table is not None:
            contigs = attach_rna([], build_rna_by_contig(rna_table))
        nucleic_map = build_nucleic_by_contig(nucleic_tables) if nucleic_tables else None
        if nucleic_map:
            contigs = attach_nucleic(contigs, nucleic_map)
        if motif_tables:
            contigs = attach_motifs(contigs, build_motifs_by_contig(list(motif_tables)))
        if original_ids:
            for contig in contigs:
                contig["raw_id"] = original_ids.get(contig["contig"])
        contigs.sort(key=lambda c: -c["length"])
        html = render_html(contigs, kwargs.get("palette"), title=title,
                           initial_tab=kwargs.get("initial_tab", "table"),
                           nucleic=nucleic_map, run_stats=run_stats, extra_tabs=extra_tabs,
                           command_line=command_line, log_text=log_text,
                           source_files=source_files)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html, encoding="utf-8")
        logger.info("genome_maps: wrote %s (no protein table; %d contigs)", output, len(contigs))
        return output

    return write_genome_maps(
        str(marker_table), output, title=title,
        rna=str(rna_table) if rna_table else None,
        nucleic=nucleic_tables,
        motifs=list(motif_tables) if motif_tables else None,
        run_stats=run_stats, extra_tabs=extra_tabs, original_ids=original_ids,
        command_line=command_line, log_text=log_text, source_files=source_files,
        **kwargs,
    )


# The browser engine. Data injected at /*__DATA__*/.
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>RolyPoly — Genome maps</title>
<style>
 :root{--bg:#f6f7f9;--panel:#fff;--ink:#1f2430;--mut:#6b7280;--line:#e5e7eb;--accent:#33507e;}
 *{box-sizing:border-box}
 body{margin:0;font-family:'Segoe UI',system-ui,-apple-system,Arial,sans-serif;background:var(--bg);color:var(--ink)}
 header{background:linear-gradient(120deg,#1f2a44,#33507e);color:#fff;padding:14px 26px}
 header h1{margin:0;font-size:19px;font-weight:700}
 header p{margin:5px 0 0;font-size:13px;color:#cdd6e6}
 .commandline{display:none;padding:7px 26px;background:#e5e7eb;color:#6b7280;border-bottom:1px solid #d1d5db;font:11px ui-monospace,Consolas,monospace;white-space:nowrap;overflow-x:auto}
 .tabs{display:flex;gap:2px;background:#26324d;padding:0 20px;flex-wrap:wrap}
 .tabs button{background:transparent;border:0;color:#c8d3e6;font-size:14px;font-weight:600;padding:11px 16px;cursor:pointer;border-bottom:3px solid transparent}
 .tabs button.on{color:#fff;border-bottom-color:#7fa8e6}
 .toolbar{display:flex;flex-wrap:wrap;gap:14px;align-items:center;padding:10px 20px;background:#eef1f6;border-bottom:1px solid var(--line);font-size:13px}
 .toolbar .grp{display:flex;align-items:center;gap:6px}
 .seg{display:flex}
 .seg button{padding:6px 12px;border:1px solid var(--line);background:#fff;cursor:pointer;font-size:13px}
 .seg button:first-child{border-radius:8px 0 0 8px}
 .seg button:last-child{border-radius:0 8px 8px 0;border-left:0}
 .seg button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
 select,input[type=text],input[type=number]{padding:7px 9px;border:1px solid var(--line);border-radius:8px;font-size:13px;background:#fff;color:var(--ink)}
 .chk{display:flex;align-items:center;gap:6px;cursor:pointer}
 .wrap{display:flex;gap:16px;padding:16px 20px;align-items:flex-start}
 .side{flex:0 0 300px;position:sticky;top:16px}
 .main{flex:1 1 auto;min-width:0}
 .pane{display:none} .pane.on{display:block}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
 .card h2{margin:0 0 10px;font-size:13px;letter-spacing:.04em;text-transform:uppercase;color:var(--mut)}
 .side select,.side input[type=text]{width:100%}
 .row{display:flex;justify-content:space-between;font-size:13px;padding:4px 0;border-bottom:1px dashed var(--line)}
 .row:last-child{border-bottom:0}
 .dbtoggle{display:flex;align-items:center;gap:8px;font-size:13px;padding:5px 0;cursor:pointer}
 .sw{width:14px;height:14px;border-radius:3px;display:inline-block}
 .nav{display:flex;gap:8px;margin-top:10px}
 .nav button{flex:1;padding:8px;border:1px solid var(--line);background:#fff;border-radius:8px;cursor:pointer;font-size:13px}
 .nav button:hover{background:#eef2f8}
 .fbar{display:flex;gap:8px;margin-top:8px}
 .fbar label{flex:1;font-size:12px;color:var(--mut)}
 .fbar input{width:100%}
 #maptitle{font-size:17px;font-weight:700;margin:2px 0;word-break:break-all}
 #mapsub{font-size:13px;color:var(--mut);margin-bottom:10px}
 svg{width:100%;display:block}
 .tip{position:fixed;pointer-events:none;z-index:99;background:#111826;color:#eef;font-size:12px;line-height:1.5;padding:9px 11px;border-radius:8px;max-width:400px;box-shadow:0 6px 20px rgba(0,0,0,.3);opacity:0;transition:opacity .08s}
 .tip b{color:#fff}.tip .k{color:#9fb3d1}
 .tip .aln{font-family:ui-monospace,Consolas,monospace;font-size:10.5px;color:#bfe3c0;word-break:break-all;display:block;margin-top:4px}
 .legend-note{font-size:12px;color:var(--mut);margin-top:6px}
 table{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:4px}
 th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
 th{color:var(--mut);font-weight:600;text-transform:uppercase;font-size:11px;letter-spacing:.03em}
 tr:hover td{background:#f2f6fc}
 .pill{display:inline-block;padding:1px 8px;border-radius:20px;color:#fff;font-size:11px;font-weight:600}
 .mono{font-family:ui-monospace,Consolas,monospace}
 .cid{color:#2456a6;cursor:pointer;font-weight:600}
 .cid:hover{text-decoration:underline}
 .statcards{display:flex;flex-wrap:wrap;gap:12px}
 .statcard{flex:1 1 150px;background:#f2f6fc;border:1px solid var(--line);border-radius:10px;padding:12px 14px}
 .statcard .v{font-size:20px;font-weight:700}
 .statcard .l{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.03em}
 #exp{margin-top:8px;width:100%;padding:8px;border:1px solid var(--line);background:#fff;border-radius:8px;cursor:pointer;font-size:13px}
 #exp:hover{background:#eef2f8}
 #logtext{margin:0;white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.45 ui-monospace,Consolas,monospace;color:#374151}
 .exportbar{display:flex;flex-wrap:wrap;gap:7px;align-items:center;margin:0 0 10px}
 .btn,.exportbar button,.exportbar a{display:inline-block;padding:6px 10px;border:1px solid var(--line);border-radius:7px;background:#fff;color:#334155;text-decoration:none;cursor:pointer;font-size:12px}
 .btn:hover,.exportbar button:hover,.exportbar a:hover{background:#eef2f8}
 #seqtext{max-height:360px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;font:11px/1.4 ui-monospace,Consolas,monospace;color:#26324d}
</style></head>
<body>
<header><h1 id="hdrtitle"></h1><p id="hdrsub"></p></header>
<div class="commandline" id="commandline"></div>
<div class="tabs" id="tabs"></div>
<div class="toolbar" id="toolbar">
  <div class="grp"><span>Show</span>
    <div class="seg" id="modeSeg">
      <button data-mode="all" class="on">All hits</button>
      <button data-mode="best">Best only</button>
    </div>
  </div>
  <div class="grp"><span>Best by</span><select id="critSel"></select></div>
  <label class="chk"><input type="checkbox" id="rnaToggle" checked> RNA track</label>
  <label class="chk"><input type="checkbox" id="nucToggle" checked> Nucleic track</label>
  <label class="chk"><input type="checkbox" id="motifToggle" checked> RdRp motifs</label>
  <div class="grp legend-note" id="tbmeta"></div>
</div>

<div id="pane-table" class="pane on">
  <div class="wrap"><div class="main" style="flex:1 1 100%">
    <div class="card">
      <div class="grp" style="display:flex;gap:0;margin-bottom:10px">
        <div class="seg" id="tblSeg">
          <button data-view="contigs" class="on">Contigs</button>
          <button data-view="hits">All hits</button>
        </div>
        <input type="text" id="tblSearch" placeholder="Filter…" style="margin-left:10px;flex:1">
        <label class="chk" id="rawIdControl" style="margin-left:12px;display:none"><input type="checkbox" id="rawIdToggle"> Show raw contig ID</label>
      </div>
      <div class="exportbar"><button onclick="exportCurrentTable()">Export shown TSV</button><span id="tableOriginalLinks"></span></div>
      <div id="bigtable"></div>
    </div>
  </div></div>
</div>

<div id="pane-maps" class="pane">
  <div class="wrap">
    <div class="side">
      <div class="card">
        <h2>Select genome / contig</h2>
        <input type="text" id="search" placeholder="Filter contigs…"/>
        <select id="picker" style="margin-top:8px" size="1"></select>
        <div class="nav"><button id="prev">&larr; Prev</button><button id="next">Next &rarr;</button></div>
        <div class="fbar">
          <label>min score<input type="number" id="fscore" value="0" step="1" min="0"></label>
          <label>max E (10^)<input type="number" id="fevalue" value="0" step="1"></label>
        </div>
        <button id="exp">⬇ Export current map as SVG</button>
      </div>
      <div class="card"><h2>Sequence files</h2>
        <div id="fastaLinks" class="exportbar"></div>
        <button class="btn" id="loadFastas">Load referenced FASTA</button>
        <button class="btn" onclick="if(contigs[idx])showContigSequence(contigs[idx].contig)">Show current contig</button>
        <label class="btn" style="margin-left:4px">Choose FASTA…<input id="fastaPicker" type="file" accept=".fa,.faa,.fna,.fasta" multiple hidden></label>
        <div class="legend-note" id="fastaStatus">Sequences are not embedded in this report.</div>
      </div>
      <div class="card"><h2>Contig details</h2><div id="details"></div></div>
      <div class="card"><h2>Protein sources</h2><div id="dbtoggles"></div>
        <div class="legend-note">Click to show / hide a source.</div></div>
    </div>
    <div class="main">
      <div class="card"><div id="maptitle"></div><div id="mapsub"></div><div id="mapholder"></div></div>
      <div class="card" id="seqcard" style="display:none"><div class="exportbar"><h2 id="seqtitle" style="margin:0;flex:1"></h2><button id="seqexport">Export FASTA</button><button onclick="document.getElementById('seqcard').style.display='none'">Close</button></div><pre id="seqtext"></pre></div>
      <div class="card"><h2>Hits in this contig</h2><div id="tablebox"></div></div>
    </div>
  </div>
</div>

<div id="pane-nucleic" class="pane">
  <div class="wrap"><div class="main" style="flex:1 1 100%">
    <div class="card">
      <div class="exportbar"><button onclick="exportNucleicTable()">Export shown TSV</button><span id="nucOriginalLinks"></span></div>
      <input type="text" id="nucSearch" placeholder="Filter nucleic hits…" style="width:100%;margin-bottom:10px">
      <div id="nuctable"></div>
    </div>
  </div></div>
</div>

<div id="pane-stats" class="pane">
  <div class="wrap"><div class="main" style="flex:1 1 100%"><div id="statsbox"></div></div></div>
</div>

<div id="pane-log" class="pane">
  <div class="wrap"><div class="main" style="flex:1 1 100%"><div class="card"><pre id="logtext"></pre></div></div></div>
</div>

<div id="extra-panes"></div>

<div class="tip" id="tip"></div>
<script>
const DATA=/*__DATA__*/;
const SVGNS="http://www.w3.org/2000/svg";
const colors=DATA.colors, rnaColors=DATA.rna_colors||{}, nucColors=DATA.nucleic_colors||{}, motifColors=DATA.motif_colors||{}, contigs=DATA.contigs;
const CRIT=DATA.criteria;
const NUCLEIC=DATA.nucleic||[], STATS=DATA.stats||null, EXTRA=DATA.extra_tabs||[];
const FILES=DATA.source_files||{tables:[],fastas:[]};
const hasMaps=contigs.length>0, hasNucleic=NUCLEIC.length>0, hasStats=!!STATS, hasLog=!!DATA.log_text;
let active=new Set(Object.keys(colors)), idx=0, mode=DATA.initial_mode||"all", showRNA=true, showNuc=true, showMotif=true;
let crit=DATA.initial_criterion||Object.keys(CRIT)[0]||"score", minScore=0, maxEexp=0, showRawIds=false;
let currentTableExport={columns:[],rows:[],name:'contigs.tsv'},currentNucleicExport={columns:[],rows:[]};
let currentMapExport={columns:[],rows:[],name:'contig_hits.tsv'};
const sequenceCache=new Map();let displayedSequence=null;

document.getElementById('hdrtitle').textContent=DATA.title;
document.getElementById('hdrsub').textContent=DATA.subtitle;
if(DATA.command_line){const command=document.getElementById('commandline');command.textContent='Command called: '+DATA.command_line;command.style.display='block';}
if(hasLog)document.getElementById('logtext').textContent=DATA.log_text;

// Create panes for extra tabs.
const extraPanes=document.getElementById('extra-panes');
EXTRA.forEach(t=>{const d=document.createElement('div');d.className='pane';d.id='pane-'+t.id;
  d.innerHTML=`<div class="wrap"><div class="main" style="flex:1 1 100%"><div class="card">`+
    `<div id="xchart-${t.id}"></div>`+
    `<div class="exportbar"><button onclick="exportExtraTable('${t.id}')">Export shown TSV</button><span id="xoriginal-${t.id}"></span></div>`+
    `<input type="text" class="xsearch" data-for="${t.id}" placeholder="Filter ${t.label}…" style="width:100%;margin-bottom:10px">`+
    `<div id="xtable-${t.id}"></div></div></div></div>`;
  extraPanes.appendChild(d);});

// Build the tab bar from available data (+ extra tabs).
const TABS=[];
if(hasMaps){TABS.push(["table","▤ Contig Table"]);TABS.push(["maps","🧬 Genome maps"]);}
if(hasNucleic)TABS.push(["nucleic","🧷 Nucleic hits"]);
if(hasStats)TABS.push(["stats","📊 Run stats"]);
if(hasLog)TABS.push(["log","▧ Log"]);
EXTRA.forEach(t=>TABS.push([t.id, t.label]));
const tabsEl=document.getElementById('tabs');
TABS.forEach(([id,label])=>{const b=document.createElement('button');b.dataset.tab=id;b.textContent=label;
  b.onclick=()=>showTab(id);tabsEl.appendChild(b);});
const BUILTIN=new Set(["table","maps","nucleic","stats","log"]);
function showTab(name){
  document.querySelectorAll('.tabs button').forEach(x=>x.classList.toggle('on',x.dataset.tab===name));
  document.querySelectorAll('.pane').forEach(p=>p.classList.remove('on'));
  const el=document.getElementById('pane-'+name); if(el)el.classList.add('on');
  document.getElementById('toolbar').style.display=name==='maps'?'flex':'none';
  if(name==='table')renderTable(); else if(name==='maps')render();
  else if(name==='nucleic')renderNucleic(); else if(name==='stats')renderStats();
  else if(!BUILTIN.has(name))renderExtra(name);
}

const critSel=document.getElementById('critSel');
Object.entries(CRIT).forEach(([k,v])=>{const o=document.createElement('option');o.value=k;o.textContent=v;critSel.appendChild(o);});
critSel.value=crit; critSel.onchange=()=>{crit=critSel.value;renderAll();};
document.querySelectorAll('#modeSeg button').forEach(x=>x.classList.toggle('on',x.dataset.mode===mode));
document.querySelectorAll('#modeSeg button').forEach(b=>{b.onclick=()=>{mode=b.dataset.mode;
  document.querySelectorAll('#modeSeg button').forEach(x=>x.classList.toggle('on',x===b));renderAll();};});
document.getElementById('rnaToggle').onchange=e=>{showRNA=e.target.checked;render();};
document.getElementById('nucToggle').onchange=e=>{showNuc=e.target.checked;render();};
document.getElementById('motifToggle').onchange=e=>{showMotif=e.target.checked;render();};

const dbBox=document.getElementById('dbtoggles');
Object.keys(colors).forEach(db=>{const d=document.createElement('div');d.className='dbtoggle';
 d.innerHTML=`<span class="sw" style="background:${colors[db]}"></span><span style="flex:1">${db}</span><span id="cnt_${db}" style="color:#6b7280"></span>`;
 d.onclick=()=>{active.has(db)?active.delete(db):active.add(db);d.style.opacity=active.has(db)?1:.35;renderAll();};dbBox.appendChild(d);});

const picker=document.getElementById('picker');
function fillPicker(f=''){picker.innerHTML='';f=f.toLowerCase();
 contigs.forEach((c,i)=>{const hay=(c.short+' '+c.top_profile+' '+c.contig).toLowerCase();
  if(f&&!hay.includes(f))return;const o=document.createElement('option');o.value=i;
  o.textContent=`${c.short} · ${c.top_profile} · score ${c.best_score}`;picker.appendChild(o);});}
if(hasMaps){fillPicker();picker.value=0;}
picker.onchange=()=>{idx=+picker.value;render();};
document.getElementById('search').oninput=e=>{fillPicker(e.target.value);if(picker.options.length){idx=+picker.options[0].value;picker.value=idx;render();}};
document.getElementById('prev').onclick=()=>{if(idx>0){idx--;picker.value=idx;render();}};
document.getElementById('next').onclick=()=>{if(idx<contigs.length-1){idx++;picker.value=idx;render();}};
document.getElementById('fscore').oninput=e=>{minScore=+e.target.value||0;render();};
document.getElementById('fevalue').oninput=e=>{maxEexp=+e.target.value||0;render();};
document.getElementById('exp').onclick=()=>{const svg=document.querySelector('#mapholder svg');if(!svg)return;
 const blob=new Blob([svg.outerHTML],{type:'image/svg+xml'});const a=document.createElement('a');
 a.href=URL.createObjectURL(blob);a.download=contigs[idx].contig+'_map.svg';a.click();};

let tblView="contigs";
if(contigs.some(c=>c.raw_id))document.getElementById('rawIdControl').style.display='flex';
document.getElementById('rawIdToggle').onchange=e=>{showRawIds=e.target.checked;renderTable();};
document.querySelectorAll('#tblSeg button').forEach(b=>{b.onclick=()=>{tblView=b.dataset.view;
  document.querySelectorAll('#tblSeg button').forEach(x=>x.classList.toggle('on',x===b));renderTable();};});
document.getElementById('tblSearch').oninput=renderTable;
document.getElementById('nucSearch').oninput=renderNucleic;
document.querySelectorAll('.xsearch').forEach(inp=>inp.oninput=()=>renderExtra(inp.dataset.for));

function openContig(cid){const i=contigs.findIndex(c=>c.contig===cid);if(i<0)return;idx=i;picker.value=i;showTab('maps');}
window.openContig=openContig;

const tip=document.getElementById('tip');
function showTip(h,x,y){tip.innerHTML=h;tip.style.opacity=1;let tx=x+14;if(tx+410>innerWidth)tx=x-410;tip.style.left=tx+'px';tip.style.top=(y+14)+'px';}
function hideTip(){tip.style.opacity=0;}
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function safeName(s){return String(s||'export').replace(/[^A-Za-z0-9._-]+/g,'_');}
function downloadText(name,text,type='text/plain'){
 const blob=new Blob([text],{type:type+';charset=utf-8'}),a=document.createElement('a');
 a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
}
function tsvCell(v){const s=String(v??'');return /[\t\n\r"]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s;}
function downloadTsv(name,columns,rows){const text=[columns,...rows].map(r=>r.map(tsvCell).join('\t')).join('\n')+'\n';downloadText(name,text,'text/tab-separated-values');}
function exportCurrentTable(){downloadTsv(currentTableExport.name,currentTableExport.columns,currentTableExport.rows);}
function exportNucleicTable(){downloadTsv('nucleic_hits_shown.tsv',currentNucleicExport.columns,currentNucleicExport.rows);}
function exportMapTable(){downloadTsv(currentMapExport.name,currentMapExport.columns,currentMapExport.rows);}
function exportExtraTable(id){const t=EXTRA.find(x=>x.id===id);if(!t)return;const inp=document.querySelector('.xsearch[data-for="'+id+'"]'),f=(inp&&inp.value||'').toLowerCase();downloadTsv(safeName(t.label)+'_shown.tsv',t.columns,t.rows.filter(r=>!f||r.join(' ').toLowerCase().includes(f)));}
function sourceLinks(kind){return FILES.tables.filter(f=>f.kind===kind).map(f=>`<a href="${encodeURI(f.path)}" download title="Untouched pipeline output">${esc(f.label)}</a>`).join(' ');}
document.getElementById('tableOriginalLinks').innerHTML=[sourceLinks('protein'),sourceLinks('rna')].filter(Boolean).join(' ');
document.getElementById('nucOriginalLinks').innerHTML=sourceLinks('nucleic');
EXTRA.forEach(t=>{const box=document.getElementById('xoriginal-'+t.id);if(box)box.innerHTML=sourceLinks(t.id);});
document.getElementById('fastaLinks').innerHTML=FILES.fastas.map(f=>`<a href="${encodeURI(f.path)}" download>${esc(f.label)}</a>`).join(' ');

function parseFasta(text,source){let header=null,parts=[],count=0;const save=()=>{if(!header)return;const seq=parts.join('').replace(/\s+/g,'');if(!seq)return;const id=header.split(/\s+/)[0];const record={id,header,seq,source};sequenceCache.set(id,record);sequenceCache.set(header,record);count++;};
 text.split(/\r?\n/).forEach(line=>{if(line.startsWith('>')){save();header=line.slice(1).trim();parts=[];}else if(header)parts.push(line.trim());});save();return count;}
async function loadReferencedFastas(kinds=null){const status=document.getElementById('fastaStatus');let loaded=0,failed=0;status.textContent='Loading referenced FASTA files…';
 const selected=kinds?FILES.fastas.filter(f=>kinds.includes(f.kind)):FILES.fastas;
 for(const f of selected){try{const response=await fetch(f.path);if(!response.ok)throw new Error(response.statusText);loaded+=parseFasta(await response.text(),f.label);}catch(e){failed++;}}
 status.textContent=loaded?`Loaded ${loaded.toLocaleString()} sequences${failed?' · '+failed+' file(s) require manual selection':''}.`:'Browser access to local FASTA files was blocked. Use “Choose FASTA…” to authorize them.';return loaded;}
document.getElementById('loadFastas').onclick=()=>loadReferencedFastas();
document.getElementById('fastaPicker').onchange=async e=>{let loaded=0;for(const file of e.target.files)loaded+=parseFasta(await file.text(),file.name);document.getElementById('fastaStatus').textContent=`Loaded ${loaded.toLocaleString()} sequences from ${e.target.files.length} selected file(s).`;};
function findSequence(ids){for(const id of ids.filter(Boolean)){if(sequenceCache.has(id))return sequenceCache.get(id);}return null;}
async function showSequence(ids,title,kind,range=null){let record=findSequence(ids);if(!record){await loadReferencedFastas([kind]);record=findSequence(ids);}if(!record){document.getElementById('fastaStatus').textContent=`Sequence ${ids.filter(Boolean).join(' / ')} is not loaded. Choose the matching FASTA file.`;return;}
 if(range){const start=Math.max(1,Math.min(range[0],range[1])),end=Math.min(record.seq.length,Math.max(range[0],range[1]));record={id:record.id+'_'+start+'-'+end,header:record.id+'|coords='+start+'-'+end,seq:record.seq.slice(start-1,end),source:record.source};}
 displayedSequence=record;document.getElementById('seqtitle').textContent=title+' · '+record.id;document.getElementById('seqtext').textContent='>'+record.header+'\n'+record.seq.match(/.{1,80}/g).join('\n');document.getElementById('seqcard').style.display='block';document.getElementById('seqcard').scrollIntoView({behavior:'smooth',block:'start'});}
function focusSequencePane(cid){const i=contigs.findIndex(c=>c.contig===cid);if(i>=0){idx=i;picker.value=i;}showTab('maps');}
function showContigSequence(cid){const c=contigs.find(x=>x.contig===cid);focusSequencePane(cid);showSequence([cid,c&&c.raw_id], 'Contig sequence','contigs');}
function showOrfSequence(orfId){const c=contigs.find(c=>c.orfs.some(o=>o.orf_id===orfId));if(c)focusSequencePane(c.contig);showSequence([orfId], 'ORF amino-acid sequence','orfs');}
function showOrfHit(orfId,start,end){const c=contigs.find(c=>c.orfs.some(o=>o.orf_id===orfId));if(c)focusSequencePane(c.contig);showSequence([orfId], 'Matched amino-acid region','orfs',[start,end]);}
document.getElementById('seqexport').onclick=()=>{if(displayedSequence)downloadText(safeName(displayedSequence.id)+'.fasta','>'+displayedSequence.header+'\n'+displayedSequence.seq.match(/.{1,80}/g).join('\n')+'\n','text/x-fasta');};
function fmtE(e){if(e===null||e===undefined||e==='')return'–';const n=+e;if(!isFinite(n))return e;if(n===0)return'0';const ex=Math.floor(Math.log10(n));return (n/Math.pow(10,ex)).toFixed(1)+'e'+ex;}
function fmtBytes(n){if(!n)return'–';const u=['B','KB','MB','GB'];let i=0;while(n>=1024&&i<u.length-1){n/=1024;i++;}return n.toFixed(1)+u[i];}
function isBest(h){return h.best&&h.best[crit];}
function hitVisible(h){if(!active.has(h.source))return false;if(mode==='best'&&!isBest(h))return false;
  if(minScore>0&&(h.score||0)<minScore)return false;if(maxEexp!==0&&h.evalue!==null&&h.evalue>Math.pow(10,maxEexp))return false;return true;}
function featVisible(f){if(mode==='best'&&f.best&&!f.best[crit])return false;return true;}
function packLanes(hits){const lanes=[],out=[];hits.forEach(h=>{let p=false;
 for(let l=0;l<lanes.length;l++){if((h.nt_from||h.qstart)>lanes[l]+18){lanes[l]=(h.nt_to||h.qend);out.push({h,lane:l});p=true;break;}}
 if(!p){lanes.push(h.nt_to||h.qend);out.push({h,lane:lanes.length-1});}});return {rows:out,nlanes:Math.max(1,lanes.length)};}
function densColor(v){const t=Math.max(0,Math.min(1,v));const r=Math.round(245-165*t),g=Math.round(245-70*t),b=Math.round(245-100*t);return`rgb(${r},${g},${b})`;}

function renderAll(){if(document.getElementById('pane-table').classList.contains('on'))renderTable();
  if(document.getElementById('pane-maps').classList.contains('on'))render();}

function renderExtra(id){
  const t=EXTRA.find(x=>x.id===id); if(!t)return;
  const inp=document.querySelector('.xsearch[data-for="'+id+'"]');
  const f=(inp&&inp.value||'').toLowerCase();
  let html=`<table><thead><tr>`+t.columns.map(c=>`<th>${esc(c)}</th>`).join('')+`</tr></thead><tbody>`;
  t.rows.forEach(r=>{if(f&&!r.join(' ').toLowerCase().includes(f))return;
    html+=`<tr>`+r.map(v=>`<td>${esc(v)}</td>`).join('')+`</tr>`;});
  html+=`</tbody></table>`;
  document.getElementById('xtable-'+id).innerHTML=html;
  renderComposition(t);
}

function renderComposition(t){
  const box=document.getElementById('xchart-'+t.id);if(!box||!t.composition||!t.composition.length)return;
  const entries=t.composition,total=entries.reduce((sum,x)=>sum+x.count,0),W=760,H=260,cx=135,cy=125,r=92;
  const palette=['#33507e','#2e8b57','#c65d21','#7b5ea7','#d19a00','#208a9b','#ad3d66','#6b7280'];
  let angle=-Math.PI/2,svg=`<div class="legend-note" style="margin-bottom:6px">Composition by ${esc(t.composition_label)}</div>`+
    `<svg viewBox="0 0 ${W} ${H}" style="max-width:760px" xmlns="${SVGNS}">`;
  entries.forEach((entry,i)=>{const frac=entry.count/total,next=angle+frac*Math.PI*2,x1=cx+r*Math.cos(angle),y1=cy+r*Math.sin(angle),x2=cx+r*Math.cos(next),y2=cy+r*Math.sin(next),large=frac>.5?1:0,col=palette[i%palette.length];
    if(frac>=.999999)svg+=`<circle cx="${cx}" cy="${cy}" r="${r}" fill="${col}"/>`;
    else svg+=`<path d="M${cx},${cy} L${x1},${y1} A${r},${r} 0 ${large},1 ${x2},${y2} Z" fill="${col}"/>`;
    const ly=20+i*22;svg+=`<rect x="270" y="${ly-11}" width="12" height="12" rx="2" fill="${col}"/>`+
      `<text x="290" y="${ly}" font-size="12" fill="#374151">${esc(entry.label)} — ${entry.count} (${(frac*100).toFixed(1)}%)</text>`;angle=next;});
  svg+=`<circle cx="${cx}" cy="${cy}" r="48" fill="#fff"/><text x="${cx}" y="${cy+5}" text-anchor="middle" font-size="16" font-weight="700">${total}</text></svg>`;
  box.innerHTML=svg;
}

function renderTable(){
  const f=(document.getElementById('tblSearch').value||'').toLowerCase();let html;const exportRows=[];
  if(tblView==='contigs'){
    const columns=['contig','raw_contig_id','length','orfs','hits','best','sources','top_score','top_profile','rna','nucleic_hits'];
    html=`<table><thead><tr><th>Contig</th><th>Len</th><th>ORFs</th><th>Hits</th><th>Best (${CRIT[crit]})</th><th>Sources</th><th>Top score</th><th>Top profile</th><th>RNA</th><th>Nucl.</th><th>Sequence</th></tr></thead><tbody>`;
    contigs.forEach((c)=>{if(f&&!(c.contig+' '+(c.raw_id||'')+' '+c.top_profile).toLowerCase().includes(f))return;
      const nb=(c.n_best&&c.n_best[crit]!=null)?c.n_best[crit]:'';
      const displayId=showRawIds&&c.raw_id?c.raw_id:c.short;
      exportRows.push([c.contig,c.raw_id||'',c.length,c.n_orfs,c.n_hits,nb,c.sources.join(';'),c.best_score,c.top_profile,c.rna?'true':'false',c.nucleic?c.nucleic.length:0]);
      html+=`<tr><td><span class="cid" onclick="openContig('${c.contig}')">${esc(displayId)}</span></td>`+
        `<td>${c.length.toLocaleString()}</td><td>${c.n_orfs}</td><td>${c.n_hits}</td><td>${nb}</td>`+
        `<td>${c.sources.map(s=>`<span class="pill" style="background:${colors[s]||'#888'}">${s}</span>`).join(' ')}</td>`+
        `<td>${c.best_score}</td><td class="mono">${c.top_profile}</td><td>${c.rna?'✓':''}</td><td>${c.nucleic?c.nucleic.length:''}</td>`+
        `<td><button class="btn" onclick="showContigSequence('${c.contig}')">Show</button></td></tr>`;});
    html+=`</tbody></table>`;
    currentTableExport={columns,rows:exportRows,name:'contigs_shown.tsv'};
  }else{
    const columns=['contig','raw_contig_id','orf','source','profile','score','evalue','coverage','aa_from','aa_to','best'];
    html=`<table><thead><tr><th>Contig</th><th>ORF</th><th>Source</th><th>Profile</th><th>Score</th><th>E-value</th><th>Cov</th><th>aa span</th><th>Best</th><th>Sequence</th></tr></thead><tbody>`;
    contigs.forEach((c)=>c.orfs.forEach(o=>o.hits.forEach(h=>{
      if(mode==='best'&&!isBest(h))return; if(!active.has(h.source))return;
      const hay=(c.contig+' '+h.profile+' '+h.source).toLowerCase(); if(f&&!hay.includes(f))return;
      const displayId=showRawIds&&c.raw_id?c.raw_id:c.short;
      exportRows.push([c.contig,c.raw_id||'',o.orf_id,h.source,h.profile,h.score,h.evalue,h.cov,h.aa_from,h.aa_to,isBest(h)?'true':'false']);
      html+=`<tr><td><span class="cid" onclick="openContig('${c.contig}')">${esc(displayId)}</span></td>`+
        `<td class="mono" style="font-size:11px">${o.orf_id.split('_').slice(-1)[0]}</td>`+
        `<td><span class="pill" style="background:${colors[h.source]||'#888'}">${h.source}</span></td>`+
        `<td class="mono">${h.profile}</td><td>${h.score}</td><td class="mono">${fmtE(h.evalue)}</td>`+
        `<td>${h.cov}</td><td class="mono">${h.aa_from}–${h.aa_to}</td><td>${isBest(h)?'✓':''}</td>`+
        `<td><button class="btn" onclick="showOrfHit('${o.orf_id}',${h.aa_from},${h.aa_to})">Show hit</button></td></tr>`;})));
    html+=`</tbody></table>`;
    currentTableExport={columns,rows:exportRows,name:'protein_hits_shown.tsv'};
  }
  document.getElementById('bigtable').innerHTML=html;
  document.getElementById('tbmeta').textContent=`${contigs.length} contigs · criterion: ${CRIT[crit]} · mode: ${mode==='best'?'best only':'all hits'}`;
}

function render(){
  const c=contigs[idx];if(!c)return;
  const hasRNA=!!c.rna, rnaOn=hasRNA&&showRNA;
  const cnuc=c.nucleic||[], nucOn=cnuc.length>0&&showNuc;
  const cmot=(c.motifs&&c.motifs.motifs)||[], motOn=cmot.length>0&&showMotif;
  document.getElementById('maptitle').textContent=c.contig;
  const nb=(c.n_best&&c.n_best[crit]!=null)?c.n_best[crit]:0;
  let sub=`${c.length.toLocaleString()} bp · ${c.n_orfs} ORF(s) · ${c.n_hits} hits (best ${nb}) · ${c.n_source} source(s)`;
  if(hasRNA)sub+=` · RNA ✓`; if(cnuc.length)sub+=` · ${cnuc.length} nucleic`;
  if(cmot.length)sub+=` · RdRp ${c.motifs.conformation||cmot.length}`;
  document.getElementById('mapsub').textContent=sub;

  const det=document.getElementById('details');
  let dh=`<div class="row"><span>Length</span><b>${c.length.toLocaleString()} bp</b></div>`+
   `<div class="row"><span>ORFs</span><b>${c.n_orfs}</b></div>`+
   `<div class="row"><span>Hits (all / best)</span><b>${c.n_hits} / ${nb}</b></div>`+
   `<div class="row"><span>Sources</span><b>${c.n_source}</b></div>`+
   `<div class="row"><span>Top score</span><b>${c.best_score}</b></div>`+
   `<div class="row"><span>Top profile</span><b>${c.top_profile}</b></div>`;
  if(hasRNA&&c.rna.structure){const s=c.rna.structure;
   dh+=`<div class="row"><span>RNA MFE</span><b>${s.mfe??'–'}</b></div>`+
       `<div class="row"><span>Paired</span><b>${s.paired_fraction!=null?(s.paired_fraction*100).toFixed(0)+'%':'–'}</b></div>`+
       `<div class="row"><span>GC</span><b>${s.gc!=null?(s.gc*100).toFixed(0)+'%':'–'}</b></div>`;}
  if(hasRNA&&c.rna.features.length)dh+=`<div class="row"><span>RNA features</span><b>${c.rna.features.length}</b></div>`;
  if(cnuc.length)dh+=`<div class="row"><span>Nucleic hits</span><b>${cnuc.length}</b></div>`;
  if(cmot.length)dh+=`<div class="row"><span>RdRp motifs</span><b>${c.motifs.conformation||cmot.length}</b></div>`;
  det.innerHTML=dh;

  const cnt={};Object.keys(colors).forEach(d=>cnt[d]=0);
  c.orfs.forEach(o=>o.hits.forEach(h=>{if(cnt[h.source]!==undefined&&hitVisible(h))cnt[h.source]++;}));
  Object.keys(colors).forEach(d=>{const el=document.getElementById('cnt_'+d);if(el)el.textContent=cnt[d]||'';});

  const W=Math.max(900,document.getElementById('mapholder').clientWidth||960);
  const padL=60,padR=30,padT=26,plotW=W-padL-padR,L=c.length;const x=nt=>padL+plotW*(nt/L);
  let y=padT;const bands=[];
  c.orfs.forEach(o=>{const vis=o.hits.filter(hitVisible);const {rows,nlanes}=packLanes(vis);
   const arrowH=22,laneH=17;bands.push({o,rows,nlanes,y0:y,arrowH,laneH});y+=arrowH+6+nlanes*laneH+24;});
  let rnaY=y,rnaH=0;
  if(rnaOn){const hs=!!c.rna.structure,hf=c.rna.features.length>0;rnaH=18+(hs?20:0)+(hf?18:0);y+=rnaH+16;}
  let nucY=y;
  if(nucOn){const {nlanes}=packLanes(cnuc);y+=18+nlanes*16+16;}
  let motY=y;
  if(motOn){const {nlanes}=packLanes(cmot);y+=18+nlanes*15+16;}
  const H=y+20;
  let s=`<svg viewBox="0 0 ${W} ${H}" xmlns="${SVGNS}"><rect width="${W}" height="${H}" fill="#fff"/>`;
  s+=`<line x1="${padL}" y1="16" x2="${padL+plotW}" y2="16" stroke="#c9ced8"/>`;
  for(let i=0;i<=8;i++){const p=Math.round(L*i/8),px=x(p);
   s+=`<line x1="${px}" y1="12" x2="${px}" y2="16" stroke="#9aa2b1"/>`;
   s+=`<text x="${px}" y="9" font-size="9" fill="#8a92a3" text-anchor="middle">${p.toLocaleString()}</text>`;}
  bands.forEach(b=>{const o=b.o,ay=b.y0,xa=x(o.start),xb=x(o.end),hh=b.arrowH;let path;
   if(o.strand===-1){const t=Math.min(xb-6,xa+12);path=`M${xb},${ay} L${t},${ay} L${xa},${ay+hh/2} L${t},${ay+hh} L${xb},${ay+hh} Z`;}
   else{const t=Math.max(xa+6,xb-12);path=`M${xa},${ay} L${t},${ay} L${xb},${ay+hh/2} L${t},${ay+hh} L${xa},${ay+hh} Z`;}
   s+=`<path d="${path}" fill="#dfe4ec" stroke="#aeb6c4"/>`;
   s+=`<text x="${(xa+xb)/2}" y="${ay+hh/2+4}" font-size="11" fill="#4a5266" text-anchor="middle" font-weight="600">${o.qlen||'?'} aa (${o.strand===-1?'−':'+'})</text>`;
   b.rows.forEach(({h,lane})=>{const dy=ay+hh+6+lane*b.laneH,xf=x(h.nt_from),xt=x(h.nt_to),w=Math.max(3,xt-xf),col=colors[h.source]||'#888';
    const meta=JSON.stringify(h).replace(/"/g,'&quot;');const op=(mode==='all'&&!isBest(h))?0.4:0.9;
    s+=`<rect class="dom" data-m="${meta}" x="${xf}" y="${dy}" width="${w}" height="${b.laneH-4}" rx="2" fill="${col}" fill-opacity="${op}" stroke="${col}" stroke-width="0.8"/>`;
    if(w>44){const mx=Math.floor(w/6);const lbl=h.profile.length>mx?h.profile.slice(0,mx)+'…':h.profile;
     s+=`<text x="${xf+4}" y="${dy+b.laneH-8}" font-size="9.5" fill="#fff" pointer-events="none">${lbl}</text>`;}});});
  if(rnaOn){let ry=rnaY;
   s+=`<line x1="${padL}" y1="${ry}" x2="${padL+plotW}" y2="${ry}" stroke="#e5e7eb"/>`;
   s+=`<text x="${padL}" y="${ry+13}" font-size="10.5" fill="#6b7280" font-weight="600">RNA</text>`;ry+=18;
   const R=c.rna;
   if(R.structure){const st=R.structure,bins=st.density;const spanFrom=x(st.start||1),spanTo=x(st.end||L),spanW=Math.max(1,spanTo-spanFrom),cw=spanW/bins.length;
    for(let i=0;i<bins.length;i++){const meta=JSON.stringify({kind:'dens',v:bins[i],
      nt_from:Math.round((st.start||1)+st.len*i/bins.length),nt_to:Math.round((st.start||1)+st.len*(i+1)/bins.length),
      mfe:st.mfe,pf:st.paired_fraction,gc:st.gc,src:st.source}).replace(/"/g,'&quot;');
     s+=`<rect class="dens" data-m="${meta}" x="${spanFrom+i*cw}" y="${ry}" width="${Math.ceil(cw)}" height="14" fill="${densColor(bins[i])}"/>`;}
    s+=`<rect x="${spanFrom}" y="${ry}" width="${spanW}" height="14" fill="none" stroke="#cfd6e0"/>`;
    s+=`<text x="${Math.min(spanTo+4,W-70)}" y="${ry+11}" font-size="9" fill="#8a92a3">pairing density</text>`;ry+=20;}
   if(R.features.length){R.features.filter(featVisible).forEach(fe=>{if(fe.start==null||fe.end==null)return;
     const xf=x(fe.start),xt=x(fe.end),w=Math.max(4,xt-xf),col=rnaColors[fe.klass]||rnaColors.other||'#7F7F7F';
     const meta=JSON.stringify(Object.assign({kind:'feat'},fe)).replace(/"/g,'&quot;');
     const op=(mode==='all'&&fe.best&&!fe.best[crit])?0.4:0.9;
     s+=`<rect class="rnafeat" data-m="${meta}" x="${xf}" y="${ry}" width="${w}" height="13" rx="3" fill="${col}" fill-opacity="${op}" stroke="${col}"/>`;
     if(w>40)s+=`<text x="${xf+3}" y="${ry+10}" font-size="9" fill="#fff" pointer-events="none">${fe.klass}</text>`;});}
  }
  if(nucOn){let ny=nucY;
   s+=`<line x1="${padL}" y1="${ny}" x2="${padL+plotW}" y2="${ny}" stroke="#e5e7eb"/>`;
   s+=`<text x="${padL}" y="${ny+13}" font-size="10.5" fill="#6b7280" font-weight="600">Nucleic</text>`;ny+=18;
   const {rows}=packLanes(cnuc);
   rows.forEach(({h,lane})=>{const dy=ny+lane*16,xf=x(Math.min(h.qstart,h.qend)),xt=x(Math.max(h.qstart,h.qend)),w=Math.max(4,xt-xf),col=nucColors[h.source]||nucColors.other||'#6b7280';
     const meta=JSON.stringify(Object.assign({kind:'nuc'},h)).replace(/"/g,'&quot;');
     s+=`<rect class="nuc" data-m="${meta}" x="${xf}" y="${dy}" width="${w}" height="12" rx="2" fill="${col}" fill-opacity="0.85" stroke="${col}"/>`;
     if(w>60){const lbl=(h.target||'').slice(0,Math.floor(w/6));s+=`<text x="${xf+3}" y="${dy+9}" font-size="8.5" fill="#fff" pointer-events="none">${esc(lbl)}</text>`;}});
  }
  if(motOn){let my=motY;
   s+=`<line x1="${padL}" y1="${my}" x2="${padL+plotW}" y2="${my}" stroke="#e5e7eb"/>`;
   s+=`<text x="${padL}" y="${my+13}" font-size="10.5" fill="#6b7280" font-weight="600">RdRp motifs</text>`;my+=18;
   // Small blocks per catalytic motif (A/B/C/D...), coloured by letter; motifs are
   // short so we widen them a touch to stay clickable/labelled.
   const {rows}=packLanes(cmot);
   rows.forEach(({h,lane})=>{const dy=my+lane*15,xf=x(h.nt_from),xt=x(h.nt_to),w=Math.max(6,xt-xf),col=motifColors[h.letter]||motifColors.other||'#7F7F7F';
     const meta=JSON.stringify(Object.assign({kind:'motif'},h)).replace(/"/g,'&quot;');
     s+=`<rect class="motif" data-m="${meta}" x="${xf}" y="${dy}" width="${w}" height="12" rx="2" fill="${col}" fill-opacity="0.9" stroke="${col}"/>`;
     s+=`<text x="${xf+Math.max(2,w/2-3)}" y="${dy+9.5}" font-size="9" fill="#fff" pointer-events="none">${esc(h.letter)}</text>`;});
  }
  s+=`</svg>`;
  document.getElementById('mapholder').innerHTML=s;

  document.querySelectorAll('rect.dom').forEach(el=>{el.style.cursor='pointer';
   el.addEventListener('mousemove',ev=>{const m=JSON.parse(el.getAttribute('data-m').replace(/&quot;/g,'"'));
    let html=`<b>${m.profile}</b>`+(m.acc?` <span class="k">(${m.acc})</span>`:'')+
     (m.best&&m.best[crit]?` <span class="pill" style="background:#2e8b57">best·${CRIT[crit]}</span>`:'')+
     `<br><span class="pill" style="background:${colors[m.source]||'#888'}">${m.source}</span>`+
     `<br><span class="k">E</span> ${fmtE(m.evalue)} · <span class="k">score</span> ${m.score} · <span class="k">len</span> ${m.ali_len??'–'} · <span class="k">cov</span> ${m.cov}`+
     `<br><span class="k">HMM</span> ${m.hmm_from}–${m.hmm_to}/${m.hmm_len} · <span class="k">aa</span> ${m.aa_from}–${m.aa_to} · <span class="k">nt</span> ${(m.nt_from||0).toLocaleString()}–${(m.nt_to||0).toLocaleString()}`;
    if(m.desc)html+=`<br><span class="k">${m.desc}</span>`;
    if(m.aln)html+=`<span class="k" style="display:block;margin-top:4px">aligned region:</span><span class="aln">${m.aln}</span>`;
    showTip(html,ev.clientX,ev.clientY);});
   el.addEventListener('mouseleave',hideTip);});
  document.querySelectorAll('rect.dens').forEach(el=>{el.addEventListener('mousemove',ev=>{
    const m=JSON.parse(el.getAttribute('data-m').replace(/&quot;/g,'"'));
    showTip(`<b>RNA secondary structure</b> <span class="k">(${m.src})</span>`+
     `<br><span class="k">window</span> nt ${m.nt_from.toLocaleString()}–${m.nt_to.toLocaleString()}`+
     `<br><span class="k">paired in window</span> ${(m.v*100).toFixed(0)}%`+
     `<br><span class="k">contig MFE</span> ${m.mfe??'–'} · <span class="k">paired</span> ${m.pf!=null?(m.pf*100).toFixed(0)+'%':'–'} · <span class="k">GC</span> ${m.gc!=null?(m.gc*100).toFixed(0)+'%':'–'}`,ev.clientX,ev.clientY);});
   el.addEventListener('mouseleave',hideTip);});
  document.querySelectorAll('rect.rnafeat').forEach(el=>{el.style.cursor='pointer';
   el.addEventListener('mousemove',ev=>{const m=JSON.parse(el.getAttribute('data-m').replace(/&quot;/g,'"'));
    let html=`<b>${m.klass}</b> <span class="k">(${m.type})</span>`+(m.profile?` ${m.profile}`:'')+
     (m.best&&m.best[crit]?` <span class="pill" style="background:#2e8b57">best·${CRIT[crit]}</span>`:'')+
     `<br><span class="k">source</span> ${m.source} · <span class="k">strand</span> ${m.strand||'?'}`+
     `<br><span class="k">nt</span> ${(m.start||0).toLocaleString()}–${(m.end||0).toLocaleString()} · <span class="k">score</span> ${m.score??'–'} · <span class="k">E</span> ${fmtE(m.evalue)}`;
    if(m.note)html+=`<br><span class="k">${m.note}</span>`;if(m.dbn)html+=`<span class="aln">${m.dbn}</span>`;
    showTip(html,ev.clientX,ev.clientY);});
   el.addEventListener('mouseleave',hideTip);});
  document.querySelectorAll('rect.nuc').forEach(el=>{el.style.cursor='pointer';
   el.addEventListener('mousemove',ev=>{const m=JSON.parse(el.getAttribute('data-m').replace(/&quot;/g,'"'));
    showTip(`<b>${esc(m.target)}</b>`+
     `<br><span class="pill" style="background:${nucColors[m.source]||'#6b7280'}">${m.source}</span>`+
     `<br><span class="k">identity</span> ${m.pident}% · <span class="k">score</span> ${m.score} · <span class="k">E</span> ${fmtE(m.evalue)} · <span class="k">strand</span> ${m.strand||'?'}`+
     `<br><span class="k">contig</span> ${m.qstart.toLocaleString()}–${m.qend.toLocaleString()} · <span class="k">target</span> ${m.tstart.toLocaleString()}–${m.tend.toLocaleString()}`+
     `<br><span class="k">qcov</span> ${(m.qcov*100).toFixed(0)}% · <span class="k">tcov</span> ${(m.tcov*100).toFixed(0)}%`,ev.clientX,ev.clientY);});
   el.addEventListener('mouseleave',hideTip);});
  document.querySelectorAll('rect.motif').forEach(el=>{el.style.cursor='pointer';
   el.addEventListener('mousemove',ev=>{const m=JSON.parse(el.getAttribute('data-m').replace(/&quot;/g,'"'));
    let html=`<b>RdRp motif ${esc(m.letter)}</b>`+(m.profile?` <span class="k">(${esc(m.profile)})</span>`:'')+
     `<br><span class="k">frame</span> ${m.frame} · <span class="k">score</span> ${m.score} · <span class="k">E</span> ${fmtE(m.evalue)}`+
     `<br><span class="k">aa</span> ${m.aa_from}–${m.aa_to} · <span class="k">nt</span> ${(m.nt_from||0).toLocaleString()}–${(m.nt_to||0).toLocaleString()}`;
    if(m.alignment)html+=`<span class="aln">${esc(m.alignment)}</span>`;
    showTip(html,ev.clientX,ev.clientY);});
   el.addEventListener('mouseleave',hideTip);});

  const allh=[];c.orfs.forEach(o=>o.hits.forEach(h=>{if(hitVisible(h))allh.push({h,o});}));
  allh.sort((a,b)=>(b.h.score||0)-(a.h.score||0));
  const hitColumns=['contig','orf','source','profile','score','evalue','coverage','alignment_length','aa_from','aa_to','best','description'];
  currentMapExport={columns:hitColumns,rows:allh.map(({h,o})=>[c.contig,o.orf_id,h.source,h.profile,h.score,h.evalue,h.cov,h.ali_len,h.aa_from,h.aa_to,isBest(h)?'true':'false',h.desc||'']),name:safeName(c.contig)+'_hits_shown.tsv'};
  let t=`<div class="exportbar"><button onclick="exportMapTable()">Export shown hits TSV</button>${sourceLinks('protein')}${sourceLinks('rna')}${sourceLinks('motif')}</div>`+
    `<table><thead><tr><th>Source</th><th>Profile</th><th>Score</th><th>E-value</th><th>Cov</th><th>Len</th><th>aa span</th><th>Best</th><th>Description</th><th>Sequence</th></tr></thead><tbody>`;
  allh.forEach(({h,o})=>{t+=`<tr><td><span class="pill" style="background:${colors[h.source]||'#888'}">${h.source}</span></td>`+
   `<td class="mono">${h.profile}</td><td>${h.score}</td><td class="mono">${fmtE(h.evalue)}</td><td>${h.cov}</td>`+
   `<td>${h.ali_len??'–'}</td><td class="mono">${h.aa_from}–${h.aa_to}</td><td>${isBest(h)?'✓':''}</td>`+
   `<td style="max-width:300px">${h.desc?h.desc:'<span style="color:#9aa2b1">—</span>'}</td>`+
   `<td><button class="btn" onclick="showOrfHit('${o.orf_id}',${h.aa_from},${h.aa_to})">Show hit</button></td></tr>`;});
  t+=`</tbody></table>`;
  if(c.rna&&c.rna.features.length){t+=`<h2 style="margin-top:14px">RNA features</h2><table><thead><tr><th>Class</th><th>Source</th><th>nt span</th><th>Score</th><th>Best</th><th>Profile / note</th></tr></thead><tbody>`;
   c.rna.features.forEach(fe=>{t+=`<tr><td><span class="pill" style="background:${rnaColors[fe.klass]||'#7F7F7F'}">${fe.klass}</span></td>`+
     `<td>${fe.source}</td><td class="mono">${(fe.start||0).toLocaleString()}–${(fe.end||0).toLocaleString()}</td>`+
     `<td>${fe.score??'–'}</td><td>${fe.best&&fe.best[crit]?'✓':''}</td><td>${[fe.profile,fe.note].filter(Boolean).join(' · ')}</td></tr>`;});
   t+=`</tbody></table>`;}
  if(cnuc.length){t+=`<h2 style="margin-top:14px">Nucleic hits</h2><table><thead><tr><th>Source</th><th>Target</th><th>%id</th><th>Score</th><th>E</th><th>contig span</th><th>strand</th></tr></thead><tbody>`;
   cnuc.forEach(h=>{t+=`<tr><td><span class="pill" style="background:${nucColors[h.source]||'#6b7280'}">${h.source}</span></td>`+
     `<td>${esc(h.target)}</td><td>${h.pident}</td><td>${h.score}</td><td class="mono">${fmtE(h.evalue)}</td>`+
     `<td class="mono">${h.qstart.toLocaleString()}–${h.qend.toLocaleString()}</td><td>${h.strand||''}</td></tr>`;});
   t+=`</tbody></table>`;}
  if(cmot.length){t+=`<h2 style="margin-top:14px">RdRp motifs${c.motifs.conformation?' ('+esc(c.motifs.conformation)+')':''}</h2>`+
     `<table><thead><tr><th>Motif</th><th>Frame</th><th>Score</th><th>E</th><th>nt span</th><th>Profile</th><th>Alignment</th></tr></thead><tbody>`;
   cmot.forEach(m=>{t+=`<tr><td><span class="pill" style="background:${motifColors[m.letter]||'#7F7F7F'}">${esc(m.letter)}</span></td>`+
     `<td>${m.frame}</td><td>${m.score}</td><td class="mono">${fmtE(m.evalue)}</td>`+
     `<td class="mono">${(m.nt_from||0).toLocaleString()}–${(m.nt_to||0).toLocaleString()}</td>`+
     `<td class="mono" style="font-size:11px">${esc(m.profile||'')}</td><td class="mono" style="font-size:11px;word-break:break-all">${esc(m.alignment||'')}</td></tr>`;});
   t+=`</tbody></table>`;}
  document.getElementById('tablebox').innerHTML=t;
}

function renderNucleic(){
  const f=(document.getElementById('nucSearch').value||'').toLowerCase();
  const columns=['contig','source','target','pident','score','evalue','qstart','qend','tstart','tend','qcov','tcov','strand'],rows=[];
  let t=`<table><thead><tr><th>Contig</th><th>Source</th><th>Target</th><th>%id</th><th>Score</th><th>E-value</th><th>contig span</th><th>target span</th><th>qcov</th><th>strand</th><th>Sequence</th></tr></thead><tbody>`;
  NUCLEIC.forEach(h=>{const hay=(h.contig+' '+h.target+' '+h.source).toLowerCase();if(f&&!hay.includes(f))return;
    rows.push([h.contig,h.source,h.target,h.pident,h.score,h.evalue,h.qstart,h.qend,h.tstart,h.tend,h.qcov,h.tcov,h.strand||'']);
    const link=hasMaps&&contigs.some(c=>c.contig===h.contig)?`<span class="cid" onclick="openContig('${h.contig}')">${h.contig}</span>`:h.contig;
    t+=`<tr><td>${link}</td><td><span class="pill" style="background:${nucColors[h.source]||'#6b7280'}">${h.source}</span></td>`+
      `<td>${esc(h.target)}</td><td>${h.pident}</td><td>${h.score}</td><td class="mono">${fmtE(h.evalue)}</td>`+
      `<td class="mono">${h.qstart.toLocaleString()}–${h.qend.toLocaleString()}</td>`+
      `<td class="mono">${h.tstart.toLocaleString()}–${h.tend.toLocaleString()}</td>`+
      `<td>${(h.qcov*100).toFixed(0)}%</td><td>${h.strand||''}</td>`+
      `<td><button class="btn" onclick="showContigSequence('${h.contig}')">Show query</button></td></tr>`;});
  t+=`</tbody></table>`;
  document.getElementById('nuctable').innerHTML=t;
  currentNucleicExport={columns,rows};
}

function renderStats(){
  if(!STATS){document.getElementById('statsbox').innerHTML='';return;}
  let html='';
  const a=STATS.assembly||{};
  if(a&&(a.n_contigs||a.assemblers)){
    const sub=a.source?` <span class="legend-note">(${esc(a.source)})</span>`:'';
    html+=`<div class="card"><h2>Assembly${sub}</h2><div class="statcards">`+
      `<div class="statcard"><div class="v">${(a.n_contigs||0).toLocaleString()}</div><div class="l">contigs</div></div>`+
      `<div class="statcard"><div class="v">${(a.total_bp||0).toLocaleString()}</div><div class="l">total bp</div></div>`+
      `<div class="statcard"><div class="v">${(a.n50||0).toLocaleString()}</div><div class="l">N50</div></div>`+
      `<div class="statcard"><div class="v">${(a.max||0).toLocaleString()}</div><div class="l">max</div></div>`+
      `<div class="statcard"><div class="v">${(a.mean||0).toLocaleString()}</div><div class="l">mean</div></div></div>`;
    // When more than one assembler ran, show each assembler's own contigs too.
    if(a.assemblers&&a.assemblers.length){
      html+=`<table style="margin-top:10px"><thead><tr><th>Assembler</th><th>Contigs</th><th>Total bp</th><th>N50</th><th>Max</th><th>Mean</th></tr></thead><tbody>`;
      a.assemblers.forEach(s=>{html+=`<tr><td>${esc(s.name)}</td><td>${s.n_contigs.toLocaleString()}</td>`+
        `<td>${s.total_bp.toLocaleString()}</td><td>${s.n50.toLocaleString()}</td>`+
        `<td>${s.max.toLocaleString()}</td><td>${s.mean.toLocaleString()}</td></tr>`;});
      html+=`</tbody></table><div class="legend-note">Per-assembler counts are raw contigs (before dereplication / clustering / length filtering).</div>`;
    }
    html+=`</div>`;
  }
  const mg=STATS.merge||null;
  if((STATS.reads&&STATS.reads.length)||(mg&&mg.pct!=null)){
    html+=`<div class="card"><h2>Read filtering (per step)</h2>`;
    if(mg&&mg.pct!=null){
      const joined=mg.joined!=null?`, ~ ${mg.joined.toLocaleString()} joined`:'';
      html+=`<div style="margin-bottom:10px;font-size:14px"><b>${mg.pct.toFixed(1)}%</b> reads merged (bbmerge${joined})</div>`;
    }
    if(STATS.reads&&STATS.reads.length){
      html+=`<table><thead><tr><th>Step</th><th>Reads in</th><th>Removed</th><th>Removed %</th><th>Kept</th><th>Kept %</th></tr></thead><tbody>`;
      STATS.reads.forEach(r=>{html+=`<tr><td>${r.step}</td><td>${r.total!=null?r.total.toLocaleString():'–'}</td>`+
        `<td>${r.matched!=null?r.matched.toLocaleString():'–'}</td><td>${r.pct||'–'}</td>`+
        `<td>${r.kept!=null?r.kept.toLocaleString():'–'}</td><td>${r.kept_pct!=null?r.kept_pct.toFixed(1)+'%':'–'}</td></tr>`;});
      html+=`</tbody></table>`;
    }
    html+=`</div>`;
  }
  if(STATS.adapters&&STATS.adapters.length){
    html+=`<div class="card"><h2>Discovered adapters (bbmerge)</h2><table><thead><tr><th>Name</th><th>Sequence</th></tr></thead><tbody>`;
    STATS.adapters.forEach(ad=>{html+=`<tr><td>${esc(ad.name)}</td><td class="mono" style="font-size:11px;word-break:break-all">${esc(ad.sequence)}</td></tr>`;});
    html+=`</tbody></table></div>`;
  }
  const dom=STATS.rrna_domain||{};
  if(STATS.rrna_top&&STATS.rrna_top.length){
    html+=`<div class="card"><h2>Top rRNA matches (decontamination)</h2>`;
    if(dom.eukaryotic!=null||dom.prokaryotic!=null){
      html+=`<div class="statcards" style="margin-bottom:10px">`+
        `<div class="statcard"><div class="v">${(dom.eukaryotic||0).toFixed(1)}%</div><div class="l">eukaryotic</div></div>`+
        `<div class="statcard"><div class="v">${(dom.prokaryotic||0).toFixed(1)}%</div><div class="l">prokaryotic</div></div>`+
        `<div class="statcard"><div class="v">${(dom.unknown||0).toFixed(1)}%</div><div class="l">unknown</div></div></div>`;
    }
    const hasNames=STATS.rrna_top.some(r=>r.query_name);
    html+=`<table><thead><tr><th>taxid</th>${hasNames?'<th>organism</th>':''}<th>subunit</th><th>domain</th><th>rRNA type</th><th>reads</th><th>%</th>${hasNames?'<th>rel.</th><th>assembly</th>':''}</tr></thead><tbody>`;
    STATS.rrna_top.forEach(r=>{html+=`<tr><td class="mono">${r.taxid??'–'}</td>`+
      (hasNames?`<td>${esc(r.query_name||'')}</td>`:'')+
      `<td>${r.subunit}</td><td>${r.domain}</td><td class="mono" style="font-size:11px">${esc(r.rrna_type||'')}</td>`+
      `<td>${r.reads}</td><td>${r.pct}</td>`+
      (hasNames?`<td>${r.relationship||''}</td><td>${r.assembly_level||''}</td>`:'')+`</tr>`;});
    html+=`</tbody></table></div>`;
  }
  if(STATS.falco&&STATS.falco.length){
    // Per-module PASS/WARN/FAIL as labelled coloured tags (so each test's status
    // is actually readable, not a single anonymous colour bar).
    const tag=(m,v)=>{const c=v==='PASS'?'#2E8B57':(v==='WARN'?'#CCB974':'#C44E52');
      return `<span class="pill" style="background:${c};margin:1px" title="${v}">${esc(m)}</span>`;};
    html+=`<div class="card"><h2>Read QC (falco / FastQC)</h2>`;
    STATS.falco.forEach((fq,i)=>{
      html+=`<div style="margin-bottom:6px"><b class="mono" style="font-size:12px">${esc(fq.file)}</b> — `+
        `${fq.total_sequences!=null?fq.total_sequences.toLocaleString():'?'} seqs, `+
        `${esc(fq.total_bases||'?')}, GC ${fq.gc!=null?fq.gc:'?'}%, length ${esc(fq.length||'?')}</div>`;
      html+=`<div style="margin-bottom:6px">`+
        Object.entries(fq.flags||{}).map(([m,v])=>tag(m,v)).join('')+`</div>`;
      // Embed the full falco HTML report in an iframe when we captured it.
      // Falco's in-report navigation uses bare fragment links (href="#module").
      // Inside a srcdoc iframe those resolve against the *parent* page URL, so a
      // click would reload the whole report page into the iframe. Inject a tiny
      // click-interceptor that turns fragment links into in-iframe scrolls, and
      // a <base target="_self"> so nothing escapes to the parent.
      if(fq.report_html){
        const guard='<base target="_self">'+
          '<script>document.addEventListener("click",function(e){'+
          'var a=e.target.closest&&e.target.closest(\'a[href^="#"]\');if(!a)return;'+
          'e.preventDefault();var el=document.getElementById(decodeURIComponent(a.getAttribute("href").slice(1)));'+
          'if(el)el.scrollIntoView();},true);<\/script>';
        const doc=/<head[^>]*>/i.test(fq.report_html)
          ? fq.report_html.replace(/<head[^>]*>/i, m=>m+guard)
          : guard+fq.report_html;
        html+=`<iframe sandbox="allow-scripts allow-same-origin" `+
          `style="width:100%;height:520px;border:1px solid var(--line);border-radius:8px;margin-bottom:14px" `+
          `srcdoc="${doc.replace(/"/g,'&quot;')}"></iframe>`;
      }
    });
    html+=`</div>`;
  }
  if(STATS.files&&STATS.files.length){
    // Point #5: show a real % merged (from bbmerge) in the merged column when known.
    const mergedPct=(mg&&mg.pct!=null)?`${mg.pct.toFixed(1)}%`:null;
    html+=`<div class="card"><h2>Intermediate files</h2><table><thead><tr><th>Step</th><th>Type</th><th>Size</th><th>Merged</th></tr></thead><tbody>`;
    STATS.files.forEach(fr=>{const cell=fr.merged?(mergedPct||'✓'):'';
      html+=`<tr><td>${fr.step}</td><td>${fr.type}</td><td>${fmtBytes(fr.size)}</td><td>${cell}</td></tr>`;});
    html+=`</tbody></table></div>`;
  }
  document.getElementById('statsbox').innerHTML=html||'<div class="card">No run statistics found.</div>';
}

addEventListener('resize',()=>{if(document.getElementById('pane-maps').classList.contains('on'))render();});
let startTab=DATA.initial_tab;
const available=TABS.map(t=>t[0]);
if(!available.includes(startTab))startTab=available[0]||"table";
showTab(startTab);
</script>
</body></html>
"""
