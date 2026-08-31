import os
import re
from pathlib import Path

from rich.console import Console
from rich_click import Choice, command, option

from rolypoly.utils.logging.config import BaseConfig


class RVirusSearchConfig(BaseConfig):
    def __init__(self, **kwargs):
        # Always treat output as a directory
        output_path = Path(kwargs.get("output", ""))
        kwargs["output_dir"] = str(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        super().__init__(
            input=kwargs.get("input", ""),
            output=kwargs.get("output", ""),
            keep_tmp=kwargs.get("keep_tmp", False),
            log_file=kwargs.get("log_file"),
            threads=kwargs.get("threads", 1),
            memory=kwargs.get("memory"),
            config_file=kwargs.get("config_file", None),
            overwrite=kwargs.get("overwrite", False),
            log_level=kwargs.get("log_level", "INFO"),
            temp_dir=kwargs.get("temp_dir", "marker_search_tmp/"),
        )  # initialize the BaseConfig class
        # initialize the rest of the parameters (i.e. the ones that are not in the BaseConfig class)
        self.database = kwargs.get("database", "RVMT,genomad")
        self.search_tool = kwargs.get("search_tool", "hmmsearch")
        self.inc_evalue = kwargs.get("inc_evalue", 0.001)
        self.score = kwargs.get("score", 20)
        self.min_ali_len = kwargs.get("min_ali_len", 15)
        self.aa_method = kwargs.get("aa_method", "six_frame")
        self.resolve_mode = kwargs.get("resolve_mode") or "simple"
        self.min_overlap_positions = kwargs.get("min_overlap_positions") or 10
        self.repeat_filter = kwargs.get("repeat_filter", True)
        self.name = kwargs.get("name") or None
        self.write_matched_regions = kwargs.get("write_matched_regions", True)
        self.matched_regions_output = (
            kwargs.get("matched_regions_output") or None
        )
        self.include_aligned_region = kwargs.get("include_aligned_region", True)
        self.include_alignment_string = kwargs.get(
            "include_alignment_string", False
        )
        self.write_matched_input_seqs = kwargs.get(
            "write_matched_input_seqs", False
        )
        self.matched_input_seqs_output = (
            kwargs.get("matched_input_seqs_output") or None
        )


def search_mmseqs_marker_db(
    amino_file,
    db_path,
    output,
    temp_dir,
    threads,
    inc_evalue,
    score,
    min_ali_len,
    include_aligned_region,
    include_alignment_string,
    include_full_query,
    logger,
    include_alignment_path=False,
):
    """Search an MMseqs profile database and write marker-search hit columns."""
    import polars as pl

    from rolypoly.utils.bio.alignments import mmseqs_easy_search

    output = Path(output)
    raw_output = output.with_suffix(".mmseqs.tsv")
    format_columns = [
        "query",
        "qheader",
        "target",
        "theader",
        "evalue",
        "bits",
        "qstart",
        "qend",
        "tstart",
        "tend",
        "qlen",
        "tlen",
        "alnlen",
        "qcov",
        "tcov",
        "qaln",
        "taln",
    ]
    if include_full_query:
        format_columns.append("qseq")

    mmseqs_easy_search(
        query=amino_file,
        target=db_path,
        output=raw_output,
        tmp_dir=Path(temp_dir) / f"mmseqs_{output.stem}",
        threads=threads,
        evalue=inc_evalue,
        format_mode=4,
        format_output=format_columns,
        logger=logger,
    )
    if not raw_output.exists():
        raise RuntimeError(
            f"MMseqs2 marker search did not create its output: {raw_output}"
        )

    raw_hits = pl.read_csv(raw_output, separator="\t")
    hits = raw_hits.filter(
        (pl.col("evalue") <= inc_evalue)
        & (pl.col("bits") >= score)
        & (pl.col("alnlen") >= min_ali_len)
    )
    profile_id = pl.col("target").str.replace(r"_seed$", "")
    output_columns = [
        pl.col("qheader").alias("query_full_name"),
        profile_id.alias("hmm_full_name"),
        profile_id.alias("profile_accession"),
        pl.col("tlen").alias("hmm_len"),
        pl.col("qlen"),
        pl.col("evalue").alias("full_hmm_evalue"),
        pl.col("bits").alias("full_hmm_score"),
        pl.lit(0.0).alias("full_hmm_bias"),
        pl.col("bits").alias("this_dom_score"),
        pl.lit(0.0).alias("this_dom_bias"),
        pl.col("tstart").alias("hmm_from"),
        pl.col("tend").alias("hmm_to"),
        pl.col("qstart").alias("q1"),
        pl.col("qend").alias("q2"),
        pl.col("qstart").alias("env_from"),
        pl.col("qend").alias("env_to"),
        pl.col("tcov").alias("hmm_cov"),
        pl.col("alnlen").alias("ali_len"),
        pl.col("theader").str.replace(r"^\S+\s*", "").alias("dom_desc"),
    ]
    if include_aligned_region:
        output_columns.append(pl.col("qaln").alias("aligned_region"))
    if include_alignment_path:
        output_columns.extend(
            (
                pl.lit("1").alias("domain_index"),
                pl.col("taln").alias("profile_alignment"),
                pl.col("qaln").alias("query_alignment"),
            )
        )
    if include_full_query:
        output_columns.append(pl.col("qseq").alias("full_qseq"))
    if include_alignment_string:
        output_columns.append(
            pl.struct("qaln", "taln")
            .map_elements(
                lambda row: "".join(
                    "|" if query == target and query != "-" else " "
                    for query, target in zip(row["qaln"], row["taln"])
                ),
                return_dtype=pl.String,
            )
            .alias("identity_str")
        )

    hits.select(output_columns).write_csv(output, separator="\t")
    return output


FEATURE_PROJECTION_COLUMNS = (
    "marker_path_id",
    "database_id",
    "query_full_name",
    "hmm_full_name",
    "profile_accession",
    "domain_index",
    "feature_id",
    "canonical_term_id",
    "raw_source_label",
    "backend",
    "full_hmm_evalue",
    "full_hmm_score",
    "this_dom_score",
    "profile_states_1based",
    "mapped_profile_states_1based",
    "projected_query_positions_1based",
    "projected_query_residues",
    "unresolved_profile_states_1based",
    "unresolved_query_positions_1based",
    "unresolved_query_residues",
    "deleted_profile_states_1based",
    "out_of_span_profile_states_1based",
    "feature_query_start_1based",
    "feature_query_end_1based",
    "insertion_query_positions_1based",
    "insertion_query_residues",
    "mapped_state_count",
    "unresolved_state_count",
    "deleted_state_count",
    "out_of_span_state_count",
    "projection_status",
    "projection_reason",
    "hmm_from",
    "hmm_to",
    "q1",
    "q2",
)


def project_marker_features(
    hit_df, feature_path: Path, output_path: Path, backend: str, logger=None
) -> None:
    """Project labelled profile states through retained local alignments.

    This is deliberately an annotation/reporting pass.  It does not alter the
    marker hit table or apply motif order, completeness, or residue filters.
    ``profile_alignment`` and ``query_alignment`` use the same profile/query
    orientation for HMMER and MMseqs2, so one coordinate walker handles both.
    """
    import polars as pl

    if backend not in {"hmmsearch", "mmseqs2"}:
        raise ValueError(f"Unsupported marker projection backend: {backend}")
    features = pl.read_csv(feature_path, separator="\t")
    required = {
        "marker_db",
        "profile_id",
        "feature_id",
        "canonical_term_id",
        "raw_source_label",
    }
    state_columns = {
        "hmmer_states_1based",
        "hmm_states_1based",
        "mmseqs_states_1based",
    }
    missing = required.difference(features.columns)
    backend_state_column_present = (
        (
            "hmmer_states_1based" in features.columns
            or "hmm_states_1based" in features.columns
        )
        if backend == "hmmsearch"
        else "mmseqs_states_1based" in features.columns
    )
    if (
        missing
        or not state_columns.intersection(features.columns)
        or not backend_state_column_present
    ):
        missing_text = sorted(missing)
        if not backend_state_column_present:
            missing_text.append(f"{backend} state column")
        raise ValueError(
            f"Marker feature table {feature_path} is missing columns: "
            f"{missing_text}"
        )

    def parse_states(value, field: str, row_number: int) -> list[int]:
        if value is None:
            return []
        text = str(value).strip()
        if not text or text.lower() in {"na", "none", "null", "not_applicable"}:
            return []
        try:
            states = [int(token) for token in text.replace(",", ";").split(";")]
        except ValueError as exc:
            raise ValueError(
                f"Invalid {field} in {feature_path} row {row_number}: {text!r}"
            ) from exc
        if any(state < 1 for state in states):
            raise ValueError(
                f"{field} must contain positive 1-based states in "
                f"{feature_path} row {row_number}"
            )
        if len(states) != len(set(states)):
            raise ValueError(
                f"{field} contains duplicate states in {feature_path} "
                f"row {row_number}"
            )
        return states

    feature_rows = []
    feature_keys = set()
    for row_number, row in enumerate(features.iter_rows(named=True), start=2):
        marker_db = str(row["marker_db"] or "").strip().lower()
        profile_id = (
            str(row["profile_id"] or "").strip().lower().removesuffix("_seed")
        )
        feature_id = str(row["feature_id"] or "").strip()
        canonical_term_id = str(row["canonical_term_id"] or "").strip()
        raw_source_label = str(row["raw_source_label"] or "").strip()
        if (
            not marker_db
            or not profile_id
            or not feature_id
            or not canonical_term_id
            or not raw_source_label
        ):
            raise ValueError(
                f"Marker feature row {row_number} has an empty database, profile, "
                "feature identity, canonical term, or source label"
            )
        key = (marker_db, profile_id, feature_id)
        if key in feature_keys:
            raise ValueError(
                f"Duplicate marker feature identity in {feature_path}: {key}"
            )
        feature_keys.add(key)
        feature_rows.append(
            {
                "marker_db": marker_db,
                "profile_id": profile_id,
                "feature_id": feature_id,
                "canonical_term_id": canonical_term_id,
                "raw_source_label": raw_source_label,
                "hmmer_states": parse_states(
                    row.get(
                        "hmmer_states_1based", row.get("hmm_states_1based")
                    ),
                    "hmmer_states_1based",
                    row_number,
                ),
                "mmseqs_states": parse_states(
                    row.get("mmseqs_states_1based"),
                    "mmseqs_states_1based",
                    row_number,
                ),
            }
        )

    by_profile = {}
    for feature in feature_rows:
        by_profile.setdefault(
            (feature["marker_db"], feature["profile_id"]), []
        ).append(feature)
    annotated_databases = {key[0] for key in by_profile}

    profile_states = (
        "hmmer_states" if backend == "hmmsearch" else "mmseqs_states"
    )
    canonical_residues = set("ACDEFGHIKLMNPQRSTVWY")

    def walk_alignment(
        profile_alignment: str,
        query_alignment: str,
        profile_start: int,
        query_start: int,
        profile_end: int,
        query_end: int,
    ):
        if len(profile_alignment) != len(query_alignment):
            raise ValueError(
                "Profile/query alignment paths have different lengths"
            )
        profile_position = profile_start - 1
        query_position = query_start - 1
        state_map = {}
        insertions = []
        for profile_residue, query_residue in zip(
            profile_alignment, query_alignment
        ):
            profile_gap = profile_residue in {"-", "."}
            query_gap = query_residue == "-"
            if profile_gap and query_gap:
                raise ValueError("Alignment path contains a double gap")
            if not profile_gap:
                profile_position += 1
            if not query_gap:
                query_position += 1
            if profile_gap:
                if not query_gap:
                    insertions.append(
                        (
                            profile_position,
                            profile_position + 1,
                            query_position,
                            query_residue.upper(),
                        )
                    )
                continue
            if query_gap:
                state_map[profile_position] = ("deleted", None, None)
            elif query_residue.upper() in canonical_residues:
                state_map[profile_position] = (
                    "mapped",
                    query_position,
                    query_residue.upper(),
                )
            else:
                state_map[profile_position] = (
                    "unresolved",
                    query_position,
                    query_residue.upper(),
                )
        if profile_position != profile_end or query_position != query_end:
            raise ValueError(
                "Alignment path coordinates do not match its endpoint columns"
            )
        return state_map, insertions

    rows = []
    for hit in hit_df.iter_rows(named=True):
        database_id = str(hit.get("database_id", "")).strip()
        database_key = database_id.lower()
        profile_aliases = {
            str(hit.get(alias) or "").strip().lower().removesuffix("_seed")
            for alias in ("profile_accession", "hmm_full_name")
        }
        profile_aliases.discard("")
        profile_id = next(iter(profile_aliases), "")
        # The currently bundled sidecar is specifically for RVMT.  Other
        # marker DBs remain ordinary broad detection results.
        candidate_keys = {
            (database_key, alias)
            for alias in profile_aliases
            if (database_key, alias) in by_profile
        }
        if len(candidate_keys) > 1:
            raise ValueError(
                f"Hit profile aliases disagree with marker feature identities: "
                f"{sorted(candidate_keys)}"
            )
        candidates = (
            by_profile.get(next(iter(candidate_keys)), [])
            if candidate_keys
            else []
        )
        if not candidates and database_key not in annotated_databases:
            continue
        if not profile_aliases:
            raise ValueError(
                f"Marker hit for {database_id} has no profile identity"
            )

        common = {
            "marker_path_id": str(hit.get("marker_path_id") or ""),
            "database_id": database_id,
            "query_full_name": str(hit.get("query_full_name", "")),
            "hmm_full_name": str(hit.get("hmm_full_name", "")),
            "profile_accession": str(hit.get("profile_accession") or ""),
            "domain_index": str(hit.get("domain_index", "1")),
            "backend": backend,
            "full_hmm_evalue": str(
                hit.get("full_hmm_evalue")
                if hit.get("full_hmm_evalue") is not None
                else ""
            ),
            "full_hmm_score": str(
                hit.get("full_hmm_score")
                if hit.get("full_hmm_score") is not None
                else ""
            ),
            "this_dom_score": str(
                hit.get("this_dom_score")
                if hit.get("this_dom_score") is not None
                else ""
            ),
            "hmm_from": str(hit.get("hmm_from", "")),
            "hmm_to": str(hit.get("hmm_to", "")),
            "q1": str(hit.get("q1", "")),
            "q2": str(hit.get("q2", "")),
        }
        if not candidates:
            rows.append(
                {
                    **common,
                    "feature_id": "",
                    "canonical_term_id": "",
                    "raw_source_label": "",
                    "profile_states_1based": "",
                    "mapped_profile_states_1based": "",
                    "projected_query_positions_1based": "",
                    "projected_query_residues": "",
                    "unresolved_profile_states_1based": "",
                    "unresolved_query_positions_1based": "",
                    "unresolved_query_residues": "",
                    "deleted_profile_states_1based": "",
                    "out_of_span_profile_states_1based": "",
                    "feature_query_start_1based": "",
                    "feature_query_end_1based": "",
                    "insertion_query_positions_1based": "",
                    "insertion_query_residues": "",
                    "mapped_state_count": "0",
                    "unresolved_state_count": "0",
                    "deleted_state_count": "0",
                    "out_of_span_state_count": "0",
                    "projection_status": "unlabelled_profile",
                    "projection_reason": "no_feature_definition_for_profile",
                }
            )
            continue

        try:
            profile_start = int(hit["hmm_from"])
            profile_end = int(hit["hmm_to"])
            query_start = int(hit["q1"])
            query_end = int(hit["q2"])
            profile_length = int(hit["hmm_len"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Marker hit lacks valid alignment coordinates for "
                f"{sorted(profile_aliases)}"
            ) from exc
        if not (1 <= profile_start <= profile_end <= profile_length):
            raise ValueError(
                f"Invalid profile alignment span {profile_start}-{profile_end} "
                f"for {profile_id} (length {profile_length})"
            )
        if not 1 <= query_start <= query_end:
            raise ValueError(
                f"Invalid query alignment span {query_start}-{query_end} "
                f"for {profile_id}"
            )
        profile_alignment = str(hit.get("profile_alignment", ""))
        query_alignment = str(hit.get("query_alignment", ""))
        if not profile_alignment or not query_alignment:
            raise ValueError(
                "Feature projection requires retained profile/query alignment paths"
            )

        for feature in candidates:
            states = feature[profile_states]
            if any(state > profile_length for state in states):
                raise ValueError(
                    f"Feature {feature['feature_id']} state exceeds profile length "
                    f"for {profile_id}"
                )
        state_map, insertions = walk_alignment(
            profile_alignment,
            query_alignment,
            profile_start,
            query_start,
            profile_end,
            query_end,
        )

        for feature in candidates:
            states = feature[profile_states]
            state_status = {
                state: state_map.get(state, ("out_of_span", None, None))
                for state in states
            }
            mapped_values = [
                state_status[state][1:]
                for state in states
                if state_status[state][0] == "mapped"
            ]
            unresolved_values = [
                state_status[state][1:]
                for state in states
                if state_status[state][0] == "unresolved"
            ]
            mapped_states = [
                state
                for state in states
                if state_status[state][0] == "mapped"
            ]
            unresolved_states = [
                state
                for state in states
                if state_status[state][0] == "unresolved"
            ]
            deleted_states = [
                state
                for state in states
                if state_status[state][0] == "deleted"
            ]
            out_of_span_states = [
                state
                for state in states
                if state_status[state][0] == "out_of_span"
            ]
            deleted_count = len(deleted_states)
            out_of_span_count = len(out_of_span_states)
            if not states:
                status = "absent"
                reason = "no_backend_state_map"
            elif len(mapped_values) == len(states):
                status = "complete"
                reason = "all_feature_states_projected"
            elif mapped_values:
                status = "partial"
                reason = "some_feature_states_not_projected"
            else:
                status = "absent"
                reason_parts = []
                if unresolved_states:
                    reason_parts.append("unresolved")
                if deleted_states:
                    reason_parts.append("deleted")
                if out_of_span_states:
                    reason_parts.append("out_of_span")
                reason = "feature_states_" + "_and_".join(reason_parts)
            state_set = set(states)
            feature_insertions = [
                (query_position, residue)
                for left, right, query_position, residue in insertions
                if left in state_set and right in state_set
            ]
            feature_query_positions = [
                value[0] for value in mapped_values + unresolved_values
            ] + [value[0] for value in feature_insertions]
            rows.append(
                {
                    **common,
                    "feature_id": feature["feature_id"],
                    "canonical_term_id": feature["canonical_term_id"],
                    "raw_source_label": feature["raw_source_label"],
                    "profile_states_1based": ";".join(map(str, states)),
                    "mapped_profile_states_1based": ";".join(
                        map(str, mapped_states)
                    ),
                    "projected_query_positions_1based": ";".join(
                        map(str, (value[0] for value in mapped_values))
                    ),
                    "projected_query_residues": "".join(
                        value[1] for value in mapped_values
                    ),
                    "unresolved_profile_states_1based": ";".join(
                        map(str, unresolved_states)
                    ),
                    "unresolved_query_positions_1based": ";".join(
                        map(str, (value[0] for value in unresolved_values))
                    ),
                    "unresolved_query_residues": "".join(
                        value[1] for value in unresolved_values
                    ),
                    "deleted_profile_states_1based": ";".join(
                        map(str, deleted_states)
                    ),
                    "out_of_span_profile_states_1based": ";".join(
                        map(str, out_of_span_states)
                    ),
                    "feature_query_start_1based": (
                        str(min(feature_query_positions))
                        if feature_query_positions
                        else ""
                    ),
                    "feature_query_end_1based": (
                        str(max(feature_query_positions))
                        if feature_query_positions
                        else ""
                    ),
                    "insertion_query_positions_1based": ";".join(
                        map(str, (value[0] for value in feature_insertions))
                    ),
                    "insertion_query_residues": "".join(
                        value[1] for value in feature_insertions
                    ),
                    "mapped_state_count": str(len(mapped_values)),
                    "unresolved_state_count": str(len(unresolved_values)),
                    "deleted_state_count": str(deleted_count),
                    "out_of_span_state_count": str(out_of_span_count),
                    "projection_status": status,
                    "projection_reason": reason,
                }
            )

    projection_df = pl.DataFrame(
        rows,
        schema={column: pl.String for column in FEATURE_PROJECTION_COLUMNS},
    )
    projection_df.write_csv(output_path, separator="\t")
    if logger:
        logger.info(
            "Marker feature projection table written to %s (%d rows)",
            output_path,
            projection_df.height,
        )


def write_matched_regions_fasta(
    hit_df, output_path: Path, include_aligned_region: bool
) -> None:
    """Write matched query regions from search hits to FASTA."""
    if hit_df.is_empty():
        output_path.touch()
        return

    required_cols = {"query_full_name", "hmm_full_name", "q1", "q2"}
    missing = required_cols.difference(set(hit_df.columns))
    if missing:
        raise ValueError(
            f"Cannot write matched regions, missing required columns: {missing}"
        )

    use_full_qseq = "full_qseq" in hit_df.columns
    use_aligned_region = (
        include_aligned_region and "aligned_region" in hit_df.columns
    )

    with open(output_path, "w") as fout:
        for row in hit_df.iter_rows(named=True):
            query_id = str(row["query_full_name"]).strip()
            hmm_name = str(row["hmm_full_name"]).strip().replace(" ", "_")
            q1 = int(row["q1"])
            q2 = int(row["q2"])

            # Many FASTA parsers only keep text before first whitespace as the ID.
            # Extract frame metadata from the full query label and encode it in the ID token.
            frame_match = re.search(
                r"(?:^|[\s;|,_])(?:frame|rf)[:=]([+-]?[1-3])",
                query_id,
                re.IGNORECASE,
            )
            frame_suffix = (
                f"|frame={frame_match.group(1)}" if frame_match else ""
            )
            query_token = query_id.split()[0].replace(" ", "_")
            token_parts = query_token.split("|")
            orf_suffix = (
                f"|orf={token_parts[-1]}"
                if not frame_suffix
                and len(token_parts) >= 3
                and token_parts[-1].isdigit()
                else ""
            )

            if use_full_qseq:
                full_qseq = str(row.get("full_qseq", ""))
                region_seq = full_qseq[max(0, q1 - 1) : max(0, q2)]
            elif use_aligned_region:
                region_seq = str(row.get("aligned_region", ""))
            else:
                # If only the summary table is kept, at least preserve a traceable record.
                region_seq = ""

            # Matched-region FASTA output should be raw query segments, not gapped alignments.
            region_seq = region_seq.replace("-", "").replace(".", "")

            if not region_seq:
                continue

            region_id = f"{query_token}{frame_suffix}{orf_suffix}|hit={hmm_name}|coords={q1}-{q2}"
            fout.write(f">{region_id}\n{region_seq}\n")


def write_matched_input_seqs_fasta(
    hit_df, input_file: str, output_path: Path, input_alpha: str, aa_method: str
) -> None:
    """Write full original input sequences that had marker hits to FASTA.

    For protein input the query IDs match the input FASTA headers directly.
    For nucleotide input the query IDs come from the translated ORF/frame file
    and are mapped back to the original contig IDs.  The exact suffix that must
    be stripped depends on the translation tool:

    - six_frame (seqkit --append-frame): appends a frame suffix (e.g. ``_frame:+1``)
      to each sequence ID; strip it to recover the contig name.
    - pyrodigal / bbmap: append a numeric ORF ordinal (e.g. ``_1``, ``_2``);
      strip the trailing ``_<digits>`` to recover the contig name.

    Note:
        The exact suffix formats should be confirmed experimentally; see inline
        TODO comments.
    """
    import re as _re

    from rolypoly.utils.bio.sequences import filter_fasta_by_headers

    if hit_df.is_empty():
        output_path.touch()
        return

    if "query_full_name" not in hit_df.columns:
        output_path.touch()
        return

    raw_ids = (
        hit_df["query_full_name"]
        .str.extract(r"^([^|\s]+)")
        .drop_nulls()
        .unique()
        .to_list()
    )
    if not raw_ids:
        output_path.touch()
        return

    if input_alpha == "aa":
        matched_ids = raw_ids
    elif aa_method in ("pyrodigal", "bbmap"):
        # pyrodigal: <contig>_<orf_ordinal>
        # TODO: confirm bbmap callgenes.sh follows the same <contig>_<N> convention.
        matched_ids = list({_re.sub(r"_\d+$", "", sid) for sid in raw_ids})
    else:  # six_frame (seqkit)
        # seqkit --append-frame: <contig>_frame=<N>
        matched_ids = list(
            {_re.sub(r"_frame=[+-]?\d+$", "", sid) for sid in raw_ids}
        )

    filter_fasta_by_headers(
        fasta_file=input_file, headers=matched_ids, output_file=str(output_path)
    )


global tools
tools = []

console = Console(width=150)


@command()
@option(
    "-i",
    "--input",
    required=True,
    help="Input fasta file. Preferably nucleotide contigs, but you can provide amino acid input too (the script would skip 6 frame translation)",
)
@option(
    "-o",
    "--output",
    default=lambda: f"{os.getcwd()}/marker_search_out",
    help="Path to output directory. Note - if multiple DBs are used and the resolve-mode is `none`, multiple outputs are made (DB name appended as suffix).",
)
@option(
    "-rm",
    "--resolve-mode",
    default="simple",
    type=Choice(
        [
            "merge",
            "one_per_range",
            "one_per_query",
            "split",
            "drop_contained",
            "none",
            "simple",
        ]
    ),
    help="""How to deal with regions in your query that match multiple profiles? \n
        - merge: all overlapping hits are merged into one range \n
        - one_per_range: one hit per range (ali_from-ali_to) is reported \n
        - one_per_query: one hit per query sequence is reported \n
        - split: each overlapping domain is split into a new row \n
        - drop_contained: hits that are contained within (i.e. enveloped by) other hits are dropped. \n
        - none: no resolution of overlapping hits is performed. NOTE - EXPECT A POTENTIALLY LARGE OUTPUT \n
        - simple: heuristic/personal observation based - chains drop_contained output with split mode. \n
        """,
)
@option(
    "-mo",
    "--min-overlap-positions",
    default=10,
    help="Minimal number of overlapping positions between two intersecting ranges before they are considered as overlapping (used in some resolve_mode(s)",
)
@option(
    "--repeat-filter/--no-repeat-filter",
    default=True,
    show_default=True,
    help="Filter hits where the same profile region repeatedly matches distinct parts of one query.",
)
@option(
    "-ie",
    "--inc-evalue",
    default=0.001,
    help="Maximal e-value for including a domain match in the results",
)  #  for HMM reporting
@option(
    "-s",
    "--score",
    default=20,
    help="Minimal score for including a domain match in the results",
)
@option(
    "-mla",
    "--min-ali-len",
    default=15,
    help="Minimal alignment length for including a domain match in the results",
)
@option(
    "-am",
    "--aa-method",
    default="six_frame",
    type=Choice(["six_frame", "pyrodigal", "bbmap"]),
    help="Method to translate nucleotide sequences into amino acids. Options: six frame translation using seqkit, pyrodigal-rv uses pyrodigal-meta with additional genetic codes, bbmap callgenes.sh (quick but less accurate for metagenomic data)",
)
@option(
    "-db",
    "--database",
    type=str,
    default="RVMT,genomad",
    help="""comma separated list of databases to search against (or `all`), or path to a custom database. \n
        options: NeoRdRp_v2.1, RdRp-scan, RVMT, Pfam_RTs_RdRp, genomad, all. Availability depends on the selected backend. \n
        With hmmsearch, a custom path may be an HMM, an MSA, or a directory of either. With mmseqs2, provide an MMseqs database prefix.""",
)
@option(
    "-st",
    "--search-tool",
    type=Choice(["hmmsearch", "mmseqs2"]),
    default="hmmsearch",
    show_default=True,
    help="Profile-search backend. MMseqs2 uses the corresponding prebuilt MMseqs profile databases.",
)
@option(
    "-cf",
    "--config-file",
    hidden=True,
    default=None,
    help="path to a json config file with parameters for the search - overrides command line parameters",
)
@option(
    "-n",
    "--name",
    hidden=True,
    default=None,
    help="basename for the output files (default is the basename of the input file)",
)
@option(
    "-ow",
    "--overwrite",
    is_flag=True,
    default=False,
    help="Do not overwrite the output directory if it already exists",
)
@option(
    "--write-matched-regions/--no-write-matched-regions",
    default=True,
    help="Write matched query regions to FASTA (enabled by default; disable with --no-write-matched-regions)",
)
@option(
    "-mro",
    "--matched-regions-output",
    default=None,
    help="Output FASTA path for matched regions (default: <output>/marker_search_matched_regions.faa)",
)
@option(
    "--include-aligned-region/--no-include-aligned-region",
    default=True,
    help="Include aligned query region sequence in marker_search_results.tsv (enabled by default)",
)
@option(
    "--include-alignment-string/--no-include-alignment-string",
    default=False,
    help="Include alignment identity string in marker_search_results.tsv (disabled by default)",
)
@option(
    "--write-matched-input-seqs/--no-write-matched-input-seqs",
    default=False,
    help="Write full original input sequences (contigs for nucleotide input, whole proteins for AA input) that had at least one marker hit to FASTA (disabled by default)",
)
@option(
    "-miso",
    "--matched-input-seqs-output",
    default=None,
    help="Output FASTA path for matched input sequences (default: <output>/marker_search_matched_input_seqs.fna|faa)",
)
def marker_search(
    input,
    output,
    resolve_mode,
    min_overlap_positions,
    repeat_filter,
    inc_evalue,
    score,
    aa_method,
    database,
    search_tool,
    threads,
    log_file,
    memory,
    config_file,
    name,
    keep_tmp,
    overwrite,
    log_level,
    temp_dir,
    write_matched_regions,
    matched_regions_output,
    min_ali_len,
    include_aligned_region,
    include_alignment_string,
    write_matched_input_seqs,
    matched_input_seqs_output,
):
    """RNA virus marker protein search using HMMER or MMseqs2 profile databases.
    Most pre-made DBs are based on RdRp domain (except for geNomad).
    Input can be nucleotide contigs or amino acid seqs.
    If nucleotide, by default all contigs will be translated to six end-to-end frames (with stops replaced by `X`), or into ORFs called by pyrodigal (meta) or callgenes.sh \n
    Pre-compiled options are: \n
    • NeoRdRp2.1 \n
        GitHub: https://github.com/shoichisakaguchi/NeoRdRp  | Paper: https://doi.org/10.1264/jsme2.ME22001 \n
    • RVMT \n
        GitHub: https://github.com/UriNeri/RVMT  | Zenodo: https://zenodo.org/record/7368133  |  Paper: https://doi.org/10.1016/j.cell.2022.08.023 \n
    • RdRp-Scan \n
        GitHub: https://github.com/JustineCharon/RdRp-scan  |  Paper: https://doi.org/10.1093/ve/veac082 \n
            ⤷ (which IIRC incorporated PALMdb, GitHub: https://github.com/rcedgar/palmdb, Paper: https://doi.org/10.7717/peerj.14055 \n
    • Pfam_RTs_RdRp \n
        RdRp and RT profiles from Pfam 38.2 --- PF04197.18,PF04196.18,PF22212.2,PF22152.2,PF22260.2,PF00680.26,PF00978.27,PF00998.29,PF02123.22,PF07925.16,PF00078.33,PF07727.20,PF13456.13
        Data: https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam38.2/ | Paper https://doi.org/10.1093/nar/gkaa913
    • geNomad \n
        RNA virus marker genes from geNomad v1.9 --- https://zenodo.org/records/14886553
    For a custom path, use an HMM/MSA source with hmmsearch or an MMseqs database prefix with mmseqs2.
    Please cite accordingly based on the DBs you select.
    """
    import json

    import polars as pl

    from rolypoly.utils.bio.alignments import (
        hmm_from_msa,
        hmmdb_from_directory,
        search_hmmdb,
    )
    from rolypoly.utils.bio.interval_ops import consolidate_hits
    from rolypoly.utils.bio.sequences import guess_fasta_alpha
    from rolypoly.utils.bio.translation import (
        pyro_predict_orfs,
        translate_6frx_seqkit,
        translate_with_bbmap,
    )
    from rolypoly.utils.logging.citation_reminder import remind_citations
    from rolypoly.utils.logging.loggit import log_start_info

    # Determine if output should be treated as directory based on resolve_mode and path
    output = str(Path(output).absolute())
    is_directory_output = resolve_mode == "none" or output.endswith("/")

    if is_directory_output:
        # Ensure output ends with '/' to signal directory to config
        if not output.endswith("/"):
            output = output + "/"
    else:
        # Ensure parent directory exists for file output
        Path(output).parent.mkdir(parents=True, exist_ok=True)

    if config_file:
        config = RVirusSearchConfig(**json.load(open(config_file)))
    else:
        if not name:
            name = Path(input).stem
        config = RVirusSearchConfig(
            input=input,
            output=output,
            inc_evalue=inc_evalue,
            score=score,
            aa_method=aa_method,
            temp_dir=temp_dir,
            database=database,
            search_tool=search_tool,
            overwrite=overwrite,
            log_level=log_level,
            threads=threads,
            log_file=log_file,
            name=name,
            keep_tmp=keep_tmp,
            resolve_mode=resolve_mode,
            min_overlap_positions=min_overlap_positions,
            repeat_filter=repeat_filter,
            memory=memory,
            write_matched_regions=write_matched_regions,
            matched_regions_output=matched_regions_output,
            include_aligned_region=include_aligned_region,
            include_alignment_string=include_alignment_string,
            write_matched_input_seqs=write_matched_input_seqs,
            matched_input_seqs_output=matched_input_seqs_output,
            min_ali_len=min_ali_len,
        )

    # Logging
    log_start_info(config.logger, config.to_dict())

    config.logger.info(
        f"Starting RNA virus marker protein search with: {config.input}"
    )

    # Determine the databases to use
    hmmdbdir = Path(os.environ["ROLYPOLY_DATA"]) / "profiles/hmmdbs"
    mmseqsdbdir = Path(os.environ["ROLYPOLY_DATA"]) / "profiles/mmseqs_dbs"

    database_paths_by_tool = {
        "hmmsearch": {
            "NeoRdRp_v2.1".lower(): hmmdbdir / "neordrp2.1.hmm",
            "RdRp-scan".lower(): hmmdbdir / "rdrp_scan.hmm",
            "RVMT".lower(): hmmdbdir / "rvmt.hmm",
            "Pfam_RTs_RdRp".lower(): hmmdbdir / "pfam_rdrps_and_rts.hmm",
            "genomad".lower(): hmmdbdir / "genomad_rna_viral_markers.hmm",
        },
        "mmseqs2": {
            "RdRp-scan".lower(): mmseqsdbdir / "rdrp_scan/rdrp_scan",
            "RVMT".lower(): mmseqsdbdir / "RVMT/RVMT",
            "Pfam_RTs_RdRp".lower(): mmseqsdbdir
            / "pfam_rdrps_and_rts/pfam_rdrps_and_rts",
            "genomad".lower(): mmseqsdbdir / "genomad/rna_viral_markers",
        },
    }
    db_paths = database_paths_by_tool[config.search_tool]
    marker_features_path = (
        Path(os.environ["ROLYPOLY_DATA"])
        / "profiles"
        / "marker_features.tsv.gz"
    )

    requested_database = config.database
    if requested_database == "all":
        database_paths = db_paths
    elif requested_database.startswith("/") or requested_database.startswith(
        "./"
    ):
        custom_database = str(Path(requested_database).resolve())
        if not Path(custom_database).exists():
            config.logger.error(
                f"Custom database path {custom_database} does not exist"
            )
            return
        elif config.search_tool == "mmseqs2":
            database_paths = {"Custom": custom_database}
        else:
            # check if a file it's an hmm or an msa file
            if custom_database.endswith(".hmm"):
                database_paths = {"Custom": custom_database}
            elif custom_database.endswith((".faa", ".fasta", ".afa")):
                from rolypoly.utils.bio.alignments import hmm_from_msa

                database_paths = {
                    "Custom": hmm_from_msa(
                        msa_file=requested_database,
                        output=requested_database.replace(".faa", ".hmm"),
                        name=Path(requested_database).stem,
                    )
                }
            # if it's a directory:
            elif Path(custom_database).is_dir():
                from rolypoly.utils.bio.library_detection import (
                    validate_database_directory,
                )

                db_info = validate_database_directory(
                    custom_database, logger=config.logger
                )
                config.logger.info(
                    f"Database directory analysis: {db_info['message']}"
                )

                if db_info["type"] == "hmm_directory":
                    # concatenate all hmms into one file
                    with open(
                        Path(custom_database) / "concatenated.hmm", "w"
                    ) as f:
                        for hmm_file in db_info["files"]:
                            with open(hmm_file, "r") as hmm_file_obj:
                                f.write(hmm_file_obj.read())
                    database_paths = {
                        "Custom": str(
                            Path(custom_database) / "concatenated.hmm"
                        )
                    }
                elif db_info["type"] == "msa_directory":
                    from rolypoly.utils.bio.alignments import (
                        hmmdb_from_directory,
                    )

                    hmmdb_from_directory(
                        msa_dir=custom_database,
                        output=Path(custom_database) / "all_msa_built.hmm",
                        # alphabet="aa",
                    )
                    database_paths = {
                        "Custom": str(
                            Path(custom_database) / "all_msa_built.hmm"
                        )
                    }
                else:
                    config.logger.error(
                        f"Unsupported database directory type: {db_info['type']}"
                    )
                    return
            else:
                config.logger.error(
                    f"Invalid custom database path: {custom_database}"
                )
                return
    else:
        databases = [
            db.strip() for db in requested_database.split(",") if db.strip()
        ]
        unsupported = [db for db in databases if db.lower() not in db_paths]
        if unsupported:
            raise ValueError(
                f"Databases {unsupported} are not available for {config.search_tool}. "
                f"Supported databases: {', '.join(db_paths)}"
            )
        database_paths = {db: db_paths[db.lower()] for db in databases}

    feature_projection_enabled = marker_features_path.exists() and any(
        db_name.lower() == "rvmt" for db_name in database_paths
    )

    input_alpha = guess_fasta_alpha(input)
    if input_alpha == "amino":
        input_alpha = "aa"

    if input_alpha == "nucl":
        config.logger.info("Input identified as nucl")
        amino_file = str(config.temp_dir / f"{config.name}")
        if config.aa_method == "pyrodigal":
            config.logger.info("Predicting ORFs using pyrodigal-rv")
            amino_file = amino_file + "_pyro.faa"
            pyro_predict_orfs(input, amino_file, threads)
            tools.append("pyrodigal")
        elif config.aa_method == "bbmap":
            config.logger.info("Using BBMap's callgenes.sh for translation")
            amino_file = amino_file + "_cg.faa"
            translate_with_bbmap(input, amino_file, threads)
            tools.append("bbmap")
        else:
            config.logger.info("Using seqkit for 6 frames translation")
            amino_file = amino_file + "_6frx.faa"
            translate_6frx_seqkit(input, amino_file, threads)
            tools.append("seqkit")
    elif input_alpha == "aa":
        config.logger.info(
            "Using supplied amino acid fasta file, skipping translation"
        )
        amino_file = input
    else:
        config.logger.error(
            "Input is not in fasta format or seqs not recognized as nucleotide or amino acid"
        )
        return

    all_outputs = []
    config.logger.info(f"Searching with {amino_file}")
    for db_name, db_path in database_paths.items():
        # Search translated sequences against viral marker databases
        config.logger.info(f"Searching {db_name}")
        tools.append(f"{db_name}") if db_name != "Custom" else None

        tmp_output = config.temp_dir / f"raw_{config.name}_vs_{db_name}.tsv"
        retain_feature_path = (
            feature_projection_enabled and db_name.lower() == "rvmt"
        )
        if config.search_tool == "mmseqs2":
            search_mmseqs_marker_db(
                amino_file=amino_file,
                db_path=db_path,
                output=tmp_output,
                temp_dir=config.temp_dir,
                threads=threads,
                logger=config.logger,
                inc_evalue=config.inc_evalue,
                score=config.score,
                min_ali_len=config.min_ali_len,
                include_aligned_region=config.include_aligned_region,
                include_alignment_string=config.include_alignment_string,
                include_full_query=config.write_matched_regions,
                include_alignment_path=retain_feature_path,
            )
        else:
            search_hmmdb(
                amino_file=amino_file,
                db_path=db_path,
                output=tmp_output,
                threads=threads,
                logger=config.logger,
                inc_e=config.inc_evalue,
                mscore=config.score,
                min_ali_len=config.min_ali_len,
                output_format="modomtblout",
                ali_str=config.include_alignment_string,
                full_qseq=config.write_matched_regions,
                match_region=config.include_aligned_region,
                include_alignment_path=retain_feature_path,
            )
        config.logger.debug(f"temp output: {tmp_output}")
        all_outputs.append((db_name, tmp_output))

    # read all output files, stack them, and resolve overlaps
    config.logger.debug(f"Reading {len(all_outputs)} output files")
    stack_df = pl.concat(
        [
            pl.scan_csv(
                output_path, separator="\t", infer_schema_length=123123
            ).with_columns(pl.lit(db_name).alias("database_id"))
            for db_name, output_path in all_outputs
        ],
        how="diagonal_relaxed",
    ).collect()

    # Profile-class metadata is a build artifact rather than a collection of
    # command-specific name checks. RT hits remain reportable evidence, but do
    # not nominate a contig as a viral candidate.
    profile_manifest = (
        Path(os.environ["ROLYPOLY_DATA"])
        / "profiles"
        / "pfam_rdrps_and_rts_profiles.tsv.gz"
    )
    if profile_manifest.exists():
        profile_metadata = pl.read_csv(profile_manifest, separator="\t")
        required_manifest_columns = {
            "database_id",
            "profile_name",
            "profile_accession",
            "profile_accession_base",
            "mmseqs_target",
            "profile_class",
        }
        missing_manifest_columns = required_manifest_columns.difference(
            profile_metadata.columns
        )
        if missing_manifest_columns:
            raise ValueError(
                f"Profile metadata {profile_manifest} is missing columns: "
                f"{sorted(missing_manifest_columns)}"
            )
        profile_classes = {}
        for row in profile_metadata.iter_rows(named=True):
            database_key = row["database_id"].lower()
            for alias_column in (
                "profile_name",
                "profile_accession",
                "profile_accession_base",
                "mmseqs_target",
            ):
                alias = str(row[alias_column]).lower().removesuffix("_seed")
                profile_classes[(database_key, alias)] = row["profile_class"]
        stack_df = stack_df.with_columns(
            pl.struct("database_id", "profile_accession", "hmm_full_name")
            .map_elements(
                lambda row: profile_classes.get(
                    (
                        row["database_id"].lower(),
                        str(row["profile_accession"] or row["hmm_full_name"])
                        .lower()
                        .removesuffix("_seed"),
                    ),
                    "unclassified",
                ),
                return_dtype=pl.String,
            )
            .alias("profile_class")
        )
        unmapped_pfam = stack_df.filter(
            (pl.col("database_id").str.to_lowercase() == "pfam_rts_rdrp")
            & (pl.col("profile_class") == "unclassified")
        )
        if not unmapped_pfam.is_empty():
            config.logger.warning(
                "%d Pfam RdRp/RT hits could not be classified from %s; "
                "they remain candidate evidence",
                unmapped_pfam.height,
                profile_manifest,
            )
    else:
        stack_df = stack_df.with_columns(
            pl.lit("unclassified").alias("profile_class")
        )
        if any(
            db_name.lower() == "pfam_rts_rdrp" for db_name, _ in all_outputs
        ):
            config.logger.warning(
                "Pfam profile metadata is absent at %s; RT hits cannot be separated "
                "from candidate evidence with this database bundle",
                profile_manifest,
            )
    stack_df = stack_df.with_columns(
        pl.when(pl.col("profile_class") == "rt")
        .then(pl.lit("rt_evidence"))
        .otherwise(pl.lit("candidate"))
        .alias("marker_role")
    )
    config.logger.debug(stack_df)
    if stack_df.is_empty():
        config.logger.info("No hits found in any DB")
        config.logger.info("skipping resolution of overlaps")
        config.resolve_mode = "none"

    if config.repeat_filter and not stack_df.is_empty():
        from rolypoly.utils.bio.interval_ops import (
            filter_repeated_profile_regions,
        )

        raw_hit_count = stack_df.height
        stack_df, repeat_filtered_df = filter_repeated_profile_regions(stack_df)
        if repeat_filtered_df.is_empty():
            config.logger.info(
                "Repeat filter found no repeated profile-region hits"
            )
        else:
            repeat_filtered_file = (
                Path(output) / "marker_search_repeat_filtered.tsv"
            )
            repeat_filtered_df.write_csv(repeat_filtered_file, separator="\t")
            config.logger.info(
                "Repeat filter removed %d/%d hits; audit table written to %s",
                repeat_filtered_df.height,
                raw_hit_count,
                repeat_filtered_file,
            )
            if stack_df.is_empty():
                config.logger.info(
                    "No hits remain after repeat filtering; skipping overlap resolution"
                )
                config.resolve_mode = "none"
    elif not config.repeat_filter:
        config.logger.info("Repeat filter disabled by --no-repeat-filter")

    if feature_projection_enabled:
        stack_df = stack_df.with_row_index("marker_path_id", offset=1)
        # Project while every row still represents one authentic backend
        # alignment path. Overlap modes such as ``merge`` may synthesize a
        # wider interval without synthesizing a corresponding traceback.
        project_marker_features(
            hit_df=stack_df,
            feature_path=marker_features_path,
            output_path=Path(output) / "marker_evidence.tsv",
            backend=config.search_tool,
            logger=config.logger,
        )

    # Tracebacks are internal projection inputs, not part of the established
    # marker-search result-table contract.
    helper_columns = [
        column
        for column in (
            "domain_index",
            "profile_alignment",
            "query_alignment",
        )
        if column in stack_df.columns
    ]
    if helper_columns:
        stack_df = stack_df.drop(helper_columns)

    rt_evidence_df = stack_df.filter(pl.col("marker_role") == "rt_evidence")
    stack_df = stack_df.filter(pl.col("marker_role") == "candidate")
    if not rt_evidence_df.is_empty():
        config.logger.info(
            "Retaining %d RT hits as evidence without nominating their contigs",
            rt_evidence_df.height,
        )
    if stack_df.is_empty():
        config.logger.info(
            "No candidate marker hits remain; skipping overlap resolution"
        )
        config.resolve_mode = "none"

    results_file = Path(output) / "marker_search_results.tsv"

    if config.resolve_mode == "simple":
        config.logger.info(
            "Using adaptive 'simple' mode for overlap resolution with polyprotein detection"
        )

        # Run in two passes because drop_contained returns early and cannot be
        # combined with one_per_range in a single consolidate_hits call.
        stage_df = consolidate_hits(
            input=stack_df,
            one_per_query=False,
            one_per_range=False,
            min_overlap_positions=config.min_overlap_positions,  # Will be overridden by adaptive logic
            merge=False,
            split=False,
            column_specs="query_full_name,hmm_full_name",
            rank_columns="-full_hmm_score,+full_hmm_evalue,-hmm_cov",
            drop_contained=True,
            alphabet="aa",
            adaptive_overlap=True,
        )

        testdf = consolidate_hits(
            input=stage_df,
            one_per_query=False,
            one_per_range=True,
            min_overlap_positions=config.min_overlap_positions,  # Will be overridden by adaptive logic
            merge=False,
            split=False,
            column_specs="query_full_name,hmm_full_name",
            rank_columns="-full_hmm_score,+full_hmm_evalue,-hmm_cov",
            drop_contained=False,
            alphabet="aa",
            adaptive_overlap=True,
        )

    elif config.resolve_mode != "none":
        resolve_mode_dict = {
            "split": False,
            "one_per_range": False,
            "one_per_query": False,
            "merge": False,
            "drop_contained": False,
        }
        resolve_mode_dict[config.resolve_mode] = True
        testdf = consolidate_hits(
            input=stack_df,
            min_overlap_positions=config.min_overlap_positions,
            column_specs="query_full_name,hmm_full_name",
            rank_columns="-full_hmm_score,+full_hmm_evalue",
            **resolve_mode_dict,
        )
    else:
        testdf = stack_df

    # RT evidence is kept in the result table, but is deliberately excluded
    # from candidate overlap resolution and downstream contig nomination.
    if not rt_evidence_df.is_empty():
        testdf = pl.concat([testdf, rt_evidence_df], how="diagonal_relaxed")

    # Write optional matched regions from the retained marker hits.
    if config.write_matched_regions:
        matched_region_output = (
            Path(config.matched_regions_output)
            if config.matched_regions_output
            else Path(output) / "marker_search_matched_regions.faa"
        )
        matched_region_output.parent.mkdir(parents=True, exist_ok=True)
        write_matched_regions_fasta(
            hit_df=testdf,
            output_path=matched_region_output,
            include_aligned_region=config.include_aligned_region,
        )
        config.logger.info(
            f"Matched regions written to {matched_region_output.absolute()}"
        )

    if config.write_matched_input_seqs:
        ext = "faa" if input_alpha == "aa" else "fna"
        input_seqs_output = (
            Path(config.matched_input_seqs_output)
            if config.matched_input_seqs_output
            else Path(output) / f"marker_search_matched_input_seqs.{ext}"
        )
        input_seqs_output.parent.mkdir(parents=True, exist_ok=True)
        write_matched_input_seqs_fasta(
            hit_df=testdf,
            input_file=input,
            output_path=input_seqs_output,
            input_alpha=input_alpha,
            aa_method=config.aa_method,
        )
        config.logger.info(
            f"Matched input sequences written to {input_seqs_output.absolute()}"
        )

    if "full_qseq" in testdf.columns and not config.write_matched_regions:
        testdf = testdf.drop("full_qseq")

    if "full_qseq" in testdf.columns and config.write_matched_regions:
        # Keep result tables compact while preserving full sequences in region FASTA output.
        testdf = testdf.drop("full_qseq")

    # Add explicit trace columns for downstream joins/provenance.
    # Naming conventions:
    #   six_frame (seqkit --append-frame): <contig>_frame=<N>
    #   pyrodigal:                         <contig>_<orf_ordinal>  (e.g. _1, _2)
    #   bbmap callgenes.sh:                assumed same as pyrodigal (TODO: confirm)
    #   aa input:                          query ID = original protein FASTA header
    # Note: query_full_name = "<hit_name> <hit_description>" (joined them),
    # so the ID token is only the part before the first space.
    if "query_full_name" in testdf.columns:
        if input_alpha == "nucl":
            if aa_method in ("pyrodigal", "bbmap"):
                testdf = testdf.with_columns(
                    pl.col("query_full_name")
                    .str.extract(r"^([^\s]+)", group_index=1)
                    .str.replace(r"_\d+$", "")
                    .alias("source_seq_id"),
                    pl.col("query_full_name")
                    .str.extract(r"^([^\s]+)", group_index=1)
                    .str.extract(r"_(\d+)$", group_index=1)
                    .alias("orf_id"),
                )
            else:  # six_frame (seqkit)
                testdf = testdf.with_columns(
                    pl.col("query_full_name")
                    .str.extract(r"^([^\s]+)", group_index=1)
                    .str.replace(r"_frame=[+-]?\d+$", "")
                    .alias("source_seq_id"),
                    pl.col("query_full_name")
                    .str.extract(r"_frame=([+-]?\d+)", group_index=1)
                    .alias("frame_id"),
                )
        else:  # aa input
            testdf = testdf.with_columns(
                pl.col("query_full_name")
                .str.extract(r"^([^\s]+)", group_index=1)
                .alias("source_seq_id")
            )

    # Write to a file in the output directory instead of the directory itself
    testdf.write_csv(results_file, separator="\t")

    # Remove temporary directory if keep_tmp is False
    if not config.keep_tmp:
        import shutil

        shutil.rmtree(config.temp_dir)
        config.logger.info(f"Removed temporary directory: {config.temp_dir}")

    config.logger.info(
        f"""Finished RNA virus marker protein search using : {input}"""
    )
    output_files = [ix.absolute() for ix in Path(output).glob("*.tsv")]
    config.logger.info(f"""Outputs saved to {output_files}""")

    if config.search_tool == "mmseqs2":
        tools.append("mmseqs2")
    else:
        tools.append("pyhmmer")
        tools.append("hmmer")

    with open(f"{config.log_file}", "a") as f_out:
        f_out.write(remind_citations(tools, return_bibtex=True) or "")


if __name__ == "__main__":
    marker_search()
