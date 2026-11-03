"""Export motif detector footprints and projected feature spans as GFF3."""

from pathlib import Path

from rolypoly.utils.bio.polars_fastx import convert_record_to_gff3_record


def write_motif_gff(contigs, output):
    """Write detector alignments and, where available, projected motif children."""
    output = Path(output)
    with output.open("w") as handle:
        handle.write("##gff-version 3\n")
        index = 0
        for contig in contigs:
            for motif in (contig.get("motifs") or {}).get("motifs", []):
                index += 1
                identifier = f"motif_match_{index}"
                strand = motif.get("strand")
                if strand is None and motif.get("frame") is not None:
                    strand = 1 if int(motif["frame"]) > 0 else -1
                strand = "+" if str(strand) in {"1", "+"} else "-" if str(strand) in {"-1", "-"} else "."
                common = {
                    "sequence_id": contig["contig"],
                    "source": "rolypoly",
                    "score": motif.get("score") if motif.get("score") is not None else ".",
                    "strand": strand, "phase": ".",
                }
                parent = {
                    **common, "type": "protein_match",
                    "start": motif.get("context_nt_from", motif["nt_from"]),
                    "end": motif.get("context_nt_to", motif["nt_to"]),
                    "ID": identifier, "Name": f"RdRp motif {motif['letter']}",
                    "evidence_source": motif.get("source", "short motif"),
                    "profile": motif.get("profile"), "database": motif.get("database"),
                    "evalue": motif.get("evalue"), "alignment": motif.get("alignment"),
                }
                handle.write(convert_record_to_gff3_record(parent) + "\n")
                if motif.get("projected_positions"):
                    feature = {
                        **common, "type": "polypeptide_motif",
                        "start": motif["nt_from"], "end": motif["nt_to"],
                        "ID": f"{identifier}_feature", "Parent": identifier,
                        "Name": f"RdRp motif {motif['letter']}",
                        "feature_id": motif.get("feature_id"),
                        "source_label": motif.get("source_label"),
                        "projection_status": motif.get("projection_status"),
                        "query_aa_positions": motif["projected_positions"],
                        "query_residues": motif.get("projected_residues"),
                        "profile_states": motif.get("profile_states"),
                        "unresolved_aa_positions": motif.get("unresolved_positions"),
                        "deleted_profile_states": motif.get("deleted_states"),
                        "out_of_span_profile_states": motif.get("out_of_span_states"),
                        "insertion_aa_positions": motif.get("insertion_positions"),
                        "span_status": "approximate_hull",
                    }
                    handle.write(convert_record_to_gff3_record(feature) + "\n")
