from __future__ import annotations


def compute_self_dotplot_track_spans(
    sequence: str,
    k: int,
) -> dict[str, int]:
    """Estimate strongest forward/inverted self-dotplot tracks from exact k-mer runs."""
    normalized = str(sequence).upper()
    n = len(normalized)
    if n < max(2, k):
        return {
            "dotplot_forward_max_span": 0,
            "dotplot_inverted_max_span": 0,
        }

    kmer_positions: dict[str, list[int]] = {}
    for start in range(0, n - k + 1):
        kmer = normalized[start : start + k]
        kmer_positions.setdefault(kmer, []).append(start)

    def max_span_from_offset_positions(
        offset_positions: dict[int, list[int]],
    ) -> int:
        max_span = 0
        for positions in offset_positions.values():
            if not positions:
                continue
            unique_positions = sorted(set(positions))
            run_start = unique_positions[0]
            run_prev = unique_positions[0]
            for pos in unique_positions[1:]:
                if pos == run_prev + 1:
                    run_prev = pos
                    continue
                run_len = (run_prev - run_start) + 1
                max_span = max(max_span, run_len + k - 1)
                run_start = pos
                run_prev = pos
            run_len = (run_prev - run_start) + 1
            max_span = max(max_span, run_len + k - 1)
        return max_span

    offset_to_positions_forward: dict[int, list[int]] = {}
    for positions in kmer_positions.values():
        if len(positions) < 2:
            continue
        for left_idx in range(0, len(positions) - 1):
            left_pos = positions[left_idx]
            for right_pos in positions[left_idx + 1 :]:
                offset = right_pos - left_pos
                if offset <= 0:
                    continue
                offset_to_positions_forward.setdefault(offset, []).append(
                    left_pos
                )
    forward_max_span = max_span_from_offset_positions(offset_to_positions_forward)

    rc_sequence = normalized[::-1].translate(str.maketrans("ACGT", "TGCA"))
    rc_kmer_positions: dict[str, list[int]] = {}
    for start in range(0, n - k + 1):
        kmer = rc_sequence[start : start + k]
        rc_kmer_positions.setdefault(kmer, []).append(start)

    offset_to_positions_inverted: dict[int, list[int]] = {}
    for kmer, query_positions in kmer_positions.items():
        target_positions = rc_kmer_positions.get(kmer)
        if not target_positions:
            continue
        for query_pos in query_positions:
            for target_pos in target_positions:
                delta = target_pos - query_pos
                offset_to_positions_inverted.setdefault(delta, []).append(
                    query_pos
                )
    inverted_max_span = max_span_from_offset_positions(offset_to_positions_inverted)

    return {
        "dotplot_forward_max_span": int(forward_max_span),
        "dotplot_inverted_max_span": int(inverted_max_span),
    }
