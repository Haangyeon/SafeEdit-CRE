"""Audit low-Hamming-distance leakage across chromosome-based splits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .sequence import normalize_sequence
from .split import assign_upstream_split


def hamming_distance_at_most(left: str, right: str, threshold: int) -> int | None:
    """Return distance when it is at most the threshold, otherwise ``None``."""
    if len(left) != len(right):
        return None
    distance = 0
    for left_base, right_base in zip(left, right):
        distance += left_base != right_base
        if distance > threshold:
            return None
    return distance


def segment_keys(sequence: str, segments: int) -> tuple[tuple[int, str], ...]:
    """Partition a sequence into equal indexed segments for pigeonhole lookup."""
    if len(sequence) % segments:
        raise ValueError("sequence length must be divisible by the segment count")
    width = len(sequence) // segments
    return tuple(
        (index, sequence[index * width : (index + 1) * width])
        for index in range(segments)
    )


def load_holdouts(
    source: Path, sequence_length: int, segments: int
) -> tuple[list[tuple[str, str, str]], dict[tuple[int, str], list[int]], Counter]:
    records: list[tuple[str, str, str]] = []
    index: dict[tuple[int, str], list[int]] = defaultdict(list)
    length_counts = Counter()
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            sequence = normalize_sequence(row["sequence"])
            length_counts[len(sequence)] += 1
            split = assign_upstream_split(row["chr"])
            if split not in {"validation", "test"} or len(sequence) != sequence_length:
                continue
            record_index = len(records)
            records.append((sequence, split, row["IDs"]))
            for key in segment_keys(sequence, segments):
                index[key].append(record_index)
    return records, index, length_counts


def audit_near_duplicates(
    source: Path,
    sequence_length: int = 200,
    max_hamming: int = 3,
) -> dict[str, object]:
    segments = max_hamming + 1
    if sequence_length % segments:
        raise ValueError("sequence length must be divisible by max_hamming + 1")
    holdouts, index, length_counts = load_holdouts(source, sequence_length, segments)
    distance_counts = Counter()
    pair_counts = Counter()
    matches: list[dict[str, object]] = []
    candidate_comparisons = 0
    query_rows = 0
    matched_pairs: set[tuple[str, str]] = set()

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            sequence = normalize_sequence(row["sequence"])
            split = assign_upstream_split(row["chr"])
            if len(sequence) != sequence_length or split not in {"train", "validation"}:
                continue
            query_rows += 1
            candidate_indices: set[int] = set()
            for key in segment_keys(sequence, segments):
                candidate_indices.update(index.get(key, ()))

            for candidate_index in candidate_indices:
                candidate_sequence, candidate_split, candidate_id = holdouts[candidate_index]
                if split == candidate_split:
                    continue
                if split == "validation" and candidate_split != "test":
                    continue
                candidate_comparisons += 1
                distance = hamming_distance_at_most(sequence, candidate_sequence, max_hamming)
                if distance is None:
                    continue
                pair_key = tuple(sorted((row["IDs"], candidate_id)))
                if pair_key in matched_pairs:
                    continue
                matched_pairs.add(pair_key)
                distance_counts[distance] += 1
                split_pair = "--".join(sorted((split, candidate_split)))
                pair_counts[split_pair] += 1
                matches.append(
                    {
                        "query_id": row["IDs"],
                        "query_split": split,
                        "candidate_id": candidate_id,
                        "candidate_split": candidate_split,
                        "hamming_distance": distance,
                    }
                )

    return {
        "source": str(source),
        "scope": f"equal-length {sequence_length}-nt sequences",
        "method": (
            f"exact pigeonhole search using {segments} non-overlapping segments; "
            f"all cross-split pairs with Hamming distance <= {max_hamming} are detectable"
        ),
        "audited_row_count": length_counts[sequence_length],
        "excluded_variable_length_row_count": sum(
            count for length, count in length_counts.items() if length != sequence_length
        ),
        "query_row_count": query_rows,
        "holdout_index_row_count": len(holdouts),
        "candidate_comparison_count": candidate_comparisons,
        "cross_split_near_duplicate_pair_count": len(matched_pairs),
        "distance_counts": dict(sorted(distance_counts.items())),
        "split_pair_counts": dict(sorted(pair_counts.items())),
        "matches": matches,
        "limitation": (
            "Variable-length indel constructs are covered by exact/RC and variant-key audits, "
            "but not by this fixed-length Hamming search."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--max-hamming", type=int, default=3)
    args = parser.parse_args()
    report = audit_near_duplicates(args.source, args.sequence_length, args.max_hamming)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
