"""Create immutable split metadata and a deterministic pilot subset."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .g0_audit import LABEL_COLUMNS, SE_COLUMNS
from .sequence import normalize_sequence
from .split import assign_upstream_split, normalize_chromosome


EMBEDDED_CHROMOSOME = re.compile(r"(?:^|,)(?:chr)?([0-9]+|X|Y):", re.IGNORECASE)
INDEX_FIELDS = (
    "source_row",
    "IDs",
    "chr",
    "split",
    "data_project",
    "sequence_length",
    "max_lfcSE",
    "quality_max_se_lt_1",
    "near_duplicate_holdout_conflict",
    "eligible_for_modeling",
    "exact_blake2b_128",
)
PILOT_FIELDS = (
    "source_row",
    "IDs",
    "chr",
    "split",
    "data_project",
    *LABEL_COLUMNS,
    *SE_COLUMNS,
    "sequence",
)


def primary_variant_key(identifier: str) -> str | None:
    """Return the first chr:position:ref:alt key when one is present."""
    first = identifier.split(",", 1)[0]
    tokens = first.split(":")
    if len(tokens) < 4 or not tokens[1].isdigit():
        return None
    return ":".join(tokens[:4])


def update_reservoir(
    reservoir: list[dict[str, str]],
    row: dict[str, str],
    seen: int,
    target: int,
    rng: random.Random,
) -> None:
    """Update a fixed-size reservoir after observing ``seen`` eligible rows."""
    if len(reservoir) < target:
        reservoir.append(row)
        return
    replacement = rng.randrange(seen)
    if replacement < target:
        reservoir[replacement] = row


def derive(
    source: Path,
    index_path: Path,
    pilot_path: Path,
    report_path: Path,
    sizes: dict[str, int],
    seed: int,
    near_duplicate_report: Path | None = None,
) -> dict[str, object]:
    """Stream the source once and write split index, pilot, and ID-pair audit."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pilot_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    reservoirs = {split: [] for split in sizes}
    eligible_seen = Counter()
    rngs = {split: random.Random(f"{seed}:{split}") for split in sizes}
    variant_splits: dict[str, set[str]] = defaultdict(set)
    variant_rows = Counter()
    embedded_chromosome_mismatches = 0
    rows_with_multiple_embedded_chromosomes = 0
    total_rows = 0
    conflict_variant_keys: set[str] = set()
    if near_duplicate_report is not None:
        near_report = json.loads(near_duplicate_report.read_text(encoding="utf-8"))
        for match in near_report.get("matches", []):
            for id_field, split_field in (
                ("query_id", "query_split"),
                ("candidate_id", "candidate_split"),
            ):
                if match.get(split_field) == "train":
                    key = primary_variant_key(str(match[id_field]))
                    if key is not None:
                        conflict_variant_keys.add(key)
    conflict_rows = 0
    conflict_quality_rows = 0

    with (
        source.open("r", encoding="utf-8-sig", newline="") as source_handle,
        gzip.open(index_path, "wt", encoding="utf-8", newline="") as index_handle,
    ):
        reader = csv.DictReader(source_handle, delimiter="\t")
        index_writer = csv.DictWriter(index_handle, fieldnames=INDEX_FIELDS, delimiter="\t")
        index_writer.writeheader()

        for source_row, row in enumerate(reader, start=1):
            total_rows += 1
            chrom = normalize_chromosome(row["chr"])
            split = assign_upstream_split(chrom)
            sequence = normalize_sequence(row["sequence"])
            max_se = max(float(row[column]) for column in SE_COLUMNS)
            quality = max_se < 1.0
            identifier = row["IDs"]

            embedded = {
                normalize_chromosome(match.group(1))
                for match in EMBEDDED_CHROMOSOME.finditer(identifier)
            }
            if len(embedded) > 1:
                rows_with_multiple_embedded_chromosomes += 1
            if embedded and embedded != {chrom}:
                embedded_chromosome_mismatches += 1

            key = primary_variant_key(identifier)
            if key is not None:
                variant_splits[key].add(split)
                variant_rows[key] += 1
            near_conflict = split == "train" and key in conflict_variant_keys
            if near_conflict:
                conflict_rows += 1
                conflict_quality_rows += quality
            eligible = quality and not near_conflict

            digest = hashlib.blake2b(sequence.encode("ascii"), digest_size=16).hexdigest()
            index_writer.writerow(
                {
                    "source_row": source_row,
                    "IDs": identifier,
                    "chr": chrom,
                    "split": split,
                    "data_project": row["data_project"],
                    "sequence_length": len(sequence),
                    "max_lfcSE": f"{max_se:.12g}",
                    "quality_max_se_lt_1": int(quality),
                    "near_duplicate_holdout_conflict": int(near_conflict),
                    "eligible_for_modeling": int(eligible),
                    "exact_blake2b_128": digest,
                }
            )

            if eligible and split in sizes:
                eligible_seen[split] += 1
                pilot_row = {field: row.get(field, "") for field in PILOT_FIELDS}
                pilot_row.update(
                    {
                        "source_row": str(source_row),
                        "chr": chrom,
                        "split": split,
                        "sequence": sequence,
                    }
                )
                update_reservoir(
                    reservoirs[split],
                    pilot_row,
                    eligible_seen[split],
                    sizes[split],
                    rngs[split],
                )

    with gzip.open(pilot_path, "wt", encoding="utf-8", newline="") as pilot_handle:
        writer = csv.DictWriter(pilot_handle, fieldnames=PILOT_FIELDS, delimiter="\t")
        writer.writeheader()
        for split in ("train", "validation", "test"):
            for row in sorted(reservoirs[split], key=lambda item: int(item["source_row"])):
                writer.writerow(row)

    report: dict[str, object] = {
        "source": str(source),
        "seed": seed,
        "total_rows": total_rows,
        "eligible_quality_rows_by_split": dict(sorted(eligible_seen.items())),
        "pilot_target_rows_by_split": sizes,
        "pilot_actual_rows_by_split": {
            split: len(rows) for split, rows in reservoirs.items()
        },
        "variant_key_count": len(variant_splits),
        "variant_keys_with_multiple_rows": sum(count > 1 for count in variant_rows.values()),
        "variant_keys_crossing_splits": sum(
            len(split_set) > 1 for split_set in variant_splits.values()
        ),
        "rows_with_multiple_embedded_chromosomes": rows_with_multiple_embedded_chromosomes,
        "embedded_chromosome_mismatch_rows": embedded_chromosome_mismatches,
        "near_duplicate_conflict_variant_key_count": len(conflict_variant_keys),
        "near_duplicate_conflict_train_row_count": conflict_rows,
        "near_duplicate_conflict_quality_train_row_count": conflict_quality_rows,
        "near_duplicate_report": str(near_duplicate_report) if near_duplicate_report else None,
        "index_path": str(index_path),
        "pilot_path": str(pilot_path),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=30_000)
    parser.add_argument("--validation-size", type=int, default=5_000)
    parser.add_argument("--test-size", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--near-duplicate-report", type=Path)
    args = parser.parse_args()
    report = derive(
        args.source,
        args.index,
        args.pilot,
        args.report,
        {
            "train": args.train_size,
            "validation": args.validation_size,
            "test": args.test_size,
        },
        args.seed,
        args.near_duplicate_report,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
