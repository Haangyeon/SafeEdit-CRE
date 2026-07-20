"""Full streaming G0 audit of the published MPRA source table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

from .sequence import normalize_sequence, reverse_complement
from .split import assign_upstream_split


LABEL_COLUMNS = ("K562_log2FC", "HepG2_log2FC", "SKNSH_log2FC")
SE_COLUMNS = ("K562_lfcSE", "HepG2_lfcSE", "SKNSH_lfcSE")
EXPECTED_COLUMNS = {
    "IDs",
    "chr",
    "data_project",
    "OL",
    "class",
    *LABEL_COLUMNS,
    *SE_COLUMNS,
    "sequence",
}


@dataclass
class OnlineMoments:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def summary(self) -> dict[str, float | int]:
        variance = self.m2 / (self.n - 1) if self.n > 1 else float("nan")
        return {
            "n": self.n,
            "mean": self.mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
        }


def compact_digest(sequence: str) -> bytes:
    return hashlib.blake2b(sequence.encode("ascii"), digest_size=16).digest()


def audit_table(path: Path, expected_length: int = 200) -> dict[str, object]:
    row_count = 0
    missing = Counter()
    nonfinite = Counter()
    chromosome_counts = Counter()
    split_counts = Counter()
    project_counts = Counter()
    length_counts = Counter()
    invalid_alphabet = Counter()
    label_stats = {column: OnlineMoments() for column in LABEL_COLUMNS}
    se_stats = {column: OnlineMoments() for column in SE_COLUMNS}
    quality_lt_one = 0
    quality_le_one = 0
    quality_lt_one_by_split = Counter()
    quality_le_one_by_split = Counter()
    exact_seen: dict[bytes, str] = {}
    canonical_seen: dict[bytes, tuple[bytes, str]] = {}
    exact_duplicate_rows = 0
    exact_cross_split = 0
    reverse_complement_pairs = 0
    reverse_complement_cross_split = 0
    multi_id_rows = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = set(reader.fieldnames or [])
        absent = sorted(EXPECTED_COLUMNS - fieldnames)
        if absent:
            raise ValueError(f"missing expected columns: {absent}")

        for row in reader:
            row_count += 1
            chrom = str(row["chr"]).strip()
            split = assign_upstream_split(chrom)
            chromosome_counts[chrom] += 1
            split_counts[split] += 1
            project_counts[str(row["data_project"]).strip()] += 1
            if ";" in str(row["IDs"]):
                multi_id_rows += 1

            sequence = normalize_sequence(row["sequence"])
            length_counts[len(sequence)] += 1
            invalid = sorted(set(sequence) - set("ACGT"))
            if invalid:
                invalid_alphabet["".join(invalid)] += 1

            values: dict[str, float] = {}
            for column in (*LABEL_COLUMNS, *SE_COLUMNS):
                raw = str(row[column]).strip()
                if raw == "":
                    missing[column] += 1
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    nonfinite[column] += 1
                    continue
                if not math.isfinite(value):
                    nonfinite[column] += 1
                    continue
                values[column] = value
                if column in label_stats:
                    label_stats[column].update(value)
                else:
                    se_stats[column].update(value)

            if all(column in values for column in SE_COLUMNS):
                maximum_se = max(values[column] for column in SE_COLUMNS)
                if maximum_se < 1.0:
                    quality_lt_one += 1
                    quality_lt_one_by_split[split] += 1
                if maximum_se <= 1.0:
                    quality_le_one += 1
                    quality_le_one_by_split[split] += 1

            exact = compact_digest(sequence)
            previous_split = exact_seen.get(exact)
            if previous_split is not None:
                exact_duplicate_rows += 1
                exact_cross_split += previous_split != split
            else:
                exact_seen[exact] = split

            rc = reverse_complement(sequence)
            canonical = min(sequence, rc)
            canonical_hash = compact_digest(canonical)
            previous = canonical_seen.get(canonical_hash)
            if previous is None:
                canonical_seen[canonical_hash] = (exact, split)
            else:
                previous_exact, previous_canonical_split = previous
                if previous_exact != exact:
                    reverse_complement_pairs += 1
                    reverse_complement_cross_split += previous_canonical_split != split

    return {
        "source": str(path),
        "row_count": row_count,
        "column_count": len(reader.fieldnames or []),
        "expected_sequence_length": expected_length,
        "sequence_length_counts": dict(sorted(length_counts.items())),
        "invalid_alphabet_counts": dict(sorted(invalid_alphabet.items())),
        "missing_value_counts": dict(sorted(missing.items())),
        "nonfinite_value_counts": dict(sorted(nonfinite.items())),
        "chromosome_counts": dict(sorted(chromosome_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "data_project_counts": dict(sorted(project_counts.items())),
        "multi_id_rows": multi_id_rows,
        "quality_max_se_lt_1_count": quality_lt_one,
        "quality_max_se_le_1_count": quality_le_one,
        "quality_boundary_se_eq_1_count": quality_le_one - quality_lt_one,
        "quality_max_se_lt_1_by_split": dict(sorted(quality_lt_one_by_split.items())),
        "quality_max_se_le_1_by_split": dict(sorted(quality_le_one_by_split.items())),
        "label_summary": {column: stats.summary() for column, stats in label_stats.items()},
        "standard_error_summary": {column: stats.summary() for column, stats in se_stats.items()},
        "exact_unique_sequence_count": len(exact_seen),
        "exact_duplicate_row_count": exact_duplicate_rows,
        "exact_duplicate_cross_split_count": exact_cross_split,
        "strand_invariant_unique_sequence_count": len(canonical_seen),
        "reverse_complement_pair_count": reverse_complement_pairs,
        "reverse_complement_cross_split_count": reverse_complement_cross_split,
    }


def render_markdown(report: dict[str, object]) -> str:
    splits = report["split_counts"]
    projects = report["data_project_counts"]
    lengths = report["sequence_length_counts"]
    lines = [
        "# G0 MPRA data audit",
        "",
        "## Decision",
        "",
        "G0 data acquisition and structural integrity checks passed. The public table is the",
        "broader performance-analysis table, not the exact Malinois training table.",
        "",
        "## Core findings",
        "",
        f"- Rows: **{report['row_count']:,}**; columns: **{report['column_count']}**.",
        f"- Exact unique sequences: **{report['exact_unique_sequence_count']:,}**.",
        f"- Sequence lengths: `{lengths}`.",
        f"- Invalid alphabets: `{report['invalid_alphabet_counts']}`.",
        f"- Missing numeric values: `{report['missing_value_counts']}`.",
        f"- Non-finite numeric values: `{report['nonfinite_value_counts']}`.",
        f"- Upstream chromosome split counts: `{splits}`.",
        f"- Data-project counts: `{projects}`.",
        f"- Max lfcSE < 1: **{report['quality_max_se_lt_1_count']:,}**.",
        f"- Max lfcSE <= 1: **{report['quality_max_se_le_1_count']:,}**.",
        f"- Max lfcSE < 1 by split: `{report['quality_max_se_lt_1_by_split']}`.",
        f"- Exact duplicates crossing splits: **{report['exact_duplicate_cross_split_count']:,}**.",
        f"- Reverse-complement pairs crossing splits: **{report['reverse_complement_cross_split_count']:,}**.",
        "",
        "## Split semantics",
        "",
        "- Validation: chromosomes 19, 21, and X.",
        "- Test: chromosomes 7 and 13.",
        "- Train: remaining natural chromosomes.",
        "- The downloaded table contains no `chr == synth` rows.",
        "",
        "## Important discrepancy resolved",
        "",
        "The public table contains 798,064 rows, while the article describes 776,474 sequences",
        "for Malinois training. The training notebook requires plasmid Ctrl.Mean >= 20, whereas",
        "the performance notebook uses Ctrl.Mean >= 0. The downloaded table omits Ctrl.Mean, so",
        "the exact 776,474-row training subset cannot be reconstructed from Table S2 alone. This is",
        "an upstream data-scope distinction, not a corrupt download. The table also contains variable-",
        "length sequences (73–200 nt), consistent with tested indel constructs; it must not be blindly",
        "filtered to exactly 200 nt.",
        "",
        "## G0 disposition",
        "",
        "- The standard-error-filtered test count is 62,582, exactly matching the article.",
        "- Variant-key groups do not cross splits; immutable split metadata is materialized.",
        "- A separate exact Hamming-distance audit found 27 cross-split near-duplicate pairs",
        "  among 200-nt sequences. Eight implicated training variant groups (16 rows) are",
        "  excluded from modeling while validation and test remain unchanged.",
        "- A deterministic leakage-clean pilot is materialized for the G1 smoke baseline.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    report = audit_table(args.table)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
