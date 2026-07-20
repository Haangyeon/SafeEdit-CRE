"""Evaluate published Malinois weights on the complete quality-filtered test split."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .baseline import mean_correlation, regression_metrics
from .g0_audit import LABEL_COLUMNS, SE_COLUMNS
from .sequence import normalize_sequence
from .split import assign_upstream_split


def load_complete_test(source: Path) -> dict[str, list[object]]:
    fields = {
        "IDs": [],
        "chr": [],
        "data_project": [],
        "sequence": [],
        "sequence_length": [],
        "labels": [],
    }
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if assign_upstream_split(row["chr"]) != "test":
                continue
            if max(float(row[column]) for column in SE_COLUMNS) >= 1.0:
                continue
            sequence = normalize_sequence(row["sequence"])
            fields["IDs"].append(row["IDs"])
            fields["chr"].append(row["chr"])
            fields["data_project"].append(row["data_project"])
            fields["sequence"].append(sequence)
            fields["sequence_length"].append(len(sequence))
            fields["labels"].append([float(row[column]) for column in LABEL_COLUMNS])
    return fields


def subset_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    indices: np.ndarray,
) -> dict[str, object]:
    metrics = regression_metrics(observed[indices], predicted[indices])
    return {
        "n": int(indices.sum()),
        "mean_pearson_r": mean_correlation(metrics),
        "mean_spearman_rho": mean_correlation(metrics, "spearman_rho"),
        "metrics": metrics,
    }


def stratified_metrics(
    values: list[object], observed: np.ndarray, predicted: np.ndarray
) -> dict[str, object]:
    array = np.asarray(values)
    return {
        str(value): subset_metrics(observed, predicted, array == value)
        for value in sorted(set(values), key=str)
        if int(np.sum(array == value)) >= 3
    }


def rejection_curve(
    observed: np.ndarray,
    predicted: np.ndarray,
    uncertainty: np.ndarray,
) -> list[dict[str, object]]:
    order = np.argsort(uncertainty)
    rows = []
    for retained_fraction in (1.0, 0.95, 0.9, 0.8, 0.5):
        retained = order[: max(3, int(len(order) * retained_fraction))]
        mask = np.zeros(len(order), dtype=bool)
        mask[retained] = True
        metrics = subset_metrics(observed, predicted, mask)
        rows.append(
            {
                "retained_fraction": retained_fraction,
                "uncertainty_threshold": float(np.max(uncertainty[retained])),
                **metrics,
            }
        )
    return rows


def write_predictions(
    path: Path,
    table: dict[str, list[object]],
    observed: np.ndarray,
    forward: np.ndarray,
    reverse: np.ndarray,
) -> None:
    averaged = (forward + reverse) / 2.0
    uncertainty = np.mean(np.abs(forward - reverse), axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    fieldnames = [
        "IDs",
        "chr",
        "data_project",
        "sequence_length",
        "strand_disagreement",
    ]
    for label in LABEL_COLUMNS:
        fieldnames.extend((f"observed_{label}", f"predicted_{label}"))
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for index in range(len(observed)):
            row = {
                "IDs": table["IDs"][index],
                "chr": table["chr"][index],
                "data_project": table["data_project"][index],
                "sequence_length": table["sequence_length"][index],
                "strand_disagreement": f"{uncertainty[index]:.8g}",
            }
            for label_index, label in enumerate(LABEL_COLUMNS):
                row[f"observed_{label}"] = f"{observed[index, label_index]:.8g}"
                row[f"predicted_{label}"] = f"{averaged[index, label_index]:.8g}"
            writer.writerow(row)
    with gzip.open(temporary, "rt", encoding="utf-8") as handle:
        observed_rows = sum(1 for _ in handle) - 1
    if observed_rows != len(observed):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"prediction artifact row mismatch: {observed_rows:,} != {len(observed):,}"
        )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch

    from .malinois import load_malinois, predict_both_strands

    table = load_complete_test(args.source)
    sequences = [str(sequence) for sequence in table["sequence"]]
    if len(sequences) != 62_582:
        raise RuntimeError(f"expected 62,582 quality test rows, observed {len(sequences):,}")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, metadata = load_malinois(args.checkpoint, device=device)
    forward, reverse = predict_both_strands(model, sequences, args.batch_size)
    predicted = (forward + reverse) / 2.0
    observed = np.asarray(table["labels"], dtype=np.float64)
    uncertainty = np.mean(np.abs(forward - reverse), axis=1)
    absolute_error = np.mean(np.abs(observed - predicted), axis=1)
    overall_metrics = regression_metrics(observed, predicted)
    uncertainty_error_rho = float(spearmanr(uncertainty, absolute_error).statistic)

    length_groups = ["200" if length == 200 else "73-199" for length in table["sequence_length"]]
    report: dict[str, object] = {
        "purpose": "complete published test-split evaluation and strand-disagreement audit",
        "source": str(args.source),
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata": metadata,
        "rows": len(sequences),
        "quality_rule": "max(K562_lfcSE, HepG2_lfcSE, SKNSH_lfcSE) < 1",
        "sequence_length_counts": dict(sorted(Counter(table["sequence_length"]).items())),
        "overall": {
            "mean_pearson_r": mean_correlation(overall_metrics),
            "mean_spearman_rho": mean_correlation(overall_metrics, "spearman_rho"),
            "metrics": overall_metrics,
        },
        "by_chromosome": stratified_metrics(table["chr"], observed, predicted),
        "by_data_project": stratified_metrics(table["data_project"], observed, predicted),
        "by_sequence_length_group": stratified_metrics(length_groups, observed, predicted),
        "strand_disagreement": {
            "mean": float(np.mean(uncertainty)),
            "median": float(np.median(uncertainty)),
            "p95": float(np.quantile(uncertainty, 0.95)),
            "spearman_with_mean_absolute_error": uncertainty_error_rho,
            "rejection_curve_descriptive_only": rejection_curve(
                observed, predicted, uncertainty
            ),
            "caution": (
                "This test-set rejection curve is descriptive, not a tuned deployment rule. "
                "Any rejection threshold must be selected on validation data."
            ),
        },
        "limitations": [
            "Table S2 is the performance table, not the exact Ctrl.Mean>=20 training subset.",
            "This is computational validation against MPRA measurements, not new wet-lab validation.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_predictions(args.predictions, table, observed, forward, reverse)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
