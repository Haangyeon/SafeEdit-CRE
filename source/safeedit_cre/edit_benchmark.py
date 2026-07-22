"""Locked pilot benchmark for constrained minimal CRE sequence editing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import load_pilot
from .sequence import normalize_sequence


BASES = "ACGT"
BUDGETS = (1, 5, 10, 20)
CELL_NAMES = ("K562", "HepG2", "SKNSH")


def specificity_score(predictions: np.ndarray, target: int) -> np.ndarray:
    """Target activity minus the strongest predicted off-target activity."""
    off_targets = [index for index in range(predictions.shape[1]) if index != target]
    return predictions[:, target] - np.max(predictions[:, off_targets], axis=1)


def enumerate_substitutions(
    sequence: str, used_positions: set[int]
) -> tuple[list[str], list[tuple[int, str, str]]]:
    """Enumerate all single substitutions at positions not previously edited."""
    sequence = normalize_sequence(sequence)
    candidates: list[str] = []
    edits: list[tuple[int, str, str]] = []
    for position, reference in enumerate(sequence):
        if position in used_positions:
            continue
        for alternate in BASES:
            if alternate == reference:
                continue
            candidates.append(sequence[:position] + alternate + sequence[position + 1 :])
            edits.append((position, reference, alternate))
    return candidates, edits


def random_edit_path(
    sequence: str, budgets: tuple[int, ...], seed: int
) -> dict[int, tuple[str, list[tuple[int, str, str]]]]:
    rng = random.Random(seed)
    current = normalize_sequence(sequence)
    available = list(range(len(current)))
    edits: list[tuple[int, str, str]] = []
    snapshots = {}
    for step in range(1, max(budgets) + 1):
        position = available.pop(rng.randrange(len(available)))
        reference = current[position]
        alternate = rng.choice([base for base in BASES if base != reference])
        current = current[:position] + alternate + current[position + 1 :]
        edits.append((position, reference, alternate))
        if step in budgets:
            snapshots[step] = (current, edits.copy())
    return snapshots


def max_homopolymer(sequence: str) -> int:
    longest = 0
    current = 0
    previous = None
    for base in sequence:
        if base == previous:
            current += 1
        else:
            previous = base
            current = 1
        longest = max(longest, current)
    return longest


def gc_fraction(sequence: str) -> float:
    sequence = normalize_sequence(sequence)
    return sum(base in "GC" for base in sequence) / len(sequence)


def kmer_index(kmer: str) -> int:
    value = 0
    mapping = {base: index for index, base in enumerate(BASES)}
    for base in kmer:
        value = value * 4 + mapping[base]
    return value


def fit_kmer_log_probabilities(
    sequences: list[str], k: int = 6, pseudocount: float = 1.0
) -> np.ndarray:
    counts = np.full(4**k, pseudocount, dtype=np.float64)
    for sequence in sequences:
        sequence = normalize_sequence(sequence)
        for start in range(len(sequence) - k + 1):
            counts[kmer_index(sequence[start : start + k])] += 1.0
    return np.log(counts / counts.sum())


def sequence_kmer_log_likelihood(
    sequence: str, log_probabilities: np.ndarray, k: int = 6
) -> float:
    sequence = normalize_sequence(sequence)
    values = [
        log_probabilities[kmer_index(sequence[start : start + k])]
        for start in range(len(sequence) - k + 1)
    ]
    return float(np.mean(values))


def deterministic_parents(
    table: pd.DataFrame, split: str, count: int
) -> pd.DataFrame:
    subset = table.loc[
        (table["split"] == split)
        & (table["data_project"] == "CRE")
        & (table["sequence"].astype(str).str.len() == 200)
    ].copy()
    subset["selection_hash"] = [
        hashlib.sha256(f"{identifier}\t{sequence}".encode("utf-8")).hexdigest()
        for identifier, sequence in zip(subset["IDs"], subset["sequence"], strict=True)
    ]
    subset = subset.sort_values(["selection_hash", "IDs"], kind="stable").head(count)
    if len(subset) != count:
        raise RuntimeError(
            f"requested {count} {split} CRE parents but only found {len(subset)}"
        )
    return subset.loc[
        :, ["IDs", "sequence", "data_project", "split", "selection_hash"]
    ].reset_index(drop=True)


class PrimaryPredictor:
    def __init__(self, checkpoint: Path, device: str, batch_size: int) -> None:
        from .malinois import load_malinois

        self.model, self.metadata = load_malinois(checkpoint, device=device)
        self.batch_size = batch_size

    def predict(self, sequences: list[str]) -> tuple[np.ndarray, np.ndarray]:
        from .malinois import predict_both_strands

        forward, reverse = predict_both_strands(
            self.model, sequences, batch_size=self.batch_size
        )
        return (forward + reverse) / 2.0, np.mean(np.abs(forward - reverse), axis=1)


class CompactEnsembleReviewer:
    def __init__(
        self,
        checkpoints: list[Path],
        ensemble_report: Path,
        device: str,
        batch_size: int,
    ) -> None:
        import torch

        from .cnn_model import HeteroscedasticCRECNN

        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.members = []
        reference_mean = None
        reference_std = None
        for checkpoint_path in checkpoints:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            model = HeteroscedasticCRECNN()
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.to(self.device).eval()
            label_mean = checkpoint["label_mean"].numpy()
            label_std = checkpoint["label_std"].numpy()
            if reference_mean is None:
                reference_mean = label_mean
                reference_std = label_std
            elif not (
                np.allclose(reference_mean, label_mean)
                and np.allclose(reference_std, label_std)
            ):
                raise RuntimeError("reviewer checkpoints use inconsistent normalization")
            self.members.append(model)
        if len(self.members) != 3:
            raise RuntimeError("the locked reviewer requires exactly three members")
        self.label_mean = np.asarray(reference_mean)
        self.label_std = np.asarray(reference_std)
        report = json.loads(ensemble_report.read_text(encoding="utf-8"))
        if report.get("calibration_source") != "validation only":
            raise RuntimeError("reviewer variance scale is not validation-only")
        self.variance_scale = np.asarray(report["variance_scale"], dtype=np.float64)

    def predict(self, sequences: list[str]) -> tuple[np.ndarray, np.ndarray]:
        from .cnn_model import encode_sequences

        torch = self.torch
        inputs = encode_sequences(sequences)
        member_means = []
        member_variances = []
        for model in self.members:
            means = []
            variances = []
            with torch.inference_mode():
                for start in range(0, len(inputs), self.batch_size):
                    batch = inputs[start : start + self.batch_size].to(self.device)
                    mean, raw_variance = model(batch)
                    means.append(mean.cpu().numpy())
                    variances.append(torch.nn.functional.softplus(raw_variance).cpu().numpy())
            mean_standardized = np.concatenate(means)
            variance_standardized = np.concatenate(variances)
            member_means.append(mean_standardized * self.label_std + self.label_mean)
            member_variances.append(variance_standardized * (self.label_std**2))
        stacked_means = np.stack(member_means)
        stacked_variances = np.stack(member_variances)
        ensemble_mean = np.mean(stacked_means, axis=0)
        predictive_variance = (
            np.mean(stacked_variances, axis=0) + np.var(stacked_means, axis=0)
        ) * self.variance_scale
        mean_uncertainty = np.mean(np.sqrt(np.maximum(predictive_variance, 1e-12)), axis=1)
        return ensemble_mean, mean_uncertainty


def greedy_paths(
    parents: pd.DataFrame,
    target: int,
    primary: PrimaryPredictor,
    budgets: tuple[int, ...],
) -> list[dict[str, object]]:
    sequences = parents["sequence"].astype(str).tolist()
    used_positions = [set() for _ in sequences]
    edit_histories: list[list[tuple[int, str, str]]] = [[] for _ in sequences]
    snapshots: list[dict[str, object]] = []
    for step in range(1, max(budgets) + 1):
        all_candidates: list[str] = []
        all_edits: list[tuple[int, str, str]] = []
        slices: list[tuple[int, int]] = []
        for sequence, used in zip(sequences, used_positions, strict=True):
            candidates, edits = enumerate_substitutions(sequence, used)
            start = len(all_candidates)
            all_candidates.extend(candidates)
            all_edits.extend(edits)
            slices.append((start, len(all_candidates)))
        predictions, _ = primary.predict(all_candidates)
        objectives = specificity_score(predictions, target)
        for parent_index, (start, stop) in enumerate(slices):
            local_index = int(np.argmax(objectives[start:stop]))
            selected_index = start + local_index
            sequences[parent_index] = all_candidates[selected_index]
            edit = all_edits[selected_index]
            used_positions[parent_index].add(edit[0])
            edit_histories[parent_index].append(edit)
        if step in budgets:
            for parent_index, sequence in enumerate(sequences):
                snapshots.append(
                    {
                        "parent_index": parent_index,
                        "target_index": target,
                        "budget": step,
                        "method": "greedy_malinois",
                        "sequence": sequence,
                        "edits": edit_histories[parent_index].copy(),
                    }
                )
    return snapshots


def random_paths(
    parents: pd.DataFrame,
    target: int,
    budgets: tuple[int, ...],
    seed: int,
) -> list[dict[str, object]]:
    snapshots = []
    for parent_index, sequence in enumerate(parents["sequence"].astype(str)):
        path = random_edit_path(
            sequence, budgets, seed=seed + target * 100_000 + parent_index
        )
        for budget in budgets:
            edited, edits = path[budget]
            snapshots.append(
                {
                    "parent_index": parent_index,
                    "target_index": target,
                    "budget": budget,
                    "method": "random_matched",
                    "sequence": edited,
                    "edits": edits,
                }
            )
    return snapshots


def review_records(
    records: list[dict[str, object]],
    parents: pd.DataFrame,
    primary: PrimaryPredictor,
    reviewer: CompactEnsembleReviewer,
    log_probabilities: np.ndarray,
) -> list[dict[str, object]]:
    parent_sequences = parents["sequence"].astype(str).tolist()
    candidate_sequences = [str(record["sequence"]) for record in records]
    parent_primary, parent_strand = primary.predict(parent_sequences)
    candidate_primary, candidate_strand = primary.predict(candidate_sequences)
    parent_review, parent_uncertainty = reviewer.predict(parent_sequences)
    candidate_review, candidate_uncertainty = reviewer.predict(candidate_sequences)
    parent_naturalness = np.asarray(
        [sequence_kmer_log_likelihood(sequence, log_probabilities) for sequence in parent_sequences]
    )
    reviewed = []
    for row_index, record in enumerate(records):
        parent_index = int(record["parent_index"])
        target = int(record["target_index"])
        parent_primary_margin = float(
            specificity_score(parent_primary[parent_index : parent_index + 1], target)[0]
        )
        candidate_primary_margin = float(
            specificity_score(candidate_primary[row_index : row_index + 1], target)[0]
        )
        parent_review_margin = float(
            specificity_score(parent_review[parent_index : parent_index + 1], target)[0]
        )
        candidate_review_margin = float(
            specificity_score(candidate_review[row_index : row_index + 1], target)[0]
        )
        sequence = candidate_sequences[row_index]
        parent_sequence = parent_sequences[parent_index]
        result = {
            **record,
            "parent_id": str(parents.iloc[parent_index]["IDs"]),
            "target_cell": CELL_NAMES[target],
            "parent_sequence": parent_sequence,
            "edit_string": ";".join(
                f"{position + 1}{reference}>{alternate}"
                for position, reference, alternate in record["edits"]
            ),
            "hamming_distance": sum(
                left != right for left, right in zip(parent_sequence, sequence, strict=True)
            ),
            "primary_parent_target": float(parent_primary[parent_index, target]),
            "primary_candidate_target": float(candidate_primary[row_index, target]),
            "primary_target_gain": float(
                candidate_primary[row_index, target] - parent_primary[parent_index, target]
            ),
            "primary_parent_margin": parent_primary_margin,
            "primary_candidate_margin": candidate_primary_margin,
            "primary_margin_gain": candidate_primary_margin - parent_primary_margin,
            "primary_strand_disagreement": float(candidate_strand[row_index]),
            "reviewer_parent_margin": parent_review_margin,
            "reviewer_candidate_margin": candidate_review_margin,
            "reviewer_margin_gain": candidate_review_margin - parent_review_margin,
            "reviewer_uncertainty": float(candidate_uncertainty[row_index]),
            "reviewer_parent_uncertainty": float(parent_uncertainty[parent_index]),
            "naturalness_delta": sequence_kmer_log_likelihood(
                sequence, log_probabilities
            )
            - parent_naturalness[parent_index],
            "absolute_gc_delta": abs(gc_fraction(sequence) - gc_fraction(parent_sequence)),
            "max_homopolymer": max_homopolymer(sequence),
        }
        reviewed.append(result)
    return reviewed


def calibration_thresholds(
    reviewed_validation: list[dict[str, object]], budgets: tuple[int, ...]
) -> dict[int, dict[str, float]]:
    thresholds = {}
    for budget in budgets:
        rows = [row for row in reviewed_validation if int(row["budget"]) == budget]
        thresholds[budget] = {
            "strand_disagreement_max": float(
                np.quantile([row["primary_strand_disagreement"] for row in rows], 0.95)
            ),
            "reviewer_uncertainty_max": float(
                np.quantile([row["reviewer_uncertainty"] for row in rows], 0.95)
            ),
            "naturalness_delta_min": float(
                np.quantile([row["naturalness_delta"] for row in rows], 0.05)
            ),
            "absolute_gc_delta_max": float(
                np.quantile([row["absolute_gc_delta"] for row in rows], 0.95)
            ),
        }
    return thresholds


def apply_audit(
    rows: list[dict[str, object]], thresholds: dict[int, dict[str, float]]
) -> None:
    for row in rows:
        threshold = thresholds[int(row["budget"])]
        checks = {
            "primary_target_nonnegative": row["primary_target_gain"] >= 0,
            "primary_margin_positive": row["primary_margin_gain"] > 0,
            "reviewer_transfer_positive": row["reviewer_margin_gain"] > 0,
            "strand_in_domain": row["primary_strand_disagreement"]
            <= threshold["strand_disagreement_max"],
            "reviewer_uncertainty_in_domain": row["reviewer_uncertainty"]
            <= threshold["reviewer_uncertainty_max"],
            "naturalness_in_domain": row["naturalness_delta"]
            >= threshold["naturalness_delta_min"],
            "gc_in_domain": row["absolute_gc_delta"]
            <= threshold["absolute_gc_delta_max"],
            "homopolymer_safe": row["max_homopolymer"] <= 6,
            "exact_edit_budget": row["hamming_distance"] == row["budget"],
        }
        row.update({f"check_{name}": bool(value) for name, value in checks.items()})
        row["accepted"] = all(checks.values())
        row["failed_checks"] = ";".join(name for name, value in checks.items() if not value)


def finite_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = []
    keys = sorted(
        {(row["method"], row["target_cell"], row["budget"]) for row in rows},
        key=lambda item: (str(item[0]), str(item[1]), int(item[2])),
    )
    for method, target, budget in keys:
        group = [
            row
            for row in rows
            if (row["method"], row["target_cell"], row["budget"])
            == (method, target, budget)
        ]
        accepted = [row for row in group if row["accepted"]]
        summaries.append(
            {
                "method": method,
                "target_cell": target,
                "budget": budget,
                "n": len(group),
                "accepted_n": len(accepted),
                "accepted_fraction": len(accepted) / len(group),
                "mean_primary_margin_gain": finite_mean(
                    [row["primary_margin_gain"] for row in group]
                ),
                "mean_primary_target_gain": finite_mean(
                    [row["primary_target_gain"] for row in group]
                ),
                "mean_reviewer_margin_gain": finite_mean(
                    [row["reviewer_margin_gain"] for row in group]
                ),
                "reviewer_transfer_positive_fraction": finite_mean(
                    [float(row["reviewer_margin_gain"] > 0) for row in group]
                ),
                "accepted_mean_primary_margin_gain": finite_mean(
                    [row["primary_margin_gain"] for row in accepted]
                ),
            }
        )
    return summaries


def paired_bootstrap(
    rows: list[dict[str, object]], repetitions: int = 2000, seed: int = 20260713
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    results = []
    for target_name in CELL_NAMES:
        for budget in BUDGETS:
            selected = [
                row
                for row in rows
                if row["target_cell"] == target_name and row["budget"] == budget
            ]
            by_key = {(row["parent_id"], row["method"]): row for row in selected}
            parent_ids = sorted({row["parent_id"] for row in selected})
            for metric in ("primary_margin_gain", "reviewer_margin_gain"):
                differences = np.asarray(
                    [
                        by_key[(parent, "greedy_malinois")][metric]
                        - by_key[(parent, "random_matched")][metric]
                        for parent in parent_ids
                    ],
                    dtype=np.float64,
                )
                samples = rng.choice(
                    differences, size=(repetitions, len(differences)), replace=True
                ).mean(axis=1)
                results.append(
                    {
                        "target_cell": target_name,
                        "budget": budget,
                        "metric": metric,
                        "greedy_minus_random_mean": float(np.mean(differences)),
                        "bootstrap_95_ci": [
                            float(np.quantile(samples, 0.025)),
                            float(np.quantile(samples, 0.975)),
                        ],
                        "n_pairs": len(differences),
                    }
                )
    return results


def serialize_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    serializable = []
    for row in rows:
        serializable.append({key: value for key, value in row.items() if key != "edits"})
    return pd.DataFrame(serializable)


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available() and args.device == "cuda":
        raise RuntimeError("CUDA is required by the locked G3 handoff")
    table = load_pilot(args.pilot)
    training_sequences = table.loc[table["split"] == "train", "sequence"].astype(str).tolist()
    log_probabilities = fit_kmer_log_probabilities(training_sequences, k=6)
    validation_parents = deterministic_parents(table, "validation", args.parent_count)
    test_parents = deterministic_parents(table, "test", args.parent_count)

    primary = PrimaryPredictor(args.malinois_checkpoint, args.device, args.batch_size)
    reviewer = CompactEnsembleReviewer(
        args.reviewer_checkpoints,
        args.reviewer_report,
        args.device,
        args.reviewer_batch_size,
    )

    validation_random = []
    for target in range(3):
        validation_random.extend(
            random_paths(validation_parents, target, BUDGETS, seed=args.seed + 10_000)
        )
    validation_reviewed = review_records(
        validation_random,
        validation_parents,
        primary,
        reviewer,
        log_probabilities,
    )
    thresholds = calibration_thresholds(validation_reviewed, BUDGETS)

    test_records = []
    for target in range(3):
        test_records.extend(greedy_paths(test_parents, target, primary, BUDGETS))
        test_records.extend(
            random_paths(test_parents, target, BUDGETS, seed=args.seed + 20_000)
        )
    test_reviewed = review_records(
        test_records, test_parents, primary, reviewer, log_probabilities
    )
    apply_audit(test_reviewed, thresholds)

    output_table = serialize_rows(test_reviewed)
    args.candidates.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.candidates.with_name(args.candidates.name + ".part")
    output_table.to_csv(temporary, sep="\t", index=False, compression="gzip")
    with pd.read_csv(temporary, sep="\t", compression="gzip", chunksize=1000) as chunks:
        written_rows = sum(len(chunk) for chunk in chunks)
    if written_rows != len(test_reviewed):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("candidate row-count verification failed")
    temporary.replace(args.candidates)

    failure_counts = Counter()
    for row in test_reviewed:
        for failure in str(row["failed_checks"]).split(";"):
            if failure:
                failure_counts[failure] += 1
    report: dict[str, object] = {
        "purpose": "locked G3 pilot benchmark of minimal substitution editing",
        "test_labels_used": False,
        "parent_selection": (
            "first SHA-256-ranked 200-nt CRE sequences; no observed activity labels"
        ),
        "parent_count_per_split": args.parent_count,
        "targets": list(CELL_NAMES),
        "budgets": list(BUDGETS),
        "primary_objective": "target prediction minus maximum off-target prediction",
        "greedy_constraint": "each position edited at most once; exact Hamming budget",
        "calibration_source": "validation random edits only",
        "audit_thresholds_by_budget": thresholds,
        "novel_sequence_variance": (
            "mean learned variance plus between-seed variance; no experimental SE term"
        ),
        "test_candidate_rows": len(test_reviewed),
        "summary": summarize(test_reviewed),
        "paired_bootstrap": paired_bootstrap(test_reviewed, seed=args.seed),
        "failed_check_counts": dict(sorted(failure_counts.items())),
        "accepted_total": int(sum(bool(row["accepted"]) for row in test_reviewed)),
        "candidate_table": str(args.candidates),
        "limitations": [
            "Edited sequences have predictor scores but no new experimental measurements.",
            "The G2 reviewer is a low-capacity pilot model trained on 30,000 sequences.",
            "This G3 pilot uses a small deterministic set of genomic CRE parents.",
            "Validation random-edit quantiles are operational audit rules, not universal biology.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pilot", type=Path)
    parser.add_argument("malinois_checkpoint", type=Path)
    parser.add_argument("--reviewer-checkpoints", nargs=3, type=Path, required=True)
    parser.add_argument("--reviewer-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--parent-count", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--reviewer-batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
