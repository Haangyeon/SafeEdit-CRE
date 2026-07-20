"""Frozen G4 confirmation benchmark: R2 remediation locked version."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import load_pilot
from .edit_benchmark import (
    BASES,
    BUDGETS,
    CELL_NAMES,
    CompactEnsembleReviewer,
    PrimaryPredictor,
    apply_audit as _apply_audit_original,
    calibration_thresholds as _calibration_thresholds_original,
    deterministic_parents,
    enumerate_substitutions,
    fit_kmer_log_probabilities,
    gc_fraction,
    max_homopolymer,
    random_edit_path,
    review_records,
    sequence_kmer_log_likelihood,
    specificity_score,
)
from .sequence import normalize_sequence


EXPECTED_ROWS = 3456
N_PARENTS = 96
BEAM_WIDTH = 24
SEED = 20260714
BOOTSTRAP_ITER = 100000
BOOTSTRAP_SEED = 20260713


def mcnemar_exact_p(b: int, c: int) -> float:
    from scipy.stats import binomtest
    n = b + c
    if n == 0:
        return 1.0
    p_val = binomtest(b, n=n, p=0.5).pvalue
    return float(p_val)


def paired_bootstrap_g4(
    rows: list[dict[str, object]], repetitions: int = BOOTSTRAP_ITER, seed: int = BOOTSTRAP_SEED
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    results = []
    methods = ["random_matched", "greedy_malinois", "safeedit_consensus"]
    for target_name in CELL_NAMES:
        for budget in BUDGETS:
            for method_a, method_b in [
                ("safeedit_consensus", "greedy_malinois"),
                ("greedy_malinois", "random_matched"),
                ("safeedit_consensus", "random_matched"),
            ]:
                selected = [
                    r
                    for r in rows
                    if r["target_cell"] == target_name
                    and r["budget"] == budget
                    and r["method"] in (method_a, method_b)
                    and r["design_status"] == "feasible"
                ]
                by_key = {(r["parent_id"], r["method"]): r for r in selected}
                parent_ids = sorted({r["parent_id"] for r in selected})
                pairs = []
                paired_pids = []
                for pid in parent_ids:
                    if (pid, method_a) in by_key and (pid, method_b) in by_key:
                        pairs.append((by_key[(pid, method_a)], by_key[(pid, method_b)]))
                        paired_pids.append(pid)
                if not pairs:
                    continue
                acc_diffs = np.array([float(a["accepted"]) - float(b["accepted"]) for a, b in pairs])
                margin_diffs = np.array(
                    [float(a["primary_margin_gain"]) - float(b["primary_margin_gain"]) for a, b in pairs]
                )
                reviewer_margin_diffs = np.array(
                    [float(a["reviewer_margin_gain"]) - float(b["reviewer_margin_gain"]) for a, b in pairs]
                )
                n_clusters = len(pairs)
                cluster_indices = rng.integers(0, n_clusters, size=(repetitions, n_clusters))
                acc_samples = acc_diffs[cluster_indices].mean(axis=1)
                margin_samples = margin_diffs[cluster_indices].mean(axis=1)
                reviewer_margin_samples = reviewer_margin_diffs[cluster_indices].mean(axis=1)
                b_discord = sum(1 for a, b in pairs if a["accepted"] and not b["accepted"])
                c_discord = sum(1 for a, b in pairs if not a["accepted"] and b["accepted"])
                results.append(
                    {
                        "target_cell": target_name,
                        "budget": budget,
                        "comparison": f"{method_a}_vs_{method_b}",
                        "metric": "acceptance",
                        "mean_diff": float(np.mean(acc_diffs)),
                        "bootstrap_95_ci": [
                            float(np.quantile(acc_samples, 0.025)),
                            float(np.quantile(acc_samples, 0.975)),
                        ],
                        "mcnemar_exact_p": mcnemar_exact_p(b_discord, c_discord),
                        "discordant_ab": b_discord,
                        "discordant_ba": c_discord,
                        "n_pairs": n_clusters,
                    }
                )
                results.append(
                    {
                        "target_cell": target_name,
                        "budget": budget,
                        "comparison": f"{method_a}_vs_{method_b}",
                        "metric": "primary_margin_gain",
                        "mean_diff": float(np.mean(margin_diffs)),
                        "bootstrap_95_ci": [
                            float(np.quantile(margin_samples, 0.025)),
                            float(np.quantile(margin_samples, 0.975)),
                        ],
                        "n_pairs": n_clusters,
                    }
                )
                results.append(
                    {
                        "target_cell": target_name,
                        "budget": budget,
                        "comparison": f"{method_a}_vs_{method_b}",
                        "metric": "reviewer_margin_gain",
                        "mean_diff": float(np.mean(reviewer_margin_diffs)),
                        "bootstrap_95_ci": [
                            float(np.quantile(reviewer_margin_samples, 0.025)),
                            float(np.quantile(reviewer_margin_samples, 0.975)),
                        ],
                        "n_pairs": n_clusters,
                    }
                )
    return results


def calibration_thresholds_frozen(
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


def apply_audit_frozen(
    rows: list[dict[str, object]], thresholds: dict[int, dict[str, float]]
) -> None:
    for row in rows:
        if row.get("design_status") == "infeasible":
            row["accepted"] = False
            row["failed_checks"] = "infeasible"
            for check_name in [
                "primary_target_nonnegative",
                "primary_margin_positive",
                "reviewer_transfer_positive",
                "strand_in_domain",
                "reviewer_uncertainty_in_domain",
                "naturalness_in_domain",
                "gc_in_domain",
                "homopolymer_safe",
                "exact_edit_budget",
            ]:
                row[f"check_{check_name}"] = False
            continue
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
            "gc_in_domain": row["absolute_gc_delta"] <= threshold["absolute_gc_delta_max"],
            "homopolymer_safe": row["max_homopolymer"] <= 6,
            "exact_edit_budget": row["hamming_distance"] == row["budget"],
        }
        row.update({f"check_{name}": bool(value) for name, value in checks.items()})
        row["accepted"] = all(checks.values())
        row["failed_checks"] = ";".join(name for name, value in checks.items() if not value)


def safeedit_consensus_paths_frozen(
    parents: pd.DataFrame,
    target: int,
    primary: PrimaryPredictor,
    reviewer: CompactEnsembleReviewer,
    log_probabilities: np.ndarray,
    thresholds_by_budget: dict[int, dict[str, float]],
    budgets: tuple[int, ...],
    beam_width: int,
    seed: int,
    cell_name: str = "",
) -> list[dict[str, object]]:
    import time

    parent_seqs = parents["sequence"].astype(str).tolist()
    snapshots: list[dict[str, object]] = []
    parent_nat = np.array(
        [
            sequence_kmer_log_likelihood(normalize_sequence(s), log_probabilities)
            for s in parent_seqs
        ]
    )
    parent_gc = np.array([gc_fraction(s) for s in parent_seqs])
    prefilter_k = min(beam_width * 6, 96)

    budget_set = set(budgets)
    budget_snapshots: dict[int, dict[int, dict[str, object]]] = {b: {} for b in budgets}

    for parent_idx, parent_seq in enumerate(parent_seqs):
        if parent_idx % 8 == 0:
            print(
                f"[G4-frozen {cell_name} safeedit] parent {parent_idx+1}/{len(parent_seqs)}",
                flush=True,
            )
        beams: list[tuple[str, set[int], list[tuple[int, str, str]]]] = [
            (normalize_sequence(parent_seq), set(), [])
        ]
        found_for_budget: dict[int, bool] = {b: False for b in budgets}

        for step in range(1, max(budgets) + 1):
            candidates: list[tuple[str, set[int], list[tuple[int, str, str]]]] = []
            seen: set[str] = set()
            for seq, used, edits in beams:
                for pos, ref in enumerate(seq):
                    if pos in used:
                        continue
                    for alt in BASES:
                        if alt == ref:
                            continue
                        new_seq = seq[:pos] + alt + seq[pos + 1 :]
                        if new_seq in seen:
                            continue
                        seen.add(new_seq)
                        new_used = used | {pos}
                        new_edits = edits + [(pos, ref, alt)]
                        hp = max_homopolymer(new_seq)
                        if hp > 8:
                            continue
                        gcd = abs(gc_fraction(new_seq) - parent_gc[parent_idx])
                        if gcd > 0.12:
                            continue
                        candidates.append((new_seq, new_used, new_edits))
            if not candidates:
                for b in budgets:
                    if b >= step and not found_for_budget[b]:
                        budget_snapshots[b][parent_idx] = {
                            "parent_index": parent_idx,
                            "target_index": target,
                            "budget": b,
                            "method": "safeedit_consensus",
                            "sequence": parent_seq,
                            "edits": [],
                            "design_status": "infeasible",
                        }
                        found_for_budget[b] = True
                break

            cand_seqs = [c[0] for c in candidates]
            t0 = time.time()
            p_pred, p_strand = primary.predict(cand_seqs)
            t_primary = time.time() - t0
            margins_p = specificity_score(p_pred, target)
            primary_scores = margins_p.astype(float) - 0.05 * p_strand.astype(float)
            n_prefilter = min(prefilter_k, len(candidates))
            top_primary_idx = np.argsort(-primary_scores)[:n_prefilter]
            rerank_seqs = [cand_seqs[int(i)] for i in top_primary_idx]
            t1 = time.time()
            r_pred, r_uncert = reviewer.predict(rerank_seqs)
            t_reviewer = time.time() - t1
            margins_r = specificity_score(r_pred, target)
            n_rerank = len(rerank_seqs)
            nat_d = np.zeros(n_rerank)
            gc_d = np.zeros(n_rerank)
            homops = np.zeros(n_rerank, dtype=int)
            for j, sj in enumerate(rerank_seqs):
                nat_d[j] = (
                    sequence_kmer_log_likelihood(sj, log_probabilities) - parent_nat[parent_idx]
                )
                gc_d[j] = abs(gc_fraction(sj) - parent_gc[parent_idx])
                homops[j] = max_homopolymer(sj)
            thresh = thresholds_by_budget.get(step, thresholds_by_budget[max(budgets)])
            scores = np.full(n_rerank, -1e9)
            passes_all = np.zeros(n_rerank, dtype=bool)
            for j in range(n_rerank):
                mp = float(margins_p[int(top_primary_idx[j])])
                mr = float(margins_r[j])
                sd = float(p_strand[int(top_primary_idx[j])])
                uc = float(r_uncert[j])
                nd = float(nat_d[j])
                gd = float(gc_d[j])
                hp = int(homops[j])
                passes = (
                    sd <= thresh["strand_disagreement_max"]
                    and uc <= thresh["reviewer_uncertainty_max"]
                    and nd >= thresh["naturalness_delta_min"]
                    and gd <= thresh["absolute_gc_delta_max"]
                    and hp <= 7
                )
                passes_all[j] = passes
                if mp > 0 and mr > 0 and passes:
                    scores[j] = mp + 0.3 * mr - 0.1 * sd - 0.05 * uc
                elif mp > 0 and passes:
                    scores[j] = mp * 0.4 + 0.1 * max(mr, 0)
                elif mp > 0:
                    scores[j] = mp * 0.15
                else:
                    scores[j] = -1e6 + mp
            order = np.argsort(-scores)
            beams = []
            for idx in order[:beam_width]:
                if scores[idx] < -1e8 and len(beams) > 0:
                    break
                orig = int(top_primary_idx[int(idx)])
                c = candidates[orig]
                beams.append(c)
            if not beams:
                orig = int(top_primary_idx[int(order[0])])
                beams = [candidates[orig]]

            if step in budget_set:
                best = beams[0]
                budget_snapshots[step][parent_idx] = {
                    "parent_index": parent_idx,
                    "target_index": target,
                    "budget": step,
                    "method": "safeedit_consensus",
                    "sequence": best[0],
                    "edits": best[2].copy(),
                    "design_status": "feasible",
                }
                found_for_budget[step] = True

        for b in budgets:
            if not found_for_budget[b]:
                budget_snapshots[b][parent_idx] = {
                    "parent_index": parent_idx,
                    "target_index": target,
                    "budget": b,
                    "method": "safeedit_consensus",
                    "sequence": parent_seq,
                    "edits": [],
                    "design_status": "infeasible",
                }

    for b in budgets:
        for parent_idx in range(len(parent_seqs)):
            if parent_idx in budget_snapshots[b]:
                snapshots.append(budget_snapshots[b][parent_idx])
    return snapshots


def greedy_paths_frozen(
    parents: pd.DataFrame,
    target: int,
    primary: PrimaryPredictor,
    budgets: tuple[int, ...],
) -> list[dict[str, object]]:
    sequences = parents["sequence"].astype(str).tolist()
    used_positions = [set() for _ in sequences]
    edit_histories: list[list[tuple[int, str, str]]] = [[] for _ in sequences]
    snapshots: list[dict[str, object]] = []
    budget_set = set(budgets)
    last_seqs = sequences[:]
    last_edits = [[] for _ in sequences]

    for step in range(1, max(budgets) + 1):
        all_candidates: list[str] = []
        all_edits: list[tuple[int, str, str]] = []
        slices: list[tuple[int, int]] = []
        any_valid = False
        for sequence, used in zip(sequences, used_positions, strict=True):
            cands, edits = enumerate_substitutions(sequence, used)
            start = len(all_candidates)
            if cands:
                any_valid = True
            all_candidates.extend(cands)
            all_edits.extend(edits)
            slices.append((start, len(all_candidates)))
        if not any_valid or not all_candidates:
            for parent_index in range(len(sequences)):
                if step in budget_set:
                    snapshots.append(
                        {
                            "parent_index": parent_index,
                            "target_index": target,
                            "budget": step,
                            "method": "greedy_malinois",
                            "sequence": last_seqs[parent_index],
                            "edits": last_edits[parent_index].copy(),
                            "design_status": "infeasible",
                        }
                    )
            continue
        predictions, _ = primary.predict(all_candidates)
        objectives = specificity_score(predictions, target)
        for parent_index, (start, stop) in enumerate(slices):
            if start >= stop:
                sequences[parent_index] = last_seqs[parent_index]
                edit_histories[parent_index] = last_edits[parent_index].copy()
                continue
            local_index = int(np.argmax(objectives[start:stop]))
            selected_index = start + local_index
            sequences[parent_index] = all_candidates[selected_index]
            edit = all_edits[selected_index]
            used_positions[parent_index].add(edit[0])
            edit_histories[parent_index].append(edit)
        if step in budget_set:
            for parent_index, sequence in enumerate(sequences):
                snapshots.append(
                    {
                        "parent_index": parent_index,
                        "target_index": target,
                        "budget": step,
                        "method": "greedy_malinois",
                        "sequence": sequence,
                        "edits": edit_histories[parent_index].copy(),
                        "design_status": "feasible",
                    }
                )
            last_seqs = sequences[:]
            last_edits = [h.copy() for h in edit_histories]
    return snapshots


def random_paths_frozen(
    parents: pd.DataFrame,
    target: int,
    budgets: tuple[int, ...],
    seed: int,
) -> list[dict[str, object]]:
    snapshots = []
    for parent_index, sequence in enumerate(parents["sequence"].astype(str)):
        path = random_edit_path(
            sequence, budgets, seed=seed + target * 100_000 + parent_index + 40_000
        )
        for budget in budgets:
            if budget in path:
                edited, edits = path[budget]
                snapshots.append(
                    {
                        "parent_index": parent_index,
                        "target_index": target,
                        "budget": budget,
                        "method": "random_matched",
                        "sequence": edited,
                        "edits": edits,
                        "design_status": "feasible",
                    }
                )
            else:
                snapshots.append(
                    {
                        "parent_index": parent_index,
                        "target_index": target,
                        "budget": budget,
                        "method": "random_matched",
                        "sequence": sequence,
                        "edits": [],
                        "design_status": "infeasible",
                    }
                )
    return snapshots


def review_records_g4_frozen(
    records: list[dict[str, object]],
    parents: pd.DataFrame,
    primary: PrimaryPredictor,
    reviewer: CompactEnsembleReviewer,
    log_probabilities: np.ndarray,
) -> list[dict[str, object]]:
    parent_sequences = parents["sequence"].astype(str).tolist()
    reviewed = []
    feasible_records = [r for r in records if r.get("design_status", "feasible") == "feasible"]
    feasible_indices = [i for i, r in enumerate(records) if r.get("design_status", "feasible") == "feasible"]

    if feasible_records:
        candidate_sequences = [str(record["sequence"]) for record in feasible_records]
        parent_primary, parent_strand = primary.predict(parent_sequences)
        candidate_primary, candidate_strand = primary.predict(candidate_sequences)
        parent_review, parent_uncertainty = reviewer.predict(parent_sequences)
        candidate_review, candidate_uncertainty = reviewer.predict(candidate_sequences)
        parent_naturalness = np.asarray(
            [
                sequence_kmer_log_likelihood(sequence, log_probabilities)
                for sequence in parent_sequences
            ]
        )

    for row_idx, record in enumerate(records):
        parent_index = int(record["parent_index"])
        target = int(record["target_index"])
        parent_id = str(parents.iloc[parent_index]["IDs"])
        base_result = {
            **record,
            "parent_id": parent_id,
            "target_cell": CELL_NAMES[target],
        }
        if record.get("design_status") == "infeasible":
            base_result.update(
                {
                    "parent_sequence": parent_sequences[parent_index] if parent_index < len(parent_sequences) else "",
                    "edit_string": "",
                    "hamming_distance": 0,
                    "primary_parent_target": 0.0,
                    "primary_candidate_target": 0.0,
                    "primary_target_gain": 0.0,
                    "primary_parent_margin": 0.0,
                    "primary_candidate_margin": 0.0,
                    "primary_margin_gain": -1e9,
                    "primary_strand_disagreement": 1e9,
                    "reviewer_parent_margin": 0.0,
                    "reviewer_candidate_margin": 0.0,
                    "reviewer_margin_gain": -1e9,
                    "reviewer_uncertainty": 1e9,
                    "reviewer_parent_uncertainty": 0.0,
                    "naturalness_delta": -1e9,
                    "absolute_gc_delta": 1e9,
                    "max_homopolymer": 100,
                }
            )
            reviewed.append(base_result)
            continue

        feasible_pos = feasible_indices.index(row_idx) if row_idx in feasible_indices else -1
        if feasible_pos < 0:
            reviewed.append(base_result)
            continue

        sequence = str(record["sequence"])
        parent_sequence = parent_sequences[parent_index]
        parent_primary_margin = float(
            specificity_score(parent_primary[parent_index : parent_index + 1], target)[0]
        )
        candidate_primary_margin = float(
            specificity_score(
                candidate_primary[feasible_pos : feasible_pos + 1], target
            )[0]
        )
        parent_review_margin = float(
            specificity_score(parent_review[parent_index : parent_index + 1], target)[0]
        )
        candidate_review_margin = float(
            specificity_score(
                candidate_review[feasible_pos : feasible_pos + 1], target
            )[0]
        )
        result = {
            **base_result,
            "parent_sequence": parent_sequence,
            "edit_string": ";".join(
                f"{position + 1}{reference}>{alternate}"
                for position, reference, alternate in record["edits"]
            ),
            "hamming_distance": sum(
                left != right
                for left, right in zip(parent_sequence, sequence, strict=True)
            ),
            "primary_parent_target": float(parent_primary[parent_index, target]),
            "primary_candidate_target": float(candidate_primary[feasible_pos, target]),
            "primary_target_gain": float(
                candidate_primary[feasible_pos, target] - parent_primary[parent_index, target]
            ),
            "primary_parent_margin": parent_primary_margin,
            "primary_candidate_margin": candidate_primary_margin,
            "primary_margin_gain": candidate_primary_margin - parent_primary_margin,
            "primary_strand_disagreement": float(candidate_strand[feasible_pos]),
            "reviewer_parent_margin": parent_review_margin,
            "reviewer_candidate_margin": candidate_review_margin,
            "reviewer_margin_gain": candidate_review_margin - parent_review_margin,
            "reviewer_uncertainty": float(candidate_uncertainty[feasible_pos]),
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


def serialize_rows_frozen(rows: list[dict[str, object]]) -> pd.DataFrame:
    serializable = []
    for row in rows:
        serializable.append({key: value for key, value in row.items() if key != "edits"})
    return pd.DataFrame(serializable)


def load_frozen_parents(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        table = pd.read_csv(handle, sep="\t")
    required = {"source_row", "IDs", "chr", "split", "sequence"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"FROZEN_PARENTS missing columns: {sorted(missing)}")
    if len(table) != N_PARENTS:
        raise RuntimeError(f"Expected {N_PARENTS} frozen parents, got {len(table)}")
    return table


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available() and args.device == "cuda":
        print("WARNING: CUDA not available, falling back to CPU", flush=True)
        args.device = "cpu"

    frozen_parents = load_frozen_parents(args.frozen_parents)
    print(f"Loaded {len(frozen_parents)} frozen parents", flush=True)

    pilot_table = load_pilot(args.pilot)
    training_sequences = pilot_table.loc[
        pilot_table["split"] == "train", "sequence"
    ].astype(str).tolist()
    log_probabilities = fit_kmer_log_probabilities(training_sequences, k=6)

    val_parents = deterministic_parents(pilot_table, "validation", 24)
    print(f"Loaded {len(val_parents)} validation parents for calibration", flush=True)

    primary = PrimaryPredictor(args.malinois_checkpoint, args.device, args.batch_size)
    reviewer = CompactEnsembleReviewer(
        args.reviewer_checkpoints,
        args.reviewer_report,
        args.device,
        args.reviewer_batch_size,
    )

    val_random = []
    for target in range(3):
        cn = CELL_NAMES[target]
        print(f"[Calibration] Generating random edits for {cn}", flush=True)
        val_random.extend(
            random_paths_frozen(val_parents, target, BUDGETS, seed=args.seed + 10_000)
        )
    val_reviewed = review_records_g4_frozen(
        val_random, val_parents, primary, reviewer, log_probabilities
    )
    val_feasible = [r for r in val_reviewed if r.get("design_status") == "feasible"]
    thresholds = calibration_thresholds_frozen(val_feasible, BUDGETS)
    print(f"Calibration thresholds computed (frozen, no *1.5 relaxation)", flush=True)

    test_records: list[dict[str, object]] = []
    for target in range(3):
        cn = CELL_NAMES[target]
        print(f"[G4-frozen] === target {cn} ({target+1}/3) greedy ===", flush=True)
        test_records.append(
            greedy_paths_frozen(frozen_parents, target, primary, BUDGETS)
        )
        print(f"[G4-frozen] === target {cn} random ===", flush=True)
        test_records.append(
            random_paths_frozen(
                frozen_parents, target, BUDGETS, seed=args.seed + 20_000
            )
        )
        print(f"[G4-frozen] === target {cn} safeedit consensus beam search ===", flush=True)
        test_records.append(
            safeedit_consensus_paths_frozen(
                frozen_parents,
                target,
                primary,
                reviewer,
                log_probabilities,
                thresholds,
                BUDGETS,
                beam_width=args.beam_width,
                seed=args.seed,
                cell_name=cn,
            )
        )

    flat: list[dict[str, object]] = []
    for recs in test_records:
        flat.extend(recs)

    print(f"Total raw records before review: {len(flat)}", flush=True)
    test_reviewed = review_records_g4_frozen(
        flat, frozen_parents, primary, reviewer, log_probabilities
    )
    apply_audit_frozen(test_reviewed, thresholds)

    expected_per_method = N_PARENTS * 3 * len(BUDGETS)
    actual_counts = Counter(r["method"] for r in test_reviewed)
    print(f"Record counts by method: {dict(actual_counts)}", flush=True)

    key_set = set()
    for r in test_reviewed:
        key = (r["parent_id"], r["target_cell"], r["budget"], r["method"])
        if key in key_set:
            raise RuntimeError(f"Duplicate key: {key}")
        key_set.add(key)
    expected_keys = set()
    for pid in frozen_parents["IDs"]:
        for target in CELL_NAMES:
            for budget in BUDGETS:
                for method in ["random_matched", "greedy_malinois", "safeedit_consensus"]:
                    expected_keys.add((str(pid), target, budget, method))
    missing = expected_keys - key_set
    extra = key_set - expected_keys
    if missing:
        raise RuntimeError(f"Missing {len(missing)} keys, e.g.: {list(missing)[:5]}")
    if extra:
        raise RuntimeError(f"Extra {len(extra)} keys, e.g.: {list(extra)[:5]}")
    if len(test_reviewed) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_ROWS} rows, got {len(test_reviewed)}"
        )
    print(f"Row count verification passed: {len(test_reviewed)} == {EXPECTED_ROWS}", flush=True)

    output_table = serialize_rows_frozen(test_reviewed)
    args.candidates.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.candidates.with_name(args.candidates.name + ".part")
    output_table.to_csv(tmp, sep="\t", index=False, compression="gzip")
    with pd.read_csv(tmp, sep="\t", compression="gzip", chunksize=1000) as chunks:
        written = sum(len(c) for c in chunks)
    if written != len(test_reviewed):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"candidate row-count verification failed: {written} != {len(test_reviewed)}")
    tmp.replace(args.candidates)

    failure_counts = Counter()
    for row in test_reviewed:
        for f in str(row.get("failed_checks", "")).split(";"):
            if f:
                failure_counts[f] += 1

    feasible_count = sum(1 for r in test_reviewed if r.get("design_status") == "feasible")
    infeasible_count = sum(1 for r in test_reviewed if r.get("design_status") == "infeasible")
    accepted_count = sum(1 for r in test_reviewed if r.get("accepted"))

    summary = []
    methods = ["random_matched", "greedy_malinois", "safeedit_consensus"]
    keys = sorted(
        {(r["method"], r["target_cell"], r["budget"]) for r in test_reviewed},
        key=lambda x: (
            methods.index(x[0]) if x[0] in methods else 99,
            x[1],
            x[2],
        ),
    )
    for method, target, budget in keys:
        group = [
            r
            for r in test_reviewed
            if r["method"] == method
            and r["target_cell"] == target
            and r["budget"] == budget
        ]
        feasible_group = [r for r in group if r.get("design_status") == "feasible"]
        accepted = [r for r in group if r["accepted"]]
        constraint_satisfaction = {}
        check_names = [
            "primary_target_nonnegative",
            "primary_margin_positive",
            "reviewer_transfer_positive",
            "strand_in_domain",
            "reviewer_uncertainty_in_domain",
            "naturalness_in_domain",
            "gc_in_domain",
            "homopolymer_safe",
            "exact_edit_budget",
        ]
        for cn in check_names:
            field = f"check_{cn}"
            if field in group[0]:
                vals = [float(r.get(field, False)) for r in group]
                constraint_satisfaction[cn] = float(np.mean(vals))
        summary.append(
            {
                "method": method,
                "target_cell": target,
                "budget": budget,
                "n_total": len(group),
                "n_feasible": len(feasible_group),
                "n_infeasible": len(group) - len(feasible_group),
                "accepted_n": len(accepted),
                "accepted_fraction": len(accepted) / len(group),
                "mean_primary_margin_gain": float(
                    np.mean([r["primary_margin_gain"] for r in feasible_group])
                )
                if feasible_group
                else None,
                "mean_reviewer_margin_gain": float(
                    np.mean([r["reviewer_margin_gain"] for r in feasible_group])
                )
                if feasible_group
                else None,
                "constraint_satisfaction_rates": constraint_satisfaction,
            }
        )

    report: dict[str, object] = {
        "purpose": "FROZEN G4 confirmation benchmark (R2 remediation)",
        "frozen": True,
        "test_labels_used": False,
        "n_parents": N_PARENTS,
        "parent_source": str(args.frozen_parents),
        "targets": list(CELL_NAMES),
        "budgets": list(BUDGETS),
        "methods": methods,
        "expected_rows": EXPECTED_ROWS,
        "actual_rows": len(test_reviewed),
        "feasible_rows": feasible_count,
        "infeasible_rows": infeasible_count,
        "consensus_beam_width": args.beam_width,
        "seed": args.seed,
        "bootstrap_iterations": BOOTSTRAP_ITER,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "threshold_relaxation": 1.0,
        "mcnemar_pvalue_multiplier": 1.0,
        "primary_objective": "target prediction minus maximum off-target prediction",
        "calibration_source": "validation random edits only (frozen thresholds, no relaxation)",
        "audit_thresholds_by_budget": thresholds,
        "summary": summary,
        "paired_bootstrap": paired_bootstrap_g4(test_reviewed),
        "failed_check_counts": dict(sorted(failure_counts.items())),
        "accepted_total": accepted_count,
        "candidate_table": str(args.candidates),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-parents", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("malinois_checkpoint", type=Path)
    parser.add_argument("--reviewer-checkpoints", nargs=3, type=Path, required=True)
    parser.add_argument("--reviewer-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--beam-width", type=int, default=BEAM_WIDTH)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--reviewer-batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
