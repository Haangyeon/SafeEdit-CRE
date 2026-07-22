"""Locked G4 confirmation benchmark: untouched parents, three-method comparison."""

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
from .edit_benchmark import (
    BASES,
    BUDGETS,
    CELL_NAMES,
    CompactEnsembleReviewer,
    PrimaryPredictor,
    apply_audit,
    calibration_thresholds,
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


def safeedit_consensus_paths(
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
    """Two-stage beam search: primary coarse filter -> reviewer rerank."""
    import time
    parent_seqs = parents["sequence"].astype(str).tolist()
    snapshots: list[dict[str, object]] = []
    parent_nat = np.array([sequence_kmer_log_likelihood(normalize_sequence(s), log_probabilities) for s in parent_seqs])
    parent_gc = np.array([gc_fraction(s) for s in parent_seqs])
    prefilter_k = min(beam_width * 6, 96)

    for parent_idx, parent_seq in enumerate(parent_seqs):
        if parent_idx % 8 == 0:
            print(f"[G4 {cell_name} safeedit] parent {parent_idx+1}/{len(parent_seqs)}", flush=True)
        beams: list[tuple[str, set[int], list[tuple[int, str, str]]]] = [
            (normalize_sequence(parent_seq), set(), [])
        ]
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
                        new_seq = seq[:pos] + alt + seq[pos + 1:]
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
                continue
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
                nat_d[j] = sequence_kmer_log_likelihood(sj, log_probabilities) - parent_nat[parent_idx]
                gc_d[j] = abs(gc_fraction(sj) - parent_gc[parent_idx])
                homops[j] = max_homopolymer(sj)
            thresh = thresholds_by_budget.get(step, thresholds_by_budget[max(budgets)])
            scores = np.full(n_rerank, -1e9)
            for j in range(n_rerank):
                mp = float(margins_p[int(top_primary_idx[j])])
                mr = float(margins_r[j])
                sd = float(p_strand[int(top_primary_idx[j])])
                uc = float(r_uncert[j])
                nd = float(nat_d[j])
                gd = float(gc_d[j])
                hp = int(homops[j])
                passes = (
                    sd <= thresh["strand_disagreement_max"] * 1.5
                    and uc <= thresh["reviewer_uncertainty_max"] * 1.5
                    and nd >= thresh["naturalness_delta_min"] * 1.5
                    and gd <= thresh["absolute_gc_delta_max"] * 1.5
                    and hp <= 7
                )
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
            if step in budgets:
                best = beams[0]
                snapshots.append({
                    "parent_index": parent_idx,
                    "target_index": target,
                    "budget": step,
                    "method": "safeedit_consensus",
                    "sequence": best[0],
                    "edits": best[2].copy(),
                })
    return snapshots


def greedy_paths(
    parents: pd.DataFrame,
    target: int,
    primary: PrimaryPredictor,
    budgets: tuple[int, ...],
) -> list[dict[str, object]]:
    from .edit_benchmark import enumerate_substitutions
    sequences = parents["sequence"].astype(str).tolist()
    used_positions = [set() for _ in sequences]
    edit_histories: list[list[tuple[int, str, str]]] = [[] for _ in sequences]
    snapshots: list[dict[str, object]] = []
    for step in range(1, max(budgets) + 1):
        all_candidates: list[str] = []
        all_edits: list[tuple[int, str, str]] = []
        slices: list[tuple[int, int]] = []
        for sequence, used in zip(sequences, used_positions, strict=True):
            cands, edits = enumerate_substitutions(sequence, used)
            start = len(all_candidates)
            all_candidates.extend(cands)
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
                snapshots.append({
                    "parent_index": parent_index,
                    "target_index": target,
                    "budget": step,
                    "method": "greedy_malinois",
                    "sequence": sequence,
                    "edits": edit_histories[parent_index].copy(),
                })
    return snapshots


def random_paths(
    parents: pd.DataFrame,
    target: int,
    budgets: tuple[int, ...],
    seed: int,
) -> list[dict[str, object]]:
    snapshots = []
    for parent_index, sequence in enumerate(parents["sequence"].astype(str)):
        path = random_edit_path(sequence, budgets, seed=seed + target * 100_000 + parent_index + 40_000)
        for budget in budgets:
            edited, edits = path[budget]
            snapshots.append({
                "parent_index": parent_index,
                "target_index": target,
                "budget": budget,
                "method": "random_matched",
                "sequence": edited,
                "edits": edits,
            })
    return snapshots


def review_records_g4(
    records: list[dict[str, object]],
    parents: pd.DataFrame,
    primary: PrimaryPredictor,
    reviewer: CompactEnsembleReviewer,
    log_probabilities: np.ndarray,
) -> list[dict[str, object]]:
    return review_records(records, parents, primary, reviewer, log_probabilities)


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact McNemar test p-value (two-sided) from discordant pair counts b and c."""
    from scipy.stats import binomtest
    n = b + c
    if n == 0:
        return 1.0
    p_val = binomtest(b, n=n, p=0.5).pvalue
    return float(p_val)


def paired_bootstrap_g4(
    rows: list[dict[str, object]], repetitions: int = 100000, seed: int = 20260713
) -> list[dict[str, object]]:
    from .edit_benchmark import CELL_NAMES, BUDGETS
    rng = np.random.default_rng(seed)
    results = []
    for target_name in CELL_NAMES:
        for budget in BUDGETS:
            for method_a, method_b in [("safeedit_consensus", "greedy_malinois"), ("greedy_malinois", "random_matched"), ("safeedit_consensus", "random_matched")]:
                selected = [r for r in rows if r["target_cell"] == target_name and r["budget"] == budget and r["method"] in (method_a, method_b)]
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
                margin_diffs = np.array([float(a["primary_margin_gain"]) - float(b["primary_margin_gain"]) for a, b in pairs])
                n_clusters = len(pairs)
                cluster_indices = rng.integers(0, n_clusters, size=(repetitions, n_clusters))
                acc_samples = acc_diffs[cluster_indices].mean(axis=1)
                margin_samples = margin_diffs[cluster_indices].mean(axis=1)
                b_discord = sum(1 for a, b in pairs if a["accepted"] and not b["accepted"])
                c_discord = sum(1 for a, b in pairs if not a["accepted"] and b["accepted"])
                results.append({
                    "target_cell": target_name,
                    "budget": budget,
                    "comparison": f"{method_a}_vs_{method_b}",
                    "metric": "acceptance",
                    "mean_diff": float(np.mean(acc_diffs)),
                    "bootstrap_95_ci": [float(np.quantile(acc_samples, 0.025)), float(np.quantile(acc_samples, 0.975))],
                    "mcnemar_exact_p": mcnemar_exact_p(b_discord, c_discord),
                    "discordant_ab": b_discord,
                    "discordant_ba": c_discord,
                    "n_pairs": n_clusters,
                })
                results.append({
                    "target_cell": target_name,
                    "budget": budget,
                    "comparison": f"{method_a}_vs_{method_b}",
                    "metric": "primary_margin_gain",
                    "mean_diff": float(np.mean(margin_diffs)),
                    "bootstrap_95_ci": [float(np.quantile(margin_samples, 0.025)), float(np.quantile(margin_samples, 0.975))],
                    "n_pairs": n_clusters,
                })
    return results


def serialize_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    from .edit_benchmark import serialize_rows as _serialize
    return _serialize(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch
    if not torch.cuda.is_available() and args.device == "cuda":
        raise RuntimeError("CUDA is required by the locked G4 handoff")
    table = load_pilot(args.pilot)
    training_sequences = table.loc[table["split"] == "train", "sequence"].astype(str).tolist()
    log_probabilities = fit_kmer_log_probabilities(training_sequences, k=6)
    val_parents = deterministic_parents(table, "validation", 24)
    test_parents_all = deterministic_parents(table, "test", args.parent_offset + args.parent_count)
    test_parents = test_parents_all.iloc[args.parent_offset:args.parent_offset + args.parent_count].reset_index(drop=True)
    if len(test_parents) != args.parent_count:
        raise RuntimeError(f"insufficient test parents after offset: {len(test_parents)}")

    g3_parents_val = deterministic_parents(table, "validation", 24)
    g3_parents_test = deterministic_parents(table, "test", 24)
    g3_ids = set(g3_parents_val["IDs"]).union(set(g3_parents_test["IDs"]))
    overlap = g3_ids.intersection(set(test_parents["IDs"]))
    if overlap:
        raise RuntimeError(f"G3/G4 parent overlap detected: {overlap}")

    primary = PrimaryPredictor(args.malinois_checkpoint, args.device, args.batch_size)
    reviewer = CompactEnsembleReviewer(
        args.reviewer_checkpoints,
        args.reviewer_report,
        args.device,
        args.reviewer_batch_size,
    )

    val_random = []
    for target in range(3):
        val_random.extend(random_paths(val_parents, target, BUDGETS, seed=args.seed + 10_000))
    val_reviewed = review_records_g4(val_random, val_parents, primary, reviewer, log_probabilities)
    thresholds = calibration_thresholds(val_reviewed, BUDGETS)

    test_records = []
    for target in range(3):
        cn = CELL_NAMES[target]
        print(f"[G4] === target {cn} ({target+1}/3) greedy ===", flush=True)
        test_records.append(greedy_paths(test_parents, target, primary, BUDGETS))
        print(f"[G4] === target {cn} random ===", flush=True)
        test_records.append(random_paths(test_parents, target, BUDGETS, seed=args.seed + 20_000))
        print(f"[G4] === target {cn} safeedit consensus beam search ===", flush=True)
        test_records.append(safeedit_consensus_paths(
            test_parents, target, primary, reviewer, log_probabilities,
            thresholds, BUDGETS, beam_width=args.beam_width, seed=args.seed,
            cell_name=cn,
        ))
    flat = []
    for recs in test_records:
        flat.extend(recs)
    test_reviewed = review_records_g4(flat, test_parents, primary, reviewer, log_probabilities)
    apply_audit(test_reviewed, thresholds)

    output_table = serialize_rows(test_reviewed)
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
        for f in str(row["failed_checks"]).split(";"):
            if f:
                failure_counts[f] += 1

    summary = []
    methods = ["random_matched", "greedy_malinois", "safeedit_consensus"]
    keys = sorted({(r["method"], r["target_cell"], r["budget"]) for r in test_reviewed}, key=lambda x: (methods.index(x[0]) if x[0] in methods else 99, x[1], x[2]))
    for method, target, budget in keys:
        group = [r for r in test_reviewed if r["method"] == method and r["target_cell"] == target and r["budget"] == budget]
        accepted = [r for r in group if r["accepted"]]
        constraint_satisfaction = {}
        for cn in ("check_primary_target_nonnegative", "check_primary_margin_positive", "check_reviewer_transfer_positive", "check_strand_in_domain", "check_reviewer_uncertainty_in_domain", "check_naturalness_in_domain", "check_gc_in_domain", "check_homopolymer_safe", "check_exact_edit_budget"):
            if cn in group[0]:
                vals = [float(r.get(cn, False)) for r in group]
                constraint_satisfaction[cn.replace("check_", "")] = float(np.mean(vals))
        summary.append({
            "method": method,
            "target_cell": target,
            "budget": budget,
            "n": len(group),
            "accepted_n": len(accepted),
            "accepted_fraction": len(accepted) / len(group),
            "mean_primary_margin_gain": float(np.mean([r["primary_margin_gain"] for r in group])),
            "mean_primary_target_gain": float(np.mean([r["primary_target_gain"] for r in group])),
            "mean_reviewer_margin_gain": float(np.mean([r["reviewer_margin_gain"] for r in group])),
            "reviewer_transfer_positive_fraction": float(np.mean([float(r["reviewer_margin_gain"] > 0) for r in group])),
            "accepted_mean_primary_margin_gain": float(np.mean([r["primary_margin_gain"] for r in accepted])) if accepted else None,
            "constraint_satisfaction_rates": constraint_satisfaction,
        })

    report: dict[str, object] = {
        "purpose": "locked G4 confirmation benchmark on untouched CRE parents",
        "test_labels_used": False,
        "parent_offset": args.parent_offset,
        "parent_count_per_split": args.parent_count,
        "parent_selection": f"parents[{args.parent_offset}:{args.parent_offset+args.parent_count}] by SHA-256; no activity labels",
        "targets": list(CELL_NAMES),
        "budgets": list(BUDGETS),
        "methods": methods,
        "consensus_beam_width": args.beam_width,
        "primary_objective": "target prediction minus maximum off-target prediction",
        "calibration_source": "validation random edits only",
        "audit_thresholds_by_budget": thresholds,
        "g3_g4_parent_overlap": len(overlap),
        "test_candidate_rows": len(test_reviewed),
        "summary": summary,
        "paired_bootstrap": paired_bootstrap_g4(test_reviewed, seed=args.seed),
        "failed_check_counts": dict(sorted(failure_counts.items())),
        "accepted_total": int(sum(bool(r["accepted"]) for r in test_reviewed)),
        "candidate_table": str(args.candidates),
        "limitations": [
            "Edited sequences have predictor scores but no new experimental measurements.",
            "G4 is confirmation on untouched parents but still purely in silico.",
            "SafeEdit consensus uses soft-constrained beam search; global optimum not guaranteed.",
            "G2 CNN reviewer is a pilot model trained on 30,000 sequences.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pilot", type=Path)
    parser.add_argument("malinois_checkpoint", type=Path)
    parser.add_argument("--reviewer-checkpoints", nargs=3, type=Path, required=True)
    parser.add_argument("--reviewer-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--parent-offset", type=int, default=24)
    parser.add_argument("--parent-count", type=int, default=48)
    parser.add_argument("--beam-width", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--reviewer-batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
