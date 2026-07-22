"""G8: Run 4 editing methods on 600 new parents.

Methods:
  - random_matched (3 replicates, seeds 20260801-20260803)
  - greedy_malinois
  - primary_beam (beam search with Malinois only, no reviewer)
  - safeedit_consensus (Malinois + G7 reviewer ensemble)

Output: results/g8_candidates_presealed.tsv.gz
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .cnn_model import encode_sequences, MAX_SEQUENCE_LENGTH
from .edit_benchmark import (
    BASES,
    BUDGETS,
    CELL_NAMES,
    PrimaryPredictor,
    enumerate_substitutions,
    fit_kmer_log_probabilities,
    gc_fraction,
    max_homopolymer,
    random_edit_path,
    sequence_kmer_log_likelihood,
    specificity_score,
)
from .g0_audit import LABEL_COLUMNS, SE_COLUMNS
from .g7_models import ARCHITECTURES, load_g7_model
from .g7_train import load_full_table
from .sequence import normalize_sequence


BEAM_WIDTH = 24
PREFILTER_K = min(BEAM_WIDTH * 6, 96)
RANDOM_SEEDS = [20260801, 20260802, 20260803]
REVIEWER_ARCH = "ResidualDilatedCNN"
REVIEWER_SEEDS = [20260721, 20260722, 20260723]


class G7ReviewerEnsemble:
    """3-member ResidualDilatedCNN ensemble reviewer trained on full data."""

    def __init__(self, checkpoint_dir: Path, device: str, batch_size: int = 256):
        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.members = []
        reference_mean = None
        reference_std = None
        for seed in REVIEWER_SEEDS:
            ckpt_path = checkpoint_dir / f"{REVIEWER_ARCH}_seed_{seed}.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"reviewer checkpoint: {ckpt_path}")
            model, ckpt = load_g7_model(REVIEWER_ARCH, ckpt_path, device=device)
            self.members.append(model)
            label_mean = ckpt["label_mean"].numpy()
            label_std = ckpt["label_std"].numpy()
            if reference_mean is None:
                reference_mean = label_mean
                reference_std = label_std
            elif not (np.allclose(reference_mean, label_mean) and np.allclose(reference_std, label_std)):
                raise RuntimeError("reviewer checkpoints use inconsistent normalization")
            variance_scale = ckpt["variance_scale"].numpy()
        if len(self.members) != 3:
            raise RuntimeError(f"expected 3 reviewer members, got {len(self.members)}")
        self.label_mean = np.asarray(reference_mean)
        self.label_std = np.asarray(reference_std)
        self.variance_scale = np.asarray(variance_scale)

    def predict(self, sequences: list[str]) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        member_means = []
        member_variances = []
        for model in self.members:
            means = []
            variances = []
            with torch.inference_mode():
                for start in range(0, len(sequences), self.batch_size):
                    batch_seqs = sequences[start:start + self.batch_size]
                    inputs = encode_sequences(batch_seqs).to(self.device)
                    mean, raw_lv = model(inputs)
                    means.append(mean.cpu().numpy())
                    variances.append(torch.nn.functional.softplus(raw_lv).cpu().numpy())
            mean_std = np.concatenate(means)
            var_std = np.concatenate(variances)
            member_means.append(mean_std * self.label_std + self.label_mean)
            member_variances.append(var_std * (self.label_std ** 2))
        stacked_means = np.stack(member_means)
        stacked_variances = np.stack(member_variances)
        ensemble_mean = np.mean(stacked_means, axis=0)
        predictive_variance = (
            np.mean(stacked_variances, axis=0) + np.var(stacked_means, axis=0)
        ) * self.variance_scale
        mean_uncertainty = np.mean(np.sqrt(np.maximum(predictive_variance, 1e-12)), axis=1)
        return ensemble_mean, mean_uncertainty


def beam_search(
    parents: pd.DataFrame,
    target: int,
    primary: PrimaryPredictor,
    reviewer: G7ReviewerEnsemble | None,
    log_probabilities: np.ndarray,
    thresholds_by_budget: dict[int, dict[str, float]],
    budgets: tuple[int, ...] = BUDGETS,
    beam_width: int = BEAM_WIDTH,
    seed: int = 0,
    cell_name: str = "",
    use_reviewer: bool = True,
    ablation_config: str = "full",
) -> list[dict[str, object]]:
    """Beam search for SafeEdit or primary_beam (no reviewer).

    ablation_config controls which components are disabled:
      - full: all components active
      - no_reviewer_score: skip reviewer reranking
      - no_uncertainty_penalty: drop uncertainty penalty
      - no_strand_penalty: drop strand disagreement penalty
      - no_naturalness_rerank: skip naturalness filter
      - no_sequence_prefilter: skip homopolymer/GC prefilter
    """
    parent_seqs = parents["sequence"].astype(str).tolist()
    snapshots: list[dict[str, object]] = []
    parent_nat = np.array([
        sequence_kmer_log_likelihood(normalize_sequence(s), log_probabilities)
        for s in parent_seqs
    ])
    parent_gc = np.array([gc_fraction(s) for s in parent_seqs])
    prefilter_k = min(beam_width * 6, 96)
    budget_set = set(budgets)
    budget_snapshots: dict[int, dict[int, dict[str, object]]] = {b: {} for b in budgets}

    effective_use_reviewer = use_reviewer and ablation_config != "no_reviewer_score"

    for parent_idx, parent_seq in enumerate(parent_seqs):
        if parent_idx % 16 == 0:
            print(
                f"[G8 {cell_name} {'safeedit' if effective_use_reviewer else 'primary_beam'}] "
                f"parent {parent_idx+1}/{len(parent_seqs)}",
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
                        new_seq = seq[:pos] + alt + seq[pos + 1:]
                        if new_seq in seen:
                            continue
                        seen.add(new_seq)
                        if ablation_config != "no_sequence_prefilter":
                            hp = max_homopolymer(new_seq)
                            if hp > 8:
                                continue
                            gcd = abs(gc_fraction(new_seq) - parent_gc[parent_idx])
                            if gcd > 0.12:
                                continue
                        new_used = used | {pos}
                        new_edits = edits + [(pos, ref, alt)]
                        candidates.append((new_seq, new_used, new_edits))

            if not candidates:
                for b in budgets:
                    if b >= step and not found_for_budget[b]:
                        budget_snapshots[b][parent_idx] = {
                            "parent_index": parent_idx,
                            "target_index": target,
                            "budget": b,
                            "method": "safeedit_consensus" if effective_use_reviewer else "primary_beam",
                            "sequence": parent_seq,
                            "edits": [],
                            "design_status": "infeasible",
                        }
                        found_for_budget[b] = True
                break

            cand_seqs = [c[0] for c in candidates]
            t0 = time.time()
            p_pred, p_strand = primary.predict(cand_seqs)
            margins_p = specificity_score(p_pred, target)
            primary_scores = margins_p.astype(float) - 0.05 * p_strand.astype(float)
            n_prefilter = min(prefilter_k, len(candidates))
            top_primary_idx = np.argsort(-primary_scores)[:n_prefilter]

            if effective_use_reviewer:
                rerank_seqs = [cand_seqs[int(i)] for i in top_primary_idx]
                t1 = time.time()
                r_pred, r_uncert = reviewer.predict(rerank_seqs)
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
                    if ablation_config == "no_naturalness_rerank":
                        passes = True
                    else:
                        passes = (
                            sd <= thresh["strand_disagreement_max"]
                            and uc <= thresh["reviewer_uncertainty_max"]
                            and nd >= thresh["naturalness_delta_min"]
                            and gd <= thresh["absolute_gc_delta_max"]
                            and hp <= 7
                        )
                    penalty_sd = 0.0 if ablation_config == "no_strand_penalty" else 0.1 * sd
                    penalty_uc = 0.0 if ablation_config == "no_uncertainty_penalty" else 0.05 * uc
                    if mp > 0 and mr > 0 and passes:
                        scores[j] = mp + 0.3 * mr - penalty_sd - penalty_uc
                    elif mp > 0 and passes:
                        scores[j] = mp * 0.4 + 0.1 * max(mr, 0)
                    elif mp > 0:
                        scores[j] = mp * 0.15
                    else:
                        scores[j] = -1e6 + mp
            else:
                scores = primary_scores[top_primary_idx]
                rerank_seqs = [cand_seqs[int(i)] for i in top_primary_idx]

            order = np.argsort(-scores)
            beams = []
            for idx in order[:beam_width]:
                if scores[idx] < -1e8 and len(beams) > 0:
                    break
                orig = int(top_primary_idx[int(idx)])
                beams.append(candidates[orig])
            if not beams:
                orig = int(top_primary_idx[int(order[0])])
                beams = [candidates[orig]]

            if step in budget_set:
                best = beams[0]
                budget_snapshots[step][parent_idx] = {
                    "parent_index": parent_idx,
                    "target_index": target,
                    "budget": step,
                    "method": "safeedit_consensus" if effective_use_reviewer else "primary_beam",
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
                    "method": "safeedit_consensus" if effective_use_reviewer else "primary_beam",
                    "sequence": parent_seq,
                    "edits": [],
                    "design_status": "infeasible",
                }

    for b in budgets:
        for parent_idx in range(len(parent_seqs)):
            if parent_idx in budget_snapshots[b]:
                snapshots.append(budget_snapshots[b][parent_idx])
    return snapshots


def greedy_paths(
    parents: pd.DataFrame,
    target: int,
    primary: PrimaryPredictor,
    budgets: tuple[int, ...] = BUDGETS,
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
                    snapshots.append({
                        "parent_index": parent_index,
                        "target_index": target,
                        "budget": step,
                        "method": "greedy_malinois",
                        "sequence": last_seqs[parent_index],
                        "edits": last_edits[parent_index].copy(),
                        "design_status": "infeasible",
                    })
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
                snapshots.append({
                    "parent_index": parent_index,
                    "target_index": target,
                    "budget": step,
                    "method": "greedy_malinois",
                    "sequence": sequence,
                    "edits": edit_histories[parent_index].copy(),
                    "design_status": "feasible",
                })
            last_seqs = sequences[:]
            last_edits = [h.copy() for h in edit_histories]
    return snapshots


def random_paths(
    parents: pd.DataFrame,
    target: int,
    budgets: tuple[int, ...],
    seed: int,
    replicate: int = 0,
) -> list[dict[str, object]]:
    snapshots = []
    for parent_index, sequence in enumerate(parents["sequence"].astype(str)):
        path = random_edit_path(
            sequence, budgets, seed=seed + target * 100_000 + parent_index + 40_000
        )
        for budget in budgets:
            if budget in path:
                edited, edits = path[budget]
                snapshots.append({
                    "parent_index": parent_index,
                    "target_index": target,
                    "budget": budget,
                    "method": "random_matched",
                    "replicate": replicate,
                    "sequence": edited,
                    "edits": edits,
                    "design_status": "feasible",
                })
            else:
                snapshots.append({
                    "parent_index": parent_index,
                    "target_index": target,
                    "budget": budget,
                    "method": "random_matched",
                    "replicate": replicate,
                    "sequence": sequence,
                    "edits": [],
                    "design_status": "infeasible",
                })
    return snapshots


def review_records(
    records: list[dict[str, object]],
    parents: pd.DataFrame,
    primary: PrimaryPredictor,
    reviewer: G7ReviewerEnsemble | None,
    log_probabilities: np.ndarray,
) -> list[dict[str, object]]:
    parent_sequences = parents["sequence"].astype(str).tolist()
    parent_ids = parents["parent_id"].astype(str).tolist() if "parent_id" in parents.columns else None

    feasible_records = [r for r in records if r.get("design_status", "feasible") == "feasible"]
    feasible_indices = [i for i, r in enumerate(records) if r.get("design_status", "feasible") == "feasible"]

    parent_primary, parent_strand = primary.predict(parent_sequences)
    parent_naturalness = np.asarray([
        sequence_kmer_log_likelihood(s, log_probabilities) for s in parent_sequences
    ])

    if reviewer is not None:
        parent_review, parent_uncertainty = reviewer.predict(parent_sequences)
    else:
        parent_review = np.zeros((len(parent_sequences), 3))
        parent_uncertainty = np.zeros(len(parent_sequences))

    if feasible_records:
        candidate_sequences = [str(record["sequence"]) for record in feasible_records]
        candidate_primary, candidate_strand = primary.predict(candidate_sequences)
        if reviewer is not None:
            candidate_review, candidate_uncertainty = reviewer.predict(candidate_sequences)
        else:
            candidate_review = np.zeros((len(candidate_sequences), 3))
            candidate_uncertainty = np.zeros(len(candidate_sequences))

    reviewed = []
    for row_idx, record in enumerate(records):
        parent_index = int(record["parent_index"])
        target = int(record["target_index"])
        if parent_ids is not None:
            pid = parent_ids[parent_index]
        else:
            pid = str(parents.iloc[parent_index].get("IDs", parent_index))
        base_result = {
            **record,
            "parent_id": pid,
            "target_cell": CELL_NAMES[target],
            "replicate": record.get("replicate", 0),
        }
        if record.get("design_status") == "infeasible":
            base_result.update({
                "parent_sequence": parent_sequences[parent_index] if parent_index < len(parent_sequences) else "",
                "edit_string": "",
                "hamming_distance": 0,
                "primary_target_gain": 0.0,
                "primary_margin_gain": -1e9,
                "reviewer_margin_gain": -1e9 if reviewer is not None else 0.0,
                "reviewer_uncertainty": 1e9 if reviewer is not None else 0.0,
                "strand_disagreement": 1e9,
                "naturalness_delta": -1e9,
                "absolute_gc_delta": 1e9,
                "max_homopolymer": 100,
                "runtime_seconds": 0.0,
                "model_calls": 0,
            })
            reviewed.append(base_result)
            continue

        feasible_pos = feasible_indices.index(row_idx) if row_idx in feasible_indices else -1
        if feasible_pos < 0:
            reviewed.append(base_result)
            continue

        sequence = str(record["sequence"])
        parent_sequence = parent_sequences[parent_index]
        parent_primary_margin = float(specificity_score(parent_primary[parent_index:parent_index+1], target)[0])
        candidate_primary_margin = float(specificity_score(candidate_primary[feasible_pos:feasible_pos+1], target)[0])
        if reviewer is not None:
            parent_review_margin = float(specificity_score(parent_review[parent_index:parent_index+1], target)[0])
            candidate_review_margin = float(specificity_score(candidate_review[feasible_pos:feasible_pos+1], target)[0])
        else:
            parent_review_margin = 0.0
            candidate_review_margin = 0.0

        result = {
            **base_result,
            "parent_sequence": parent_sequence,
            "edit_string": ";".join(
                f"{pos+1}{ref}>{alt}" for pos, ref, alt in record["edits"]
            ),
            "hamming_distance": sum(a != b for a, b in zip(parent_sequence, sequence, strict=True)),
            "primary_target_gain": float(candidate_primary[feasible_pos, target] - parent_primary[parent_index, target]),
            "primary_margin_gain": candidate_primary_margin - parent_primary_margin,
            "reviewer_margin_gain": (candidate_review_margin - parent_review_margin) if reviewer is not None else 0.0,
            "reviewer_uncertainty": float(candidate_uncertainty[feasible_pos]) if reviewer is not None else 0.0,
            "strand_disagreement": float(candidate_strand[feasible_pos]),
            "naturalness_delta": sequence_kmer_log_likelihood(sequence, log_probabilities) - parent_naturalness[parent_index],
            "absolute_gc_delta": abs(gc_fraction(sequence) - gc_fraction(parent_sequence)),
            "max_homopolymer": max_homopolymer(sequence),
            "runtime_seconds": float(record.get("runtime_seconds", 0.0)),
            "model_calls": int(record.get("model_calls", 0)),
        }
        reviewed.append(result)
    return reviewed


def calibration_thresholds(
    reviewed_validation: list[dict[str, object]], budgets: tuple[int, ...]
) -> dict[int, dict[str, float]]:
    thresholds = {}
    for budget in budgets:
        rows = [row for row in reviewed_validation if int(row["budget"]) == budget]
        if not rows:
            thresholds[budget] = {
                "strand_disagreement_max": 1.0,
                "reviewer_uncertainty_max": 10.0,
                "naturalness_delta_min": -10.0,
                "absolute_gc_delta_max": 0.5,
            }
            continue
        thresholds[budget] = {
            "strand_disagreement_max": float(np.quantile([row["strand_disagreement"] for row in rows], 0.95)),
            "reviewer_uncertainty_max": float(np.quantile([row["reviewer_uncertainty"] for row in rows], 0.95)),
            "naturalness_delta_min": float(np.quantile([row["naturalness_delta"] for row in rows], 0.05)),
            "absolute_gc_delta_max": float(np.quantile([row["absolute_gc_delta"] for row in rows], 0.95)),
        }
    return thresholds


def main():
    parser = argparse.ArgumentParser(description="G8 methods on 600 parents")
    parser.add_argument("table_s2", type=Path)
    parser.add_argument("--g8-parents", type=Path, required=True)
    parser.add_argument("--malinois-checkpoint", type=Path, required=True)
    parser.add_argument("--reviewer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reviewer-batch-size", type=int, default=256)
    parser.add_argument("--target-cell", type=str, default=None,
                        help="Run only this cell (K562/HepG2/SKNSH). If None, run all 3.")
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}", flush=True)

    if args.target_cell is not None:
        target_map = {name: i for i, name in enumerate(CELL_NAMES)}
        if args.target_cell not in target_map:
            raise ValueError(f"Unknown target_cell: {args.target_cell}. Must be one of {CELL_NAMES}")
        target_indices = [target_map[args.target_cell]]
    else:
        target_indices = list(range(3))
    print(f"Targets: {[CELL_NAMES[t] for t in target_indices]}", flush=True)

    g8_parents = pd.read_csv(args.g8_parents, sep="\t")
    g8_parents = g8_parents.rename(columns={"parent_id": "parent_id"})
    print(f"Loaded {len(g8_parents)} G8 parents", flush=True)

    table = load_full_table(args.table_s2)
    training_sequences = table.loc[table["split"] == "train", "sequence"].astype(str).tolist()
    log_probabilities = fit_kmer_log_probabilities(training_sequences, k=6)
    print(f"K-mer naturalness fitted on {len(training_sequences)} training sequences", flush=True)

    val_df = table[table["split"] == "validation"]
    val_parents = val_df[val_df["sequence"].str.len() == 200].copy()
    val_parents = val_parents.head(min(96, len(val_parents)))
    val_parents = val_parents.rename(columns={"IDs": "parent_id"})
    if "parent_id" not in val_parents.columns:
        val_parents["parent_id"] = val_parents["IDs"]
    print(f"Validation parents for calibration: {len(val_parents)}", flush=True)

    primary = PrimaryPredictor(args.malinois_checkpoint, args.device, args.batch_size)
    reviewer = G7ReviewerEnsemble(args.reviewer_dir, args.device, args.reviewer_batch_size)
    print("Models loaded.", flush=True)

    print("\n=== Calibration ===", flush=True)
    val_random = []
    for target in range(3):
        val_random.extend(random_paths(val_parents, target, BUDGETS, seed=20260740 + 40_000))
    val_reviewed = review_records(val_random, val_parents, primary, reviewer, log_probabilities)
    val_feasible = [r for r in val_reviewed if r.get("design_status") == "feasible"]
    thresholds = calibration_thresholds(val_feasible, BUDGETS)
    print(f"Thresholds: {json.dumps(thresholds, indent=2)}", flush=True)

    all_records = []
    for target in target_indices:
        cn = CELL_NAMES[target]
        print(f"\n=== Target {cn} ===", flush=True)

        for rep_idx, rseed in enumerate(RANDOM_SEEDS):
            t0 = time.time()
            recs = random_paths(g8_parents, target, BUDGETS, seed=rseed, replicate=rep_idx + 1)
            dt = time.time() - t0
            for r in recs:
                r["runtime_seconds"] = dt / max(len(recs), 1)
                r["model_calls"] = 0
            all_records.extend(recs)
            print(f"  random replicate {rep_idx+1}: {len(recs)} records ({dt:.1f}s)", flush=True)

        t0 = time.time()
        recs = greedy_paths(g8_parents, target, primary, BUDGETS)
        dt = time.time() - t0
        for r in recs:
            r["runtime_seconds"] = dt / max(len(recs), 1)
            r["model_calls"] = max(budget for budget in BUDGETS)
        all_records.extend(recs)
        print(f"  greedy: {len(recs)} records ({dt:.1f}s)", flush=True)

        t0 = time.time()
        recs = beam_search(
            g8_parents, target, primary, None, log_probabilities,
            thresholds, use_reviewer=False, cell_name=cn,
        )
        dt = time.time() - t0
        for r in recs:
            r["runtime_seconds"] = dt / max(len(recs), 1)
            r["model_calls"] = max(budget for budget in BUDGETS) * BEAM_WIDTH
        all_records.extend(recs)
        print(f"  primary_beam: {len(recs)} records ({dt:.1f}s)", flush=True)

        t0 = time.time()
        recs = beam_search(
            g8_parents, target, primary, reviewer, log_probabilities,
            thresholds, use_reviewer=True, cell_name=cn,
        )
        dt = time.time() - t0
        for r in recs:
            r["runtime_seconds"] = dt / max(len(recs), 1)
            r["model_calls"] = max(budget for budget in BUDGETS) * BEAM_WIDTH * 2
        all_records.extend(recs)
        print(f"  safeedit_consensus: {len(recs)} records ({dt:.1f}s)", flush=True)

    print("\n=== Reviewing all records ===", flush=True)
    all_reviewed = review_records(all_records, g8_parents, primary, reviewer, log_probabilities)

    serializable = [{k: v for k, v in row.items() if k != "edits"} for row in all_reviewed]
    df = pd.DataFrame(serializable)
    print(f"Total rows: {len(df)}", flush=True)
    print(f"Methods: {df['method'].value_counts().to_dict()}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".part")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        df.to_csv(f, sep="\t", index=False)
    tmp.rename(args.output)
    print(f"Output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
