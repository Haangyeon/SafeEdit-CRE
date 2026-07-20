"""G9: True ablation re-search on 180 parents.

Selects first 180 G8 parents by hash order and re-runs 6 ablation configs
plus primary_beam. Each is an independent search, not post-hoc filtering.
After candidate freezing, runs sealed evaluator inference.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .cnn_model import encode_sequences
from .edit_benchmark import (
    BUDGETS,
    CELL_NAMES,
    PrimaryPredictor,
    fit_kmer_log_probabilities,
    specificity_score,
)
from .g7_train import load_full_table
from .g8_methods import (
    G7ReviewerEnsemble,
    beam_search,
    review_records,
    calibration_thresholds,
    random_paths,
)
from .g8_sealed_inference import SealedEvaluatorEnsemble


N_ABLATION_PARENTS = 180
ABLATION_CONFIGS = [
    "safeedit_full",
    "safeedit_no_reviewer_score",
    "safeedit_no_uncertainty_penalty",
    "safeedit_no_strand_penalty",
    "safeedit_no_naturalness_rerank",
    "safeedit_no_sequence_prefilter",
    "primary_beam",
]


def run_ablation(
    config: str,
    parents: pd.DataFrame,
    target: int,
    primary: PrimaryPredictor,
    reviewer: G7ReviewerEnsemble | None,
    log_probabilities: np.ndarray,
    thresholds: dict,
    cell_name: str,
) -> list[dict[str, object]]:
    use_reviewer = config != "primary_beam"
    ablation_key = "full" if config == "safeedit_full" else config.replace("safeedit_", "")
    if config == "primary_beam":
        ablation_key = "full"
        use_reviewer = False

    method_name = "safeedit_consensus" if use_reviewer else "primary_beam"
    if config.startswith("safeedit_") and config != "safeedit_full":
        method_name = f"safeedit_{ablation_key}"

    t0 = time.time()
    recs = beam_search(
        parents, target, primary,
        reviewer if use_reviewer else None,
        log_probabilities,
        thresholds,
        use_reviewer=use_reviewer,
        cell_name=f"{cell_name}/{config}",
        ablation_config=ablation_key,
    )
    dt = time.time() - t0
    for r in recs:
        r["method"] = method_name
        r["ablation_config"] = config
        r["runtime_seconds"] = dt / max(len(recs), 1)
    print(f"  {config} ({cell_name}): {len(recs)} records ({dt:.1f}s)", flush=True)
    return recs


def main():
    parser = argparse.ArgumentParser(description="G9 true ablation re-search")
    parser.add_argument("table_s2", type=Path)
    parser.add_argument("--g8-parents", type=Path, required=True)
    parser.add_argument("--malinois-checkpoint", type=Path, required=True)
    parser.add_argument("--reviewer-dir", type=Path, required=True)
    parser.add_argument("--sealed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reviewer-batch-size", type=int, default=256)
    parser.add_argument("--target-cell", type=str, default=None,
                        help="Run only this cell (K562/HepG2/SKNSH). If None, run all 3.")
    parser.add_argument("--output-mode", choices=("full", "partial"), default="full",
                        help="partial: skip sealed inference (for parallel merge)")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated list of ablation configs to run. If None, run all 7.")
    args = parser.parse_args()

    configs_to_run = args.configs.split(",") if args.configs else list(ABLATION_CONFIGS)
    for c in configs_to_run:
        if c not in ABLATION_CONFIGS:
            raise ValueError(f"Unknown config: {c}. Must be one of {ABLATION_CONFIGS}")
    print(f"Configs to run: {configs_to_run}", flush=True)

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
    if "selection_hash" in g8_parents.columns:
        g8_parents = g8_parents.sort_values("selection_hash", kind="stable")
    ablation_parents = g8_parents.head(N_ABLATION_PARENTS).copy()
    print(f"Selected {len(ablation_parents)} ablation parents", flush=True)

    table = load_full_table(args.table_s2)
    training_sequences = table.loc[table["split"] == "train", "sequence"].astype(str).tolist()
    log_probabilities = fit_kmer_log_probabilities(training_sequences, k=6)

    val_df = table[table["split"] == "validation"]
    val_parents = val_df[val_df["sequence"].str.len() == 200].head(96).copy()
    if "IDs" in val_parents.columns:
        val_parents = val_parents.rename(columns={"IDs": "parent_id"})

    primary = PrimaryPredictor(args.malinois_checkpoint, args.device, args.batch_size)
    reviewer = G7ReviewerEnsemble(args.reviewer_dir, args.device, args.reviewer_batch_size)
    print("Models loaded.", flush=True)

    print("\n=== Calibration ===", flush=True)
    val_random = []
    for target in range(3):
        val_random.extend(random_paths(val_parents, target, BUDGETS, seed=20260750 + 40_000))
    val_reviewed = review_records(val_random, val_parents, primary, reviewer, log_probabilities)
    val_feasible = [r for r in val_reviewed if r.get("design_status") == "feasible"]
    thresholds = calibration_thresholds(val_feasible, BUDGETS)

    all_records = []
    for target in target_indices:
        cn = CELL_NAMES[target]
        print(f"\n=== Target {cn} ===", flush=True)
        for config in configs_to_run:
            recs = run_ablation(
                config, ablation_parents, target, primary, reviewer,
                log_probabilities, thresholds, cn,
            )
            all_records.extend(recs)

    print("\n=== Reviewing ===", flush=True)
    all_reviewed = review_records(
        all_records, ablation_parents, primary, reviewer, log_probabilities
    )

    serializable = [{k: v for k, v in row.items() if k != "edits"} for row in all_reviewed]
    df = pd.DataFrame(serializable)
    print(f"Total rows: {len(df)}", flush=True)

    if args.output_mode == "partial":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(args.output.suffix + ".part")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            df.to_csv(f, sep="\t", index=False)
        tmp.rename(args.output)
        print(f"Partial output (no sealed): {args.output}", flush=True)
        return

    print("\n=== Sealed evaluator inference ===", flush=True)
    sealed = SealedEvaluatorEnsemble(args.sealed_dir, args.device, args.reviewer_batch_size)
    parent_seqs = df["parent_sequence"].astype(str).tolist()
    candidate_seqs = df["sequence"].astype(str).tolist()
    p_mean, _, _ = sealed.predict(parent_seqs)
    c_mean, c_unc, c_sd = sealed.predict(candidate_seqs)

    sealed_parent_target = []
    sealed_candidate_target = []
    sealed_target_gain = []
    sealed_parent_margin = []
    sealed_candidate_margin = []
    sealed_margin_gain = []
    sealed_uncertainty = []
    sealed_member_sd = []
    for idx, row in df.iterrows():
        target = CELL_NAMES.index(row["target_cell"])
        off = [j for j in range(3) if j != target]
        p_t = float(p_mean[idx, target])
        c_t = float(c_mean[idx, target])
        p_m = p_t - max(float(p_mean[idx, j]) for j in off)
        c_m = c_t - max(float(c_mean[idx, j]) for j in off)
        sealed_parent_target.append(p_t)
        sealed_candidate_target.append(c_t)
        sealed_target_gain.append(c_t - p_t)
        sealed_parent_margin.append(p_m)
        sealed_candidate_margin.append(c_m)
        sealed_margin_gain.append(c_m - p_m)
        sealed_uncertainty.append(float(c_unc[idx]))
        sealed_member_sd.append(float(np.mean(c_sd[idx])))

    df["sealed_parent_target"] = sealed_parent_target
    df["sealed_candidate_target"] = sealed_candidate_target
    df["sealed_target_gain"] = sealed_target_gain
    df["sealed_parent_margin"] = sealed_parent_margin
    df["sealed_candidate_margin"] = sealed_candidate_margin
    df["sealed_margin_gain"] = sealed_margin_gain
    df["sealed_uncertainty"] = sealed_uncertainty
    df["sealed_member_sd"] = sealed_member_sd

    forbidden = [c for c in df.columns if "log2FC" in c or "lfcSE" in c]
    if forbidden:
        raise RuntimeError(f"activity labels present: {forbidden}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".part")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        df.to_csv(f, sep="\t", index=False)
    tmp.rename(args.output)
    print(f"Output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
