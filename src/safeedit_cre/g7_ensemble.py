"""G7 ensemble evaluation and performance gate check.

Evaluates all reviewer (ResidualDilatedCNN, seeds 20260721-23) and sealed
evaluator (MultiKernelCNN, seeds 20260731-33) checkpoints on the test set.
If mean Pearson r >= 0.75 for both families, writes manifests and G7_PASS.
Otherwise writes G7_STOP with diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from .cnn_model import encode_sequences, heteroscedastic_nll
from .g0_audit import LABEL_COLUMNS, SE_COLUMNS
from .g7_models import ARCHITECTURES, load_g7_model
from .g7_train import load_full_table, predict_batched, regression_metrics, mean_pearson
from .sequence import normalize_sequence


REVIEWER_SEEDS = [20260721, 20260722, 20260723]
SEALED_SEEDS = [20260731, 20260732, 20260733]
REVIEWER_ARCH = "ResidualDilatedCNN"
SEALED_ARCH = "MultiKernelCNN"
PEARSON_GATE = 0.75


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate_checkpoint(
    architecture: str,
    seed: int,
    checkpoint_dir: Path,
    table_s2: Path,
    device: str = "auto",
) -> dict:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    ckpt_path = checkpoint_dir / f"{architecture}_seed_{seed}.pt"
    report_path = checkpoint_dir / f"{architecture}_seed_{seed}_report.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    model, ckpt = load_g7_model(architecture, ckpt_path, device=device)
    label_mean = ckpt["label_mean"].numpy()
    label_std = ckpt["label_std"].numpy()
    variance_scale = ckpt["variance_scale"].numpy()

    table = load_full_table(table_s2)
    test_df = table[table["split"] == "test"]
    test_seqs = test_df["sequence"].tolist()
    test_y = test_df[list(LABEL_COLUMNS)].to_numpy(dtype=np.float32)
    test_se = test_df[list(SE_COLUMNS)].to_numpy(dtype=np.float32)

    mean_std, var_std = predict_batched(model, test_seqs, 512, dev)
    mean_raw = mean_std * label_std + label_mean
    var_raw = var_std * (label_std ** 2)
    base_var = var_raw + test_se ** 2

    metrics = regression_metrics(test_y, mean_raw)
    mp = mean_pearson(metrics)

    existing_report = {}
    if report_path.exists():
        existing_report = json.loads(report_path.read_text(encoding="utf-8"))

    result = {
        "architecture": architecture,
        "seed": seed,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": file_sha256(ckpt_path),
        "test_mean_pearson_r": mp,
        "test_metrics": metrics,
        "best_epoch": ckpt.get("best_epoch", existing_report.get("best_epoch")),
        "variance_scale": variance_scale.tolist(),
        "label_mean": label_mean.tolist(),
        "label_std": label_std.tolist(),
    }
    print(f"  {architecture} seed={seed}: test mean Pearson r = {mp:.4f}", flush=True)
    return result


def build_manifest(
    architecture: str,
    seeds: list[int],
    checkpoint_dir: Path,
    table_s2: Path,
    device: str,
) -> dict:
    print(f"\nEvaluating {architecture} ensemble:", flush=True)
    members = []
    for seed in seeds:
        members.append(
            evaluate_checkpoint(architecture, seed, checkpoint_dir, table_s2, device)
        )
    mean_pearsons = [m["test_mean_pearson_r"] for m in members]
    ensemble_mean_pearson = float(np.mean(mean_pearsons))
    print(f"  Ensemble mean Pearson r: {ensemble_mean_pearson:.4f}", flush=True)

    manifest = {
        "architecture": architecture,
        "role": "reviewer" if architecture == REVIEWER_ARCH else "sealed_evaluator",
        "seeds": seeds,
        "members": members,
        "ensemble_mean_test_pearson_r": ensemble_mean_pearson,
        "pearson_gate": PEARSON_GATE,
        "gate_passed": ensemble_mean_pearson >= PEARSON_GATE,
        "calibration_source": "validation only",
    }
    return manifest


def main():
    parser = argparse.ArgumentParser(description="G7 ensemble gate check")
    parser.add_argument("table_s2", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    args.frozen_dir.mkdir(parents=True, exist_ok=True)
    args.state_dir.mkdir(parents=True, exist_ok=True)

    reviewer_manifest = build_manifest(
        REVIEWER_ARCH, REVIEWER_SEEDS, args.checkpoint_dir, args.table_s2, args.device
    )
    sealed_manifest = build_manifest(
        SEALED_ARCH, SEALED_SEEDS, args.checkpoint_dir, args.table_s2, args.device
    )

    reviewer_path = args.frozen_dir / "reviewer_manifest.json"
    sealed_path = args.frozen_dir / "sealed_evaluator_manifest.json"
    reviewer_path.write_text(
        json.dumps(reviewer_manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    sealed_path.write_text(
        json.dumps(sealed_manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    gate_ok = (
        reviewer_manifest["gate_passed"] and sealed_manifest["gate_passed"]
    )

    if gate_ok:
        (args.state_dir / "G7_PASS").write_text(
            f"reviewer_mean_pearson={reviewer_manifest['ensemble_mean_test_pearson_r']:.4f}\n"
            f"sealed_mean_pearson={sealed_manifest['ensemble_mean_test_pearson_r']:.4f}\n"
            f"gate={PEARSON_GATE}\n",
            encoding="utf-8",
        )
        print("\nG7_PASS written.", flush=True)
    else:
        diag = {
            "status": "G7_STOP",
            "reviewer_mean_pearson": reviewer_manifest["ensemble_mean_test_pearson_r"],
            "sealed_mean_pearson": sealed_manifest["ensemble_mean_test_pearson_r"],
            "gate": PEARSON_GATE,
            "reviewer_passed": reviewer_manifest["gate_passed"],
            "sealed_passed": sealed_manifest["gate_passed"],
        }
        (args.state_dir / "G7_STOP").write_text(
            json.dumps(diag, indent=2) + "\n", encoding="utf-8"
        )
        print("\nG7_STOP: gate not met. Diagnostic written.", flush=True)
        print(json.dumps(diag, indent=2), flush=True)


if __name__ == "__main__":
    main()
