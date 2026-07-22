"""G8 sealed evaluator inference on frozen candidates.

Loads the MultiKernelCNN sealed evaluator ensemble (seeds 20260731-33) and
runs inference on all frozen candidate and parent sequences. Appends sealed_*
columns. This runs ONLY after candidates are frozen.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .cnn_model import encode_sequences
from .edit_benchmark import CELL_NAMES, specificity_score
from .g7_models import load_g7_model


SEALED_ARCH = "MultiKernelCNN"
SEALED_SEEDS = [20260731, 20260732, 20260733]


class SealedEvaluatorEnsemble:
    """3-member MultiKernelCNN sealed evaluator ensemble."""

    def __init__(self, checkpoint_dir: Path, device: str, batch_size: int = 256):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.members = []
        reference_mean = None
        reference_std = None
        for seed in SEALED_SEEDS:
            ckpt_path = checkpoint_dir / f"{SEALED_ARCH}_seed_{seed}.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"sealed evaluator checkpoint: {ckpt_path}")
            model, ckpt = load_g7_model(SEALED_ARCH, ckpt_path, device=device)
            self.members.append(model)
            label_mean = ckpt["label_mean"].numpy()
            label_std = ckpt["label_std"].numpy()
            if reference_mean is None:
                reference_mean = label_mean
                reference_std = label_std
            elif not (np.allclose(reference_mean, label_mean) and np.allclose(reference_std, label_std)):
                raise RuntimeError("sealed evaluator checkpoints use inconsistent normalization")
            variance_scale = ckpt["variance_scale"].numpy()
        if len(self.members) != 3:
            raise RuntimeError(f"expected 3 sealed evaluator members, got {len(self.members)}")
        self.label_mean = np.asarray(reference_mean)
        self.label_std = np.asarray(reference_std)
        self.variance_scale = np.asarray(variance_scale)

    def predict(self, sequences: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (ensemble_mean, mean_uncertainty, member_sd)."""
        member_means = []
        for model in self.members:
            means = []
            with torch.inference_mode():
                for start in range(0, len(sequences), self.batch_size):
                    batch_seqs = sequences[start:start + self.batch_size]
                    inputs = encode_sequences(batch_seqs).to(self.device)
                    mean, _ = model(inputs)
                    means.append(mean.cpu().numpy())
            mean_std = np.concatenate(means)
            member_means.append(mean_std * self.label_std + self.label_mean)
        stacked = np.stack(member_means)
        ensemble_mean = np.mean(stacked, axis=0)
        member_sd = np.std(stacked, axis=0)
        predictive_variance = np.var(stacked, axis=0) * self.variance_scale
        mean_uncertainty = np.mean(np.sqrt(np.maximum(predictive_variance, 1e-12)), axis=1)
        return ensemble_mean, mean_uncertainty, member_sd


def main():
    parser = argparse.ArgumentParser(description="G8 sealed evaluator inference")
    parser.add_argument("presealed", type=Path, help="g8_candidates_presealed.tsv.gz")
    parser.add_argument("--sealed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}", flush=True)

    df = pd.read_csv(args.presealed, sep="\t")
    print(f"Loaded {len(df)} presealed candidates", flush=True)

    sealed = SealedEvaluatorEnsemble(args.sealed_dir, args.device, args.batch_size)
    print("Sealed evaluator ensemble loaded.", flush=True)

    parent_seqs = df["parent_sequence"].astype(str).tolist()
    candidate_seqs = df["sequence"].astype(str).tolist()

    print("Predicting parents...", flush=True)
    p_mean, p_unc, p_sd = sealed.predict(parent_seqs)
    print("Predicting candidates...", flush=True)
    c_mean, c_unc, c_sd = sealed.predict(candidate_seqs)

    for i, cell in enumerate(CELL_NAMES):
        df[f"_sealed_pred_{cell}"] = c_mean[:, i]

    sealed_parent_target = []
    sealed_candidate_target = []
    sealed_target_gain = []
    sealed_parent_margin = []
    sealed_candidate_margin = []
    sealed_margin_gain = []
    sealed_uncertainty = []
    sealed_member_sd = []

    for idx, row in df.iterrows():
        target_cell = row["target_cell"]
        target = CELL_NAMES.index(target_cell)
        off = [j for j in range(3) if j != target]
        p_t = float(p_mean[idx, target])
        c_t = float(c_mean[idx, target])
        p_margin = p_t - max(float(p_mean[idx, j]) for j in off)
        c_margin = c_t - max(float(c_mean[idx, j]) for j in off)
        sealed_parent_target.append(p_t)
        sealed_candidate_target.append(c_t)
        sealed_target_gain.append(c_t - p_t)
        sealed_parent_margin.append(p_margin)
        sealed_candidate_margin.append(c_margin)
        sealed_margin_gain.append(c_margin - p_margin)
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

    for col in [c for c in df.columns if c.startswith("_sealed_pred_")]:
        df = df.drop(columns=[col])

    forbidden = [c for c in df.columns if "log2FC" in c or "lfcSE" in c]
    if forbidden:
        raise RuntimeError(f"activity labels present in sealed output: {forbidden}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".part")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        df.to_csv(f, sep="\t", index=False)
    tmp.rename(args.output)
    print(f"Sealed output: {args.output}", flush=True)

    presealed_sha = hashlib.sha256(args.presealed.read_bytes()).hexdigest()
    output_sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "presealed_path": str(args.presealed),
        "presealed_sha256": presealed_sha,
        "sealed_output_path": str(args.output),
        "sealed_output_sha256": output_sha,
        "sealed_architecture": SEALED_ARCH,
        "sealed_seeds": SEALED_SEEDS,
        "n_rows": len(df),
        "sealed_columns": [
            "sealed_parent_target", "sealed_candidate_target", "sealed_target_gain",
            "sealed_parent_margin", "sealed_candidate_margin", "sealed_margin_gain",
            "sealed_uncertainty", "sealed_member_sd",
        ],
        "contains_activity_labels": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Manifest: {args.manifest}", flush=True)


if __name__ == "__main__":
    main()
