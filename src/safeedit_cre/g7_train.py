"""G7: Full-data training for Reviewer and Sealed Evaluator.

Trains ResidualDilatedCNN (reviewer) or MultiKernelCNN (sealed evaluator)
on the FULL training set with chromosome-based splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .cnn_model import (
    MAX_SEQUENCE_LENGTH,
    encode_sequences,
    heteroscedastic_nll,
    reverse_complement_tensor,
)
from .g0_audit import LABEL_COLUMNS, SE_COLUMNS
from .g7_models import ARCHITECTURES
from .sequence import normalize_sequence
from .split import assign_upstream_split


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_full_table(table_s2_path: Path) -> pd.DataFrame:
    """Load the full Table S2 and assign chromosome-based splits."""
    df = pd.read_csv(table_s2_path, sep="\t", encoding="utf-8-sig")
    df["split"] = df["chr"].apply(lambda c: assign_upstream_split(str(c)))
    df = df[df["split"].isin(["train", "validation", "test"])].copy()
    df["sequence"] = df["sequence"].astype(str).apply(normalize_sequence)
    max_se = df[list(SE_COLUMNS)].max(axis=1)
    df = df[max_se < 1.0].copy()
    df = df[df["sequence"].str.len() <= MAX_SEQUENCE_LENGTH].copy()
    invalid = df["sequence"].apply(lambda s: bool(set(s) - set("ACGT")))
    df = df[~invalid].copy()
    for col in list(LABEL_COLUMNS) + list(SE_COLUMNS):
        df = df[df[col].notna() & np.isfinite(df[col])].copy()
    df = df.drop_duplicates(subset=["sequence"]).reset_index(drop=True)
    print(f"Loaded {len(df)} quality-filtered rows: "
          f"train={sum(df['split']=='train')}, "
          f"val={sum(df['split']=='validation')}, "
          f"test={sum(df['split']=='test')}", flush=True)
    return df


class SequenceDataset(Dataset):
    """On-the-fly one-hot encoding to avoid ~3GB tensor allocation."""

    def __init__(self, sequences: list[str], labels: np.ndarray, ses: np.ndarray):
        self.sequences = sequences
        self.labels = labels.astype(np.float32)
        self.ses = ses.astype(np.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        encoded = np.zeros((5, MAX_SEQUENCE_LENGTH), dtype=np.float32)
        base_map = {"A": 0, "C": 1, "G": 2, "T": 3}
        start = (MAX_SEQUENCE_LENGTH - len(seq)) // 2
        encoded[4, start:start + len(seq)] = 1.0
        for j, base in enumerate(seq):
            encoded[base_map[base], start + j] = 1.0
        return (
            torch.from_numpy(encoded),
            torch.from_numpy(self.labels[idx]),
            torch.from_numpy(self.ses[idx]),
        )


def predict_batched(
    model: nn.Module,
    sequences: list[str],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    means, variances = [], []
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch_seqs = sequences[start:start + batch_size]
            inputs = encode_sequences(batch_seqs).to(device)
            mean, raw_lv = model(inputs)
            means.append(mean.cpu().numpy())
            variances.append(torch.nn.functional.softplus(raw_lv).cpu().numpy())
    return np.concatenate(means), np.concatenate(variances)


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    results = {}
    for i, label in enumerate(LABEL_COLUMNS):
        y, yhat = observed[:, i], predicted[:, i]
        if np.ptp(y) == 0 or np.ptp(yhat) == 0:
            r, rho = None, None
        else:
            r = float(pearsonr(y, yhat).statistic)
            rho = float(spearmanr(y, yhat).statistic)
        results[label] = {
            "pearson_r": r,
            "spearman_rho": rho,
            "mae": float(np.mean(np.abs(y - yhat))),
            "rmse": float(np.sqrt(np.mean((y - yhat) ** 2))),
        }
    return results


def mean_pearson(metrics: dict) -> float:
    vals = [v["pearson_r"] for v in metrics.values() if v["pearson_r"] is not None]
    return float(np.mean(vals)) if vals else 0.0


def coverage(observed, mean, variance, level: float) -> list[float]:
    z_values = {0.8: 1.2815515655, 0.9: 1.6448536270, 0.95: 1.9599639845}
    z = z_values[level]
    hw = z * np.sqrt(np.maximum(variance, 1e-12))
    return np.mean(np.abs(observed - mean) <= hw, axis=0).tolist()


def uncertainty_report(observed, mean, base_variance, scale) -> dict:
    cal_var = base_variance * scale
    abs_err = np.abs(observed - mean)
    corrs = []
    for t in range(observed.shape[1]):
        corrs.append(float(spearmanr(np.sqrt(cal_var[:, t]), abs_err[:, t]).statistic))
    return {
        "variance_scale": scale.tolist(),
        "spearman_uncertainty_vs_absolute_error": corrs,
        "coverage_80": coverage(observed, mean, cal_var, 0.8),
        "coverage_90": coverage(observed, mean, cal_var, 0.9),
        "coverage_95": coverage(observed, mean, cal_var, 0.95),
        "mean_gaussian_nll": float(np.mean(
            0.5 * ((observed - mean) ** 2 / np.maximum(cal_var, 1e-12)
                   + np.log(np.maximum(cal_var, 1e-12)))
        )),
    }


def train_one_model(
    architecture_name: str,
    table_s2: Path,
    seed: int,
    output_dir: Path,
    epochs: int = 20,
    patience: int = 6,
    batch_size: int = 256,
    eval_batch_size: int = 512,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.25,
    rc_weight: float = 0.1,
    gradient_clip: float = 5.0,
    device: str = "auto",
) -> dict:
    seed_everything(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    print(f"\n{'='*60}", flush=True)
    print(f"G7 Training: {architecture_name} seed={seed} device={dev}", flush=True)
    print(f"{'='*60}", flush=True)

    table = load_full_table(table_s2)
    train_df = table[table["split"] == "train"]
    val_df = table[table["split"] == "validation"]
    test_df = table[table["split"] == "test"]

    label_mean = train_df[list(LABEL_COLUMNS)].to_numpy(dtype=np.float32).mean(axis=0)
    label_std = train_df[list(LABEL_COLUMNS)].to_numpy(dtype=np.float32).std(axis=0)

    def make_dataset(df):
        labels = (df[list(LABEL_COLUMNS)].to_numpy(dtype=np.float32) - label_mean) / label_std
        ses = df[list(SE_COLUMNS)].to_numpy(dtype=np.float32) / label_std
        return SequenceDataset(df["sequence"].tolist(), labels, ses)

    train_ds = make_dataset(train_df)
    val_ds = make_dataset(val_df)
    test_ds = make_dataset(test_df)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )

    model_cls = ARCHITECTURES[architecture_name]
    model = model_cls(dropout=dropout).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Architecture: {architecture_name}, params: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_score = -math.inf
    best_epoch = 0
    best_state = None
    history = []
    patience_used = 0

    val_seqs = val_df["sequence"].tolist()
    val_y_raw = val_df[list(LABEL_COLUMNS)].to_numpy(dtype=np.float32)
    val_se_raw = val_df[list(SE_COLUMNS)].to_numpy(dtype=np.float32)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_examples = 0
        for inputs, observed, measurement_se in train_loader:
            inputs, observed, measurement_se = (
                inputs.to(dev), observed.to(dev), measurement_se.to(dev)
            )
            optimizer.zero_grad(set_to_none=True)
            mean, raw_lv = model(inputs)
            nll = heteroscedastic_nll(observed, mean, raw_lv, measurement_se)
            rc_mean, _ = model(reverse_complement_tensor(inputs))
            rc_loss = torch.mean((mean - rc_mean).square())
            loss = nll + rc_weight * rc_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(inputs)
            n_examples += len(inputs)
        scheduler.step()

        val_mean_std, _ = predict_batched(model, val_seqs, eval_batch_size, dev)
        val_mean_raw = val_mean_std * label_std + label_mean
        val_metrics = regression_metrics(val_y_raw, val_mean_raw)
        score = mean_pearson(val_metrics)
        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss / n_examples,
            "validation_mean_pearson_r": score,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        print(json.dumps(history[-1]), flush=True)

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_used = 0
        else:
            patience_used += 1
            if patience_used >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})", flush=True)
                break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(dev)

    test_seqs = test_df["sequence"].tolist()
    test_y_raw = test_df[list(LABEL_COLUMNS)].to_numpy(dtype=np.float32)
    test_se_raw = test_df[list(SE_COLUMNS)].to_numpy(dtype=np.float32)

    test_mean_std, test_var_std = predict_batched(model, test_seqs, eval_batch_size, dev)
    test_mean_raw = test_mean_std * label_std + label_mean
    test_var_raw = test_var_std * (label_std ** 2)
    test_base_var = test_var_raw + test_se_raw ** 2

    val_mean_std2, val_var_std2 = predict_batched(model, val_seqs, eval_batch_size, dev)
    val_mean_raw2 = val_mean_std2 * label_std + label_mean
    val_var_raw2 = val_var_std2 * (label_std ** 2)
    val_base_var2 = val_var_raw2 + val_se_raw ** 2
    val_sq_err = (val_y_raw - val_mean_raw2) ** 2
    variance_scale = np.sum(val_sq_err, axis=0) / np.maximum(np.sum(val_base_var2, axis=0), 1e-12)
    variance_scale = np.clip(variance_scale, 0.05, 20.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"{architecture_name}_seed_{seed}.pt"
    torch.save({
        "model_state_dict": best_state,
        "label_mean": torch.from_numpy(label_mean),
        "label_std": torch.from_numpy(label_std),
        "variance_scale": torch.from_numpy(variance_scale.astype(np.float32)),
        "seed": seed,
        "best_epoch": best_epoch,
        "architecture": architecture_name,
    }, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}", flush=True)

    test_metrics = regression_metrics(test_y_raw, test_mean_raw)
    test_mean_pearson = mean_pearson(test_metrics)
    val_metrics_final = regression_metrics(val_y_raw, val_mean_raw2)

    report = {
        "architecture": architecture_name,
        "seed": seed,
        "device": str(dev),
        "n_params": n_params,
        "rows_by_split": {
            "train": len(train_df), "validation": len(val_df), "test": len(test_df),
        },
        "best_epoch": best_epoch,
        "history": history,
        "calibration_source": "validation only",
        "variance_scale": variance_scale.tolist(),
        "validation_metrics": val_metrics_final,
        "validation_mean_pearson_r": mean_pearson(val_metrics_final),
        "test_metrics": test_metrics,
        "test_mean_pearson_r": test_mean_pearson,
        "test_uncertainty": uncertainty_report(
            test_y_raw, test_mean_raw, test_base_var, variance_scale
        ),
        "checkpoint_path": str(ckpt_path),
    }
    report_path = output_dir / f"{architecture_name}_seed_{seed}_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Test mean Pearson r: {test_mean_pearson:.4f}", flush=True)
    print(f"Report saved: {report_path}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description="G7 full-data training")
    parser.add_argument("table_s2", type=Path, help="Path to DATA-Table_S2__MPRA_dataset.txt")
    parser.add_argument("--architecture", required=True, choices=list(ARCHITECTURES))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--rc-weight", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    train_one_model(
        args.architecture, args.table_s2, args.seed, args.output_dir,
        args.epochs, args.patience, args.batch_size, args.eval_batch_size,
        args.learning_rate, args.weight_decay, args.dropout,
        args.rc_weight, args.gradient_clip, args.device,
    )


if __name__ == "__main__":
    main()
