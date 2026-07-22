"""Train and calibrate the compact uncertainty-aware CRE CNN on a fixed pilot."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .baseline import load_pilot, mean_correlation, regression_metrics
from .cnn_model import (
    HeteroscedasticCRECNN,
    encode_sequences,
    heteroscedastic_nll,
    reverse_complement_tensor,
)
from .g0_audit import LABEL_COLUMNS, SE_COLUMNS


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_split_tensors(table, split: str, label_mean, label_std):
    subset = table.loc[table["split"] == split]
    inputs = encode_sequences(subset["sequence"].astype(str).tolist())
    observed = subset.loc[:, LABEL_COLUMNS].to_numpy(dtype=np.float32)
    measurement_se = subset.loc[:, SE_COLUMNS].to_numpy(dtype=np.float32)
    standardized = (observed - label_mean) / label_std
    standardized_se = measurement_se / label_std
    return (
        inputs,
        torch.from_numpy(standardized),
        torch.from_numpy(standardized_se),
        observed,
        measurement_se,
    )


def predict(
    model: nn.Module,
    inputs: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    means = []
    variances = []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size].to(device)
            mean, raw_log_variance = model(batch)
            means.append(mean.cpu().numpy())
            variances.append(
                torch.nn.functional.softplus(raw_log_variance).cpu().numpy()
            )
    return np.concatenate(means), np.concatenate(variances)


def coverage(observed, mean, variance, level: float) -> list[float]:
    z_values = {0.8: 1.2815515655, 0.9: 1.6448536270, 0.95: 1.9599639845}
    z = z_values[level]
    half_width = z * np.sqrt(np.maximum(variance, 1e-12))
    return np.mean(np.abs(observed - mean) <= half_width, axis=0).tolist()


def uncertainty_report(observed, mean, base_variance, scale) -> dict[str, object]:
    calibrated_variance = base_variance * scale
    absolute_error = np.abs(observed - mean)
    correlations = []
    for task in range(observed.shape[1]):
        correlations.append(
            float(spearmanr(np.sqrt(calibrated_variance[:, task]), absolute_error[:, task]).statistic)
        )
    return {
        "variance_scale": scale.tolist(),
        "spearman_uncertainty_vs_absolute_error": correlations,
        "coverage_80": coverage(observed, mean, calibrated_variance, 0.8),
        "coverage_90": coverage(observed, mean, calibrated_variance, 0.9),
        "coverage_95": coverage(observed, mean, calibrated_variance, 0.95),
        "mean_gaussian_nll": float(
            np.mean(
                0.5
                * (
                    (observed - mean) ** 2 / np.maximum(calibrated_variance, 1e-12)
                    + np.log(np.maximum(calibrated_variance, 1e-12))
                )
            )
        ),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    seed_everything(args.seed)
    table = load_pilot(args.pilot)
    train_table = table.loc[table["split"] == "train"]
    label_mean = train_table.loc[:, LABEL_COLUMNS].to_numpy(dtype=np.float32).mean(axis=0)
    label_std = train_table.loc[:, LABEL_COLUMNS].to_numpy(dtype=np.float32).std(axis=0)
    tensors = {
        split: make_split_tensors(table, split, label_mean, label_std)
        for split in ("train", "validation", "test")
    }
    train_inputs, train_y, train_se, _, _ = tensors["train"]
    train_loader = DataLoader(
        TensorDataset(train_inputs, train_y, train_se),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = HeteroscedasticCRECNN(dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_score = -math.inf
    best_epoch = 0
    best_state = None
    history = []
    patience_used = 0
    validation_inputs, _, _, validation_y_raw, _ = tensors["validation"]

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        examples = 0
        for inputs, observed, measurement_se in train_loader:
            inputs = inputs.to(device)
            observed = observed.to(device)
            measurement_se = measurement_se.to(device)
            optimizer.zero_grad(set_to_none=True)
            mean, raw_log_variance = model(inputs)
            nll = heteroscedastic_nll(
                observed, mean, raw_log_variance, measurement_se
            )
            rc_mean, _ = model(reverse_complement_tensor(inputs))
            rc_consistency = torch.mean((mean - rc_mean).square())
            loss = nll + args.rc_weight * rc_consistency
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(inputs)
            examples += len(inputs)
        scheduler.step()

        validation_mean_std, _ = predict(
            model, validation_inputs, args.eval_batch_size, device
        )
        validation_mean = validation_mean_std * label_std + label_mean
        validation_metrics = regression_metrics(validation_y_raw, validation_mean)
        score = float(mean_correlation(validation_metrics))
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / examples,
                "validation_mean_pearson_r": score,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(json.dumps(history[-1]), flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            patience_used = 0
        else:
            patience_used += 1
            if patience_used >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)

    predictions = {}
    for split in ("validation", "test"):
        inputs, _, _, observed_raw, measurement_se_raw = tensors[split]
        mean_std, learned_variance_std = predict(
            model, inputs, args.eval_batch_size, device
        )
        mean_raw = mean_std * label_std + label_mean
        learned_variance_raw = learned_variance_std * (label_std**2)
        base_variance = learned_variance_raw + measurement_se_raw**2
        predictions[split] = (observed_raw, mean_raw, base_variance)

    validation_observed, validation_mean, validation_variance = predictions["validation"]
    squared_error = (validation_observed - validation_mean) ** 2
    variance_scale = np.sum(squared_error, axis=0) / np.maximum(
        np.sum(validation_variance, axis=0), 1e-12
    )
    variance_scale = np.clip(variance_scale, 0.05, 20.0)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "label_mean": torch.from_numpy(label_mean),
            "label_std": torch.from_numpy(label_std),
            "variance_scale": torch.from_numpy(variance_scale.astype(np.float32)),
            "seed": args.seed,
            "best_epoch": best_epoch,
            "architecture": "HeteroscedasticCRECNN",
        },
        args.checkpoint,
    )

    split_reports = {}
    for split in ("validation", "test"):
        observed_raw, mean_raw, base_variance = predictions[split]
        metrics = regression_metrics(observed_raw, mean_raw)
        split_reports[split] = {
            "metrics": metrics,
            "mean_pearson_r": mean_correlation(metrics),
            "mean_spearman_rho": mean_correlation(metrics, "spearman_rho"),
            "uncertainty": uncertainty_report(
                observed_raw, mean_raw, base_variance, variance_scale
            ),
        }

    report: dict[str, object] = {
        "purpose": "pilot prototype of the SafeEdit-CRE uncertainty-aware predictor",
        "pilot": str(args.pilot),
        "device": str(device),
        "rows_by_split": {
            split: int(np.sum(table["split"] == split))
            for split in ("train", "validation", "test")
        },
        "model": {
            "architecture": "3-block 1D CNN with explicit sequence mask",
            "outputs": "three standardized means and three learned variances",
            "loss": "heteroscedastic Gaussian NLL + reported SE + RC consistency",
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        "best_epoch": best_epoch,
        "history": history,
        "calibration_source": "validation only",
        "variance_scale": variance_scale.tolist(),
        **split_reports,
        "limitations": [
            "Prototype trained on the deterministic 30k pilot, not the full training cohort.",
            "A single model estimates data-dependent variance but not full epistemic uncertainty.",
            "Test metrics are locked evaluation and are not used to alter hyperparameters.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pilot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--rc-weight", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
