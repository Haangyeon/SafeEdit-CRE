"""Validation-calibrate a three-seed CNN ensemble and evaluate locked pilot test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .baseline import load_pilot, mean_correlation, regression_metrics
from .cnn_model import HeteroscedasticCRECNN, encode_sequences
from .cnn_train import predict, uncertainty_report
from .g0_audit import LABEL_COLUMNS, SE_COLUMNS


def load_member(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = HeteroscedasticCRECNN()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def evaluate_ensemble(args: argparse.Namespace) -> dict[str, object]:
    table = load_pilot(args.pilot)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    split_data = {}
    for split in ("validation", "test"):
        subset = table.loc[table["split"] == split]
        split_data[split] = {
            "inputs": encode_sequences(subset["sequence"].astype(str).tolist()),
            "observed": subset.loc[:, LABEL_COLUMNS].to_numpy(dtype=np.float64),
            "measurement_se": subset.loc[:, SE_COLUMNS].to_numpy(dtype=np.float64),
        }

    members = []
    reference_mean = None
    reference_std = None
    for path in args.checkpoints:
        model, checkpoint = load_member(path, device)
        label_mean = checkpoint["label_mean"].numpy()
        label_std = checkpoint["label_std"].numpy()
        if reference_mean is None:
            reference_mean = label_mean
            reference_std = label_std
        elif not (
            np.allclose(reference_mean, label_mean) and np.allclose(reference_std, label_std)
        ):
            raise RuntimeError("ensemble checkpoints use inconsistent label normalization")
        member = {
            "path": str(path),
            "seed": int(checkpoint["seed"]),
            "best_epoch": int(checkpoint["best_epoch"]),
            "predictions": {},
        }
        for split, data in split_data.items():
            mean_std, variance_std = predict(model, data["inputs"], args.batch_size, device)
            member["predictions"][split] = {
                "mean": mean_std * label_std + label_mean,
                "learned_variance": variance_std * (label_std**2),
            }
        members.append(member)

    individual_reports = []
    for member in members:
        test_mean = member["predictions"]["test"]["mean"]
        metrics = regression_metrics(split_data["test"]["observed"], test_mean)
        individual_reports.append(
            {
                "checkpoint": member["path"],
                "seed": member["seed"],
                "best_epoch": member["best_epoch"],
                "test_mean_pearson_r": mean_correlation(metrics),
                "test_metrics": metrics,
            }
        )

    ensemble_data = {}
    for split, data in split_data.items():
        means = np.stack(
            [member["predictions"][split]["mean"] for member in members], axis=0
        )
        learned_variances = np.stack(
            [member["predictions"][split]["learned_variance"] for member in members],
            axis=0,
        )
        ensemble_mean = np.mean(means, axis=0)
        aleatoric_variance = np.mean(learned_variances, axis=0)
        epistemic_variance = np.var(means, axis=0)
        base_variance = aleatoric_variance + epistemic_variance + data["measurement_se"] ** 2
        ensemble_data[split] = {
            "mean": ensemble_mean,
            "aleatoric_variance": aleatoric_variance,
            "epistemic_variance": epistemic_variance,
            "base_variance": base_variance,
        }

    validation = split_data["validation"]
    validation_prediction = ensemble_data["validation"]
    squared_error = (validation["observed"] - validation_prediction["mean"]) ** 2
    variance_scale = np.sum(squared_error, axis=0) / np.maximum(
        np.sum(validation_prediction["base_variance"], axis=0), 1e-12
    )
    variance_scale = np.clip(variance_scale, 0.05, 20.0)

    split_reports = {}
    for split in ("validation", "test"):
        observed = split_data[split]["observed"]
        ensemble = ensemble_data[split]
        metrics = regression_metrics(observed, ensemble["mean"])
        split_reports[split] = {
            "mean_pearson_r": mean_correlation(metrics),
            "mean_spearman_rho": mean_correlation(metrics, "spearman_rho"),
            "metrics": metrics,
            "uncertainty": uncertainty_report(
                observed, ensemble["mean"], ensemble["base_variance"], variance_scale
            ),
            "mean_aleatoric_variance": np.mean(
                ensemble["aleatoric_variance"], axis=0
            ).tolist(),
            "mean_epistemic_variance": np.mean(
                ensemble["epistemic_variance"], axis=0
            ).tolist(),
        }

    report: dict[str, object] = {
        "purpose": "three-seed SafeEdit-CRE CNN ensemble prototype",
        "pilot": str(args.pilot),
        "device": str(device),
        "member_count": len(members),
        "member_seeds": [member["seed"] for member in members],
        "individual_members": individual_reports,
        "calibration_source": "validation only",
        "variance_components": "mean learned variance + between-seed variance + reported SE^2",
        "variance_scale": variance_scale.tolist(),
        **split_reports,
        "locked_test_statement": (
            "No model, threshold, or hyperparameter was selected using test metrics."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pilot", type=Path)
    parser.add_argument("--checkpoints", nargs=3, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    report = evaluate_ensemble(args)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
