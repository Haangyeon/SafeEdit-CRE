"""Leakage-safe 4-mer ridge smoke baseline on the deterministic pilot."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import kmer_frequency_matrix
from .g0_audit import LABEL_COLUMNS


def safe_correlation(observed: np.ndarray, predicted: np.ndarray, method: str) -> float | None:
    if np.ptp(observed) == 0 or np.ptp(predicted) == 0:
        return None
    if method == "pearson":
        return float(pearsonr(observed, predicted).statistic)
    if method == "spearman":
        return float(spearmanr(observed, predicted).statistic)
    raise ValueError(f"unsupported correlation method: {method}")


def regression_metrics(
    observed: np.ndarray, predicted: np.ndarray
) -> dict[str, dict[str, float | None]]:
    results: dict[str, dict[str, float | None]] = {}
    for index, label in enumerate(LABEL_COLUMNS):
        y = observed[:, index]
        y_hat = predicted[:, index]
        results[label] = {
            "pearson_r": safe_correlation(y, y_hat, "pearson"),
            "spearman_rho": safe_correlation(y, y_hat, "spearman"),
            "mae": float(np.mean(np.abs(y - y_hat))),
            "rmse": float(np.sqrt(np.mean((y - y_hat) ** 2))),
        }
    return results


def mean_correlation(
    metrics: dict[str, dict[str, float | None]], key: str = "pearson_r"
) -> float | None:
    values = [values[key] for values in metrics.values() if values[key] is not None]
    return float(np.mean(values)) if values else None


def load_pilot(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        table = pd.read_csv(handle, sep="\t")
    expected = {"split", "sequence", *LABEL_COLUMNS}
    missing = expected - set(table.columns)
    if missing:
        raise ValueError(f"pilot is missing columns: {sorted(missing)}")
    return table


def run_baseline(pilot_path: Path, output_path: Path, alphas: tuple[float, ...]) -> dict[str, object]:
    table = load_pilot(pilot_path)
    matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in ("train", "validation", "test"):
        subset = table.loc[table["split"] == split]
        x = kmer_frequency_matrix(subset["sequence"].astype(str), k=4)
        y = subset.loc[:, LABEL_COLUMNS].to_numpy(dtype=np.float64)
        matrices[split] = (x, y)

    x_train, y_train = matrices["train"]
    x_validation, y_validation = matrices["validation"]
    validation_trials = []
    for alpha in alphas:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=alpha, solver="lsqr")),
            ]
        )
        model.fit(x_train, y_train)
        metrics = regression_metrics(y_validation, model.predict(x_validation))
        validation_trials.append(
            {
                "alpha": alpha,
                "mean_pearson_r": mean_correlation(metrics),
                "metrics": metrics,
            }
        )

    selected = max(validation_trials, key=lambda trial: float(trial["mean_pearson_r"]))
    x_test, y_test = matrices["test"]
    final_model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=float(selected["alpha"]), solver="lsqr")),
        ]
    )
    final_model.fit(
        np.concatenate([x_train, x_validation], axis=0),
        np.concatenate([y_train, y_validation], axis=0),
    )
    test_metrics = regression_metrics(y_test, final_model.predict(x_test))

    train_mean = y_train.mean(axis=0, keepdims=True)
    mean_baseline_metrics = regression_metrics(y_test, np.repeat(train_mean, len(y_test), axis=0))
    report: dict[str, object] = {
        "purpose": "CPU smoke baseline; not the final publication model",
        "pilot_path": str(pilot_path),
        "feature_set": "stranded normalized 4-mer frequencies (256 features)",
        "selection_rule": "maximize mean validation Pearson r; test untouched until final fit",
        "rows_by_split": {
            split: int(len(values[1])) for split, values in matrices.items()
        },
        "validation_trials": validation_trials,
        "selected_alpha": selected["alpha"],
        "test_metrics": test_metrics,
        "test_mean_pearson_r": mean_correlation(test_metrics),
        "test_mean_spearman_rho": mean_correlation(test_metrics, "spearman_rho"),
        "test_train_mean_baseline": mean_baseline_metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pilot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alphas", nargs="+", type=float, default=(0.1, 1.0, 10.0, 100.0))
    args = parser.parse_args()
    report = run_baseline(args.pilot, args.output, tuple(args.alphas))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
