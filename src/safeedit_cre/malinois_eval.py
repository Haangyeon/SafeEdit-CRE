"""Evaluate the published Malinois weights on the cleaned pilot test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baseline import load_pilot, mean_correlation, regression_metrics
from .g0_audit import LABEL_COLUMNS
from .malinois import load_malinois, predict_sequences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pilot", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-strand-average", action="store_true")
    args = parser.parse_args()

    table = load_pilot(args.pilot)
    test = table.loc[table["split"] == "test"]
    if args.limit is not None:
        test = test.iloc[: args.limit]
    model, metadata = load_malinois(args.checkpoint)
    predictions = predict_sequences(
        model,
        test["sequence"].astype(str).tolist(),
        batch_size=args.batch_size,
        strand_average=not args.no_strand_average,
    )
    observed = test.loc[:, LABEL_COLUMNS].to_numpy(dtype=float)
    metrics = regression_metrics(observed, predictions)
    report = {
        "purpose": "published Malinois weight compatibility and pilot evaluation",
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata": metadata,
        "rows": len(test),
        "strand_average": not args.no_strand_average,
        "metrics": metrics,
        "mean_pearson_r": mean_correlation(metrics),
        "mean_spearman_rho": mean_correlation(metrics, "spearman_rho"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
