"""G8: Select 600 new parents from test split, excluding all historical parents.

Uses sha256(ID + "\\t" + sequence) ordering for deterministic, label-free selection.
Output FROZEN_G8_PARENTS.tsv.gz contains NO activity labels.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .g0_audit import LABEL_COLUMNS, SE_COLUMNS
from .g7_train import load_full_table
from .sequence import normalize_sequence
from .split import assign_upstream_split


N_PARENTS = 600


def load_denylist(denylist_paths: list[Path]) -> set[str]:
    """Load all historical parent IDs and sequence hashes to exclude."""
    excluded_ids: set[str] = set()
    excluded_seq_hashes: set[str] = set()
    for path in denylist_paths:
        if not path.exists():
            print(f"  WARN: denylist not found: {path}", flush=True)
            continue
        if path.suffix == ".gz":
            df = pd.read_csv(path, sep="\t")
        else:
            df = pd.read_csv(path, sep="\t")
        if "IDs" in df.columns:
            excluded_ids.update(df["IDs"].astype(str).tolist())
        if "parent_id" in df.columns:
            excluded_ids.update(df["parent_id"].astype(str).tolist())
        if "sequence" in df.columns:
            for seq in df["sequence"].astype(str):
                excluded_seq_hashes.add(hashlib.sha256(seq.encode("utf-8")).hexdigest())
        if "sequence_sha256" in df.columns:
            excluded_seq_hashes.update(df["sequence_sha256"].astype(str).tolist())
        print(f"  Loaded {len(df)} rows from {path.name}", flush=True)
    return excluded_ids, excluded_seq_hashes


def select_g8_parents(
    table_s2: Path,
    denylist_paths: list[Path],
    output_path: Path,
    n_parents: int = N_PARENTS,
) -> dict:
    table = load_full_table(table_s2)
    test_df = table[
        (table["split"] == "test")
        & (table["sequence"].astype(str).str.len() == 200)
    ].copy()

    print(f"Test split 200nt CREs: {len(test_df)}", flush=True)

    excluded_ids, excluded_seq_hashes = load_denylist(denylist_paths)
    print(f"Excluded IDs: {len(excluded_ids)}, sequence hashes: {len(excluded_seq_hashes)}", flush=True)

    test_df = test_df.copy()
    test_df["seq_hash"] = test_df["sequence"].apply(
        lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
    )
    mask_id = ~test_df["IDs"].astype(str).isin(excluded_ids)
    mask_seq = ~test_df["seq_hash"].isin(excluded_seq_hashes)
    test_df = test_df[mask_id & mask_seq].copy()
    print(f"After exclusion: {len(test_df)}", flush=True)

    test_df["selection_hash"] = [
        hashlib.sha256(f"{identifier}\t{sequence}".encode("utf-8")).hexdigest()
        for identifier, sequence in zip(test_df["IDs"], test_df["sequence"], strict=True)
    ]
    test_df = test_df.sort_values(["selection_hash", "IDs"], kind="stable").head(n_parents)

    if len(test_df) < n_parents:
        print(f"  WARN: only {len(test_df)} parents available (requested {n_parents})", flush=True)

    output_df = test_df[["IDs", "sequence", "chr"]].copy()
    output_df.columns = ["parent_id", "sequence", "chr"]
    output_df["selection_hash"] = test_df["selection_hash"].values
    output_df = output_df.reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        output_df.to_csv(f, sep="\t", index=False)

    seq_hashes = [
        hashlib.sha256(s.encode("utf-8")).hexdigest()
        for s in output_df["sequence"].tolist()
    ]
    report = {
        "n_selected": len(output_df),
        "n_requested": n_parents,
        "source_split": "test",
        "sequence_length": 200,
        "excluded_id_count": len(excluded_ids),
        "excluded_seq_hash_count": len(excluded_seq_hashes),
        "selection_method": "sha256(ID + '\\t' + sequence) ascending",
        "output_path": str(output_path),
        "parent_ids": output_df["parent_id"].tolist(),
        "sequence_hashes": seq_hashes,
        "contains_activity_labels": False,
    }
    print(f"Selected {len(output_df)} parents -> {output_path}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description="G8 parent selection")
    parser.add_argument("table_s2", type=Path)
    parser.add_argument("--denylist", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--n-parents", type=int, default=N_PARENTS)
    args = parser.parse_args()

    report = select_g8_parents(args.table_s2, args.denylist, args.output, args.n_parents)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Report saved: {args.report}", flush=True)


if __name__ == "__main__":
    main()
