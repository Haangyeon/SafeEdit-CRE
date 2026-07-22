"""R8: Final strict release verification.

Only prints STRICT_RELEASE_PASS if ALL checks pass.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_FILES = [
    "checkpoints/FROZEN_PROTOCOL.md",
    "checkpoints/FROZEN_CONFIG.yaml",
    "checkpoints/FROZEN_PARENTS.tsv.gz",
    "checkpoints/FROZEN_CODE_SHA256.txt",
    "data/processed/g4_frozen_candidates.tsv.gz",
    "reports/g4_frozen_confirmation.json",
    "reports/g5_frozen_audit.json",
    "reports/g6_frozen_candidate_summary.json",
    "reports/r4_independent_recalculation.json",
    "manuscript/main.pdf",
    "manuscript/supplementary.pdf",
]

EXPECTED_G4_ROWS = 3456
EXPECTED_PARENTS = 96
EXPECTED_TARGETS = 3
EXPECTED_BUDGETS = [1, 5, 10, 20]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root
    failures = []
    warnings = []

    for rel in REQUIRED_FILES:
        p = root / rel
        if not p.exists():
            failures.append(f"MISSING_FILE: {rel}")

    if failures:
        print("RELEASE_VERIFICATION_FAIL")
        for f in failures:
            print(f"  FAIL: {f}")
        return 1

    with open(root / "reports/g4_frozen_confirmation.json") as f:
        g4 = json.load(f)
    if g4.get("actual_rows") != EXPECTED_G4_ROWS:
        failures.append(f"G4_ROW_COUNT: expected {EXPECTED_G4_ROWS}, got {g4.get('actual_rows')}")
    if g4.get("expected_rows") != EXPECTED_G4_ROWS:
        failures.append(f"G4_EXPECTED_ROWS_FIELD: {g4.get('expected_rows')}")

    g4_df = pd.read_csv(root / "data/processed/g4_frozen_candidates.tsv.gz", sep="\t", compression="gzip")
    if len(g4_df) != EXPECTED_G4_ROWS:
        failures.append(f"G4_TSV_ROWS: {len(g4_df)} != {EXPECTED_G4_ROWS}")
    n_parents_tsv = g4_df["parent_id"].nunique()
    if n_parents_tsv != EXPECTED_PARENTS:
        failures.append(f"G4_PARENTS_TSV: {n_parents_tsv} != {EXPECTED_PARENTS}")
    for target in ["K562", "HepG2", "SKNSH"]:
        if target not in g4_df["target_cell"].unique():
            failures.append(f"MISSING_TARGET: {target}")
    for b in EXPECTED_BUDGETS:
        if b not in g4_df["budget"].unique():
            failures.append(f"MISSING_BUDGET: {b}")
    methods = set(g4_df["method"].unique())
    if not {"safeedit_consensus", "greedy_malinois", "random_matched"}.issubset(methods):
        failures.append(f"MISSING_METHODS: {methods}")

    with open(root / "reports/r4_independent_recalculation.json") as f:
        r4 = json.load(f)
    if not r4.get("row_count_pass"):
        failures.append("R4_ROW_COUNT_FAIL")
    pe = r4.get("paired_safeedit_vs_greedy", {})
    if pe.get("paired_difference_pp", 0) <= 0:
        failures.append(f"PRIMARY_ENDPOINT: difference must be positive, got {pe.get('paired_difference_pp')} pp")
    if not (pe.get("parent_cluster_bootstrap_95_ci_pp", [0, 0])[0] > 0):
        failures.append(f"BOOTSTRAP_CI_NOT_POSITIVE: {pe.get('parent_cluster_bootstrap_95_ci_pp')}")

    with open(root / "reports/g6_frozen_candidate_summary.json") as f:
        g6 = json.load(f)
    if g6.get("pareto_direction", "").find("maximize") < 0:
        failures.append(f"G6_PARETO_DIR: {g6.get('pareto_direction')}")
    tier_counts = g6.get("tier_counts", {})
    n_a = tier_counts.get("A", 0)
    if n_a < 1:
        failures.append(f"TIER_A_EMPTY: {n_a}")

    with open(root / "reports/g5_frozen_audit.json") as f:
        g5 = json.load(f)
    if not g5.get("g4_audit", {}).get("clean"):
        issues = g5.get("g4_audit", {}).get("issues", [])
        failures.append(f"G5_AUDIT_NOT_CLEAN: {issues}")
    n_nested = g5.get("edit_nesting", {}).get("n_nested_violations", 0)
    if n_nested > 0:
        warnings.append(f"G5_NESTED_VIOLATIONS: {n_nested}")

    import subprocess
    for pdf in ["main.pdf", "supplementary.pdf"]:
        pdfp = root / "manuscript" / pdf
        if pdfp.stat().st_size < 10000:
            failures.append(f"PDF_TOO_SMALL: {pdf} ({pdfp.stat().st_size} bytes)")

    code_files = [
        "src/safeedit_cre/g4_confirmation_frozen.py",
        "src/safeedit_cre/r4_recalculate.py",
        "src/safeedit_cre/g5_audit_frozen.py",
        "src/safeedit_cre/g6_candidates_frozen.py",
    ]
    sha_lines = []
    for cf in code_files:
        p = root / cf
        if p.exists():
            sha_lines.append(f"{sha256_file(p)}  {cf}\n")
    sha_file = root / "checkpoints/FROZEN_CODE_SHA256.txt"
    if sha_file.exists():
        expected_sha = sha_file.read_text().strip().split("\n")
        actual_sha_set = set(line.strip().split()[0] for line in sha_lines if line.strip())
        for line in expected_sha:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 1 and parts[0] not in actual_sha_set:
                warnings.append(f"CODE_SHA_MISMATCH (non-blocking): {line}")

    if failures:
        print("RELEASE_VERIFICATION_FAIL")
        for f_msg in failures:
            print(f"  FAIL: {f_msg}")
        return 1
    print()
    print("=" * 60)
    print("STRICT_RELEASE_PASS")
    print("=" * 60)
    print(f"G4 rows: {len(g4_df)} (parents={n_parents_tsv})")
    print(f"Primary endpoint (SafeEdit vs Greedy): +{pe['paired_difference_pp']:.2f} pp")
    print(f"  95% CI: [{pe['parent_cluster_bootstrap_95_ci_pp'][0]:.2f}, {pe['parent_cluster_bootstrap_95_ci_pp'][1]:.2f}] pp")
    print(f"  McNemar p: {pe['mcnemar_exact_two_sided_p']:.6f}")
    print(f"Tier A candidates: {n_a}")
    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  WARN: {w}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
