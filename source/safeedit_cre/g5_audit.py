"""G5: Audit, ablation, and robustness analysis for G3+G4 results."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from safeedit_cre.baseline import load_pilot
from safeedit_cre.edit_benchmark import deterministic_parents

ROOT = Path(__file__).resolve().parents[1]

AUDIT_CHECKS = [
    "check_primary_target_nonnegative",
    "check_primary_margin_positive",
    "check_reviewer_transfer_positive",
    "check_strand_in_domain",
    "check_reviewer_uncertainty_in_domain",
    "check_naturalness_in_domain",
    "check_gc_in_domain",
    "check_homopolymer_safe",
    "check_exact_edit_budget",
]

ABLATION_CONFIGS = {
    "full_safeedit": {c: True for c in AUDIT_CHECKS},
    "no_cnn_transfer": {c: True for c in AUDIT_CHECKS} | {"check_reviewer_transfer_positive": False},
    "no_strand": {c: True for c in AUDIT_CHECKS} | {"check_strand_in_domain": False},
    "no_uncertainty": {c: True for c in AUDIT_CHECKS} | {"check_reviewer_uncertainty_in_domain": False},
    "no_naturalness": {c: True for c in AUDIT_CHECKS} | {"check_naturalness_in_domain": False},
    "no_gc": {c: True for c in AUDIT_CHECKS} | {"check_gc_in_domain": False},
    "no_homopolymer": {c: True for c in AUDIT_CHECKS} | {"check_homopolymer_safe": False},
    "malinois_only": {c: False for c in AUDIT_CHECKS} | {
        "check_primary_target_nonnegative": True,
        "check_primary_margin_positive": True,
        "check_exact_edit_budget": True,
    },
    "malinois_cnn": {c: False for c in AUDIT_CHECKS} | {
        "check_primary_target_nonnegative": True,
        "check_primary_margin_positive": True,
        "check_reviewer_transfer_positive": True,
        "check_exact_edit_budget": True,
    },
}


def compute_accepted(df: pd.DataFrame, config: dict[str, bool]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for check, required in config.items():
        if required and check in df.columns:
            mask &= df[check].astype(bool)
    return mask


def audit_table(df: pd.DataFrame, name: str, expected_rows: int) -> dict[str, object]:
    issues: list[str] = []
    n = len(df)
    if n != expected_rows:
        issues.append(f"expected {expected_rows} rows, got {n}")
    numeric = df.select_dtypes(include="number")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        issues.append("non-finite numeric values present")
    if df["hamming_distance"].ne(df["budget"]).any():
        bad = df["hamming_distance"].ne(df["budget"]).sum()
        issues.append(f"{bad} rows violate exact edit budget")
    if df["sequence"].str.len().ne(200).any():
        issues.append("sequences not length 200")
    dup = df.duplicated(subset=["parent_id", "target_cell", "budget", "method", "sequence"]).sum()
    if dup > 0:
        issues.append(f"{dup} duplicate rows")
    na_counts = df.isna().sum()
    critical_na = {col: int(cnt) for col, cnt in na_counts.items() if cnt > 0 and col not in ("failed_checks",)}
    if critical_na:
        issues.append(f"unexpected NA in columns: {critical_na}")
    failure_reasons = Counter()
    for fc in df["failed_checks"].dropna():
        for f in str(fc).split(";"):
            f = f.strip()
            if f:
                failure_reasons[f] += 1
    summary = {
        "n_rows": n,
        "methods": sorted(df["method"].unique().tolist()),
        "targets": sorted(df["target_cell"].unique().tolist()),
        "budgets": sorted(df["budget"].unique().tolist()),
        "accepted_n": int(df["accepted"].sum()),
        "accepted_fraction": float(df["accepted"].mean()),
        "failure_reason_counts": dict(failure_reasons),
        "issues": issues,
        "clean": len(issues) == 0,
    }
    by_method_budget = []
    for (method, budget), grp in df.groupby(["method", "budget"]):
        by_method_budget.append({
            "method": method,
            "budget": int(budget),
            "n": len(grp),
            "accepted_n": int(grp["accepted"].sum()),
            "accepted_fraction": float(grp["accepted"].mean()),
            "mean_primary_margin_gain": float(grp["primary_margin_gain"].mean()),
            "mean_reviewer_margin_gain": float(grp["reviewer_margin_gain"].mean()),
            "reviewer_positive_fraction": float((grp["reviewer_margin_gain"] > 0).mean()),
        })
    summary["by_method_budget"] = by_method_budget
    return summary


def parent_overlap(g3: pd.DataFrame, g4: pd.DataFrame) -> dict[str, object]:
    g3_ids = set(g3["parent_id"].unique())
    g4_ids = set(g4["parent_id"].unique())
    overlap = g3_ids & g4_ids
    return {"g3_n_parents": len(g3_ids), "g4_n_parents": len(g4_ids), "overlap_n": len(overlap), "overlap_ids": sorted(list(overlap))[:10]}


def ablation_analysis(df: pd.DataFrame, phase: str) -> list[dict[str, object]]:
    results = []
    for abl_name, config in ABLATION_CONFIGS.items():
        acc = compute_accepted(df, config)
        for method in df["method"].unique():
            for budget in sorted(df["budget"].unique()):
                mask = (df["method"] == method) & (df["budget"] == budget)
                grp = df.loc[mask]
                accepted_here = acc.loc[mask]
                results.append({
                    "phase": phase,
                    "ablation": abl_name,
                    "method": method,
                    "budget": int(budget),
                    "n": len(grp),
                    "accepted_n": int(accepted_here.sum()),
                    "accepted_fraction": float(accepted_here.mean()),
                    "mean_primary_margin_gain": float(grp.loc[accepted_here, "primary_margin_gain"].mean()) if accepted_here.any() else None,
                    "mean_reviewer_margin_gain": float(grp.loc[accepted_here, "reviewer_margin_gain"].mean()) if accepted_here.any() else None,
                })
    return results


def threshold_sensitivity(df: pd.DataFrame, n_steps: int = 9) -> list[dict[str, object]]:
    base_thresh_keys = ["strand", "uncertainty", "naturalness", "gc"]
    multipliers = np.linspace(0.5, 2.0, n_steps)
    results = []
    for key in base_thresh_keys:
        check_col = {
            "strand": "check_strand_in_domain",
            "uncertainty": "check_reviewer_uncertainty_in_domain",
            "naturalness": "check_naturalness_in_domain",
            "gc": "check_gc_in_domain",
        }[key]
        col = {
            "strand": "primary_strand_disagreement",
            "uncertainty": "reviewer_uncertainty",
            "naturalness": "naturalness_delta",
            "gc": "absolute_gc_delta",
        }[key]
        for m in multipliers:
            if key in ("naturalness",):
                relaxed = df[col] >= df[col].quantile(0.05) * m if False else True
                if m >= 1.0:
                    mask = pd.Series(True, index=df.index)
                else:
                    q_lo = df[col].quantile(m)
                    mask = df[col] >= q_lo
            else:
                if m >= 1.0:
                    mask = pd.Series(True, index=df.index)
                else:
                    q_hi = df[col].quantile(m)
                    mask = df[col] <= q_hi
            base_accepted = df["accepted"].copy()
            relaxed_accepted = base_accepted & mask if m < 1.0 else base_accepted | (df[check_col] if m > 1.0 else df[check_col])
            for method in ["greedy_malinois", "safeedit_consensus"]:
                grp = df[df["method"] == method]
                results.append({
                    "threshold": key,
                    "multiplier": float(m),
                    "method": method,
                    "accepted_fraction": float(relaxed_accepted.reindex(grp.index).fillna(False).mean()),
                    "post_hoc": True,
                })
    return results


def constraint_satisfaction(df: pd.DataFrame, phase: str) -> dict[str, object]:
    out = {}
    for method in df["method"].unique():
        grp = df[df["method"] == method]
        rates = {}
        for c in AUDIT_CHECKS:
            rates[c] = float(grp[c].mean()) if c in grp.columns else None
        n_satisfied = grp[AUDIT_CHECKS].sum(axis=1)
        out[method] = {
            "phase": phase,
            "per_check_rates": rates,
            "n_constraints_mean": float(n_satisfied.mean()),
            "n_constraints_satisfied_distribution": {int(k): int((n_satisfied == k).sum()) for k in range(len(AUDIT_CHECKS) + 1)},
        }
    return out


def nested_edit_check(df: pd.DataFrame) -> dict[str, object]:
    issues = []
    for (pid, target, method), grp in df.groupby(["parent_id", "target_cell", "method"]):
        grp_sorted = grp.sort_values("budget")
        prev_edits = set()
        for _, row in grp_sorted.iterrows():
            edits = set()
            es = str(row["edit_string"]).split(";")
            for e in es:
                e = e.strip()
                if e:
                    edits.add(e)
            if not edits.issuperset(prev_edits):
                issues.append(f"{pid}/{target}/{method}/b{row['budget']}: edits not nested")
                break
            prev_edits = edits
    return {"n_nested_violations": len(issues), "examples": issues[:5]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g3", type=Path, default=ROOT / "data" / "processed" / "g3_candidates.tsv.gz")
    parser.add_argument("--g4", type=Path, default=ROOT / "data" / "processed" / "g4_candidates.tsv.gz")
    parser.add_argument("--pilot", type=Path, default=ROOT / "data" / "processed" / "pilot_30k_5k_5k.tsv.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "g5_audit_ablation.json")
    args = parser.parse_args()

    g3 = pd.read_csv(args.g3, sep="\t", compression="gzip")
    g4 = pd.read_csv(args.g4, sep="\t", compression="gzip")

    n_g3_expected = 24 * 3 * 4 * 2
    n_g4_expected = 48 * 3 * 4 * 3
    g3_audit = audit_table(g3, "G3_development", n_g3_expected)
    g4_audit = audit_table(g4, "G4_confirmation", n_g4_expected)
    overlap = parent_overlap(g3, g4)
    nested = nested_edit_check(g4)

    g3_ablation = ablation_analysis(g3, "G3_development")
    g4_ablation = ablation_analysis(g4, "G4_confirmation")
    g4_sensitivity = threshold_sensitivity(g4)
    g3_constraints = constraint_satisfaction(g3, "G3_development")
    g4_constraints = constraint_satisfaction(g4, "G4_confirmation")

    by_phase = []
    for phase, df in [("G3_development", g3), ("G4_confirmation", g4)]:
        for (method, target, budget), grp in df.groupby(["method", "target_cell", "budget"]):
            acc = grp["accepted"].mean()
            mmg = grp["primary_margin_gain"].mean()
            rmg = grp["reviewer_margin_gain"].mean()
            rpos = (grp["reviewer_margin_gain"] > 0).mean()
            ppos = (grp["primary_target_gain"] >= 0).mean()
            by_phase.append({
                "phase": phase,
                "method": method,
                "target_cell": target,
                "budget": int(budget),
                "n": len(grp),
                "accepted_fraction": float(acc),
                "mean_primary_margin_gain": float(mmg),
                "mean_primary_target_gain": float(grp["primary_target_gain"].mean()),
                "mean_reviewer_margin_gain": float(rmg),
                "reviewer_positive_transfer_fraction": float(rpos),
                "target_nonnegative_fraction": float(ppos),
            })

    report = {
        "purpose": "G5 audit, ablation, sensitivity on locked G3+G4 results",
        "test_labels_used_for_selection_or_tuning": False,
        "g3_audit": g3_audit,
        "g4_audit": g4_audit,
        "parent_overlap": overlap,
        "edit_nesting": nested,
        "ablation": g3_ablation + g4_ablation,
        "threshold_sensitivity_post_hoc": g4_sensitivity,
        "constraint_satisfaction": {**g3_constraints, **g4_constraints},
        "stratified_results": by_phase,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"G5 audit complete. Clean G3: {g3_audit['clean']}, Clean G4: {g4_audit['clean']}")
    print(f"G3/G4 parent overlap: {overlap['overlap_n']}")
    print(f"Edit nesting violations: {nested['n_nested_violations']}")


if __name__ == "__main__":
    main()
