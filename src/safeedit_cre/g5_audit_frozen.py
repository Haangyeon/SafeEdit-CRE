"""G5 Frozen: Audit and robustness analysis for frozen G4 confirmation results."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

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
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        if col == "primary_margin_gain" or col == "reviewer_margin_gain":
            continue
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            n_bad = (~np.isfinite(df[col].to_numpy(dtype=float))).sum()
            if n_bad > 0:
                issues.append(f"non-finite values in {col}: {n_bad}")
    feasible = df[df["design_status"] == "feasible"]
    if "hamming_distance" in feasible.columns:
        if feasible["hamming_distance"].ne(feasible["budget"]).any():
            bad = feasible["hamming_distance"].ne(feasible["budget"]).sum()
            issues.append(f"{bad} feasible rows violate exact edit budget")
    dup = df.duplicated(subset=["parent_id", "target_cell", "budget", "method", "sequence"]).sum()
    if dup > 0:
        issues.append(f"{dup} duplicate sequence rows")
    failure_reasons = Counter()
    for fc in df["failed_checks"].dropna():
        for f in str(fc).split(";"):
            f = f.strip()
            if f and f != "infeasible":
                failure_reasons[f] += 1
    infeasible_n = int((df["design_status"] == "infeasible").sum())
    summary = {
        "name": name,
        "n_rows": n,
        "n_feasible": int(len(df) - infeasible_n),
        "n_infeasible": infeasible_n,
        "methods": sorted(df["method"].unique().tolist()),
        "targets": sorted(df["target_cell"].unique().tolist()),
        "budgets": sorted(int(b) for b in df["budget"].unique()),
        "accepted_n": int(df["accepted"].sum()),
        "accepted_fraction": float(df["accepted"].mean()),
        "feasible_accepted_fraction": float(df.loc[df["design_status"] == "feasible", "accepted"].mean()) if (df["design_status"] == "feasible").any() else None,
        "failure_reason_counts": dict(failure_reasons),
        "issues": issues,
        "clean": len(issues) == 0,
    }
    by_method_budget = []
    for (method, budget), grp in df.groupby(["method", "budget"]):
        feasible_grp = grp[grp["design_status"] == "feasible"]
        by_method_budget.append({
            "method": method,
            "budget": int(budget),
            "n_total": len(grp),
            "n_feasible": len(feasible_grp),
            "accepted_n": int(grp["accepted"].sum()),
            "accepted_fraction": float(grp["accepted"].mean()),
            "mean_primary_margin_gain": float(feasible_grp["primary_margin_gain"].replace([np.inf, -np.inf], np.nan).dropna().mean()) if len(feasible_grp) > 0 and feasible_grp["primary_margin_gain"].replace([np.inf, -np.inf], np.nan).notna().any() else None,
            "mean_reviewer_margin_gain": float(feasible_grp["reviewer_margin_gain"].replace([np.inf, -np.inf], np.nan).dropna().mean()) if len(feasible_grp) > 0 and feasible_grp["reviewer_margin_gain"].replace([np.inf, -np.inf], np.nan).notna().any() else None,
        })
    summary["by_method_budget"] = by_method_budget
    return summary


def ablation_analysis(df: pd.DataFrame) -> list[dict[str, object]]:
    results = []
    for abl_name, config in ABLATION_CONFIGS.items():
        acc = compute_accepted(df, config)
        for method in df["method"].unique():
            for budget in sorted(df["budget"].unique()):
                mask = (df["method"] == method) & (df["budget"] == budget) & (df["design_status"] == "feasible")
                grp = df.loc[mask]
                accepted_here = acc.loc[mask]
                results.append({
                    "ablation": abl_name,
                    "method": method,
                    "budget": int(budget),
                    "n": len(grp),
                    "accepted_n": int(accepted_here.sum()),
                    "accepted_fraction": float(accepted_here.mean()) if len(grp) > 0 else 0.0,
                })
    return results


def constraint_satisfaction(df: pd.DataFrame) -> dict[str, object]:
    out = {}
    feasible = df[df["design_status"] == "feasible"]
    for method in df["method"].unique():
        grp = feasible[feasible["method"] == method]
        rates = {}
        for c in AUDIT_CHECKS:
            rates[c] = float(grp[c].mean()) if c in grp.columns and len(grp) > 0 else None
        n_satisfied = grp[AUDIT_CHECKS].sum(axis=1) if all(c in grp.columns for c in AUDIT_CHECKS) else pd.Series([0]*len(grp))
        out[method] = {
            "per_check_rates": rates,
            "n_constraints_mean": float(n_satisfied.mean()) if len(grp) > 0 else None,
        }
    return out


def nested_edit_check(df: pd.DataFrame) -> dict[str, object]:
    issues = []
    feasible = df[df["design_status"] == "feasible"]
    for (pid, target, method), grp in feasible.groupby(["parent_id", "target_cell", "method"]):
        grp_sorted = grp.sort_values("budget")
        prev_edits: set[str] = set()
        for _, row in grp_sorted.iterrows():
            edits: set[str] = set()
            es = str(row.get("edit_string", "")).split(";")
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
    parser.add_argument("--g4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    g4 = pd.read_csv(args.g4, sep="\t", compression="gzip")
    n_expected = 3456
    g4_audit = audit_table(g4, "G4_frozen_confirmation", n_expected)
    nested = nested_edit_check(g4)
    g4_ablation = ablation_analysis(g4)
    g4_constraints = constraint_satisfaction(g4)

    by_method_target_budget = []
    for (method, target, budget), grp in g4.groupby(["method", "target_cell", "budget"]):
        feasible_grp = grp[grp["design_status"] == "feasible"]
        by_method_target_budget.append({
            "method": method,
            "target_cell": target,
            "budget": int(budget),
            "n_total": len(grp),
            "n_feasible": len(feasible_grp),
            "accepted_fraction": float(grp["accepted"].mean()),
            "mean_primary_margin_gain": float(feasible_grp["primary_margin_gain"].replace([np.inf, -np.inf], np.nan).dropna().mean()) if len(feasible_grp) > 0 else None,
            "mean_reviewer_margin_gain": float(feasible_grp["reviewer_margin_gain"].replace([np.inf, -np.inf], np.nan).dropna().mean()) if len(feasible_grp) > 0 else None,
        })

    report = {
        "purpose": "G5 frozen audit and ablation on 96-parent confirmation set",
        "frozen": True,
        "expected_rows": n_expected,
        "g4_audit": g4_audit,
        "edit_nesting": nested,
        "ablation": g4_ablation,
        "constraint_satisfaction": g4_constraints,
        "stratified_results": by_method_target_budget,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"G5 frozen audit complete. Clean: {g4_audit['clean']}")
    print(f"Accepted: {g4_audit['accepted_n']}/{g4_audit['n_rows']} ({g4_audit['accepted_fraction']:.2%})")
    print(f"Nested violations: {nested['n_nested_violations']}")


if __name__ == "__main__":
    main()
