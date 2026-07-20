"""R4: CPU-only independent statistical recalculation from frozen G4 candidates TSV.

This script performs all statistical calculations independently using only
numpy, scipy, and pandas. It does NOT import torch or any model code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, binom


def mcnemar_exact_two_sided(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(b, n=n, p=0.5).pvalue)


def parent_cluster_bootstrap_ci(
    parent_ids: np.ndarray,
    safeedit_pass: np.ndarray,
    greedy_pass: np.ndarray,
    n_bootstrap: int = 100000,
    seed: int = 20260714,
    alpha: float = 0.05,
) -> tuple[float, float, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique_parents = np.unique(parent_ids)
    n_clusters = len(unique_parents)
    parent_to_idx = {pid: i for i, pid in enumerate(unique_parents)}
    cluster_safe = np.zeros(n_clusters, dtype=np.int64)
    cluster_greedy = np.zeros(n_clusters, dtype=np.int64)
    cluster_n = np.zeros(n_clusters, dtype=np.int64)
    for i, pid in enumerate(parent_ids):
        cidx = parent_to_idx[pid]
        cluster_safe[cidx] += int(safeedit_pass[i])
        cluster_greedy[cidx] += int(greedy_pass[i])
        cluster_n[cidx] += 1
    boot_diffs = np.zeros(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        sel = rng.integers(0, n_clusters, size=n_clusters)
        bs = cluster_safe[sel].sum()
        bg = cluster_greedy[sel].sum()
        bn = cluster_n[sel].sum()
        if bn > 0:
            boot_diffs[b] = (bs - bg) / bn
        else:
            boot_diffs[b] = 0.0
    lo = float(np.percentile(boot_diffs, 100 * alpha / 2))
    hi = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))
    return lo, hi, boot_diffs


def compute_method_summary(df: pd.DataFrame) -> list[dict]:
    out = []
    for method, grp in df.groupby("method"):
        feasible = grp[grp["design_status"] == "feasible"]
        n_total = len(grp)
        n_feasible = len(feasible)
        n_accepted = int(grp["accepted"].sum())
        n_reviewer_transfer = int(feasible["check_reviewer_transfer_positive"].sum()) if "check_reviewer_transfer_positive" in feasible.columns else None
        out.append({
            "method": method,
            "n_total": n_total,
            "n_feasible": n_feasible,
            "accepted_n": n_accepted,
            "accepted_fraction": float(grp["accepted"].mean()),
            "feasible_accepted_fraction": float(feasible["accepted"].mean()) if n_feasible > 0 else None,
            "reviewer_transfer_fraction": float(n_reviewer_transfer / n_feasible) if n_reviewer_transfer is not None and n_feasible > 0 else None,
        })
    return out


def compute_paired_endpoint(df: pd.DataFrame) -> dict:
    safe_all = df[df["method"] == "safeedit_consensus"].copy()
    greedy_all = df[df["method"] == "greedy_malinois"].copy()
    merge_keys = ["parent_id", "target_cell", "budget"]
    safe = safe_all[safe_all["design_status"] == "feasible"].set_index(merge_keys)
    greedy = greedy_all.set_index(merge_keys)
    common = safe.index.intersection(greedy.index)
    s_pass = safe.loc[common, "accepted"].astype(bool).to_numpy()
    g_pass = greedy.loc[common, "accepted"].astype(bool).to_numpy()
    parent_ids = safe.loc[common].index.get_level_values("parent_id").to_numpy()
    n_pairs = len(common)
    s_frac = float(s_pass.mean())
    g_frac = float(g_pass.mean())
    diff = s_frac - g_frac
    b = int(((s_pass) & (~g_pass)).sum())
    c = int(((~s_pass) & (g_pass)).sum())
    mcnemar_p = mcnemar_exact_two_sided(b, c)
    n_parent_clusters = len(np.unique(parent_ids))
    ci_lo, ci_hi, _ = parent_cluster_bootstrap_ci(parent_ids, s_pass.astype(int), g_pass.astype(int))
    return {
        "n_condition_pairs": n_pairs,
        "n_parent_clusters": n_parent_clusters,
        "safeedit_fraction": s_frac,
        "greedy_fraction_on_same_pairs": g_frac,
        "paired_difference": diff,
        "paired_difference_pp": diff * 100,
        "parent_cluster_bootstrap_95_ci": [ci_lo, ci_hi],
        "parent_cluster_bootstrap_95_ci_pp": [ci_lo * 100, ci_hi * 100],
        "discordant_safeedit_only": b,
        "discordant_greedy_only": c,
        "mcnemar_exact_two_sided_p": mcnemar_p,
        "note": "Condition-level McNemar ignores within-parent dependence; parent-cluster bootstrap is primary inference.",
    }


def compute_stratified_paired(df: pd.DataFrame) -> list[dict]:
    out = []
    safe = df[(df["method"] == "safeedit_consensus") & (df["design_status"] == "feasible")].copy()
    greedy = df[df["method"] == "greedy_malinois"].copy()
    merge_keys = ["parent_id", "target_cell", "budget"]
    safe_idx = safe.set_index(merge_keys)
    greedy_idx = greedy.set_index(merge_keys)
    for (target, budget), grp_s in safe.groupby(["target_cell", "budget"]):
        grp_g = greedy_idx[(greedy_idx["target_cell"] == target) & (greedy_idx["budget"] == budget)] if "target_cell" in greedy_idx.columns else greedy.loc[greedy["budget"] == budget]
        s_idx = grp_s.set_index(merge_keys).index
        g_sub = greedy[greedy["budget"] == budget]
        g_sub = g_sub[g_sub["target_cell"] == target].set_index(merge_keys)
        common = s_idx.intersection(g_sub.index)
        if len(common) == 0:
            continue
        sv = grp_s.set_index(merge_keys).loc[common, "accepted"].astype(bool).to_numpy()
        gv = g_sub.loc[common, "accepted"].astype(bool).to_numpy()
        s_n = int(sv.sum())
        g_n = int(gv.sum())
        n_pairs = len(common)
        b = int(((sv) & (~gv)).sum())
        c = int(((~sv) & (gv)).sum())
        out.append({
            "target_cell": target,
            "budget": int(budget),
            "n_pairs": n_pairs,
            "safeedit_accepted_n": s_n,
            "greedy_accepted_n": g_n,
            "acceptance_difference": (s_n - g_n) / n_pairs,
            "discordant_safeedit_only": b,
            "discordant_greedy_only": c,
            "mcnemar_exact_two_sided_p": mcnemar_exact_two_sided(b, c),
        })
    return out


def compute_safeedit_vs_random_paired(df: pd.DataFrame) -> dict:
    safe = df[(df["method"] == "safeedit_consensus") & (df["design_status"] == "feasible")].copy()
    rand = df[df["method"] == "random_matched"].copy()
    merge_keys = ["parent_id", "target_cell", "budget"]
    s_idx = safe.set_index(merge_keys)
    r_idx = rand.set_index(merge_keys)
    common = s_idx.index.intersection(r_idx.index)
    sv = s_idx.loc[common, "accepted"].astype(bool).to_numpy()
    rv = r_idx.loc[common, "accepted"].astype(bool).to_numpy()
    parent_ids = s_idx.loc[common].index.get_level_values("parent_id").to_numpy()
    b = int((sv & ~rv).sum())
    c = int((~sv & rv).sum())
    ci_lo, ci_hi, _ = parent_cluster_bootstrap_ci(parent_ids, sv.astype(int), rv.astype(int))
    return {
        "n_pairs": len(common),
        "safeedit_fraction": float(sv.mean()),
        "random_fraction": float(rv.mean()),
        "paired_difference": float(sv.mean() - rv.mean()),
        "paired_difference_pp": float((sv.mean() - rv.mean()) * 100),
        "discordant_safeedit_only": b,
        "discordant_random_only": c,
        "mcnemar_exact_two_sided_p": mcnemar_exact_two_sided(b, c),
        "parent_cluster_bootstrap_95_ci": [ci_lo, ci_hi],
    }


def check_edit_nesting(df: pd.DataFrame) -> dict:
    feasible = df[df["design_status"] == "feasible"]
    violations = 0
    total_transitions = 0
    violation_examples = []
    for (pid, target, method), grp in feasible.groupby(["parent_id", "target_cell", "method"]):
        grp_sorted = grp.sort_values("budget")
        prev_edits: set = set()
        for _, row in grp_sorted.iterrows():
            edits_str = str(row.get("edit_string", ""))
            edits = set(e.strip() for e in edits_str.split(";") if e.strip())
            total_transitions += 1
            if not edits.issuperset(prev_edits):
                violations += 1
                if len(violation_examples) < 5:
                    violation_examples.append(f"{pid}/{target}/{method}/b{row['budget']}")
            prev_edits = edits
    return {
        "nonnested_transitions": violations,
        "total_cross_budget_transitions": total_transitions,
        "examples": violation_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g4", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=3456)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.g4, sep="\t", compression="gzip")
    n = len(df)
    assert n == args.expected_rows, f"Expected {args.expected_rows} rows, got {n}"
    assert {"parent_id", "target_cell", "budget", "method", "accepted", "design_status"}.issubset(df.columns)
    n_parents = df["parent_id"].nunique()
    n_targets = df["target_cell"].nunique()
    n_methods = df["method"].nunique()
    n_feasible = int((df["design_status"] == "feasible").sum())
    n_infeasible = int((df["design_status"] == "infeasible").sum())
    summary = {
        "status": "R4_INDEPENDENT_RECALCULATION",
        "frozen": True,
        "expected_rows": args.expected_rows,
        "actual_rows": n,
        "row_count_pass": n == args.expected_rows,
        "n_parents": n_parents,
        "n_targets": n_targets,
        "n_methods": n_methods,
        "n_feasible": n_feasible,
        "n_infeasible": n_infeasible,
        "chr_distribution": df.drop_duplicates("parent_id")["chr"].value_counts().to_dict() if "chr" in df.columns else None,
    }
    summary["method_summary"] = compute_method_summary(df)
    summary["paired_safeedit_vs_greedy"] = compute_paired_endpoint(df)
    summary["paired_safeedit_vs_random"] = compute_safeedit_vs_random_paired(df)
    summary["stratified_paired_safeedit_vs_greedy"] = compute_stratified_paired(df)
    summary["edit_nesting"] = check_edit_nesting(df)
    total_accepted = int(df["accepted"].sum())
    summary["total_accepted"] = total_accepted
    summary["total_accepted_fraction"] = float(df["accepted"].mean())
    feasible = df[df["design_status"] == "feasible"]
    summary["feasible_accepted_fraction_overall"] = float(feasible["accepted"].mean()) if n_feasible > 0 else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("=== R4 CPU Independent Recalculation ===")
    print(f"Rows: {n} (expected {args.expected_rows}) -> {'PASS' if summary['row_count_pass'] else 'FAIL'}")
    print(f"Parents: {n_parents}, Feasible: {n_feasible}, Infeasible: {n_infeasible}")
    ms = summary["method_summary"]
    for m in ms:
        print(f"  {m['method']}: {m['accepted_n']}/{m['n_total']} accepted ({m['accepted_fraction']:.2%})")
    pe = summary["paired_safeedit_vs_greedy"]
    print(f"\nPrimary endpoint (SafeEdit vs Greedy, paired):")
    print(f"  SafeEdit: {pe['safeedit_fraction']:.2%}, Greedy: {pe['greedy_fraction_on_same_pairs']:.2%}")
    print(f"  Difference: +{pe['paired_difference_pp']:.2f} pp")
    print(f"  95% CI (parent-cluster bootstrap): [{pe['parent_cluster_bootstrap_95_ci_pp'][0]:.2f}, {pe['parent_cluster_bootstrap_95_ci_pp'][1]:.2f}] pp")
    print(f"  Discordant: SE-only={pe['discordant_safeedit_only']}, Greedy-only={pe['discordant_greedy_only']}")
    print(f"  McNemar exact two-sided p: {pe['mcnemar_exact_two_sided_p']:.6f}")
    pr = summary["paired_safeedit_vs_random"]
    print(f"\nSafeEdit vs Random: +{pr['paired_difference_pp']:.2f} pp, p={pr['mcnemar_exact_two_sided_p']:.2e}")
    print(f"\nR4 results written to {args.output}")


if __name__ == "__main__":
    main()
