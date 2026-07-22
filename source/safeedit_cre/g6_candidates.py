"""G6: Candidate tiering, Pareto analysis, and biological interpretation."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASES = list("ACGT")
BUDGETS = (1, 5, 10, 20)

SYNTHETIC_RISK_FLAGS = [
    ("homopolymer_run_ge_8", lambda df: df["max_homopolymer"] >= 8),
    ("extreme_gc_delta_gt_0p08", lambda df: df["absolute_gc_delta"] > 0.08),
    ("severe_naturalness_drop_lt_m3sd", None),
    ("strand_disagreement_gt_2x_threshold", None),
    ("reviewer_uncertainty_gt_2x_threshold", None),
]


def compute_pareto_front(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    maximize_x: bool = True,
    maximize_y: bool = True,
) -> np.ndarray:
    """Return boolean mask of Pareto-optimal points (maximize both x and y by default)."""
    n = len(df)
    is_pareto = np.ones(n, dtype=bool)
    xs = df[x_col].to_numpy(dtype=float)
    ys = df[y_col].to_numpy(dtype=float)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            better_x = (xs[j] >= xs[i]) if maximize_x else (xs[j] <= xs[i])
            better_y = (ys[j] >= ys[i]) if maximize_y else (ys[j] <= ys[i])
            strict_x = (xs[j] > xs[i]) if maximize_x else (xs[j] < xs[i])
            strict_y = (ys[j] > ys[i]) if maximize_y else (ys[j] < ys[i])
            strict = strict_x or strict_y
            if better_x and better_y and strict:
                is_pareto[i] = False
                break
    return is_pareto


def compute_near_pareto(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    is_pareto: np.ndarray,
    rel_tol: float = 0.05,
    abs_tol: float = 0.1,
) -> np.ndarray:
    """Return boolean mask of near-Pareto points.
    
    A point is near-Pareto if it is either on the Pareto front, or:
      - its x value is within rel_tol (5%) of the Pareto front's max x at or above its y, OR
      - its absolute x difference from any Pareto point with >= y is within abs_tol
    """
    n = len(df)
    if n == 0:
        return is_pareto.copy()
    near_pareto = is_pareto.copy()
    xs = df[x_col].to_numpy(dtype=float)
    ys = df[y_col].to_numpy(dtype=float)
    pareto_mask = is_pareto.astype(bool)
    pareto_xs = xs[pareto_mask]
    pareto_ys = ys[pareto_mask]
    if len(pareto_xs) == 0:
        return near_pareto
    for i in range(n):
        if near_pareto[i]:
            continue
        xi, yi = xs[i], ys[i]
        dominated = False
        best_x_at_or_above_yi = -np.inf
        for px, py in zip(pareto_xs, pareto_ys, strict=True):
            if py >= yi - 1e-12:
                if px > best_x_at_or_above_yi:
                    best_x_at_or_above_yi = px
            if px >= xi - 1e-12 and py >= yi - 1e-12 and (px > xi + 1e-12 or py > yi + 1e-12):
                dominated = True
        if not dominated:
            near_pareto[i] = True
            continue
        if best_x_at_or_above_yi == -np.inf:
            continue
        x_range = max(np.ptp(pareto_xs), 1e-12)
        rel_ok = (best_x_at_or_above_yi - xi) <= rel_tol * max(abs(best_x_at_or_above_yi), 1e-12)
        abs_ok = (best_x_at_or_above_yi - xi) <= abs_tol * x_range
        if rel_ok or abs_ok:
            near_pareto[i] = True
    return near_pareto


def base_changes(edit_string: str) -> list[tuple[int, str, str]]:
    out = []
    for token in str(edit_string).split(";"):
        token = token.strip()
        if not token:
            continue
        try:
            pos_str, rest = token[:-3], token[-3:]
            pos = int(pos_str)
            ref, alt = rest[0], rest[2]
            out.append((pos, ref, alt))
        except Exception:
            continue
    return out


def substitution_type(ref: str, alt: str) -> str:
    purines = {"A", "G"}
    if (ref in purines) == (alt in purines):
        return "transition" if {ref, alt} in ({"A", "G"}, {"C", "T"}) else "transversion_sameclass"
    return "transversion"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g3", type=Path, default=ROOT / "data" / "processed" / "g3_candidates.tsv.gz")
    parser.add_argument("--g4", type=Path, default=ROOT / "data" / "processed" / "g4_candidates.tsv.gz")
    parser.add_argument("--g4-report", type=Path, default=ROOT / "reports" / "g4_confirmation.json")
    parser.add_argument("--output-library", type=Path, default=ROOT / "data" / "processed" / "final_candidate_library.tsv.gz")
    parser.add_argument("--output-summary", type=Path, default=ROOT / "reports" / "final_candidate_summary.json")
    args = parser.parse_args()

    g3 = pd.read_csv(args.g3, sep="\t", compression="gzip")
    g4 = pd.read_csv(args.g4, sep="\t", compression="gzip")
    g3["phase"] = "G3_development"
    g4["phase"] = "G4_confirmation"
    df = pd.concat([g3, g4], ignore_index=True)

    nat_mean = df["naturalness_delta"].mean()
    nat_std = df["naturalness_delta"].std()
    strand_lo = df.loc[df["accepted"], "primary_strand_disagreement"].quantile(0.99)
    uncert_hi = df.loc[df["accepted"], "reviewer_uncertainty"].quantile(0.99)

    df["synthetic_risk_flags"] = ""
    risk_counter = Counter()
    risk_mask = pd.Series(False, index=df.index)
    risk_details: list[list[str]] = [[] for _ in range(len(df))]
    for flag_name, fn in SYNTHETIC_RISK_FLAGS:
        if flag_name == "severe_naturalness_drop_lt_m3sd":
            mask = df["naturalness_delta"] < (nat_mean - 3 * nat_std)
        elif flag_name == "strand_disagreement_gt_2x_threshold":
            mask = df["primary_strand_disagreement"] > 2 * strand_lo
        elif flag_name == "reviewer_uncertainty_gt_2x_threshold":
            mask = df["reviewer_uncertainty"] > 2 * uncert_hi
        else:
            mask = fn(df)
        for idx in df.index[mask]:
            risk_details[idx].append(flag_name)
        risk_counter[flag_name] = int(mask.sum())
        risk_mask |= mask
    df["synthetic_risk_flags"] = [";".join(d) for d in risk_details]
    df["has_synthetic_risk"] = risk_mask.values

    df["pareto_front"] = False
    df["near_pareto"] = False
    for (target, budget), grp_idx in df.groupby(["target_cell", "budget"]).groups.items():
        grp = df.loc[grp_idx]
        audit_pass_mask = grp["accepted"].astype(bool)
        candidates = grp[audit_pass_mask]
        if len(candidates) < 2:
            if len(candidates) == 1:
                df.loc[candidates.index[0], "pareto_front"] = True
                df.loc[candidates.index[0], "near_pareto"] = True
            continue
        pf = compute_pareto_front(candidates, "primary_margin_gain", "reviewer_margin_gain", maximize_x=True, maximize_y=True)
        npf = compute_near_pareto(candidates, "primary_margin_gain", "reviewer_margin_gain", pf, rel_tol=0.05, abs_tol=0.1)
        df.loc[candidates.index[pf], "pareto_front"] = True
        df.loc[candidates.index[npf], "near_pareto"] = True

    df["audit_status"] = np.where(df["accepted"], "pass", "fail")
    method_map = {
        "random_matched": "random",
        "greedy_malinois": "greedy",
        "safeedit_consensus": "safeedit",
    }
    df["design_method"] = df["method"].map(method_map).fillna(df["method"])

    priority_tiers = []
    for idx, row in df.iterrows():
        audit_pass = bool(row["audit_status"] == "pass")
        is_safeedit = bool(row["design_method"] == "safeedit")
        tg_ok = bool(row["primary_target_gain"] >= 0)
        pmg_ok = bool(row["primary_margin_gain"] > 0)
        rmg_ok = bool(row["reviewer_margin_gain"] > 0)
        exact = bool(row["check_exact_edit_budget"])
        syn_safe = not bool(row["has_synthetic_risk"])
        on_pareto = bool(row["pareto_front"]) or bool(row["near_pareto"])
        core_quality = audit_pass and tg_ok and pmg_ok and rmg_ok and exact
        if is_safeedit and core_quality and syn_safe and on_pareto:
            priority_tiers.append("A")
        elif core_quality:
            priority_tiers.append("B")
        else:
            priority_tiers.append("C")
    df["priority_tier"] = priority_tiers

    cols_order = [
        "parent_id", "parent_sequence", "sequence", "target_cell", "budget", "edit_string",
        "hamming_distance", "primary_parent_target", "primary_candidate_target", "primary_target_gain",
        "primary_parent_margin", "primary_candidate_margin", "primary_margin_gain",
        "primary_strand_disagreement", "reviewer_parent_margin", "reviewer_candidate_margin",
        "reviewer_margin_gain", "reviewer_uncertainty", "reviewer_parent_uncertainty",
        "naturalness_delta", "absolute_gc_delta", "max_homopolymer",
        "audit_status", "design_method", "priority_tier",
        "accepted", "failed_checks", "check_primary_target_nonnegative", "check_primary_margin_positive",
        "check_reviewer_transfer_positive", "check_strand_in_domain", "check_reviewer_uncertainty_in_domain",
        "check_naturalness_in_domain", "check_gc_in_domain", "check_homopolymer_safe", "check_exact_edit_budget",
        "phase", "method", "pareto_front", "near_pareto", "has_synthetic_risk", "synthetic_risk_flags",
    ]
    available_cols = [c for c in cols_order if c in df.columns]
    library = df[available_cols].copy()

    computational_disclaimer = (
        "In silico computational candidate only. Predicted activity from Malinois primary model; "
        "no experimental validation performed. Requires MPRA or luciferase reporter assay confirmation."
    )
    library["computational_disclaimer"] = computational_disclaimer

    args.output_library.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output_library.with_suffix(".tsv.gz.tmp")
    library.to_csv(tmp, sep="\t", index=False, compression="gzip")
    tmp.replace(args.output_library)

    tier_counts = library.groupby(["phase", "priority_tier", "design_method"]).size().unstack(fill_value=0).to_dict()
    tier_summary = {}
    for (phase, tier), grp in library.groupby(["phase", "priority_tier"]):
        tier_summary[f"{phase}__{tier}"] = {
            "n": int(len(grp)),
            "audit_pass_n": int((grp["audit_status"] == "pass").sum()),
            "mean_primary_margin_gain": float(grp["primary_margin_gain"].mean()),
            "mean_reviewer_margin_gain": float(grp["reviewer_margin_gain"].mean()),
            "by_design_method": {m: int((grp["design_method"] == m).sum()) for m in grp["design_method"].unique()},
        }

    edit_pos_dist = Counter()
    sub_type_counts = Counter()
    cell_edit_profile = defaultdict(Counter)
    for idx, row in library.iterrows():
        changes = base_changes(row["edit_string"])
        for pos, ref, alt in changes:
            edit_pos_dist[pos] += 1
            sub_type_counts[substitution_type(ref, alt)] += 1
            cell_edit_profile[row["target_cell"]][pos] += 1

    budget_naturalness = {}
    for budget, grp in library.groupby("budget"):
        budget_naturalness[int(budget)] = {
            "mean_naturalness_delta": float(grp["naturalness_delta"].mean()),
            "std_naturalness_delta": float(grp["naturalness_delta"].std()),
            "mean_absolute_gc_delta": float(grp["absolute_gc_delta"].mean()),
            "mean_max_homopolymer": float(grp["max_homopolymer"].mean()),
        }

    correlation = {}
    for method in library["method"].unique():
        grp = library[library["method"] == method]
        if len(grp) > 2:
            correlation[method] = {
                "pearson_margin_vs_reviewer_gain": float(np.corrcoef(grp["primary_margin_gain"], grp["reviewer_margin_gain"])[0, 1]) if np.std(grp["primary_margin_gain"]) > 0 and np.std(grp["reviewer_margin_gain"]) > 0 else None,
                "pearson_uncertainty_vs_rejected": float(np.corrcoef(grp["reviewer_uncertainty"], ~grp["accepted"])[0, 1]) if np.std(grp["reviewer_uncertainty"]) > 0 else None,
                "pearson_strand_vs_rejected": float(np.corrcoef(grp["primary_strand_disagreement"], ~grp["accepted"])[0, 1]) if np.std(grp["primary_strand_disagreement"]) > 0 else None,
            }

    tierA = library[(library["priority_tier"] == "A") & (library["design_method"] == "safeedit")]
    tierA_examples = []
    for (target, budget), grp in tierA.groupby(["target_cell", "budget"]):
        best = grp.sort_values("primary_margin_gain", ascending=False).head(2)
        for _, r in best.iterrows():
            tierA_examples.append({
                "parent_id": r["parent_id"],
                "target_cell": r["target_cell"],
                "budget": int(r["budget"]),
                "edit_string": r["edit_string"],
                "primary_margin_gain": float(r["primary_margin_gain"]),
                "reviewer_margin_gain": float(r["reviewer_margin_gain"]),
                "primary_target_gain": float(r["primary_target_gain"]),
            })

    summary = {
        "purpose": "Final SafeEdit-CRE tiered candidate library",
        "total_candidates": len(library),
        "tier_summary": tier_summary,
        "synthetic_risk_flag_counts": dict(risk_counter),
        "substitution_type_counts": dict(sub_type_counts),
        "edit_position_hotspots": {int(k): int(v) for k, v in edit_pos_dist.most_common(20)},
        "cell_type_edit_profile": {cell: {int(k): int(v) for k, v in ctr.most_common(10)} for cell, ctr in cell_edit_profile.items()},
        "budget_naturalness_drift": budget_naturalness,
        "gain_correlations": correlation,
        "tier_A_examples": tierA_examples[:24],
        "computational_disclaimer": computational_disclaimer,
        "n_tier_A": int((library["priority_tier"] == "A").sum()),
        "n_tier_B": int((library["priority_tier"] == "B").sum()),
        "n_tier_C": int((library["priority_tier"] == "C").sum()),
        "pareto_near_pareto_tolerance": {
            "relative_tolerance": 0.05,
            "absolute_tolerance": 0.1,
            "tolerance_description": "primary_margin_gain within 5% relative or 0.1 absolute standardized units of Pareto front",
        },
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"G6 complete: {len(library)} candidates, Tier A={summary['n_tier_A']}, B={summary['n_tier_B']}, C={summary['n_tier_C']}")


if __name__ == "__main__":
    main()
