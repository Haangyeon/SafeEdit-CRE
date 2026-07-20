"""G6 Frozen: Candidate tiering and Pareto analysis for frozen confirmation set only.

Unlike the regular g6_candidates.py, this version:
- Uses ONLY frozen G4 data (no G3 development set mixing)
- Adapts to frozen G4 column names (design_status, edit_string)
- Applies corrected Pareto direction (maximize both margins)
- Produces Tier A/B/C with clear audit_status separation
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


SYNTHETIC_RISK_FLAGS = [
    ("homopolymer_run_ge_8", lambda df: df["max_homopolymer"] >= 8),
    ("extreme_gc_delta_gt_0p08", lambda df: df["absolute_gc_delta"] > 0.08),
]


def compute_pareto_front(
    df: pd.DataFrame, x_col: str, y_col: str,
    maximize_x: bool = True, maximize_y: bool = True,
) -> np.ndarray:
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
            bx = (xs[j] >= xs[i]) if maximize_x else (xs[j] <= xs[i])
            by = (ys[j] >= ys[i]) if maximize_y else (ys[j] <= ys[i])
            sx = (xs[j] > xs[i]) if maximize_x else (xs[j] < xs[i])
            sy = (ys[j] > ys[i]) if maximize_y else (ys[j] < ys[i])
            if bx and by and (sx or sy):
                is_pareto[i] = False
                break
    return is_pareto


def compute_near_pareto(
    df: pd.DataFrame, x_col: str, y_col: str,
    is_pareto: np.ndarray, rel_tol: float = 0.05, abs_tol: float = 0.1,
) -> np.ndarray:
    n = len(df)
    near = is_pareto.copy()
    xs = df[x_col].to_numpy(dtype=float)
    ys = df[y_col].to_numpy(dtype=float)
    p_xs = xs[is_pareto]
    p_ys = ys[is_pareto]
    if len(p_xs) == 0:
        return near
    x_range = max(np.ptp(p_xs), 1e-12)
    for i in range(n):
        if near[i]:
            continue
        xi, yi = xs[i], ys[i]
        dominated = False
        best_x = -np.inf
        for px, py in zip(p_xs, p_ys, strict=True):
            if py >= yi - 1e-12 and px > best_x:
                best_x = px
            if px >= xi - 1e-12 and py >= yi - 1e-12 and (px > xi + 1e-12 or py > yi + 1e-12):
                dominated = True
        if not dominated:
            near[i] = True
            continue
        if best_x == -np.inf:
            continue
        rel_ok = (best_x - xi) <= rel_tol * max(abs(best_x), 1e-12)
        abs_ok = (best_x - xi) <= abs_tol
        if rel_ok or abs_ok:
            near[i] = True
    return near


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
    parser.add_argument("--g4-frozen", type=Path, required=True)
    parser.add_argument("--output-library", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.g4_frozen, sep="\t", compression="gzip")
    df["phase"] = "G4_frozen_confirmation"
    nat_mean = df["naturalness_delta"].mean()
    nat_std = df["naturalness_delta"].std()
    accepted_mask = df["accepted"].astype(bool)
    strand_lo = df.loc[accepted_mask, "primary_strand_disagreement"].quantile(0.99) if accepted_mask.any() else 0.5
    uncert_hi = df.loc[accepted_mask, "reviewer_uncertainty"].quantile(0.99) if accepted_mask.any() else 0.5

    df["synthetic_risk_flags"] = ""
    risk_counter = Counter()
    risk_mask = pd.Series(False, index=df.index)
    risk_details: list[list[str]] = [[] for _ in range(len(df))]
    for flag_name, fn in SYNTHETIC_RISK_FLAGS:
        mask = fn(df)
        if flag_name == "severe_naturalness_drop_lt_m3sd":
            mask = df["naturalness_delta"] < (nat_mean - 3 * nat_std)
        elif flag_name == "strand_disagreement_gt_2x_threshold":
            mask = df["primary_strand_disagreement"] > 2 * strand_lo
        elif flag_name == "reviewer_uncertainty_gt_2x_threshold":
            mask = df["reviewer_uncertainty"] > 2 * uncert_hi
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
        feasible_audit_pass = grp[(grp["design_status"] == "feasible") & grp["accepted"].astype(bool)]
        if len(feasible_audit_pass) < 2:
            if len(feasible_audit_pass) == 1:
                df.loc[feasible_audit_pass.index[0], "pareto_front"] = True
                df.loc[feasible_audit_pass.index[0], "near_pareto"] = True
            continue
        pf = compute_pareto_front(feasible_audit_pass, "primary_margin_gain", "reviewer_margin_gain", True, True)
        npf = compute_near_pareto(feasible_audit_pass, "primary_margin_gain", "reviewer_margin_gain", pf)
        df.loc[feasible_audit_pass.index[pf], "pareto_front"] = True
        df.loc[feasible_audit_pass.index[npf], "near_pareto"] = True

    df["audit_status"] = np.where(df["accepted"], "pass", "fail")
    method_map = {
        "random_matched": "random",
        "greedy_malinois": "greedy",
        "safeedit_consensus": "safeedit",
    }
    df["design_method"] = df["method"].map(method_map).fillna(df["method"])
    tiers = []
    for _, row in df.iterrows():
        audit_pass = row["audit_status"] == "pass"
        is_safeedit = row["design_method"] == "safeedit"
        feasible = row["design_status"] == "feasible"
        tg_ok = row.get("primary_target_gain", -1) >= 0
        pmg_ok = row.get("primary_margin_gain", -1) > 0
        rmg_ok = row.get("reviewer_margin_gain", -1) > 0
        exact = bool(row.get("check_exact_edit_budget", False))
        syn_safe = not bool(row["has_synthetic_risk"])
        on_pareto = bool(row["pareto_front"]) or bool(row["near_pareto"])
        core = feasible and audit_pass and tg_ok and pmg_ok and rmg_ok and exact
        if is_safeedit and core and syn_safe and on_pareto:
            tiers.append("A")
        elif core:
            tiers.append("B")
        else:
            tiers.append("C")
    df["priority_tier"] = tiers

    cols_order = [
        "parent_id", "chr", "parent_sequence", "sequence", "target_cell", "budget",
        "edit_string", "hamming_distance", "design_status",
        "primary_parent_target", "primary_candidate_target", "primary_target_gain",
        "primary_parent_margin", "primary_candidate_margin", "primary_margin_gain",
        "primary_strand_disagreement",
        "reviewer_parent_margin", "reviewer_candidate_margin", "reviewer_margin_gain",
        "reviewer_uncertainty", "reviewer_parent_uncertainty",
        "naturalness_delta", "absolute_gc_delta", "max_homopolymer",
        "audit_status", "design_method", "priority_tier", "accepted", "failed_checks",
        "pareto_front", "near_pareto", "has_synthetic_risk", "synthetic_risk_flags",
        "phase", "method",
    ]
    available = [c for c in cols_order if c in df.columns]
    library = df[available].copy()
    disc = ("In silico computational candidate only. Predicted activity from Malinois; "
            "no experimental validation performed. Requires MPRA or luciferase reporter assay confirmation.")
    library["computational_disclaimer"] = disc
    args.output_library.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output_library.with_suffix(".tsv.gz.tmp")
    library.to_csv(tmp, sep="\t", index=False, compression="gzip")
    tmp.replace(args.output_library)

    tier_summary = {}
    for (tier, method), grp in library.groupby(["priority_tier", "design_method"]):
        key = f"tier_{tier}__{method}"
        tier_summary[key] = {
            "n": int(len(grp)),
            "audit_pass_n": int((grp["audit_status"] == "pass").sum()),
            "feasible_n": int((grp["design_status"] == "feasible").sum()),
        }
    tierA = library[(library["priority_tier"] == "A") & (library["design_method"] == "safeedit")]
    tierA_examples = []
    for (target, budget), grp in tierA.groupby(["target_cell", "budget"]):
        best = grp.sort_values("primary_margin_gain", ascending=False).head(3)
        for _, r in best.iterrows():
            tierA_examples.append({
                "parent_id": r["parent_id"], "target_cell": target, "budget": int(budget),
                "edit_string": r["edit_string"],
                "primary_margin_gain": float(r["primary_margin_gain"]),
                "reviewer_margin_gain": float(r["reviewer_margin_gain"]),
            })
    edit_dist = Counter()
    sub_counts = Counter()
    for _, row in library.iterrows():
        for pos, ref, alt in base_changes(str(row.get("edit_string", ""))):
            edit_dist[pos] += 1
            sub_counts[substitution_type(ref, alt)] += 1
    summary = {
        "purpose": "G6 frozen candidate library (96-parent confirmation only)",
        "frozen": True,
        "total_candidates": len(library),
        "tier_counts": library["priority_tier"].value_counts().to_dict(),
        "tier_by_method": {m: library[library["design_method"] == m]["priority_tier"].value_counts().to_dict() for m in library["design_method"].unique()},
        "tier_summary": tier_summary,
        "tierA_n": int(len(tierA)),
        "tierA_examples": tierA_examples[:10],
        "synthetic_risk_flag_counts": dict(risk_counter),
        "edit_position_distribution_top20": dict(edit_dist.most_common(20)),
        "substitution_type_counts": dict(sub_counts),
        "pareto_direction": "maximize_primary_and_reviewer_margin_gain (CORRECTED)",
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"G6 frozen library: {len(library)} rows, Tier A={summary['tierA_n']}")
    print(f"Tier counts: {summary['tier_counts']}")
    print(f"Written to {args.output_library}")


if __name__ == "__main__":
    main()
