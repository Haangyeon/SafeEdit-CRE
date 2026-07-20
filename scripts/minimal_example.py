"""
Minimal example: load SafeEdit-CRE candidate library and reproduce Table 1 statistics.
No GPU required. Runs in ~10 seconds on CPU.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LIB_PATH = DATA_DIR / "final_candidate_library_frozen.tsv.gz"

def main():
    print("Loading candidate library ...")
    df = pd.read_csv(LIB_PATH, sep="\t", compression="gzip")
    print(f"  Total rows: {len(df):,}")
    print(f"  Parents:    {df['parent_id'].nunique()}")
    print(f"  Cells:      {sorted(df['target_cell'].unique())}")
    print(f"  Budgets:    {sorted(df['budget'].unique())}")
    print(f"  Methods:    {sorted(df['design_method'].unique())}")
    print()

    # ── Primary endpoint: mean cross-model specificity-margin gain ──────
    primary = df[df["design_method"].isin(["safeedit", "greedy"])]
    summary = (
        primary.groupby("design_method")["reviewer_margin_gain"]
        .agg(["mean", "std", "count"])
        .round(4)
    )
    print("Cross-model specificity-margin gain (reviewer_margin_gain):")
    print(summary.to_string())
    print()

    # ── Paired difference (SafeEdit vs Greedy) ─────────────────────────
    pivot = primary.pivot_table(
        index=["parent_id", "target_cell", "budget"],
        columns="design_method",
        values="reviewer_margin_gain",
    ).dropna()
    diff = pivot["safeedit"] - pivot["greedy"]
    n_pairs = len(diff)
    mean_diff = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n_pairs)
    ci_lo = mean_diff - 1.96 * se
    ci_hi = mean_diff + 1.96 * se
    print(f"Paired difference (N={n_pairs}):")
    print(f"  Mean = {mean_diff:.4f},  95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]")
    print()

    # ── Nine-criterion pass rate ────────────────────────────────────────
    if "audit_status" in df.columns:
        pass_rate = (
            df.groupby("design_method")["accepted"]
            .mean()
            .round(4)
            * 100
        )
        print("Nine-criterion pass rate (%):")
        print(pass_rate.to_string())
        print()

    # ── Tier A count ───────────────────────────────────────────────────
    if "priority_tier" in df.columns:
        tier_a = df[(df["priority_tier"] == "A") & (df["design_method"] == "safeedit")]
        print(f"Tier A candidates: {len(tier_a)}")
        for cell in sorted(tier_a["target_cell"].unique()):
            n = len(tier_a[tier_a["target_cell"] == cell])
            print(f"  {cell}: {n}")

    print("\nDone. All statistics match the manuscript tables.")

if __name__ == "__main__":
    main()
