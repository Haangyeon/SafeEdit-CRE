"""Generate publication-quality figures for the SafeEdit-CRE manuscript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

METHOD_COLORS = {
    "random_matched": "#9e9e9e",
    "greedy_malinois": "#ef6c00",
    "safeedit_consensus": "#2e7d32",
}
METHOD_LABELS = {
    "random_matched": "Random",
    "greedy_malinois": "Greedy (Malinois)",
    "safeedit_consensus": "SafeEdit consensus",
}
CELL_COLORS = {"K562": "#1565c0", "HepG2": "#c62828", "SKNSH": "#6a1b9a"}

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "manuscript" / "figures"


def fig_budget_acceptance(g3: pd.DataFrame, g4: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5), sharey=True)
    for ax, df, title in [(axes[0], g3, "G3 development pilot"), (axes[1], g4, "G4 confirmation (untouched parents)")]:
        for method, color in METHOD_COLORS.items():
            grp = df[df["method"] == method].groupby("budget")["accepted"].agg(["mean", "count", "std"]).reset_index()
            se = grp["std"] / np.sqrt(grp["count"])
            ax.errorbar(grp["budget"], grp["mean"], yerr=1.96*se, marker="o", color=color, label=METHOD_LABELS[method], capsize=3, linewidth=1.5)
        ax.set_xlabel("Edit budget (nt)")
        ax.set_ylabel("Audit pass rate")
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([1, 5, 10, 20])
        ax.grid(alpha=0.3, linestyle=":")
    axes[1].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(outdir / "fig4_budget_acceptance.png", bbox_inches="tight")
    fig.savefig(outdir / "fig4_budget_acceptance.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_tradeoff_scatter(g4: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for method, color in METHOD_COLORS.items():
        grp = g4[g4["method"] == method]
        budget_means = grp.groupby("budget").agg(
            mmg=("primary_margin_gain", "mean"),
            rmg=("reviewer_margin_gain", "mean"),
        ).reset_index()
        ax.scatter(grp["primary_margin_gain"], grp["reviewer_margin_gain"], s=8, alpha=0.25, color=color, rasterized=True)
        ax.plot(budget_means["mmg"], budget_means["rmg"], marker="o", color=color, linewidth=2, label=METHOD_LABELS[method], markersize=6)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Primary margin gain (Malinois)")
    ax.set_ylabel("Reviewer margin gain (CNN ensemble)")
    ax.set_title("G4: primary gain vs CNN transfer")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig(outdir / "fig5_tradeoff_scatter.png", bbox_inches="tight")
    fig.savefig(outdir / "fig5_tradeoff_scatter.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_pareto_tiers(library: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    tier_colors = {"A": "#2e7d32", "B": "#f9a825", "C": "#bdbdbd"}
    safeedit = library[library["method"] == "safeedit_consensus"]
    for ax, cell in zip(axes, ["K562", "HepG2", "SKNSH"]):
        grp = safeedit[safeedit["target_cell"] == cell]
        for tier, color in tier_colors.items():
            sub = grp[grp["tier"] == tier]
            ax.scatter(sub["primary_margin_gain"], sub["reviewer_margin_gain"], s=12, alpha=0.6, color=color, label=f"Tier {tier}" if cell == "K562" else None)
        pf = grp[grp.get("pareto_front", False) == True]
        if len(pf) > 0:
            pf_sorted = pf.sort_values("primary_margin_gain")
            ax.plot(pf_sorted["primary_margin_gain"], pf_sorted["reviewer_margin_gain"], color="k", linewidth=1, linestyle=":", alpha=0.7)
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="k", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Primary margin gain")
        if cell == "K562":
            ax.set_ylabel("Reviewer margin gain")
        ax.set_title(cell)
        ax.grid(alpha=0.3, linestyle=":")
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("G4 SafeEdit: Pareto front and tier stratification", y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / "fig6_pareto_tiers.png", bbox_inches="tight")
    fig.savefig(outdir / "fig6_pareto_tiers.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_uncertainty_calibration(g2_report: dict, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    ens = g2_report.get("ensemble", {})
    bins = ens.get("uncertainty_bins", {})
    if bins:
        centers = bins.get("bin_centers", [])
        actual = bins.get("bin_abs_error", [])
        predicted = bins.get("bin_predicted_std", [])
        if centers:
            axes[0].plot(centers, actual, marker="o", color="#1565c0", label="Observed |error|")
            axes[0].plot(centers, predicted, marker="s", color="#c62828", label="Predicted σ")
            axes[0].plot([0, max(centers)], [0, max(centers)], "k--", alpha=0.4, label="y=x")
            axes[0].set_xlabel("Predicted uncertainty (σ)")
            axes[0].set_ylabel("Observed absolute error")
            axes[0].set_title("CNN uncertainty calibration")
            axes[0].legend(frameon=False, fontsize=7)
            axes[0].grid(alpha=0.3, linestyle=":")
    ax = axes[1]
    models = ["seed_20260713", "seed_20260714", "seed_20260715"]
    for i, m in enumerate(models):
        rep = g2_report.get(m, {})
        ax.bar(i, rep.get("test_spearman", 0), color=CELL_COLORS.get("K562", "#1565c0"), alpha=0.7)
    ax.axhline(g2_report.get("ensemble", {}).get("test_spearman", 0), color="#c62828", linewidth=1.5, linestyle="--", label=f"Ensemble ρ={g2_report.get('ensemble', {}).get('test_spearman', 0):.3f}")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(["seed1", "seed2", "seed3"])
    ax.set_ylabel("Test Spearman ρ")
    ax.set_title("CNN ensemble performance")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig(outdir / "fig3_calibration.png", bbox_inches="tight")
    fig.savefig(outdir / "fig3_calibration.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_failure_reasons(g4: pd.DataFrame, outdir: Path) -> None:
    from collections import Counter
    reasons = Counter()
    for fc in g4["failed_checks"].dropna():
        for f in str(fc).split(";"):
            f = f.strip()
            if f:
                reasons[f.replace("check_", "").replace("_", " ")] += 1
    fig, ax = plt.subplots(figsize=(6, 3))
    items = sorted(reasons.items(), key=lambda x: -x[1])
    labels = [i[0] for i in items]
    counts = [i[1] for i in items]
    y = np.arange(len(labels))
    ax.barh(y, counts, color="#546e7a")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of failures")
    ax.set_title("G4 audit failure reasons")
    ax.grid(alpha=0.3, linestyle=":", axis="x")
    fig.tight_layout()
    fig.savefig(outdir / "supp_failures.png", bbox_inches="tight")
    fig.savefig(outdir / "supp_failures.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_ablation(g5: dict, outdir: Path) -> None:
    abl = [r for r in g5.get("ablation", []) if r["phase"] == "G4_confirmation" and r["budget"] == 20 and r["method"] == "safeedit_consensus"]
    order = ["full_safeedit", "malinois_cnn", "no_cnn_transfer", "no_strand", "no_uncertainty", "no_naturalness", "no_gc", "no_homopolymer", "malinois_only"]
    labels = ["Full SafeEdit", "Malinois+CNN", "- CNN transfer", "- strand", "- uncertainty", "- naturalness", "- GC", "- homopolymer", "Malinois only"]
    vals = []
    for o in order:
        match = [r for r in abl if r["ablation"] == o]
        vals.append(match[0]["accepted_fraction"] if match else 0)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    colors = ["#2e7d32" if v == max(vals) else "#90a4ae" for v in vals]
    ax.bar(range(len(vals)), vals, color=colors)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Audit pass rate (budget=20)")
    ax.set_title("G4 ablation: contribution of each constraint")
    ax.grid(alpha=0.3, linestyle=":", axis="y")
    fig.tight_layout()
    fig.savefig(outdir / "supp_ablation.png", bbox_inches="tight")
    fig.savefig(outdir / "supp_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_cell_stratification(g4: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    for ax, cell in zip(axes, ["K562", "HepG2", "SKNSH"]):
        sub = g4[g4["target_cell"] == cell]
        x = np.arange(4)
        w = 0.25
        budgets = [1, 5, 10, 20]
        for i, (method, color) in enumerate(METHOD_COLORS.items()):
            means = []
            for b in budgets:
                grp = sub[(sub["method"] == method) & (sub["budget"] == b)]
                means.append(grp["accepted"].mean())
            ax.bar(x + (i - 1) * w, means, w, color=color, label=METHOD_LABELS[method] if cell == "K562" else None)
        ax.set_xticks(x)
        ax.set_xticklabels(budgets)
        ax.set_xlabel("Edit budget (nt)")
        ax.set_title(cell)
        ax.grid(alpha=0.3, linestyle=":", axis="y")
    axes[0].set_ylabel("Audit pass rate")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("G4: per-cell-type audit pass rate", y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / "supp_cell_stratification.png", bbox_inches="tight")
    fig.savefig(outdir / "supp_cell_stratification.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_schematic(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    boxes = [
        (0.3, 4.2, 2.2, 1.2, "CRE parent\nsequences", "#bbdefb"),
        (3.0, 4.2, 2.2, 1.2, "Malinois primary\n(3-cell predictor)", "#ffe0b2"),
        (5.7, 4.2, 2.2, 1.2, "SafeEdit consensus\nbeam search", "#c8e6c9"),
        (3.0, 2.2, 2.2, 1.2, "CNN ensemble\nreviewer + uncertainty", "#f8bbd0"),
        (5.7, 2.2, 2.2, 1.2, "Audit layer\n(naturalness, GC,\nstrand, homopolymer)", "#fff9c4"),
        (8.0, 3.2, 1.7, 1.2, "Tiered\ncandidates", "#e1bee7"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", facecolor=color, edgecolor="#424242", linewidth=1))
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=8)
    arrows = [
        (2.5, 4.8, 3.0, 4.8), (5.2, 4.8, 5.7, 4.8),
        (4.1, 4.2, 4.1, 3.4), (5.2, 2.8, 5.7, 2.8),
        (7.9, 3.5, 8.0, 3.8),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#424242", lw=1.2))
    ax.text(5, 5.7, "SafeEdit-CRE: multi-model consensus with uncertainty-aware minimal editing", ha="center", fontsize=11, fontweight="bold")
    ax.text(0.3, 1.0, "Key constraints: minimal substitutions (1-20 nt), monotonic audit thresholds,\nno test-label leakage, in silico predictions require experimental validation.", fontsize=7.5, color="#424242")
    fig.savefig(outdir / "fig1_schematic.png", bbox_inches="tight")
    fig.savefig(outdir / "fig1_schematic.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_data_split(outdir: Path, g3: pd.DataFrame, g4: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    phases = ["G3 development\n(48 parents)", "G4 confirmation\n(48 untouched parents)"]
    methods = ["random_matched", "greedy_malinois", "safeedit_consensus"]
    colors = [METHOD_COLORS[m] for m in methods]
    g3_by_method = [len(g3[g3["method"] == m]) for m in methods if m != "safeedit_consensus"]
    g4_by_method = [len(g4[g4["method"] == m]) for m in methods]
    ax = axes[0]
    ax.pie([24*3*4, 24*3*4], labels=["Random", "Greedy"], autopct="%1.0f%%", colors=[METHOD_COLORS["random_matched"], METHOD_COLORS["greedy_malinois"]], textprops={"fontsize": 8})
    ax.set_title("G3 pilot candidates\n(n=576)", fontsize=9)
    ax = axes[1]
    ax.pie([48*4*3, 48*4*3, 48*4*3], labels=["Random", "Greedy", "SafeEdit"], autopct="%1.0f%%", colors=colors, textprops={"fontsize": 8})
    ax.set_title("G4 confirmation candidates\n(n=1,728)", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "fig2_data_split.png", bbox_inches="tight")
    fig.savefig(outdir / "fig2_data_split.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_edit_position(g6_summary: dict, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), sharey=True)
    profile = g6_summary.get("cell_type_edit_profile", {})
    for ax, cell in zip(axes, ["K562", "HepG2", "SKNSH"]):
        pos_counts = profile.get(cell, {})
        xs = list(range(200))
        ys = [pos_counts.get(i, 0) for i in xs]
        ax.fill_between(xs, ys, alpha=0.5, color=CELL_COLORS.get(cell, "#1565c0"))
        ax.plot(xs, ys, color=CELL_COLORS.get(cell, "#1565c0"), linewidth=0.8)
        ax.set_xlabel("Position in 200 nt CRE")
        if cell == "K562":
            ax.set_ylabel("# edits at position")
        ax.set_title(cell)
        ax.grid(alpha=0.3, linestyle=":")
    fig.suptitle("Edit position distribution (SafeEdit Tier A/B)", y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / "supp_edit_positions.png", bbox_inches="tight")
    fig.savefig(outdir / "supp_edit_positions.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g3", type=Path, default=ROOT / "data" / "processed" / "g3_candidates.tsv.gz")
    parser.add_argument("--g4", type=Path, default=ROOT / "data" / "processed" / "g4_candidates.tsv.gz")
    parser.add_argument("--g5", type=Path, default=ROOT / "reports" / "g5_audit_ablation.json")
    parser.add_argument("--g6-summary", type=Path, default=ROOT / "reports" / "final_candidate_summary.json")
    parser.add_argument("--library", type=Path, default=ROOT / "data" / "processed" / "final_candidate_library.tsv.gz")
    parser.add_argument("--g2-report", type=Path, default=ROOT / "reports" / "cnn_ensemble_pilot.json")
    parser.add_argument("--outdir", type=Path, default=FIG_DIR)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    g3 = pd.read_csv(args.g3, sep="\t", compression="gzip")
    g4 = pd.read_csv(args.g4, sep="\t", compression="gzip")
    g5 = json.loads(args.g5.read_text(encoding="utf-8")) if args.g5.exists() else {}
    g6 = json.loads(args.g6_summary.read_text(encoding="utf-8")) if args.g6_summary.exists() else {}
    library = pd.read_csv(args.library, sep="\t", compression="gzip") if args.library.exists() else g4
    g2 = json.loads(args.g2_report.read_text(encoding="utf-8")) if args.g2_report.exists() else {}

    fig_schematic(args.outdir)
    fig_data_split(args.outdir, g3, g4)
    if g2:
        fig_uncertainty_calibration(g2, args.outdir)
    fig_budget_acceptance(g3, g4, args.outdir)
    fig_tradeoff_scatter(g4, args.outdir)
    if "tier_summary" in g6:
        fig_pareto_tiers(library, args.outdir)
    fig_failure_reasons(g4, args.outdir)
    if g5:
        fig_ablation(g5, args.outdir)
    fig_cell_stratification(g4, args.outdir)
    if g6:
        fig_edit_position(g6, args.outdir)
    print(f"Figures saved to {args.outdir}")


if __name__ == "__main__":
    main()
