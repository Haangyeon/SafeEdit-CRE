from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
REPORTS = BASE / "reports"
OUT = BASE / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "random_matched": "#858F98",
    "greedy_malinois": "#D97727",
    "primary_beam": "#536FB6",
    "safeedit_consensus": "#087F8C",
}
LABELS = {
    "random_matched": "Random",
    "greedy_malinois": "Greedy",
    "primary_beam": "Primary beam",
    "safeedit_consensus": "SafeEdit",
}
CELLS = ["K562", "HepG2", "SKNSH"]
CELL_LABELS = {"K562": "K562", "HepG2": "HepG2", "SKNSH": "SK-N-SH"}
BUDGETS = [1, 5, 10, 20]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.2,
    "axes.titlesize": 9.2,
    "axes.labelsize": 8.3,
    "axes.linewidth": 0.9,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "legend.fontsize": 7.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_data():
    g4 = pd.read_csv(DATA / "g4_frozen_candidates.tsv.gz", sep="\t")
    g8 = pd.read_csv(DATA / "g8_candidates_sealed_collapsed.tsv.gz", sep="\t")
    g7r = pd.read_csv(REPORTS / "g7_reviewer_test_predictions.tsv.gz", sep="\t")
    g7s = pd.read_csv(REPORTS / "g7_sealed_test_predictions.tsv.gz", sep="\t")
    cal = json.load(open(REPORTS / "g7_uncertainty_calibration.json"))
    g9 = pd.read_csv(DATA / "g9_true_ablation_sealed.tsv.gz", sep="\t")
    tiers = pd.read_csv(DATA / "final_candidate_library_frozen.tsv.gz", sep="\t")
    return g4, g8, g7r, g7s, cal, g9, tiers


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT / f"{name}.png", dpi=360, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def boxed(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#222222")
        spine.set_linewidth(0.9)
    ax.tick_params(top=False, right=False)


def panel(ax: plt.Axes, letter: str, title: str) -> None:
    # Keep the panel letter and title on one baseline, while centering the
    # descriptive title over the plotting area. This avoids the inconsistent
    # left-shifted titles that can make multi-panel figures look unbalanced.
    ax.text(-0.13, 1.06, letter, transform=ax.transAxes, fontsize=10.2,
            fontweight="bold", va="bottom", ha="left", clip_on=False)
    ax.text(0.50, 1.06, title, transform=ax.transAxes, fontsize=8.2,
            fontweight="bold", va="bottom", ha="center", clip_on=False)


def parent_ci(df: pd.DataFrame, metric: str, n: int = 20000, seed: int = 20260716):
    values = df.groupby("parent_id")[metric].mean().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    chunks = []
    for start in range(0, n, 2000):
        k = min(2000, n - start)
        idx = rng.integers(0, len(values), size=(k, len(values)))
        chunks.append(values[idx].mean(axis=1))
    boot = np.concatenate(chunks)
    return values.mean(), np.quantile(boot, 0.025), np.quantile(boot, 0.975)


def paired_ci(g8: pd.DataFrame, baseline: str, metric: str, seed: int):
    a = g8[g8.method == "safeedit_consensus"].groupby("parent_id")[metric].mean()
    b = g8[g8.method == baseline].groupby("parent_id")[metric].mean()
    values = (a - b).dropna().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    chunks = []
    for start in range(0, 20000, 2000):
        idx = rng.integers(0, len(values), size=(2000, len(values)))
        chunks.append(values[idx].mean(axis=1))
    boot = np.concatenate(chunks)
    return values.mean(), np.quantile(boot, .025), np.quantile(boot, .975)


def draw_node(ax, x, y, w, h, text, face, fontsize=8.1):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.025,rounding_size=0.06",
        facecolor=face, edgecolor="#26343B", linewidth=1.0, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, linespacing=1.12, zorder=3)
    return patch


def arrow(ax, start, end, color="#32434C"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.15, color=color, shrinkA=4, shrinkB=5,
                                 zorder=1))


def fig1_workflow():
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.set_xlim(0, 7.0); ax.set_ylim(0, 5.6); ax.axis("off")
    main_x, main_w, h = 2.58, 1.84, 0.68
    h_freeze = 0.96
    y_data, y_primary, y_search, y_freeze, y_pf = 4.50, 3.52, 2.46, 1.32, 0.22

    draw_node(ax, main_x, y_data, main_w, h, "Public MPRA data\nfixed chromosome split", "#DCEAF3")
    draw_node(ax, main_x, y_primary, main_w, h, "Malinois primary model\nspecificity objective", "#F6DFC0")
    draw_node(ax, 2.43, y_search, 2.14, h, "SafeEdit constrained\nbeam search\nbudgets: 1 / 5 / 10 / 20 nt", "#DCEBD9", 7.5)
    draw_node(ax, main_x, y_freeze, main_w, h_freeze, "Freeze candidate\nsequences\nSHA-256 manifest", "#E6E0F0", 7.6)
    draw_node(ax, main_x, y_pf, main_w, h, "Post-freeze evaluation\ntransfer endpoint", "#D9E8EC")

    draw_node(ax, 0.20, y_search, 1.78, h, "Residual-dilated CNN\nreviewer ensemble", "#F1DCE5")
    draw_node(ax, 5.02, y_search, 1.78, h, "Validation-only\nuncertainty and\nsequence-domain\nthresholds", "#F5E8B9", 7.15)
    draw_node(ax, 0.20, y_freeze, 1.78, h_freeze, "Nine-check audit\nconstraint-compliance\nendpoint", "#F4E4B2", 7.3)

    arrow(ax, (3.50, y_data), (3.50, y_primary + h))
    arrow(ax, (3.50, y_primary), (3.50, y_search + h))
    arrow(ax, (1.98, y_search + h/2), (2.43, y_search + h/2), "#914B68")
    arrow(ax, (5.02, y_search + h/2), (4.57, y_search + h/2), "#8A6C19")
    arrow(ax, (3.50, y_search), (3.50, y_freeze + h_freeze))
    arrow(ax, (3.50, y_freeze), (3.50, y_pf + h))
    arrow(ax, (2.58, y_freeze + h_freeze/2), (1.98, y_freeze + h_freeze/2), "#8A6C19")

    ax.text(0.20, 5.42, "A", fontsize=11, fontweight="bold", va="top")
    ax.text(0.55, 5.42, "Procedural separation of search, audit, and post-freeze evaluation",
            fontsize=10.2, fontweight="bold", va="top")
    save(fig, "fig1_workflow")


def fig2_validation(g7r, g7s, cal):
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.0))
    axs = axs.ravel()
    x = np.arange(3)

    # A: predictive correlation
    perf = {
        "Malinois": [0.884, 0.888, 0.879],
        "Reviewer": [np.corrcoef(g7r[f"pred_{c}"], g7r[f"true_{c}"])[0, 1] for c in CELLS],
        "Post-freeze": [np.corrcoef(g7s[f"pred_{c}"], g7s[f"true_{c}"])[0, 1] for c in CELLS],
    }
    model_colors = {"Malinois": COLORS["greedy_malinois"], "Reviewer": COLORS["safeedit_consensus"], "Post-freeze": COLORS["primary_beam"]}
    width = 0.24
    for j, name in enumerate(["Malinois", "Reviewer", "Post-freeze"]):
        off = (j - 1) * width
        axs[0].bar(x + off, perf[name], width=width*0.9, color=model_colors[name], label=name, zorder=2)
        for xx, yy in zip(x + off, perf[name]):
            axs[0].text(xx, yy + 0.006, f"{yy:.2f}", ha="center", va="bottom", fontsize=6.5)
    axs[0].set_xticks(x, [CELL_LABELS[c] for c in CELLS])
    axs[0].set_ylim(0.66, 0.98); axs[0].set_ylabel("Pearson r")
    axs[0].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.99), columnspacing=0.9)
    panel(axs[0], "A", "Held-out predictive performance")

    # B: average coverage calibration
    for key, name, color, marker in [
        ("reviewer", "Reviewer", model_colors["Reviewer"], "o"),
        ("sealed_evaluator", "Post-freeze", model_colors["Post-freeze"], "s")]:
        nominal, observed = [], []
        for lev in ["80%", "90%", "95%"]:
            nominal.append(float(lev[:-1]))
            observed.append(100*np.mean([cal[key]["per_cell_test"][c]["coverage"][lev] for c in CELLS]))
        axs[1].plot(nominal, observed, marker=marker, color=color, lw=1.5, ms=5, label=name)
    axs[1].plot([78, 97], [78, 97], "--", color="#555555", lw=1.0, label="Ideal")
    axs[1].set_xlim(78, 97); axs[1].set_ylim(78, 97)
    axs[1].set_xlabel("Nominal coverage (%)"); axs[1].set_ylabel("Observed coverage (%)")
    axs[1].legend(frameon=False, loc="lower right")
    panel(axs[1], "B", "Validation-only interval calibration")

    # C: uncertainty-error association
    width = 0.32
    for j, (key, name, color) in enumerate([
        ("reviewer", "Reviewer", model_colors["Reviewer"]),
        ("sealed_evaluator", "Post-freeze", model_colors["Post-freeze"])]):
        vals = [cal[key]["per_cell_test"][c]["uncertainty_error_spearman"] for c in CELLS]
        off = (j - 0.5) * width
        axs[2].bar(x + off, vals, width*0.9, color=color, label=name)
        for xx, yy in zip(x + off, vals):
            axs[2].text(xx, yy + 0.012, f"{yy:.2f}", ha="center", va="bottom", fontsize=6.7)
    axs[2].set_xticks(x, [CELL_LABELS[c] for c in CELLS]); axs[2].set_ylim(0, 0.66)
    axs[2].set_ylabel("Spearman rho")
    axs[2].legend(frameon=False, loc="upper right", bbox_to_anchor=(.99,.99))
    panel(axs[2], "C", "Uncertainty versus absolute error")

    # D: mean rejection curves
    for key, name, color in [
        ("reviewer", "Reviewer", model_colors["Reviewer"]),
        ("sealed_evaluator", "Post-freeze", model_colors["Post-freeze"])]:
        rows = cal[key]["per_cell_test"]
        fractions = [100*r["reject_fraction"] for r in rows[CELLS[0]]["rejection_curve"]]
        means = np.mean([[r["mae"] for r in rows[c]["rejection_curve"]] for c in CELLS], axis=0)
        axs[3].plot(fractions, means, "o-", color=color, lw=1.6, ms=3.8, label=name)
    axs[3].set_xlabel("Highest-uncertainty records rejected (%)"); axs[3].set_ylabel("Mean absolute error")
    axs[3].legend(frameon=False, loc="upper right")
    panel(axs[3], "D", "Error after uncertainty rejection")

    for ax in axs:
        boxed(ax); ax.grid(axis="y", color="#D7DCE0", lw=0.55, alpha=0.65, zorder=0)
    fig.subplots_adjust(left=.10, right=.98, bottom=.09, top=.94, wspace=.32, hspace=.40)
    save(fig, "fig2_validation")


def fig3_hard_audit(g4):
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.0))
    order = ["random_matched", "greedy_malinois", "safeedit_consensus"]
    # A pooled
    y = np.arange(len(order))
    for i, method in enumerate(order):
        q = g4[g4.method == method]
        vals = q.groupby("parent_id").accepted.mean().to_numpy()
        rng = np.random.default_rng(20260720 + i)
        boot = vals[rng.integers(0, len(vals), size=(20000, len(vals)))].mean(1)
        mu, lo, hi = 100*vals.mean(), 100*np.quantile(boot,.025), 100*np.quantile(boot,.975)
        axs[0,0].barh(i, mu, height=.55, color=COLORS[method], zorder=2)
        axs[0,0].errorbar(mu, i, xerr=[[mu-lo],[hi-mu]], fmt="o", color="#24343C",
                         ecolor="#24343C", ms=3.5, capsize=3, lw=1.0, zorder=3)
        axs[0,0].text(min(73.5, hi+2.2), i, f"{mu:.1f}%", va="center", ha="left", fontsize=7.2)
    axs[0,0].set_yticks(y, [LABELS[m] for m in order]); axs[0,0].invert_yaxis()
    axs[0,0].set_xlim(0, 76); axs[0,0].set_xlabel("Audit pass rate (%)")
    panel(axs[0,0], "A", "Pooled nine-check endpoint")

    budget_x = np.arange(len(BUDGETS))
    for ax, cell, letter in zip([axs[0,1], axs[1,0], axs[1,1]], CELLS, ["B","C","D"]):
        for method in order:
            q = (g4[(g4.method == method) & (g4.target_cell == cell)]
                 .groupby("budget").accepted.mean().reindex(BUDGETS))
            ax.plot(budget_x, 100*q.values, "o-", color=COLORS[method], lw=1.7,
                    ms=4.2, label=LABELS[method])
        ax.set_xticks(budget_x, BUDGETS); ax.set_xlim(-.25, len(BUDGETS)-.75); ax.set_ylim(0, 100)
        ax.set_xlabel("Edit budget (nt)"); ax.set_ylabel("Audit pass rate (%)")
        panel(ax, letter, CELL_LABELS[cell])
    handles = [plt.Line2D([0],[0], color=COLORS[m], marker="o", lw=1.7, ms=4,
                          label=LABELS[m]) for m in order]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 0.995), columnspacing=1.4)
    for ax in axs.ravel():
        boxed(ax); ax.grid(axis="y", color="#D7DCE0", lw=.55, alpha=.65)
    fig.subplots_adjust(left=.11, right=.98, bottom=.09, top=.91, wspace=.34, hspace=.39)
    save(fig, "fig3_hard_audit")


def forest_contrast(ax, g8, metric, title, letter, xlim, seed):
    baselines = ["greedy_malinois", "primary_beam"]
    y = np.arange(2)
    ax.axvline(0, color="#555555", ls="--", lw=.9, zorder=0)
    for i, baseline in enumerate(baselines):
        mu, lo, hi = paired_ci(g8, baseline, metric, seed+i)
        ax.errorbar(mu, i, xerr=[[mu-lo],[hi-mu]], fmt="o", color=COLORS[baseline],
                    ecolor=COLORS[baseline], ms=5, capsize=3, lw=1.4, zorder=2)
        xtext = xlim[1] - 0.025 * (xlim[1] - xlim[0])
        ax.text(xtext, i, f"{mu:+.3f}", va="center", ha="right", fontsize=7.0,
                bbox=dict(facecolor="white", edgecolor="none", pad=.12, alpha=.88), zorder=3)
    ax.set_yticks(y, ["vs Greedy", "vs Primary beam"]); ax.invert_yaxis(); ax.set_xlim(*xlim)
    ax.set_xlabel("SafeEdit minus comparator")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    # Keep narrow effect-size panels readable: two-decimal rounding can make
    # adjacent locator ticks display the same label (e.g. -0.02 twice).
    tick_fmt = "%.3f" if (xlim[1] - xlim[0]) < 0.05 else "%.2f"
    ax.xaxis.set_major_formatter(FormatStrFormatter(tick_fmt))
    panel(ax, letter, title); boxed(ax)


def fig4_sealed(g8):
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.05))
    order = ["random_matched", "greedy_malinois", "primary_beam", "safeedit_consensus"]
    values = []
    for i, method in enumerate(order):
        mu, lo, hi = parent_ci(g8[g8.method == method], "sealed_margin_gain", seed=20260730+i)
        values.append((mu,lo,hi))
    x = np.arange(4)
    for i, (method, (mu,lo,hi)) in enumerate(zip(order, values)):
        axs[0,0].bar(i, mu, width=.62, color=COLORS[method], zorder=2)
        axs[0,0].errorbar(i, mu, yerr=[[mu-lo],[hi-mu]], fmt="none", ecolor="#24343C",
                         capsize=3, lw=1.0, zorder=3)
        axs[0,0].text(i, hi+0.035, f"{mu:.3f}", ha="center", va="bottom", fontsize=7.0)
    axs[0,0].set_xticks(x, [LABELS[m] for m in order], rotation=18, ha="right")
    axs[0,0].set_ylim(-.04, 1.02); axs[0,0].set_ylabel("Post-freeze specificity-margin gain")
    panel(axs[0,0], "A", "Method means across 600 parents")
    boxed(axs[0,0]); axs[0,0].grid(axis="y", color="#D7DCE0", lw=.55, alpha=.65, zorder=0)

    forest_contrast(axs[0,1], g8, "sealed_margin_gain", "Specificity-margin contrast", "B", (-.025,.090), 20260740)
    forest_contrast(axs[1,0], g8, "sealed_target_gain", "Absolute target-gain contrast", "C", (-.28,.030), 20260750)
    forest_contrast(axs[1,1], g8, "sealed_uncertainty", "Ensemble-uncertainty contrast", "D", (-.030,.005), 20260760)
    for ax in axs.ravel()[1:]:
        ax.grid(axis="x", color="#D7DCE0", lw=.55, alpha=.65, zorder=0)
    fig.subplots_adjust(left=.12, right=.97, bottom=.10, top=.95, wspace=.42, hspace=.42)
    save(fig, "fig3_primary_endpoint")


def paired_heatmap(ax, g8, baseline, title, letter):
    rows = []
    for cell in CELLS:
        for budget in BUDGETS:
            a = g8[(g8.method == "safeedit_consensus") & (g8.target_cell == cell) & (g8.budget == budget)].set_index("parent_id").sealed_margin_gain
            b = g8[(g8.method == baseline) & (g8.target_cell == cell) & (g8.budget == budget)].set_index("parent_id").sealed_margin_gain
            rows.append((cell, budget, (a-b).dropna().mean()))
    z = pd.DataFrame(rows, columns=["cell","budget","diff"]).pivot(index="cell",columns="budget",values="diff").loc[CELLS,BUDGETS]
    vmax = float(np.max(np.abs(z.to_numpy())))
    cmap = LinearSegmentedColormap.from_list("safe_div", ["#4F75AD", "#F7F7F5", "#C64949"])
    im = ax.imshow(z.to_numpy(), cmap=cmap, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), aspect="auto")
    ax.set_xticks(range(4), BUDGETS); ax.set_yticks(range(3), [CELL_LABELS[c] for c in CELLS])
    ax.set_xlabel("Edit budget (nt)"); ax.set_ylabel("Target cell")
    for i in range(3):
        for j in range(4):
            ax.text(j, i, f"{z.iloc[i,j]:+.3f}", ha="center", va="center", fontsize=7.0)
    panel(ax, letter, title); boxed(ax)
    cb = ax.figure.colorbar(im, ax=ax, fraction=.047, pad=.035)
    cb.set_label("Mean difference", fontsize=7.4); cb.ax.tick_params(labelsize=6.8)


def fig5_budget(g8):
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.05))
    methods = ["greedy_malinois", "primary_beam", "safeedit_consensus"]
    budget_x = np.arange(len(BUDGETS))
    for method in methods:
        means = [g8[(g8.method == method) & (g8.budget == b)].sealed_margin_gain.mean() for b in BUDGETS]
        axs[0,0].plot(budget_x, means, "o-", color=COLORS[method], lw=1.7, ms=4.2, label=LABELS[method])
        means_u = [g8[(g8.method == method) & (g8.budget == b)].sealed_uncertainty.mean() for b in BUDGETS]
        axs[0,1].plot(budget_x, means_u, "o-", color=COLORS[method], lw=1.7, ms=4.2, label=LABELS[method])
    for ax, letter, title, ylabel in [
        (axs[0,0], "A", "Specificity gain by edit budget", "Mean sealed margin gain"),
        (axs[0,1], "B", "Uncertainty by edit budget", "Mean sealed uncertainty")]:
        ax.set_xticks(budget_x, BUDGETS); ax.set_xlim(-.25, len(BUDGETS)-.75); ax.set_xlabel("Edit budget (nt)"); ax.set_ylabel(ylabel)
        panel(ax, letter, title); boxed(ax); ax.grid(axis="y", color="#D7DCE0", lw=.55, alpha=.65)
    axs[0,0].legend(frameon=False, loc="upper left")
    paired_heatmap(axs[1,0], g8, "greedy_malinois", "SafeEdit versus Greedy", "C")
    paired_heatmap(axs[1,1], g8, "primary_beam", "SafeEdit versus primary beam", "D")
    fig.subplots_adjust(left=.11, right=.97, bottom=.09, top=.95, wspace=.42, hspace=.43)
    save(fig, "fig5_budget_tradeoff")


def fig6_ablation(g9):
    cfg = g9.groupby("ablation_config").agg(
        margin=("sealed_margin_gain","mean"), target=("sealed_target_gain","mean"),
        uncertainty=("sealed_uncertainty","mean"),
        infeasible=("design_status", lambda x: int((x != "feasible").sum())),
    )
    order = [
        "primary_beam", "safeedit_no_reviewer_score", "safeedit_no_uncertainty_penalty",
        "safeedit_no_strand_penalty", "safeedit_no_naturalness_rerank", "safeedit_no_sequence_prefilter",
    ]
    labels = ["Primary beam", "No reviewer", "No uncertainty", "No strand", "No naturalness", "No prefilter"]
    full = cfg.loc["safeedit_full"]
    y = np.arange(len(order))
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.1))

    for ax, metric, letter, title, xlabel, color in [
        (axs[0,0], "margin", "A", "Specificity-margin effect", "Change from full SafeEdit", COLORS["safeedit_consensus"]),
        (axs[0,1], "target", "B", "Absolute target-gain effect", "Change from full SafeEdit", COLORS["greedy_malinois"]),
    ]:
        vals = np.array([cfg.loc[o,metric] - full[metric] for o in order])
        ax.axvline(0, color="#555555", ls="--", lw=.9, zorder=0)
        ax.barh(y, vals, height=.58, color=color, alpha=.90, zorder=2)
        ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlabel(xlabel)
        panel(ax, letter, title); boxed(ax); ax.grid(axis="x", color="#D7DCE0", lw=.55, alpha=.65)

    axs[0,0].set_xlim(-.015,.128)
    axs[0,1].set_xlim(-.03,.395)
    for ax, vals, xpos in [
        (axs[0,0], np.array([cfg.loc[o,"margin"] - full["margin"] for o in order]), .122),
        (axs[0,1], np.array([cfg.loc[o,"target"] - full["target"] for o in order]), .385),
    ]:
        for yy, val in zip(y, vals):
            ax.text(xpos, yy, f"{val:+.3f}", va="center", ha="right", fontsize=6.7,
                    bbox=dict(facecolor="white", edgecolor="none", pad=.15, alpha=.78), zorder=4)

    all_order = ["safeedit_full"] + order
    all_labels = ["SafeEdit full"] + labels
    yy = np.arange(len(all_order))
    unc = [cfg.loc[o,"uncertainty"] for o in all_order]
    axs[1,0].barh(yy, unc, height=.58, color=[COLORS["safeedit_consensus"]]+["#9ABFC4"]*len(order), zorder=2)
    axs[1,0].set_yticks(yy, all_labels); axs[1,0].invert_yaxis(); axs[1,0].set_xlim(.29,.357)
    axs[1,0].set_xlabel("Mean sealed uncertainty")
    for yv, val in zip(yy, unc): axs[1,0].text(val+.001, yv, f"{val:.3f}", va="center", fontsize=6.7)
    panel(axs[1,0], "C", "Selected-candidate uncertainty"); boxed(axs[1,0])
    axs[1,0].grid(axis="x", color="#D7DCE0", lw=.55, alpha=.65)

    inf = [cfg.loc[o,"infeasible"] for o in all_order]
    axs[1,1].barh(yy, inf, height=.58, color=[COLORS["safeedit_consensus"]]+["#A7AFB6"]*len(order), zorder=2)
    axs[1,1].set_yticks(yy, all_labels); axs[1,1].invert_yaxis(); axs[1,1].set_xlim(0,55)
    axs[1,1].set_xlabel("Infeasible rows (of 2,160)")
    for yv, val in zip(yy, inf):
        xpos = val + 1.2 if val > 0 else 3.0
        axs[1,1].text(xpos, yv, str(val), va="center", fontsize=6.7)
    panel(axs[1,1], "D", "Search feasibility"); boxed(axs[1,1])
    axs[1,1].grid(axis="x", color="#D7DCE0", lw=.55, alpha=.65)
    fig.subplots_adjust(left=.18, right=.97, bottom=.09, top=.95, wspace=.48, hspace=.42)
    save(fig, "fig6_ablation")


def figS1_stratified(g8):
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.0))
    methods = ["random_matched", "greedy_malinois", "primary_beam", "safeedit_consensus"]
    budget_x = np.arange(len(BUDGETS))
    for ax, cell, letter in zip([axs[0,0], axs[0,1], axs[1,0]], CELLS, ["A","B","C"]):
        for method in methods:
            means = [g8[(g8.method == method) & (g8.target_cell == cell) & (g8.budget == b)].sealed_margin_gain.mean() for b in BUDGETS]
            ax.plot(budget_x, means, "o-", color=COLORS[method], lw=1.6, ms=4, label=LABELS[method])
        ax.set_xticks(budget_x, BUDGETS); ax.set_xlim(-.25, len(BUDGETS)-.75); ax.set_xlabel("Edit budget (nt)"); ax.set_ylabel("Mean sealed margin gain")
        panel(ax, letter, CELL_LABELS[cell]); boxed(ax); ax.grid(axis="y", color="#D7DCE0", lw=.55, alpha=.65)
    axs[1,1].axis("off")
    handles = [plt.Line2D([0],[0], color=COLORS[m], marker="o", lw=1.7, ms=4, label=LABELS[m]) for m in methods]
    axs[1,1].legend(handles=handles, frameon=True, edgecolor="#333333", facecolor="white",
                    loc="center", title="Editing method", title_fontsize=8.5)
    axs[1,1].text(.5,.17,"Each curve contains 600 parent sequences\nper target and budget.",
                  ha="center", va="center", transform=axs[1,1].transAxes, fontsize=7.4, color="#4A545B")
    fig.subplots_adjust(left=.11, right=.98, bottom=.09, top=.95, wspace=.33, hspace=.40)
    save(fig, "fig4_stratified_effects")


def figS2_uncertainty(cal):
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.0))
    cell_colors = ["#087F8C", "#D97727", "#536FB6"]
    for col, (key, title) in enumerate([("reviewer","Reviewer ensemble"),("sealed_evaluator","Post-freeze evaluator")]):
        rows = cal[key]["per_cell_test"]
        for c, color in zip(CELLS, cell_colors):
            curve = rows[c]["rejection_curve"]
            axs[0,col].plot([100*r["reject_fraction"] for r in curve], [r["mae"] for r in curve],
                            "o-", color=color, lw=1.6, ms=3.7, label=CELL_LABELS[c])
        axs[0,col].set_xlabel("Records rejected (%)"); axs[0,col].set_ylabel("Mean absolute error")
        panel(axs[0,col], "A" if col==0 else "B", f"{title}: rejection curve")
        axs[0,col].legend(frameon=False, loc="upper right")

        levels = [80,90,95]; x=np.arange(3); width=.23
        for j,(c,color) in enumerate(zip(CELLS,cell_colors)):
            vals=[100*rows[c]["coverage"][f"{lev}%"] for lev in levels]
            axs[1,col].bar(x+(j-1)*width, vals, width*.9, color=color, label=CELL_LABELS[c])
        axs[1,col].plot([-.4,2.4], [80,95], color="#555555", alpha=0)
        axs[1,col].set_xticks(x,[f"{lev}%" for lev in levels]); axs[1,col].set_ylim(75,98)
        axs[1,col].set_xlabel("Nominal interval"); axs[1,col].set_ylabel("Observed coverage (%)")
        panel(axs[1,col], "C" if col==0 else "D", f"{title}: interval coverage")
    for ax in axs.ravel():
        boxed(ax); ax.grid(axis="y", color="#D7DCE0", lw=.55, alpha=.65, zorder=0)
    fig.subplots_adjust(left=.11, right=.98, bottom=.09, top=.95, wspace=.35, hspace=.42)
    save(fig, "figS2_uncertainty_rejection")


def figS3_tiers(tiers):
    fig, ax = plt.subplots(figsize=(6.7, 5.8))
    rng = np.random.default_rng(20260716)
    c = tiers[tiers.priority_tier == "C"]
    if len(c)>1200: c=c.iloc[rng.choice(len(c),1200,replace=False)]
    b=tiers[tiers.priority_tier=="B"]; a=tiers[tiers.priority_tier=="A"]
    ax.scatter(c.primary_margin_gain,c.reviewer_margin_gain,s=8,alpha=.18,color="#89939C",
               label=f"Tier C (n={sum(tiers.priority_tier=='C'):,}; sample shown)")
    ax.scatter(b.primary_margin_gain,b.reviewer_margin_gain,s=10,alpha=.42,color=COLORS["greedy_malinois"],
               label=f"Tier B (n={len(b):,})")
    ax.scatter(a.primary_margin_gain,a.reviewer_margin_gain,s=24,alpha=.95,color=COLORS["safeedit_consensus"],
               edgecolor="white",linewidth=.35,label=f"Tier A (n={len(a):,})",zorder=3)
    pf=tiers[tiers.pareto_front.astype(bool)]
    ax.scatter(pf.primary_margin_gain,pf.reviewer_margin_gain,facecolors="none",edgecolors="#183C49",
               s=45,linewidth=.85,label="Pareto-front candidate",zorder=4)
    ax.axhline(0,color="#555555",lw=.8,ls="--"); ax.axvline(0,color="#555555",lw=.8,ls="--")
    ax.set_xlabel("Primary-model specificity-margin gain"); ax.set_ylabel("Reviewer specificity-margin gain")
    panel(ax,"A","Frozen candidate library: computational priority tiers")
    boxed(ax); ax.grid(color="#D7DCE0",lw=.5,alpha=.55,zorder=0)
    ax.legend(frameon=True,edgecolor="#333333",facecolor="white",ncol=2,loc="upper center",
              bbox_to_anchor=(.5,-.13),columnspacing=1.2,handletextpad=.5)
    fig.subplots_adjust(left=.12,right=.98,bottom=.24,top=.92)
    save(fig,"figS3_candidate_tiers")


if __name__ == "__main__":
    g4,g8,g7r,g7s,cal,g9,tiers=load_data()
    fig1_workflow()
    fig2_validation(g7r,g7s,cal)
    fig3_hard_audit(g4)
    fig4_sealed(g8)
    fig5_budget(g8)
    fig6_ablation(g9)
    figS1_stratified(g8)
    figS2_uncertainty(cal)
    figS3_tiers(tiers)
    print(f"Wrote journal-style figures to {OUT}")
