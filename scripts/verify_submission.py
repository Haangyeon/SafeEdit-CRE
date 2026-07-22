"""CPU-only integrity checks for the SafeEdit-CRE submission package.

Run from the package root. This verifier checks table dimensions, key uniqueness,
finite sealed endpoints, exact edit budgets for feasible rows, and that the corrected
G9 target-gain column is not a copy of the margin-gain column.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def read_gz(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", compression="gzip")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    checks: list[tuple[str, bool]] = []
    required = [
        ROOT / "manuscript/main.tex",
        ROOT / "manuscript/supplementary.tex",
        ROOT / "manuscript/main.pdf",
        ROOT / "manuscript/supplementary.pdf",
        ROOT / "data/g4_frozen_candidates.tsv.gz",
        ROOT / "data/final_candidate_library_frozen.tsv.gz",
        ROOT / "data/g8_candidates_sealed_collapsed.tsv.gz",
        ROOT / "data/g9_true_ablation_sealed.tsv.gz",
        ROOT / "stats/final_stats.json",
    ]
    for p in required:
        checks.append((f"exists:{p.relative_to(ROOT)}", p.exists() and p.stat().st_size > 0))

    g4 = read_gz(ROOT / "data/g4_frozen_candidates.tsv.gz")
    g8 = read_gz(ROOT / "data/g8_candidates_sealed_collapsed.tsv.gz")
    g9 = read_gz(ROOT / "data/g9_true_ablation_sealed.tsv.gz")
    checks.extend([
        ("g4_rows", len(g4) == 3240),
        ("g4_unique_keys", not g4.duplicated(["parent_id", "target_cell", "budget", "method"]).any()),
        ("g8_rows", len(g8) == 28800),
        ("g8_unique_keys", not g8.duplicated(["parent_id", "target_cell", "budget", "method"]).any()),
        ("g9_rows", len(g9) == 15120),
        ("g9_unique_keys", not g9.duplicated(["parent_id", "target_cell", "budget", "ablation_config"]).any()),
        ("g9_target_margin_not_identical", not np.allclose(g9["sealed_target_gain"], g9["sealed_margin_gain"])),
    ])
    feasible = g9[g9["design_status"] == "feasible"]
    checks.append(("g9_exact_budget", bool((feasible["hamming_distance"] == feasible["budget"]).all())))
    for col in ["sealed_target_gain", "sealed_margin_gain", "sealed_uncertainty"]:
        checks.append((f"g9_finite:{col}", bool(np.isfinite(g9[col].to_numpy(dtype=float)).all())))

    stats = json.loads((ROOT / "stats/final_stats.json").read_text(encoding="utf-8"))
    checks.append(("stats_contains_g4_g8_g9", all(k in stats for k in ("g4", "g8", "g9"))))
    final_library = read_gz(ROOT / "data/final_candidate_library_frozen.tsv.gz")
    checks.extend([
        ("final_library_rows", len(final_library) == 3240),
        ("tier_a_count", int((final_library["priority_tier"] == "A").sum()) == 74),
        ("tier_b_count", int((final_library["priority_tier"] == "B").sum()) == 1141),
        ("tier_c_count", int((final_library["priority_tier"] == "C").sum()) == 2025),
    ])

    failed = [name for name, ok in checks if not ok]
    report = {"checks": len(checks), "failed": failed, "status": "PASS" if not failed else "FAIL"}
    (ROOT / "stats/verify_submission_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
