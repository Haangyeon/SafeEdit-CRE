# SafeEdit-CRE

**Uncertainty-aware cross-model selection for cell-type-specific minimal regulatory-DNA editing**

This repository accompanies the manuscript:

> SafeEdit-CRE: uncertainty-aware cross-model selection improves minimal regulatory-DNA editing for cell-type-specific design. *BMC Bioinformatics* (submitted).

## Overview

SafeEdit-CRE is a computational framework for designing cell-type-selective variants of natural 200-nt cis-regulatory elements (CREs) under fixed nucleotide substitution budgets (1, 5, 10, or 20 edits). It couples the published Malinois cell-type-specificity predictor with an architecture-diverse independent evaluation ensemble, validation-calibrated uncertainty estimation, sequence-domain controls, and constrained beam search. Candidates are ranked by an uncertainty-aware cross-model selection strategy to reduce single-model optimization bias.

**Key results** (600 held-out natural CRE parents in K562, HepG2, and SK-N-SH):
- Cross-model specificity-margin gain vs. greedy: 0.822 → 0.877 (+0.055; 95% CI 0.040–0.071)
- vs. compute-matched beam search: equivalent specificity (Δ = −0.004), lower uncertainty (0.314 vs. 0.335)
- Nine-criterion design pass rate: 61.5% (SafeEdit-CRE) vs. 38.5% (greedy) vs. 13.0% (random)
- 78 Tier A experimentally actionable candidate sequences across all three cell types

## Repository structure

```
SafeEdit-CRE/
├── src/safeedit_cre/       # Python source code (PyTorch)
│   ├── baseline.py         # Baseline methods (random, greedy)
│   ├── cnn_model.py        # CNN architecture definitions
│   ├── cnn_ensemble_eval.py# Cross-model ensemble evaluation
│   ├── edit_benchmark.py   # Nine-criterion design benchmark
│   ├── features.py         # Sequence feature extraction
│   ├── malinois.py         # Malinois predictor interface
│   ├── sequence.py         # Sequence manipulation utilities
│   ├── split.py            # Data splitting (train/val/test)
│   └── ...                 # Additional pipeline modules
├── data/
│   ├── final_candidate_library_frozen.tsv.gz  # Full candidate library
│   └── g4_frozen_candidates.tsv.gz            # 96-parent benchmark results
├── stats/
│   ├── final_stats.json                        # All reported statistics
│   ├── table_g4_method_summary.tsv             # Nine-criterion summary
│   ├── table_g8_method_summary.tsv             # Primary endpoint summary
│   └── table_g9_ablation_summary.tsv           # Ablation results
├── scripts/
│   ├── make_final_figures.py                   # Regenerate all figures
│   └── verify_submission.py                    # Submission verification
├── provenance/
│   ├── FROZEN_CONFIG.yaml                      # Locked benchmark config
│   └── SERVER_ENVIRONMENT.txt                  # Compute environment
├── manuscript/
│   ├── main.tex / supplementary.tex            # LaTeX sources
│   ├── sections/                               # Chapter files
│   ├── figures/                                # PDF + PNG figures
│   └── references.bib                          # Bibliography
├── CITATION.cff
├── requirements.txt
└── CHECKSUMS_SHA256.txt
```

## Quick start

```bash
# Clone
git clone https://github.com/Haangyeon/SafeEdit-CRE.git
cd SafeEdit-CRE

# Environment
pip install -r requirements.txt

# Verify submission integrity
python scripts/verify_submission.py

# Regenerate figures (CPU only, no GPU required)
python scripts/make_final_figures.py
```

## Data access

The compressed candidate library (`data/final_candidate_library_frozen.tsv.gz`, ~15 MB) contains all design results from the primary 600-parent evaluation, the 96-parent benchmark, and the ablation study. Key columns:

| Column | Description |
|--------|-------------|
| `parent_id` | Genomic coordinate identifier |
| `target_cell` | K562, HepG2, or SKNSH |
| `budget` | 1, 5, 10, or 20 |
| `design_method` | safeedit_consensus, greedy_malinois, or random_matched |
| `primary_margin_gain` | Malinois predictor margin change |
| `reviewer_margin_gain` | Independent ensemble margin change |
| `reviewer_uncertainty` | Cross-model predictive uncertainty |
| `priority_tier` | A, B, or C |
| `audit_pass` | Whether all nine criteria were met |

## Reproducibility

- All random seeds are fixed (`seed = 20260714` for benchmark, `seed = 20260713` for bootstrap).
- The frozen evaluation protocol ensures the cross-model ensemble was locked before candidate search.
- Model checkpoints and their SHA-256 hashes are archived in the Zenodo release.
- See `provenance/FROZEN_CONFIG.yaml` for the complete locked configuration.

## Environment

- Python 3.10 (conda-forge)
- PyTorch 2.6.0+cu124
- CUDA 12.4 / cuDNN 9.0
- GPU: NVIDIA RTX 3080 Ti (final benchmarks) / RTX 3060 Laptop (development)

## Citation

If you use this code or data, please cite:

```bibtex
@article{safeedit_cre_2026,
  title={SafeEdit-CRE: uncertainty-aware cross-model selection improves minimal regulatory-DNA editing for cell-type-specific design},
  author={Yiwei Zhang},
  journal={BMC Bioinformatics},
  year={2026},
  doi={10.5281/zenodo.XXXXXXX}
}
```

## License

Code: MIT License. Data and candidate sequences: CC-BY 4.0.

## Contact

Yiwei Zhang — zhang_yiwei0404@163.com
