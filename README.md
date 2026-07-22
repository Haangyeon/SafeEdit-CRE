# SafeEdit-CRE v1.1.0

SafeEdit-CRE is a cross-model, uncertainty-aware minimal-editing workflow for
cell-type-selective cis-regulatory element (CRE) design.

## Frozen release contents

- `data/g4_frozen_candidates.tsv.gz`: 3,240 records from the locked 90-parent
  benchmark (90 parents × 3 methods × 3 cell types × 4 edit budgets).
- `data/final_candidate_library_frozen.tsv.gz`: the same 3,240-record library,
  with 74 Tier A, 1,141 Tier B, and 2,025 Tier C candidates.
- `data/g8_candidates_sealed_collapsed.tsv.gz`: sealed G8 candidate-level inputs.
- `data/g9_true_ablation_sealed.tsv.gz`: sealed, true re-search ablation inputs.
- `data/90_parent_manifest.tsv`, `data/excluded_6_parents.tsv`, and
  `reports/g4_verification_report.txt`: provenance and audit records.
- `source/`, `scripts/`, `provenance/`, `reports/`, and `stats/`: analysis code,
  frozen configurations, statistical summaries, and figure-generation utilities.
- `manuscript/`: the synchronized English manuscript, supplementary information,
  cover letter, sources, and journal figures.

## Locked G4 benchmark

SafeEdit-CRE passed 654/1,080 conditions (60.6%), compared with 419/1,080
(38.8%) for greedy editing and 142/1,080 (13.1%) for random substitutions.
The SafeEdit-CRE--greedy difference is 21.76 percentage points. Tier A is a
computational priority tier for reporter screening; it is not a claim of measured
biological activity.

## Reproduction

The lightweight statistics and figure scripts can be run from the released summary
tables without GPU access. The full model-training and inference scripts are under
`source/safeedit_cre/`. Checkpoint files are not embedded in this local preparation
bundle; they must be added from the retained training-server copies before creating
the public Zenodo archive if exact checkpoint redistribution is required.

## Version status

This directory is the upload-ready v1.1.0 release candidate. Publish it to the
GitHub repository and the corresponding Zenodo version only after the external
release records have been created and their final DOI has been written into
`manuscript/main.tex`, `manuscript/data_code_availability.md`, and the cover letter.
