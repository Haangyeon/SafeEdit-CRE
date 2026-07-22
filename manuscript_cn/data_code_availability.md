# Data and Code Availability Statement

**This statement will appear in the "Availability of data and materials" section of the final manuscript.**

## Source data
The MPRA activity data used to train and evaluate all models are publicly available from
Gosai et al. (2024) via the Gene Expression Omnibus (GEO) accession associated with that
publication and the corresponding Malinois model repository on GitHub
(https://github.com/harrisonliba/Malinois). No new experimental data were generated in
this study.

## Code
All source code for SafeEdit-CRE, including (i) model training, (ii) constrained beam-search
editing, (iii) uncertainty calibration, (iv) nine-criterion design benchmarking, (v) complete
re-search ablations, (vi) candidate tiering, and (vii) figure and statistical-table generation,
is released under the MIT License in a public GitHub repository:

- **Repository:** https://github.com/haangyeon/SafeEdit-CRE
- **Release tag:** v1.0.0 (commit hash fixed in the Zenodo archive)

## Permanent archive
A versioned, immutable archive of the code, environment files, pretrained model checkpoints,
candidate-level result tables, and a minimal reproduction script is deposited on Zenodo with
a permanent DOI:

- **Zenodo archive (v1.0.0):** https://doi.org/10.5281/zenodo.21458489

The Zenodo archive includes:
1. Source code (`src/`, `scripts/`)
2. Conda and pip environment files (`environment.yml`, `requirements.txt`)
3. Minimal end-to-end reproduction script (`scripts/minimal_example.py`) with a small
   example input and expected output
4. Core statistical input tables used to generate all figures and numerical results
   (`data/g8_candidates_sealed_collapsed.tsv.gz`, `data/g4_frozen_candidates.tsv.gz`,
   `data/final_candidate_library_frozen.tsv.gz`, `data/g9_true_ablation_sealed.tsv.gz`)
5. The full 74-sequence Tier A candidate library in machine-readable TSV format
   (`data/tier_a_candidates.tsv`)
6. Pretrained reviewer and cross-model evaluation ensemble checkpoints
7. SHA-256 checksums for every payload file (`CHECKSUMS_SHA256.txt`)

## Repository state at submission
The repository and Zenodo archive are public at the time of submission. Reviewers can
access the fixed v1.0.0 release directly, including the code, model checkpoints,
candidate library, summary tables, and checksums used for this manuscript. The DOI is
versioned so that the exact computational record associated with the submitted article
remains permanently identifiable.
