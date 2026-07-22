# Data and Code Availability Statement

**This statement will appear in the "Availability of data and materials" section of the final manuscript.**

## Source data
The processed MPRA activity data used to train Malinois are provided in Supplementary
Table 2 of Gosai et al. (2024). The associated raw data, processing notebooks, model
weights, and immunofluorescence images are archived at Zenodo record 10698014
(https://zenodo.org/records/10698014). The RNA-seq accession PRJNA1075667 is not used
as the MPRA source in this study. No new experimental data were generated.

## Code
All source code for SafeEdit-CRE, including (i) model training, (ii) constrained beam-search
editing, (iii) uncertainty calibration, (iv) nine-criterion design benchmarking, (v) complete
re-search ablations, (vi) candidate tiering, and (vii) figure and statistical-table generation,
is released under the MIT License in a public GitHub repository:

- **Repository:** https://github.com/haangyeon/SafeEdit-CRE
- **Release tag:** v1.1.0 (commit hash fixed in the Zenodo archive)

## Permanent archive
A versioned, immutable archive of the code, environment files, candidate-level result tables,
and a minimal reproduction script is prepared as the v1.1.0 release bundle. It will be
deposited on Zenodo and assigned a permanent DOI before publication; the DOI and fixed
GitHub commit will be added to the published version. The complete bundle is available
from the corresponding author during peer review.

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
6. Reviewer and cross-model evaluation ensemble checkpoints retained for the release archive
7. SHA-256 checksums for every payload file (`CHECKSUMS_SHA256.txt`)

## Repository state at submission
The v1.1.0 release is frozen at submission and will be made public in the GitHub and
Zenodo records before publication. Reviewers may request the complete bundle from the
corresponding author during peer review; the final DOI and commit will identify the
exact computational record associated with the published article.
