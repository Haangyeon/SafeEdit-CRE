# Decision Log G7-G10

1. G7: Two architecturally independent models trained (ResidualDilatedCNN reviewer, MultiKernelCNN sealed evaluator)
2. G7: Performance gate at mean Pearson >= 0.75 on test set
3. G7: Calibration parameters (variance scale) determined EXCLUSIVELY on validation split; test set never used for calibration, epoch selection, checkpoint choice, or architecture decisions. calibration_source = validation_only.
4. G8: 600 new parents selected from test split, zero overlap with historical denylist
5. G8: 4 methods (random x3 replicates, greedy, primary_beam, safeedit_consensus)
6. G8: Random method uses 3 INDEPENDENT replicates; NUMERIC METRICS ARE AVERAGED across replicates in the collapsed table. Best-of-3 selection is PROHIBITED to avoid selection/optimization bias. n_replicates field added (3 for Random, 1 for others).
7. G8: Collapsed table is used EXCLUSIVELY for method-level paired statistical comparisons; it is NEVER used to pick individual DNA sequences.
8. G8: primary beam and SafeEdit use matched computational budget (identical beam width, expansion depth, pre-screen top-k)
9. G8: Sealed evaluator inference only after candidate freezing (blind assessment)
10. G9: True re-search ablation (7 configs), not post-hoc filtering
11. G9: The file g9_true_ablation_reconstructed_presealed_view.tsv.gz is a COLUMN-STRIPPED RECONSTRUCTION from the sealed table for display only; it does NOT prove sealed scores were absent during search.
12. G9: Temporal ordering evidence (search completion before sealed model load) must come from logs, timestamps, code, and config hashes catalogued in g9_presealed_manifest.json
13. G10: Sealed evaluator as primary endpoint
14. G10: Motif analysis with JASPAR 2024 CORE vertebrates
15. G10: Exact model forward-pass counts NOT instrumented during pipeline; reported as NA and listed as a manuscript limitation. Runtime extracted from logs where possible.
16. G10: No independent external MPRA available for external audit
17. Packaging: Single ZIP with complete source, logs, checksums, and manifests
