# SafeEdit-CRE G7-G10 Project State

## Status: COMPLETE

## Stage States
### G10_PASS
```

```

### G7_PASS
```
reviewer_mean_pearson=0.8364
sealed_mean_pearson=0.7708
gate=0.75
```

### G8_PASS
```

```

### G9_PASS
```
merged_by_parallel_jobs_2026-07-16T21:52:32+0800
```

## Pipeline Stages
- G7: Train reviewer (ResidualDilatedCNN) and sealed evaluator (MultiKernelCNN) on full data
- G7: Performance gate at mean Pearson >= 0.75 on test set
- G7: Calibration (variance scaling) determined on VALIDATION set only; test set never used for calibration
- G8: Select 600 new parents from test split, zero overlap with historical
- G8: Run 4 methods (random x3, greedy, primary_beam, safeedit_consensus)
- G8: Random 3 replicates are AVERAGED over numeric metrics (no best-of-3 selection to avoid bias)
- G8: Collapsed table for method-level paired statistics only — NEVER for DNA sequence selection
- G8: Sealed evaluator inference only after candidate freezing
- G9: True re-search ablation on 180 parents, 7 configs
- G9: Presealed view is RECONSTRUCTED from sealed table by column removal; does NOT prove blinding by itself
- G9: Temporal ordering evidence in g9_presealed_manifest.json (logs, timestamps, hashes)
- G10: Sealed endpoint statistics + JASPAR motif analysis
- G10: Model call counts reported as NA (not instrumented) — listed as manuscript limitation
- G10: primary beam and SafeEdit matched on beam width, expansion depth, pre-screen top-k
