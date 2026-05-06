# Next Steps

This project is now reproducible and GitHub-ready. The next improvements should separate research extensions from engineering cleanup.

## Immediate checks

1. Run a fresh clone test on a clean local directory.
2. Confirm the GitHub Actions smoke test passes.
3. Verify that raw LOBSTER zip files and large `.npz` arrays are not tracked.

## Research extensions

### 1. Rolling graph experiment

Current main result uses a fixed training-period graph. The most valuable extension is to compare:

```text
G_train
G_t^return
G_t^OFI
G_t^cross-OFI
G_t^mask-aware
```

Report whether graph-robust training still reduces degradation when the graph itself is time-varying.

### 2. Transaction-cost-aware diagnostic

The current PnL score is only a sanity check. Add transaction-cost grids:

```text
c_tc in {0, 0.5 bp, 1 bp, 2 bp, 5 bp}
```

Then report whether robust models reduce turnover or stabilize PnL under perturbations.

### 3. Multiple-day validation

The current public LOBSTER sample covers one day. If more data are available, repeat the pipeline on multiple days and use day-based splits.

### 4. Model refactor

Move shared code into stable modules:

```text
src/lobster_ofi/io.py
src/lobster_ofi/features.py
src/lobster_ofi/graphs.py
src/lobster_ofi/models.py
src/lobster_ofi/robust.py
src/lobster_ofi/evaluation.py
```

The current step scripts are intentionally self-contained for benchmark-gated reproducibility.

## Suggested GitHub issues

- `[Repro] Fresh-clone reproducibility test`
- `[CI] Monitor repo smoke test`
- `[Experiment] Add rolling graph G_t benchmark`
- `[Experiment] Add transaction-cost-aware PnL diagnostic`
- `[Report] Write final LaTeX report from Step 06 results`
- `[Refactor] Split generated step scripts into reusable Python modules`

## Report-ready conclusion

Graph-robust training is the strongest current result: it consistently reduces graph-perturbation MSE degradation and graph sensitivity across tested horizons. Feature-robust training is useful mainly at longer horizons and shows a stronger clean-accuracy/robustness trade-off.
