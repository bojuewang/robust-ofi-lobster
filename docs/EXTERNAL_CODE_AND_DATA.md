# External Code and Data

This document records the external data, packages, and code-dependency assumptions for the Robust OFI LOBSTER project.

## External data

The project uses public LOBSTER sample files. The raw data are not committed to the repository.

Expected local data files:

```text
data/raw/LOBSTER_SampleFile_AMZN_2012-06-21_10.zip
data/raw/LOBSTER_SampleFile_AAPL_2012-06-21_10.zip
data/raw/LOBSTER_SampleFile_GOOG_2012-06-21_10.zip
data/raw/LOBSTER_SampleFile_INTC_2012-06-21_10.zip
data/raw/LOBSTER_SampleFile_MSFT_2012-06-21_10.zip
```

The scripts assume each zip contains a LOBSTER message CSV and a level-10 orderbook CSV for the same trading day. The data contract used by the repository is:

| Item | Value |
|---|---:|
| Trading day | 2012-06-21 |
| Continuous trading window | `[34200, 57600)` seconds after midnight |
| Bin width | 500ms |
| Number of bins | 46,800 |
| Assets | AMZN, AAPL, GOOG, INTC, MSFT |
| LOB levels | 10 |
| Feature tensor | `(46800, 5, 51)` |

## Data handling policy

Raw LOBSTER data are local-only. The `.gitignore` file excludes:

```text
data/raw/*.zip
data/raw/*.csv
data/interim/*.csv
data/processed/*.npz
```

Only lightweight reports, figures, benchmark summaries, and selected CSV/JSON summaries are intended for version control.

## External Python packages

The main dependencies are listed in `requirements.txt`:

```text
numpy
pandas
scikit-learn
matplotlib
```

The repository also uses Python standard-library modules such as `zipfile`, `json`, `csv`, `pathlib`, `math`, and `subprocess`.

## External code

No third-party source code was intentionally copied into the repository. The implementation was written for this project, with LLM-assisted drafting disclosed in `llm-usage.md`.

The repository uses standard open-source libraries through their public APIs. Any future copied or adapted code should be listed here with:

```text
source URL
license
files affected
what was adapted
reason for adaptation
```

## Random seeds and reproducibility

Stochastic training/evaluation scripts use fixed seeds in the generated Python modules, especially in feature-robust and graph-robust PGDA experiments. Numerical results may vary slightly across platforms or BLAS/LAPACK implementations, but benchmark gates are designed to catch substantive failures rather than harmless floating-point differences.

## Main reproducibility entry point

After placing raw LOBSTER zip files in `data/raw/`, run:

```bash
bash scripts/run_pipeline_to_step06.sh
```

This executes the benchmark-gated workflow through the final summary.

## Reported output locations

```text
reports/benchmarks/
reports/final/robust_ofi_final_summary.md
reports/final/final_ablation_table.csv
reports/figures/
```
