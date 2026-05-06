# Robust Optimization for Order Flow Imbalance Prediction

This repository implements a benchmark-gated, reproducible pipeline for short-horizon return prediction from multi-level LOBSTER sample limit order book data. The core modeling question is whether robust optimization improves stability under feature perturbations and cross-asset graph misspecification.

## Project summary

- Data source: public LOBSTER sample files.
- Assets: AMZN, AAPL, GOOG, INTC, MSFT.
- LOB depth: 10 levels.
- Time grid: 500ms bins over the regular trading window.
- Number of bins: \(T = 46800\).
- Feature tensor: `X_std.shape = (46800, 5, 51)`.
- Split: chronological 70% / 15% / 15%.
- Models:
  - Ridge/Lasso baselines.
  - Graph-linear ERM.
  - Feature-robust PGDA.
  - Graph-robust PGDA.

## Main finding

Graph-robust training gives the most consistent robustness improvement in the current experiment. It reduces MSE degradation under graph perturbation for all tested horizons and lowers graph sensitivity across all tested horizons. Feature-robust training is more mixed: it helps mainly at longer horizons but can sacrifice clean test accuracy.

See:

```text
reports/final/robust_ofi_final_summary.md
reports/final/final_ablation_table.csv
reports/figures/
```

## Repository layout

```text
.
├── data/
│   ├── raw/             # ignored; place LOBSTER zip files here
│   ├── interim/         # ignored intermediate CSVs
│   └── processed/       # ignored large arrays; lightweight CSV/JSON summaries may be kept
├── reports/
│   ├── benchmarks/      # benchmark gate reports
│   ├── final/           # report-ready summary tables
│   └── figures/         # generated figures
├── scripts/
│   ├── pipeline/        # step-by-step executable scripts
│   ├── dev/             # repository safety utilities
│   └── run_pipeline_to_step06.sh
├── src/lobster_ofi/     # generated Python modules used by pipeline scripts
├── README.md
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Reproduce the pipeline

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the five LOBSTER sample zip files under:

```text
data/raw/
```

Then run:

```bash
bash scripts/run_pipeline_to_step06.sh
```

Each step creates a benchmark report and a gate file under `.pipeline_state/`. A later step should only be run after the previous gate passes.

## Pipeline steps

| Step | Script | Purpose |
|---|---|---|
| 01 | `scripts/pipeline/step01_lobster_raw_audit_clean.sh` | Raw LOBSTER audit and cleaning benchmark |
| 02 | `scripts/pipeline/step02_feature_build_500ms.sh` | 500ms feature tensor construction |
| 03 | `scripts/pipeline/step03_graph_build_and_baseline.sh` | Graph construction and ERM baselines |
| 04 | `scripts/pipeline/step04_feature_robust_pgda.sh` | Initial feature-robust PGDA |
| 04b | `scripts/pipeline/step04b_feature_robust_calibrated_eval.sh` | Calibrated feature-adversarial evaluation |
| 05 | `scripts/pipeline/step05_graph_robust_pgda.sh` | Graph-robust PGDA |
| 06 | `scripts/pipeline/step06_feature_graph_robust_summary.sh` | Final summary, figures, and ablation tables |
| 07 | `scripts/pipeline/step07_github_packaging.sh` | GitHub packaging and safety checks |

## Data policy

Raw LOBSTER data and large generated arrays are not committed to GitHub. The `.gitignore` file excludes:

```text
data/raw/*.zip
data/raw/*.csv
data/interim/*.csv
data/processed/*.npz
```

Before committing, run:

```bash
bash scripts/dev/check_git_safety.sh
```

## GitHub publishing

This script initializes a local git repository if needed, but it does not push automatically.

Suggested commands after reviewing `reports/benchmarks/step07_github_packaging.md`:

```bash
git add README.md LICENSE CITATION.cff requirements.txt .gitignore scripts src reports/final reports/figures reports/benchmarks
git commit -m "Add robust OFI LOBSTER reproducible pipeline"

# Then create an empty GitHub repository named robust-ofi-lobster, and run:
git branch -M main
git remote add origin git@github.com:<your-username>/robust-ofi-lobster.git
git push -u origin main
```
