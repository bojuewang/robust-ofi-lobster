# Project Structure

The repository separates raw data, generated arrays, benchmark reports, final summaries, and executable scripts.

```text
.
├── data/
│   ├── raw/             # local LOBSTER zip files; ignored by Git
│   ├── interim/         # intermediate CSV files; ignored by Git
│   └── processed/       # large arrays ignored; small CSV/JSON summaries may be kept
├── docs/
│   ├── PIPELINE.md
│   ├── PROJECT_STRUCTURE.md
│   ├── RESULTS.md
│   └── reproducibility_manifest.md
├── reports/
│   ├── benchmarks/      # step-by-step PASS/FAIL reports
│   ├── final/           # report-ready summary and ablation table
│   └── figures/         # generated figures
├── scripts/
│   ├── pipeline/        # executable step scripts
│   ├── dev/             # safety utilities
│   └── run_pipeline_to_step06.sh
├── src/lobster_ofi/     # Python modules used by the scripts
├── README.md
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## What should be committed

Commit:

```text
README.md
LICENSE
CITATION.cff
requirements.txt
.gitignore
docs/
scripts/
src/
reports/benchmarks/
reports/final/
reports/figures/
```

Do not commit:

```text
data/raw/*.zip
data/raw/*.csv
data/interim/*.csv
data/processed/*.npz
```

Run the safety check before committing:

```bash
bash scripts/dev/check_git_safety.sh
```

## Suggested future cleanup

The current scripts are intentionally self-contained for reproducibility. After the project is graded or submitted, the next engineering cleanup is to factor shared utilities into stable Python modules such as:

```text
src/lobster_ofi/io.py
src/lobster_ofi/features.py
src/lobster_ofi/graphs.py
src/lobster_ofi/models.py
src/lobster_ofi/evaluation.py
```
