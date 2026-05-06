#!/usr/bin/env bash
set -euo pipefail

# Step 07: GitHub packaging / commit-ready repository scaffold
#
# Purpose:
#   - Make the current robust-ofi-lobster project safe and clean for GitHub.
#   - Copy pipeline step scripts into scripts/pipeline/ if available.
#   - Generate/update README, LICENSE, CITATION, requirements, .gitignore.
#   - Produce a packaging benchmark report.
#   - Initialize git repository if not already initialized.
#   - Do NOT push automatically. The final push should be user-controlled.
#
# Inputs:
#   .pipeline_state/step06_summary.PASS
#   reports/final/robust_ofi_final_summary.md
#
# Run:
#   bash step07_github_packaging.sh
# or:
#   PROJECT_DIR=/path/to/robust-ofi-lobster bash step07_github_packaging.sh
#
# Optional:
#   REPO_NAME=robust-ofi-lobster bash step07_github_packaging.sh

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
PROJECT_DIR="${PROJECT_DIR:-${ROOT_DIR}/robust-ofi-lobster}"
REPO_NAME="${REPO_NAME:-robust-ofi-lobster}"

if [[ ! -d "${PROJECT_DIR}" ]]; then
  if [[ -f "${ROOT_DIR}/reports/final/robust_ofi_final_summary.md" ]]; then
    PROJECT_DIR="${ROOT_DIR}"
  else
    echo "[ERROR] PROJECT_DIR not found: ${PROJECT_DIR}"
    echo "        Set PROJECT_DIR=/path/to/robust-ofi-lobster"
    exit 1
  fi
fi

cd "${PROJECT_DIR}"

if [[ ! -f ".pipeline_state/step06_summary.PASS" ]]; then
  echo "[ERROR] Step 06 gate missing: .pipeline_state/step06_summary.PASS"
  echo "        Run Step 06 first."
  exit 1
fi

mkdir -p scripts/pipeline scripts/dev reports/benchmarks reports/final reports/figures docs .pipeline_state

# ---------------------------------------------------------------------
# 1. Copy pipeline scripts into repository if they exist in parent/root.
# ---------------------------------------------------------------------
for s in \
  step01_lobster_raw_audit_clean.sh \
  step02_feature_build_500ms.sh \
  step03_graph_build_and_baseline.sh \
  step04_feature_robust_pgda.sh \
  step04b_feature_robust_calibrated_eval.sh \
  step05_graph_robust_pgda.sh \
  step06_feature_graph_robust_summary.sh \
  step07_github_packaging.sh
do
  if [[ -f "${ROOT_DIR}/${s}" ]]; then
    cp "${ROOT_DIR}/${s}" "scripts/pipeline/${s}"
  elif [[ -f "../${s}" ]]; then
    cp "../${s}" "scripts/pipeline/${s}"
  elif [[ -f "${s}" ]]; then
    cp "${s}" "scripts/pipeline/${s}"
  fi
done

chmod +x scripts/pipeline/*.sh 2>/dev/null || true

# ---------------------------------------------------------------------
# 2. Reproducible runner.
# ---------------------------------------------------------------------
cat > scripts/run_pipeline_to_step06.sh <<'RUN'
#!/usr/bin/env bash
set -euo pipefail

# Run from repository root after placing LOBSTER sample zip files under data/raw/.
# This executes the full benchmark-gated pipeline through final summary.

bash scripts/pipeline/step01_lobster_raw_audit_clean.sh
bash scripts/pipeline/step02_feature_build_500ms.sh
bash scripts/pipeline/step03_graph_build_and_baseline.sh
bash scripts/pipeline/step04_feature_robust_pgda.sh
bash scripts/pipeline/step04b_feature_robust_calibrated_eval.sh
bash scripts/pipeline/step05_graph_robust_pgda.sh
bash scripts/pipeline/step06_feature_graph_robust_summary.sh

echo "[DONE] Full pipeline through Step 06 completed."
RUN
chmod +x scripts/run_pipeline_to_step06.sh

cat > scripts/dev/check_git_safety.sh <<'RUN'
#!/usr/bin/env bash
set -euo pipefail

echo "[Git safety check]"
echo "Large files over 25MB not ignored/untracked:"
find . -type f -size +25M \
  -not -path "./.git/*" \
  -not -path "./data/raw/*" \
  -not -path "./data/interim/*" \
  -not -path "./data/processed/*" \
  -print || true

echo
echo "Raw LOBSTER files under data/raw are intentionally ignored:"
find data/raw -maxdepth 1 -type f 2>/dev/null | sed 's#^\./##' || true

echo
echo "Git status:"
git status --short || true
RUN
chmod +x scripts/dev/check_git_safety.sh

# ---------------------------------------------------------------------
# 3. GitHub-safe metadata files.
# ---------------------------------------------------------------------
cat > .gitignore <<'EOF'
# Python
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.ipynb_checkpoints/

# Virtual environments
.venv/
venv/
env/

# Raw/proprietary or large LOBSTER data
data/raw/*.zip
data/raw/*.csv
data/interim/*.csv
data/processed/*.npz

# Keep lightweight summary outputs
!data/processed/*_results*.csv
!data/processed/*summary*.csv
!data/processed/*metadata*.json

# Pipeline state
.pipeline_state/*.FAIL

# Logs and temporary files
*.log
tmp/
temp/

# OS/editor
.DS_Store
Thumbs.db
.vscode/
.idea/
EOF

cat > requirements.txt <<'EOF'
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
EOF

cat > LICENSE <<'EOF'
MIT License

Copyright (c) 2026 Bojue Wang and Jiayu Yang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
EOF

cat > CITATION.cff <<'EOF'
cff-version: 1.2.0
message: "If you use this project, please cite it as below."
title: "Robust Optimization for Order Flow Imbalance Prediction"
authors:
  - family-names: Wang
    given-names: Bojue
  - family-names: Yang
    given-names: Jiayu
date-released: 2026-05-06
version: "0.1.0"
license: MIT
repository-code: "https://github.com/<your-username>/robust-ofi-lobster"
EOF

cat > README.md <<EOF
# Robust Optimization for Order Flow Imbalance Prediction

This repository implements a benchmark-gated, reproducible pipeline for short-horizon return prediction from multi-level LOBSTER sample limit order book data. The core modeling question is whether robust optimization improves stability under feature perturbations and cross-asset graph misspecification.

## Project summary

- Data source: public LOBSTER sample files.
- Assets: AMZN, AAPL, GOOG, INTC, MSFT.
- LOB depth: 10 levels.
- Time grid: 500ms bins over the regular trading window.
- Number of bins: \(T = 46800\).
- Feature tensor: \`X_std.shape = (46800, 5, 51)\`.
- Split: chronological 70% / 15% / 15%.
- Models:
  - Ridge/Lasso baselines.
  - Graph-linear ERM.
  - Feature-robust PGDA.
  - Graph-robust PGDA.

## Main finding

Graph-robust training gives the most consistent robustness improvement in the current experiment. It reduces MSE degradation under graph perturbation for all tested horizons and lowers graph sensitivity across all tested horizons. Feature-robust training is more mixed: it helps mainly at longer horizons but can sacrifice clean test accuracy.

See:

\`\`\`text
reports/final/robust_ofi_final_summary.md
reports/final/final_ablation_table.csv
reports/figures/
\`\`\`

## Repository layout

\`\`\`text
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
\`\`\`

## Reproduce the pipeline

Create a virtual environment:

\`\`\`bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
\`\`\`

Place the five LOBSTER sample zip files under:

\`\`\`text
data/raw/
\`\`\`

Then run:

\`\`\`bash
bash scripts/run_pipeline_to_step06.sh
\`\`\`

Each step creates a benchmark report and a gate file under \`.pipeline_state/\`. A later step should only be run after the previous gate passes.

## Pipeline steps

| Step | Script | Purpose |
|---|---|---|
| 01 | \`scripts/pipeline/step01_lobster_raw_audit_clean.sh\` | Raw LOBSTER audit and cleaning benchmark |
| 02 | \`scripts/pipeline/step02_feature_build_500ms.sh\` | 500ms feature tensor construction |
| 03 | \`scripts/pipeline/step03_graph_build_and_baseline.sh\` | Graph construction and ERM baselines |
| 04 | \`scripts/pipeline/step04_feature_robust_pgda.sh\` | Initial feature-robust PGDA |
| 04b | \`scripts/pipeline/step04b_feature_robust_calibrated_eval.sh\` | Calibrated feature-adversarial evaluation |
| 05 | \`scripts/pipeline/step05_graph_robust_pgda.sh\` | Graph-robust PGDA |
| 06 | \`scripts/pipeline/step06_feature_graph_robust_summary.sh\` | Final summary, figures, and ablation tables |
| 07 | \`scripts/pipeline/step07_github_packaging.sh\` | GitHub packaging and safety checks |

## Data policy

Raw LOBSTER data and large generated arrays are not committed to GitHub. The \`.gitignore\` file excludes:

\`\`\`text
data/raw/*.zip
data/raw/*.csv
data/interim/*.csv
data/processed/*.npz
\`\`\`

Before committing, run:

\`\`\`bash
bash scripts/dev/check_git_safety.sh
\`\`\`

## GitHub publishing

This script initializes a local git repository if needed, but it does not push automatically.

Suggested commands after reviewing \`reports/benchmarks/step07_github_packaging.md\`:

\`\`\`bash
git add README.md LICENSE CITATION.cff requirements.txt .gitignore scripts src reports/final reports/figures reports/benchmarks
git commit -m "Add robust OFI LOBSTER reproducible pipeline"

# Then create an empty GitHub repository named ${REPO_NAME}, and run:
git branch -M main
git remote add origin git@github.com:<your-username>/${REPO_NAME}.git
git push -u origin main
\`\`\`
EOF

# ---------------------------------------------------------------------
# 4. Create a lightweight project manifest.
# ---------------------------------------------------------------------
cat > docs/reproducibility_manifest.md <<'EOF'
# Reproducibility Manifest

## Data

The project expects five LOBSTER sample zip files under `data/raw/`:

- `LOBSTER_SampleFile_AMZN_2012-06-21_10.zip`
- `LOBSTER_SampleFile_AAPL_2012-06-21_10.zip`
- `LOBSTER_SampleFile_GOOG_2012-06-21_10.zip`
- `LOBSTER_SampleFile_INTC_2012-06-21_10.zip`
- `LOBSTER_SampleFile_MSFT_2012-06-21_10.zip`

These files are not committed.

## Determinism

The scripts use fixed seeds where stochastic training is used. Numerical results may vary slightly across BLAS/LAPACK implementations, but benchmark gates are designed to tolerate harmless floating-point differences.

## Benchmark gates

Every module writes a markdown report under `reports/benchmarks/` and a PASS marker under `.pipeline_state/`.
EOF

# ---------------------------------------------------------------------
# 5. Initialize git if needed.
# ---------------------------------------------------------------------
if [[ ! -d ".git" ]]; then
  git init >/dev/null 2>&1 || true
fi

# ---------------------------------------------------------------------
# 6. Build benchmark report.
# ---------------------------------------------------------------------
python - <<'PY'
from pathlib import Path
import json
import subprocess
import os

checks = []

def add(name, ok, observed, expected):
    checks.append({
        "check": name,
        "status": "PASS" if ok else "FAIL",
        "observed": str(observed),
        "expected": str(expected),
    })

required = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "requirements.txt",
    ".gitignore",
    "docs/reproducibility_manifest.md",
    "reports/final/robust_ofi_final_summary.md",
    "reports/final/final_ablation_table.csv",
    "scripts/run_pipeline_to_step06.sh",
    "scripts/dev/check_git_safety.sh",
]

for p in required:
    add(f"{p}_exists", Path(p).exists() and Path(p).stat().st_size > 0, p, "nonempty file")

pipeline_scripts = list(Path("scripts/pipeline").glob("step*.sh"))
add("pipeline_scripts_present", len(pipeline_scripts) >= 5, len(pipeline_scripts), ">=5")

# Check that raw data and large arrays are ignored.
def git_check_ignore(path):
    try:
        r = subprocess.run(["git", "check-ignore", "-q", path], check=False)
        return r.returncode == 0
    except Exception:
        return False

add("raw_zip_ignored", git_check_ignore("data/raw/example.zip"), "ignored" if git_check_ignore("data/raw/example.zip") else "not ignored", "ignored")
add("processed_npz_ignored", git_check_ignore("data/processed/example.npz"), "ignored" if git_check_ignore("data/processed/example.npz") else "not ignored", "ignored")

# Find large non-ignored files >25MB outside data dirs.
large = []
for p in Path(".").rglob("*"):
    if not p.is_file():
        continue
    if ".git" in p.parts:
        continue
    if p.stat().st_size <= 25 * 1024 * 1024:
        continue
    ps = str(p)
    if ps.startswith("data/raw/") or ps.startswith("data/interim/") or ps.startswith("data/processed/"):
        continue
    large.append(ps)
add("no_large_nondata_files_over_25MB", len(large) == 0, large if large else "none", "none")

# Git initialized.
add("git_initialized", Path(".git").exists(), ".git exists" if Path(".git").exists() else ".git missing", ".git exists")

status = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"

Path("reports/benchmarks").mkdir(parents=True, exist_ok=True)
Path(".pipeline_state").mkdir(parents=True, exist_ok=True)

with open("reports/benchmarks/step07_github_packaging.json", "w", encoding="utf-8") as f:
    json.dump({"status": status, "checks": checks}, f, indent=2)

md = []
md.append("# Step 07 GitHub Packaging Benchmark\n")
md.append(f"Overall status: **{status}**\n")
md.append("## Purpose")
md.append("This module prepares the project as a commit-ready GitHub repository while preventing raw LOBSTER data and large generated arrays from being committed.\n")
md.append("## Generated/updated files")
for p in required:
    md.append(f"- `{p}`")
md.append("- `scripts/pipeline/step*.sh` if source scripts were available\n")

md.append("## Recommended GitHub commands")
md.append("```bash")
md.append("bash scripts/dev/check_git_safety.sh")
md.append("git add README.md LICENSE CITATION.cff requirements.txt .gitignore docs scripts src reports/final reports/figures reports/benchmarks")
md.append('git commit -m "Add robust OFI LOBSTER reproducible pipeline"')
md.append("git branch -M main")
md.append("git remote add origin git@github.com:<your-username>/robust-ofi-lobster.git")
md.append("git push -u origin main")
md.append("```\n")

md.append("## Benchmark checks")
md.append("| check | status | observed | expected |")
md.append("|---|---|---|---|")
for c in checks:
    md.append(f"| {c['check']} | {c['status']} | `{c['observed']}` | `{c['expected']}` |")
md.append("")

if status == "PASS":
    Path(".pipeline_state/step07_github_packaging.PASS").write_text("PASS\n", encoding="utf-8")
    md.append("Gate passed: `.pipeline_state/step07_github_packaging.PASS` should exist.")
    md.append("Review `git status --short` before committing.")
else:
    Path(".pipeline_state/step07_github_packaging.FAIL").write_text("FAIL\n", encoding="utf-8")
    md.append("Gate failed: inspect `reports/benchmarks/step07_github_packaging.json`.")

Path("reports/benchmarks/step07_github_packaging.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
PY

echo
echo "[DONE] Step 07 packaging report:"
echo "  ${PROJECT_DIR}/reports/benchmarks/step07_github_packaging.md"
