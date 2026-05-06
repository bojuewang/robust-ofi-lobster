#!/usr/bin/env bash
set -euo pipefail
if [[ ! -f .pipeline_state/step01_raw_audit.PASS ]]; then
  echo "[ERROR] Step 01 did not pass. Do not prepare GitHub yet." >&2
  exit 1
fi
cat > REPRODUCIBILITY.md <<'MD'
# Reproducibility

Step 01 uses fixed constants:

- tickers: AMZN, AAPL, GOOG, INTC, MSFT
- date: 2012-06-21
- levels: 10
- time window: [34200, 57600)
- bin width: 0.5 seconds
- bins: 46800
- split: 32760 / 7020 / 7020

Raw LOBSTER zip files are not committed. Place them under `data/raw/`.

Run:

```bash
bash scripts/01_raw_audit_clean.sh
```
MD
if [[ ! -d .git ]]; then git init; fi
git add README.md REPRODUCIBILITY.md requirements.txt .gitignore scripts src tests || true
git status --short
echo "Next Git commands:"
echo "  git commit -m 'Step 01 raw LOBSTER audit and cleaning gate'"
echo "  git branch -M main"
echo "  git remote add origin <YOUR_GITHUB_REPO_URL>"
echo "  git push -u origin main"
