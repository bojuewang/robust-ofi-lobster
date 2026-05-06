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
