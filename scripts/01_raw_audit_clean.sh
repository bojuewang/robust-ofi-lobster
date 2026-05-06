#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python3}"
RAW_DIR="${RAW_DIR:-data/raw}"
AUTO_INSTALL="${AUTO_INSTALL:-1}"

$PYTHON_BIN - <<'PY' || NEED_INSTALL=1
import importlib.util
missing = [m for m in ["numpy", "pandas"] if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit("missing packages: " + ", ".join(missing))
print("[INFO] Python dependencies available.")
PY

if [[ "${NEED_INSTALL:-0}" == "1" ]]; then
  if [[ "$AUTO_INSTALL" == "1" ]]; then
    echo "[INFO] Installing Python dependencies from requirements.txt"
    $PYTHON_BIN -m pip install --user --only-binary=:all: -r requirements.txt
  else
    echo "[ERROR] Missing Python dependencies. Run: $PYTHON_BIN -m pip install --user -r requirements.txt" >&2
    exit 1
  fi
fi

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
rm -f logs/step01_raw_audit_clean.log
for TICKER in AMZN AAPL GOOG INTC MSFT; do
  $PYTHON_BIN -m lobster_ofi.step01_raw_audit_clean --raw-dir "$RAW_DIR" --interim-dir data/interim --reports-dir reports/benchmarks --ticker "$TICKER" 2>&1 | tee -a logs/step01_raw_audit_clean.log
done
$PYTHON_BIN -m lobster_ofi.step01_raw_audit_clean --raw-dir "$RAW_DIR" --interim-dir data/interim --reports-dir reports/benchmarks --combine-only 2>&1 | tee -a logs/step01_raw_audit_clean.log

echo ""
echo "===== Step 01 Benchmark Markdown ====="
cat reports/benchmarks/step01_raw_audit_clean.md
