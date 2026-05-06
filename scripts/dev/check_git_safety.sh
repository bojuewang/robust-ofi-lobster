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
