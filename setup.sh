#!/usr/bin/env bash
# Phase 1 setup for the statcompare project. Run inside ~/statcompare on the EC2 box.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
echo "[setup] python: $($PY --version)"

# 1) venv
if [ ! -d .venv ]; then
  "$PY" -m venv .venv
  echo "[setup] created .venv"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip >/dev/null

# 2) core deps (CPU-fine)
pip install -r requirements.txt

# 3) torch + ML stack: install the CUDA build matching the box.
#    Check your driver with: nvidia-smi   (look at "CUDA Version")
#    Then UNCOMMENT one of the following, or install manually:
# pip install torch --index-url https://download.pytorch.org/whl/cu121
# pip install torch --index-url https://download.pytorch.org/whl/cu124
echo "[setup] NOTE: install torch for your CUDA, then: pip install transformers accelerate sentence-transformers datasets"

# 4) sanity checks (offline, no network / GPU needed)
echo "[setup] running offline self-tests..."
python tests/test_checker.py
python -m src.fetch_census --selftest
echo "[setup] Phase 1 environment ready."
