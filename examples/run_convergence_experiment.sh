#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 experiments/run_supplier_count_sensitivity.py

