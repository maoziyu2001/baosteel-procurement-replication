#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 src/ga_hybrid/main.py \
  --num-bases 2 \
  --num-suppliers 16 \
  --instance-name case_2x16_complete \
  --seed 0

