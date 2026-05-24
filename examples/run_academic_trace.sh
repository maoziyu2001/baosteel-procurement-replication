#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 src/ga_hybrid/main.py \
  --num-bases 2 \
  --num-suppliers 16 \
  --pop-size 4 \
  --max-gen 4 \
  --k-explore 2 \
  --k-ls 2 \
  --elite-ls 1 \
  --ls-max-iter 1 \
  --instance-name typical_2x16_trace

