#!/bin/bash
# Config C bench (3090): all-MoE-on-CPU via bench-expressible --n-cpu-moe 48,
# attention split GPU0/GPU1 (ts 6,8). No -ot (no comma/semicolon issue).
# NOTE: run_all.sh used --cpu-moe/--moe-cache-mib which llama-bench lacks
# (cli-only); --n-cpu-moe 48 is the bench equivalent of --cpu-moe.
# Cache-size dimension is covered separately (server + measure_server.py).
# Outputs: tag C-3090-bench. Usage: ./scripts/run_C_3090.sh
set -u
source "$HOME/memory_constrained_baseline/scripts/common-3090.sh"

TAG="C-3090-bench"

echo "=== verify placement ($TAG) ==="
"$MCB_BIN/llama-cli" -m "$MCB_MODEL" -p hi -n 1 -c 32 --temp 0 \
  --device CUDA0,CUDA1 -sm layer -ts 6,8 --cpu-moe --verbose \
  > /dev/null 2> "$MCB_ROOT/logs/place-$TAG.log" || { echo "VERIFY FAILED"; exit 1; }
grep -E "model buffer size|KV buffer|compute buffer" "$MCB_ROOT/logs/place-$TAG.log" | head -12

echo "=== bench ($TAG) ==="
mcb_bench "$TAG" --device CUDA0/CUDA1 -sm layer -ts 6,8 --n-cpu-moe 48
echo "DONE $TAG"
