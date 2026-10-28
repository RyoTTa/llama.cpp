#!/bin/bash
# Config A (3090): laptop-only — GPU1 layers 0-19 + CPU mmap rest.
# A100 placement verified (CUDA1 7200MB + CPU-mmap 10.5GB); same OT on 3090.
# Outputs use tag A-3090 so A100 logs (place-A.log, A.csv) are never overwritten.
# Usage: ./scripts/run_A_3090.sh
set -u
source "$HOME/memory_constrained_baseline/scripts/common-3090.sh"

TAG="A-3090"
OT='blk.(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19)\.=CUDA1,blk.(20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47)\.=CPU,token_embd.*=CPU,output.*=CPU'
# llama-bench splits -ot on ',' into separate tests (llama-bench.cpp:958); ';' separates
# rules within one placement. llama-cli uses ',' (arg.cpp:264). Same placement, both forms.
OT_BENCH='blk.(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19)\.=CUDA1;blk.(20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47)\.=CPU;token_embd.*=CPU;output.*=CPU'

echo "=== verify placement ($TAG) ==="
"$MCB_BIN/llama-cli" -m "$MCB_MODEL" -p hi -n 1 -c 32 --temp 0 \
  --device CUDA1 -sm none -ot "$OT" --verbose \
  > /dev/null 2> "$MCB_ROOT/logs/place-$TAG.log" || { echo "VERIFY FAILED"; exit 1; }
grep -E "model buffer size|KV buffer|compute buffer" "$MCB_ROOT/logs/place-$TAG.log" | head -12

echo "=== bench ($TAG) ==="
mcb_bench "$TAG" --device CUDA1 -sm none -ot "$OT_BENCH"
echo "DONE $TAG"
