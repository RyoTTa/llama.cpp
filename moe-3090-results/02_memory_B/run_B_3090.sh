#!/bin/bash
# Config B (3090): handheld(GPU0 0-13) + laptop(GPU1 14-34) + CPU(35-47+fixed) via localhost RPC.
# A100 run_B remapped to GPU0/GPU1 (was GPU3/GPU1: A100-side occupancy, N/A here).
# Outputs use tag B-3090; A100 logs untouched. PID-direct kill only (no pkill/pgrep -f).
# Usage: ./scripts/run_B_3090.sh
set -u
source "$HOME/memory_constrained_baseline/scripts/common-3090.sh"
RPCLIB="$HOME/moe_rpc_benchmark/rpcmon/librpcmon.so"

TAG="B-3090"
P0=50310
P1=50311
OT='blk.(0|1|2|3|4|5|6|7|8|9|10|11|12|13)\.=RPC0[127.0.0.1:50310],blk.(14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34)\.=RPC0[127.0.0.1:50311],blk.(35|36|37|38|39|40|41|42|43|44|45|46|47)\.=CPU,token_embd.*=CPU,output.*=CPU'
# llama-bench splits -ot on ',' into separate tests; ';' keeps one placement (cli keeps ',').
OT_BENCH='blk.(0|1|2|3|4|5|6|7|8|9|10|11|12|13)\.=RPC0[127.0.0.1:50310];blk.(14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34)\.=RPC0[127.0.0.1:50311];blk.(35|36|37|38|39|40|41|42|43|44|45|46|47)\.=CPU;token_embd.*=CPU;output.*=CPU'

cleanup() {
  [ -n "${S0:-}" ] && kill "$S0" 2>/dev/null
  [ -n "${P1PID:-}" ] && kill "$P1PID" 2>/dev/null
  sleep 2
}

echo "=== start RPC servers (GPU0:$P0 GPU1:$P1) ==="
CUDA_VISIBLE_DEVICES=0 "$MCB_BIN/ggml-rpc-server" -H 127.0.0.1 -p "$P0" > "$MCB_ROOT/logs/rpc-$TAG-$P0.log" 2>&1 &
S0=$!
CUDA_VISIBLE_DEVICES=1 "$MCB_BIN/ggml-rpc-server" -H 127.0.0.1 -p "$P1" > "$MCB_ROOT/logs/rpc-$TAG-$P1.log" 2>&1 &
P1PID=$!
for i in $(seq 1 30); do
  if grep -q "transport" "$MCB_ROOT/logs/rpc-$TAG-$P0.log" 2>/dev/null && grep -q "CUDA0" "$MCB_ROOT/logs/rpc-$TAG-$P0.log" 2>/dev/null \
  && grep -q "transport" "$MCB_ROOT/logs/rpc-$TAG-$P1.log" 2>/dev/null && grep -q "CUDA0" "$MCB_ROOT/logs/rpc-$TAG-$P1.log" 2>/dev/null; then
    echo "servers ready (pids $S0 $P1PID)"; break
  fi
  sleep 1
  if [ "$i" = "30" ]; then echo "SERVER START FAILED"; cleanup; exit 1; fi
done

echo "=== verify placement ($TAG) ==="
( export CUDA_VISIBLE_DEVICES=2
  "$MCB_BIN/llama-cli" -m "$MCB_MODEL" -p hi -n 1 -c 32 --temp 0 \
    --rpc 127.0.0.1:$P0,127.0.0.1:$P1 --device RPC0,RPC1 -sm layer -ot "$OT" --verbose \
    > /dev/null 2> "$MCB_ROOT/logs/place-$TAG.log" ) || { echo "VERIFY FAILED"; cleanup; exit 1; }
grep -E "model buffer size|KV buffer|compute buffer" "$MCB_ROOT/logs/place-$TAG.log" | head -14

echo "=== bench ($TAG) ==="
( export CUDA_VISIBLE_DEVICES=2
  export RPCMON_LOG="$MCB_ROOT/logs/rpcmon-$TAG-cli"
  export LD_PRELOAD="$RPCLIB"
  mcb_bench "$TAG" --rpc 127.0.0.1:$P0,127.0.0.1:$P1 --device RPC0/RPC1 -sm layer -ot "$OT_BENCH" )
rc=$?
cleanup
echo "DONE $TAG rc=$rc"
exit $rc
