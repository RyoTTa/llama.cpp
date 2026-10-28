#!/bin/bash
# collect_routing_traces.sh - MoE routing trace collection (native CUDA0, greedy)
set -u
ROOT="$HOME/moe_routing_analysis"
BIN="$HOME/git/llama.cpp/build-moe-trace/bin"
MODEL="$HOME/models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:${LD_LIBRARY_PATH:-}
mkdir -p "$ROOT/traces-3090"

run_one() { # run_one <tag> <promptfile> <ngen> <tracefile|-> 
  local tag=$1 pf=$2 ngen=$3 trace=$4
  if [ "$trace" != "-" ]; then export LLAMA_MOE_TRACE="$trace"; else unset LLAMA_MOE_TRACE; fi
  /usr/bin/time -f "wall: %es" "$BIN/llama-cli" -m "$MODEL" \
    -p "$(cat "$ROOT/prompts/$pf")" -n "$ngen" -c 4096 --temp 0 \
    --device CUDA0 --threads 16 < /dev/null > "$ROOT/traces-3090/$tag.stdout.log" 2> "$ROOT/traces-3090/$tag.stderr.log"
  echo "$tag exit: $?"
  unset LLAMA_MOE_TRACE
}

# train set (learn static placement ONLY from these)
run_one train_p1 p1_code.txt   128 "$ROOT/traces-3090/train_p1.csv"
run_one train_p2 p2_chat.txt   128 "$ROOT/traces-3090/train_p2.csv"
run_one train_p3 p3_reason.txt 256 "$ROOT/traces-3090/train_p3.csv"
# eval set (held out: unseen prompt; no leakage into static policy)
run_one eval_p4 p4_code2.txt   128 "$ROOT/traces-3090/eval_p4.csv"

# overhead: same workload with/without trace (p2 short, x3 each)
for i in 1 2 3; do run_one ov_off_$i p2_chat.txt 64 -; done
for i in 1 2 3; do run_one ov_on_$i  p2_chat.txt 64 "$ROOT/traces-3090/ov_$i.csv"; done
rm -f "$ROOT/traces-3090"/ov_*.csv
echo "collection done"
