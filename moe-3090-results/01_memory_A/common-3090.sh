#!/usr/bin/env bash
# Common helpers for memory-constrained baseline. Source, don't execute.
MCB_ROOT="$HOME/memory_constrained_baseline"
MCB_BIN="$HOME/git/llama.cpp/build-cuda-rpc/bin"
MCB_MODEL="$HOME/models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:${LD_LIBRARY_PATH:-}

# ot_args <dev0:layers0-13> <dev1:layers14-34> <devhost:layers35-47+fixed>
mcb_ot() {
  local d0=$1 d1=$2 dh=$3
  local r0="blk.(0|1|2|3|4|5|6|7|8|9|10|11|12|13)"
  local r1="blk.(14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34)"
  local r2="blk.(35|36|37|38|39|40|41|42|43|44|45|46|47)"
  echo "${r0}=${d0},${r1}=${d1},${r2}=${dh},token_embd.*=${dh},output.*=${dh}"
}

# vram sampler: mcb_vram <tag> — sets MCB_SAMPLER_VRAM in caller.
# NOTE (3090/bash 5.0): do NOT call inside $(...); bg-inside-command-subst wedges
# the caller. Call directly, then read $MCB_SAMPLER_VRAM.
mcb_vram() {
  local tag=$1
  ( while true; do nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits; echo "--- $(date +%s)"; sleep 2; done > "$MCB_ROOT/logs/vram-$tag.log" 2>&1 ) &
  MCB_SAMPLER_VRAM=$!
}

# peak RSS (MiB) of a finished /usr/bin/time -v stderr log
mcb_rss() { grep -i "maximum resident" "$1" | grep -oE "[0-9]+" | head -1 | awk '{print $1/1024}'; }

# host sampler: mcb_host <tag> <pid> — sets MCB_SAMPLER_HOST in caller (same no-$(...) rule).
mcb_host() {
  local tag=$1 pid=$2
  ( while kill -0 "$pid" 2>/dev/null; do
      awk '/^VmHWM:/{h=$2} /^VmRSS:/{r=$2} /^RssAnon:/{a=$2} /^RssFile:/{f=$2} END{printf "%d %d %d %d %d\n", systime(), h+0, r+0, a+0, f+0}' /proc/$pid/status 2>/dev/null || break
      sleep 2
    done > "$MCB_ROOT/logs/host-$tag.log" 2>&1 ) &
  MCB_SAMPLER_HOST=$!
}

# run bench: mcb_bench <tag> <extra...>; csv + stderr + vram + host-rss (VmHWM peak)
# NOTE: call as: mcb_bench tag args... (runs foreground, samples in background)
mcb_bench() {
  local tag=$1; shift
  mcb_vram "$tag"; local S=$MCB_SAMPLER_VRAM
  local T0; T0=$(date +%s)
  "$MCB_BIN/llama-bench" -m "$MCB_MODEL" -p 0 -n 0 -pg 512,0 -pg 0,128 -pg 512,128 \
    -b 2048 -ub 512 -r 5 -ngl 99 -o csv "$@" > "$MCB_ROOT/logs/$tag.csv" 2> "$MCB_ROOT/logs/$tag.stderr.log" &
  local BPID=$!
  mcb_host "$tag" "$BPID"; local H=$MCB_SAMPLER_HOST
  wait "$BPID"; local rc=$?
  local T1; T1=$(date +%s)
  kill "$H" "$S" 2>/dev/null
  echo "$T0 $T1 $rc" > "$MCB_ROOT/logs/$tag.wall"
  echo "$tag exit: $rc wall=$((T1-T0))s"
  return $rc
}
