#!/bin/bash
# moe_rpc_benchmark/run_benchmark.sh - Native CUDA vs Localhost RPC baseline
# Usage: ./run_benchmark.sh [--force] [--exp ABCD]  (default: all missing)
set -u
ROOT="$HOME/moe_rpc_benchmark"
LOGS="$ROOT/logs-3090"
CSV="$ROOT/results-3090.csv"
LLAMA="$HOME/git/llama.cpp"
BIN="$LLAMA/build-cuda-rpc/bin"
MODEL="$HOME/models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:${LD_LIBRARY_PATH:-}
export HF_HUB_ENABLE_HF_TRANSFER=0

FORCE=0; EXPS="A B C D"
for a in "$@"; do
  case "$a" in
    --force) FORCE=1;;
    --exp=*) EXPS=$(echo "${a#--exp=}" | tr ',' ' ');;
  esac
done

mkdir -p "$LOGS"
[ -f "$CSV" ] || echo "exp,config,gpus,prompt_n,gen_n,batch,ubatch,ngl,split,device_arg,prefill_mean,prefill_std,decode_mean,decode_std,combined_mean,combined_std,load_s,vram_gpu0_MiB,util_gpu0_pct,vram_gpu1_MiB,util_gpu1_pct,status" > "$CSV"
has_result() { [ "$FORCE" -eq 1 ] && return 1; grep -q "^$1," "$CSV" 2>/dev/null; }

check_env() {
  echo "=== GPU status ==="
  nvidia-smi 2>&1 | head -25
  if nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '$1>500{found=1} END{exit !found}'; then
    echo "NOTE: a GPU shows >500 MiB used; results may be noisy."
  fi
  [ -x "$BIN/llama-bench" ] || { echo "FATAL: llama-bench missing"; exit 1; }
  [ -x "$BIN/ggml-rpc-server" ] || { echo "FATAL: ggml-rpc-server missing"; exit 1; }
  [ -f "$MODEL" ] || { echo "FATAL: model missing: $MODEL"; exit 1; }
}

RPC_PIDS=""
start_rpc() { # start_rpc <gpu_id> <port> <log>
  local gpu=$1 port=$2 log=$3
  CUDA_VISIBLE_DEVICES=$gpu "$BIN/ggml-rpc-server" -H 127.0.0.1 -p "$port" > "$log" 2>&1 &
  local pid=$!
  RPC_PIDS="$RPC_PIDS $pid"
  for i in $(seq 1 30); do
    if grep -q "transport" "$log" 2>/dev/null && grep -q "CUDA0" "$log" 2>/dev/null; then echo "$pid"; return 0; fi
    sleep 1
  done
  echo "FATAL: RPC server gpu=$gpu port=$port failed to start; log=$log" >&2
  tail -20 "$log" >&2
  return 1
}
stop_rpc() {
  for p in $RPC_PIDS; do kill "$p" 2>/dev/null; done
  sleep 2
  pkill -f "ggml-rpc-server -H 127.0.0.1 -p 5005" 2>/dev/null || true
  RPC_PIDS=""
}
trap stop_rpc EXIT
gpu_stat() { nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; }

# run_bench <exp> <extra args...>; runs pp512, tg128, pp512+tg128 in one invocation
# also samples GPU VRAM/util during the run -> logs/<exp>.vram.log + <exp>.vrampeak
run_bench() {
  local exp=$1; shift
  local log="$LOGS/${exp}.log"
  echo "--- [$exp] llama-bench $* ---" | tee "$log"
  ( while true; do nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits; sleep 1; done > "$LOGS/${exp}.vram.log" 2>/dev/null ) &
  local sampler=$!
  "$BIN/llama-bench" -m "$MODEL" -p 0 -n 0 -pg 512,0 -pg 0,128 -pg 512,128 \
    -b 2048 -ub 512 -r 5 \
    -ngl 99 -o csv "$@" 2> "$LOGS/${exp}.stderr.log" | tee -a "$log" | tail -30
  local rc=${PIPESTATUS[0]}
  kill "$sampler" 2>/dev/null
  python3 - "$LOGS/${exp}.vram.log" "$LOGS/${exp}.vrampeak" <<'PYEOF'
import sys
peak = {}
try:
    with open(sys.argv[1]) as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) != 3: continue
            try: i, m, u = int(p[0]), int(p[1].strip()), int(p[2].strip())
            except ValueError: continue
            if i not in peak or m > peak[i][0]: peak[i] = (m, u)
except FileNotFoundError:
    pass
with open(sys.argv[2], 'w') as f:
    for i in (0, 1):
        m, u = peak.get(i, (0, 0))
        f.write(f"{i} {m} {u}\n")
PYEOF
  return $rc
}

# quick VRAM/util probe for an already-benched exp (same model+device, short run)
probe_vram() { # probe_vram <exp> <extra args...>
  local exp=$1; shift
  ( while true; do nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits; sleep 1; done > "$LOGS/${exp}.vram.log" 2>/dev/null ) &
  local sampler=$!
  "$BIN/llama-bench" -m "$MODEL" -p 0 -n 0 -pg 0,32 -b 2048 -ub 512 -r 1 \
    -ngl 99 -o csv "$@" > "$LOGS/${exp}.vramprobe.log" 2>&1
  local rc=$?
  kill "$sampler" 2>/dev/null
  python3 - "$LOGS/${exp}.vram.log" "$LOGS/${exp}.vrampeak" <<'PYEOF'
import sys
peak = {}
try:
    with open(sys.argv[1]) as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) != 3: continue
            try: i, m, u = int(p[0]), int(p[1].strip()), int(p[2].strip())
            except ValueError: continue
            if i not in peak or m > peak[i][0]: peak[i] = (m, u)
except FileNotFoundError:
    pass
with open(sys.argv[2], 'w') as f:
    for i in (0, 1):
        m, u = peak.get(i, (0, 0))
        f.write(f"{i} {m} {u}\n")
PYEOF
  return $rc
}

# summarize <exp.log> -> "pp_mean pp_std tg_mean tg_std comb_mean comb_std"
# csv columns: n_prompt,n_gen,avg_ts (combined t/s),stddev_ts
summarize() {
  python3 - "$1" <<'PYEOF'
import sys, csv, statistics as s
def rows_for(path, np, ng):
    with open(path) as f:
        txt = f.read()
    lines = txt.splitlines()
    try:
        hi = next(i for i,l in enumerate(lines) if 'n_prompt' in l and 'avg_ts' in l)
    except StopIteration:
        return []
    rdr = list(csv.reader(lines[hi:]))
    hdr = [h.strip() for h in rdr[0]]
    try:
        ip, ig, it, its = hdr.index('n_prompt'), hdr.index('n_gen'), hdr.index('avg_ts'), hdr.index('stddev_ts')
    except ValueError:
        return []
    out = []
    for p in rdr[1:]:
        if len(p) != len(hdr): continue
        try:
            if int(float(p[ip]))==np and int(float(p[ig]))==ng:
                out.append((float(p[it]), float(p[its])))
        except ValueError:
            continue
    return out
try:
    out = []
    for np, ng in [(512,0),(0,128),(512,128)]:
        rows = rows_for(sys.argv[1], np, ng)
        if not rows: out += ["NA","NA"]
        elif len(rows) == 1: out += [f"{rows[0][0]:.2f}", f"{rows[0][1]:.2f}"]
        else:
            m = s.mean(r[0] for r in rows); sd = s.stdev(r[0] for r in rows)
            out += [f"{m:.2f}", f"{sd:.2f}"]
    print(" ".join(out))
except Exception:
    print("NA NA NA NA NA NA")
PYEOF
}

record() { # record <exp> <config> <gpus> <devarg> <split> <pp_m pp_s tg_m tg_s comb_m comb_s> <load> <status>
  local exp=$1 config=$2 gpus=$3 devarg=$4 split=$5; shift 5
  local pp_m=$1 pp_s=$2 tg_m=$3 tg_s=$4 comb_m=$5 comb_s=$6 load=$7 status=$8
  local stat0="0 0 0" stat1="1 0 0"
  [ -f "$LOGS/${exp}.vrampeak" ] && stat0=$(sed -n '1p' "$LOGS/${exp}.vrampeak") && stat1=$(sed -n '2p' "$LOGS/${exp}.vrampeak")
  local g0v=$(echo "$stat0" | awk '{print $2}'); local g0u=$(echo "$stat0" | awk '{print $3}');
  local g1v=$(echo "$stat1" | awk '{print $2}'); local g1u=$(echo "$stat1" | awk '{print $3}');
  echo "$exp,$config,\"$gpus\",512,128,2048,512,99,$split,$devarg,$pp_m,$pp_s,$tg_m,$tg_s,$comb_m,$comb_s,$load,$g0v,$g0u,$g1v,$g1u,$status" >> "$CSV"
}

exp_A() {
  has_result A && { echo "A already done, skip"; return 0; }
  echo "===== Experiment A: Native CUDA single GPU0 ====="
  local t0=$(date +%s)
  run_bench A --device CUDA0 -sm none || return 1
  local t1=$(date +%s)
  grep -q "CUDA0" "$LOGS/A.log" || { echo "A FAIL: CUDA0 not in log"; return 1; }
  if grep -q "RPC" "$LOGS/A.log"; then echo "A FAIL: unexpected RPC"; return 1; fi
  local st=$(summarize "$LOGS/A.log")
  record A "native-cuda" "0" "CUDA0" "none" $st $((t1-t0)) "ok"
}
exp_B() {
  has_result B && { echo "B already done, skip"; return 0; }
  echo "===== Experiment B: Localhost RPC single GPU0 ====="
  start_rpc 0 50052 "$LOGS/rpc-50052.log" || return 1
  grep -q "found 1 CUDA devices" "$LOGS/rpc-50052.log" || { echo "B FAIL: server does not expose exactly 1 GPU"; return 1; }
  "$BIN/llama-cli" --rpc 127.0.0.1:50052 --list-devices > "$LOGS/B.devices.log" 2>&1
  grep -q "RPC0" "$LOGS/B.devices.log" || { echo "B FAIL: RPC0 not visible"; cat "$LOGS/B.devices.log"; return 1; }
  local t0=$(date +%s)
  run_bench B --rpc 127.0.0.1:50052 --device RPC0 -sm none || return 1
  local t1=$(date +%s)
  grep -q "RPC0" "$LOGS/B.log" || { echo "B FAIL: RPC0 not in bench log"; return 1; }
  local st=$(summarize "$LOGS/B.log")
  record B "rpc-localhost" "0" "RPC0" "none" $st $((t1-t0)) "ok"
  stop_rpc
}
exp_C() {
  has_result C && { echo "C already done, skip"; return 0; }
  echo "===== Experiment C: Native CUDA GPU0+GPU1 layer split ====="
  local t0=$(date +%s)
  run_bench C --device CUDA0/CUDA1 -sm layer || return 1
  local t1=$(date +%s)
  local st=$(summarize "$LOGS/C.log")
  record C "native-cuda" "0,1" "CUDA0/CUDA1" "layer" $st $((t1-t0)) "ok"
}
exp_D() {
  has_result D && { echo "D already done, skip"; return 0; }
  echo "===== Experiment D: Localhost RPC GPU0+GPU1 layer split ====="
  start_rpc 0 50052 "$LOGS/rpc-50052.log" || return 1
  start_rpc 1 50053 "$LOGS/rpc-50053.log" || return 1
  "$BIN/llama-cli" --rpc 127.0.0.1:50052,127.0.0.1:50053 --list-devices > "$LOGS/D.devices.log" 2>&1
  grep -q "RPC0" "$LOGS/D.devices.log" && grep -q "RPC1" "$LOGS/D.devices.log" || { echo "D FAIL: RPC0/RPC1 missing"; cat "$LOGS/D.devices.log"; return 1; }
  local t0=$(date +%s)
  run_bench D --rpc 127.0.0.1:50052,127.0.0.1:50053 --device RPC0/RPC1 -sm layer || return 1
  local t1=$(date +%s)
  local st=$(summarize "$LOGS/D.log")
  record D "rpc-localhost" "0,1" "RPC0/RPC1" "layer" $st $((t1-t0)) "ok"
  stop_rpc
}

check_env
FAIL=0
for e in $EXPS; do
  case $e in
    A) exp_A || FAIL=1;;
    B) exp_B || FAIL=1;;
    C) exp_C || FAIL=1;;
    D) exp_D || FAIL=1;;
  esac
  if [ "$FAIL" -ne 0 ]; then
    if [ "$e" = "A" ] || [ "$e" = "B" ]; then
      echo "FATAL: experiment $e failed; stopping before C/D" | tee -a "$LOGS/FAILED_$e.log"
      stop_rpc; exit 1
    else
      echo "WARN: experiment $e failed; continuing" | tee -a "$LOGS/FAILED_$e.log"
      FAIL=0
    fi
  fi
  stop_rpc
done
echo "done. results: $CSV"
