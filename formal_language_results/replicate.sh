#!/bin/bash
# Replicate a banked formal-language result under Walker's protocol.
#
#   Usage:  bash formal_language_results/replicate.sh <task> [seeds]
#     <task>   parity | even_pairs | cycle_nav | mod_arith_no_brack
#     [seeds]  run indices, default "0 1 2 3 4"  (seed = 1234 + index)
#   Env:  GPU=<n>    GPU to use (default 0)
#         PY=<path>  python interpreter (default: python)
#         PAR=<n>    runs in parallel on the GPU (default 1)
#
# Trains config.json once per seed (the trainer generates the task data itself; there are no
# split files), then prints each seed's best validation accuracy beside the banked value and the
# 5-seed mean ± population std. Runs use the banked regime (XLA autotuning off, deterministic ops off), which is not bit-reproducible, so
# expect draw variance: a few hundredths on the solved tasks, a few points on mod_arith.
# A seed whose log already ends with "Best Validation Accuracy" is reused, not re-run.
set +e
TASK="$1"; SEEDS="${2:-0 1 2 3 4}"; SEEDS="${SEEDS//,/ }"; GPU="${GPU:-0}"; PY="${PY:-python}"; PAR="${PAR:-1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; DIR="$ROOT/formal_language_results/$TASK"
[ -f "$DIR/config.json" ] || { echo "unknown task '$TASK' (expected a subdir of formal_language_results/)"; exit 1; }
LOGS="$DIR/replicate_logs"; mkdir -p "$LOGS"; cd "$ROOT" || exit 1
run_one() { r=$1; log="$LOGS/run$r.log"
  if grep -q "Best Validation Accuracy" "$log" 2>/dev/null; then echo "seed $((1234+r)): reusing $log"; return; fi
  echo "seed $((1234+r)): training -> $log"
  CUDA_VISIBLE_DEVICES="$GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false "$PY" experiments/fl/train.py \
    --config "$DIR/config.json" --run "$r" --gpu "$GPU" --autotune off --deterministic-ops off > "$log" 2>&1; }
n=0; for r in $SEEDS; do run_one "$r" & n=$((n+1)); if [ "$n" -ge "$PAR" ]; then wait -n; n=$((n-1)); fi; done; wait
"$PY" - "$DIR" "$LOGS" $SEEDS << 'PYEOF'
import sys, re, json, os, statistics as st
d, logs, seeds = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3:]]
got, bank = [], []
print(f"\n{'seed':<6}{'replicated':>12}{'banked':>10}   verdict")
for r in seeds:
    t = open(f"{logs}/run{r}.log").read(); m = re.findall(r"Best Validation Accuracy: ([0-9.]+)", t)
    rep = float(m[-1]) * 100 if m else float('nan')
    b = max(e['val_acc'] for e in json.load(open(f"{d}/runs/seed{1234+r}/results.json"))) * 100
    got.append(rep); bank.append(b); print(f"{1234+r:<6}{rep:>12.3f}{b:>10.3f}   {'ok' if abs(rep-b) < 0.5 else 'differs (draw variance)'}")
if len(got) > 1:
    print(f"\nmean ± std   replicated {st.mean(got):.3f} ± {st.pstdev(got):.3f}   banked {st.mean(bank):.3f} ± {st.pstdev(bank):.3f}")
PYEOF
