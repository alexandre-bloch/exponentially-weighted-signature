#!/bin/bash
# Replicate the banked PPG result (PPG-DaLiA 49,920 -> 390, LinOSS / D-LinOSS training protocol).
#
#   Usage:  bash ppg_results/replicate.sh [seeds] [-- extra train.py args]
#     [seeds]  default "2345 3456 4567 5678 6789" (the protocol's five seeds)
#   Env:  GPU=<n>    GPU to use (default 0)
#         PY=<path>  python interpreter (default: python)
#         PAR=<n>    runs in parallel on the GPU (default 1; each run needs ~10 GB)
#
# Trains ppg_results/config.json once per seed with experiments/ppg/train.py (XLA autotuning off, the regime the
# bank was produced in), then prints each seed's test / validation MSE beside the banked run in ppg_results/runs/
# and the five-seed mean +- std. A seed whose log already ends with FINAL is reused.
set +e
SEEDS="2345 3456 4567 5678 6789"
if [ $# -gt 0 ] && [ "$1" != "--" ]; then SEEDS="${1//,/ }"; shift; fi
[ "$1" = "--" ] && shift
EXTRA=("$@")
GPU="${GPU:-0}"; PY="${PY:-python}"; PAR="${PAR:-1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; DIR="$ROOT/ppg_results"
OUT="$DIR/replicate_runs"; mkdir -p "$OUT"; cd "$ROOT" || exit 1
run_one() { s=$1; log="$OUT/seed_$s.log"
  if grep -q "FINAL" "$log" 2>/dev/null; then echo "seed $s: reusing $log"; return; fi
  echo "seed $s: training -> $log"
  "$PY" experiments/ppg/train.py --config "$DIR/config.json" --seed "$s" --gpu "$GPU" --out "$OUT" "${EXTRA[@]}" > "$log" 2>&1; }
n=0; for s in $SEEDS; do run_one "$s" & n=$((n+1)); if [ "$n" -ge "$PAR" ]; then wait -n; n=$((n-1)); fi; done; wait
"$PY" - "$DIR" "$OUT" $SEEDS << 'PYEOF'
import json, os, statistics as st, sys
d, out, seeds = sys.argv[1], sys.argv[2], sys.argv[3:]
got = []
print(f"{'seed':>6} | {'test x1e-2':>10} | {'val x1e-2':>9} | {'banked test':>11} | {'banked val':>10} | {'best step':>9}")
for s in seeds:
    p, q = os.path.join(out, f"seed_{s}", "best.json"), os.path.join(d, "runs", f"seed_{s}", "best.json")
    b = json.load(open(p)) if os.path.exists(p) else None
    k = json.load(open(q)) if os.path.exists(q) else None
    f = lambda x, key: f"{100 * x[key]:.2f}" if x and key in x else "-"
    print(f"{s:>6} | {f(b, 'test_mse'):>10} | {f(b, 'val_mse'):>9} | {f(k, 'test_mse'):>11} | {f(k, 'val_mse'):>10} | {(b or {}).get('step', '-'):>9}")
    if b and "test_mse" in b:
        got.append(100 * b["test_mse"])
if len(got) > 1:
    print(f"test MSE x1e-2 over {len(got)} seeds: {st.mean(got):.2f} +- {st.pstdev(got):.2f} (banked: 5.90 +- 0.73)")
PYEOF
