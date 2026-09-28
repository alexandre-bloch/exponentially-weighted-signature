#!/bin/bash
# Replicate the banked weather result (Informer weather 720 -> 720, D-LinOSS training protocol: L1 loss, MAE).
#
#   Usage:  bash weather_results/replicate.sh [seeds] [-- extra train.py args]
#     [seeds]  default "2345 3456 4567 5678 6789" (the protocol's five seeds)
#   Env:  GPU=<n>    GPU to use (default 0)
#         PY=<path>  python interpreter (default: python)
#         PAR=<n>    runs in parallel on the GPU (default 1)
#
#   Examples:
#     bash weather_results/replicate.sh                                   # the selected model, five seeds
#     bash weather_results/replicate.sh 2345 -- --set model.m=32 --set model.hidden=128    # another grid cell
#
# Trains weather_results/config.json once per seed with experiments/weather/train.py (XLA autotuning off, the
# regime the bank was produced in), then prints each seed's test MAE / validation MAE beside the banked run in
# weather_results/runs/ and the five-seed mean +- population std. A seed whose log already ends with FINAL is reused.
set +e
SEEDS="2345 3456 4567 5678 6789"
if [ $# -gt 0 ] && [ "$1" != "--" ]; then SEEDS="${1//,/ }"; shift; fi
[ "$1" = "--" ] && shift
EXTRA=("$@")
GPU="${GPU:-0}"; PY="${PY:-python}"; PAR="${PAR:-1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; DIR="$ROOT/weather_results"
OUT="$DIR/replicate_runs"; mkdir -p "$OUT"; cd "$ROOT" || exit 1
run_one() { s=$1; log="$OUT/seed_$s.log"
  if grep -q "FINAL" "$log" 2>/dev/null; then echo "seed $s: reusing $log"; return; fi
  echo "seed $s: training -> $log"
  "$PY" experiments/weather/train.py --config "$DIR/config.json" --seed "$s" --gpu "$GPU" --out "$OUT" "${EXTRA[@]}" > "$log" 2>&1; }
n=0; for s in $SEEDS; do run_one "$s" & n=$((n+1)); if [ "$n" -ge "$PAR" ]; then wait -n; n=$((n-1)); fi; done; wait
"$PY" - "$DIR" "$OUT" $SEEDS << 'PYEOF'
import json, os, statistics as st, sys
d, out, seeds = sys.argv[1], sys.argv[2], sys.argv[3:]
got, ref = [], []
print(f"{'seed':>6} | {'test MAE':>9} | {'val MAE':>9} | {'banked test MAE':>15} | {'banked val MAE':>14}")
for s in seeds:
    p, q = os.path.join(out, f"seed_{s}", "best.json"), os.path.join(d, "runs", f"seed_{s}", "best.json")
    b = json.load(open(p)) if os.path.exists(p) else None
    k = json.load(open(q)) if os.path.exists(q) else None
    f = lambda x, key: f"{x[key]:.4f}" if x and key in x else "-"
    print(f"{s:>6} | {f(b, 'test_mae'):>9} | {f(b, 'val_mae'):>9} | {f(k, 'test_mae'):>15} | {f(k, 'val_mae'):>14}")
    if b and "test_mae" in b:
        got.append(b["test_mae"])
    if k and "test_mae" in k:
        ref.append(k["test_mae"])
if len(got) > 1:
    print(f"test MAE over {len(got)} seeds: {st.mean(got):.4f} +- {st.pstdev(got):.4f}"
          + (f" (banked: {st.mean(ref):.4f} +- {st.pstdev(ref):.4f})" if len(ref) == len(got) else ""))
PYEOF
