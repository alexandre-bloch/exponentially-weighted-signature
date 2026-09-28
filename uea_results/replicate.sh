#!/bin/bash
# Retrain a selected configuration once per seed on its own split and compare with the bank.
#
#   bash uea_results/replicate.sh <dataset> [seeds]      dataset: eigenworms ethanol heartbeat motor scp1 scp2
#   GPU=<n> (default 0), PY=<python> (default: python on PATH); seeds default to "0 1 2 3 4".
#
# Seed S trains on data/<stem>/splits/<stem>_seed{S}.npz, with the stem and the XLA regime taken from grid.json, and
# reports the test accuracy at its best-validation checkpoint next to grid.json's per_seed_test.
set +e
DS="$1"
SEEDS="${2:-0 1 2 3 4}"; SEEDS="${SEEDS//,/ }"
GPU="${GPU:-0}"
PY="${PY:-python}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DDIR="$ROOT/uea_results/$DS"
CFG="$DDIR/config.json"
GRID="$DDIR/grid.json"
[ -f "$CFG" ]  || { echo "ERROR: no config: $CFG"; exit 1; }
[ -f "$GRID" ] || { echo "ERROR: no grid.json: $GRID"; exit 1; }

STEM="$($PY -c "import json;print(json.load(open('$GRID'))['dataset_file_stem'])")"
# regime per entry (grid.json["regime"]); entries without it were produced with autotuning on, deterministic ops off
AUTOTUNE="$($PY -c "import json;print(json.load(open('$GRID')).get('regime',{}).get('autotune','on'))")"
DETOPS="$($PY -c "import json;print(json.load(open('$GRID')).get('regime',{}).get('deterministic_ops','off'))")"
[ -n "$STEM" ] || { echo "ERROR: no dataset_file_stem in $GRID"; exit 1; }

OUT="$ROOT/outputs/replicate_$DS"
rm -rf "$OUT"; mkdir -p "$OUT"
echo "== replicate $DS ==  stem=$STEM  seeds=[$SEEDS]  gpu=$GPU  regime: autotune $AUTOTUNE, deterministic ops $DETOPS"

for S in $SEEDS; do
  SPLIT="$ROOT/data/$STEM/splits/${STEM}_seed${S}.npz"
  if [ ! -f "$SPLIT" ]; then echo "  MISSING split: $SPLIT"; continue; fi
  "$PY" "$ROOT/experiments/uea/train.py" \
      --config "$CFG" --gpu "$GPU" --seed "$S" --autotune "$AUTOTUNE" --deterministic-ops "$DETOPS" \
      --dataset_file "$SPLIT" --out_ckpt "$OUT/s${S}.eqx" \
      > "$OUT/s${S}.log" 2>&1 &
done
wait

"$PY" - "$DDIR" "$OUT" "$SEEDS" <<'PYEOF'
import sys, re, json, os, statistics
ddir, out, seeds = sys.argv[1], sys.argv[2], sys.argv[3].split()
bank = (json.load(open(ddir + "/grid.json")).get("result") or {}).get("per_seed_test") or []
banked_mean = (json.load(open(ddir + "/grid.json")).get("result") or {}).get("method_B_5seed_test")
runs, allok = [], True
for s in seeds:
    lp = f"{out}/s{s}.log"
    x = open(lp, errors="ignore").read() if os.path.exists(lp) else ""
    m  = re.search(r"Test metrics: {'accuracy': ([\d.]+)}", x)
    mb = re.search(r"Best  step=(\d+)  val_accuracy=([\d.]+)", x)
    t  = round(float(m.group(1)) * 100, 2) if m else None
    b  = bank[int(s)] if int(s) < len(bank) else None
    if t is not None: runs.append(t)
    if b is None:
        print(f"  seed {s}: test {t}  (no banked ref)")
    else:
        ok = t is not None and abs(t - b) < 0.15
        allok = allok and ok
        step = mb.group(1) if mb else "?"
        print(f"  seed {s}: best-val@{step:>6}  test {t}   banked {b}   -> {'MATCH' if ok else 'MISMATCH'}")
if len(runs) == 5 and banked_mean is not None:
    mean = round(statistics.mean(runs), 2)
    ok = abs(mean - banked_mean) < 0.1
    allok = allok and ok
    print(f"  MEAN test {mean}   banked {banked_mean}   -> {'MATCH' if ok else 'MISMATCH'}")
print("  RESULT:", "OK — replicates" if allok else "CHECK — see mismatches above")
PYEOF
