#!/bin/bash
# Replicate one banked sde_results entry: ten seeds of its config under the paper protocol.
#
#   bash sde_results/replicate.sh expressivity/ews_efm            # all ten seeds
#   bash sde_results/replicate.sh oscillator/ews "0 1 2"          # some seeds
#
# Seed k trains on its own Brownian paths (data seed k) with model seed 100 + k. Missing data
# files are generated first (data/sde_d/sde_d_<target>_seed<k>.npz, data/sde_a/sde_a_seed<k>.npz).
#
# Env:  PY=<python>        interpreter (default: python)
#       GPUS=<list>        GPUs to spread the seeds over, e.g. "0,1" (default 0)
#       PAR=<n>            concurrent runs per GPU (default 4)
#       DATA_ROOT=<dir>    where the data files live (default <repo>/data; ~4 GB for all seeds)
#       OUT=<dir>          output root (default <repo>/outputs/sde_replicate)
#       BANK=1             write into the entry's runs/ directory instead (used to bank the numbers)
#
# Each seed writes seed<k>/results.json (evaluation history, best checkpoint, learned A and its
# eigenvalues) and seed<k>/train.log. A seed whose results.json exists is not rerun. At the end
# the per-seed numbers and the mean +/- std (population std over seeds) are printed and, unless
# banking, compared with the banked runs/.
set -u
ENTRY="${1:?usage: replicate.sh <experiment>/<name> [seeds]}"
SEEDS="${2:-0 1 2 3 4 5 6 7 8 9}"; SEEDS="${SEEDS//,/ }"
PY="${PY:-python}"; GPUS="${GPUS:-0}"; PAR="${PAR:-4}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
EDIR="$ROOT/sde_results/$ENTRY"; CFG="$EDIR/config.json"
[ -f "$CFG" ] || { echo "no config: $CFG"; exit 1; }
EXP="${ENTRY%%/*}"; NAME="${ENTRY##*/}"

if [ "$EXP" = expressivity ]; then
  case "${NAME%%_*}" in ews) T=EWS ;; efm) T=EFM ;; sig) T=Sig ;; *) echo "bad entry"; exit 1 ;; esac
  data_for() { echo "$DATA_ROOT/sde_d/sde_d_${T}_seed$1.npz"; }
  gen() { "$PY" "$ROOT/experiments/sde/sde_d/generate_sde_d.py" --a_true "$T" --seed "$1" --gpu "$2" --output "$(data_for "$1")"; }
else
  data_for() { echo "$DATA_ROOT/sde_a/sde_a_seed$1.npz"; }
  gen() { "$PY" "$ROOT/experiments/sde/sde_a/generate_sde_a.py" --seed "$1" --output "$(data_for "$1")"; }
fi

if [ "${BANK:-0}" = 1 ]; then OUTDIR="$EDIR/runs"; else OUTDIR="${OUT:-$ROOT/outputs/sde_replicate}/$EXP/$NAME"; fi
mkdir -p "$OUTDIR"
IFS=',' read -ra G <<< "$GPUS"; NG=${#G[@]}
echo "== $ENTRY ==  seeds=[$SEEDS]  gpus=$GPUS  par=$PAR  out=$OUTDIR"

i=0
for s in $SEEDS; do
  f="$(data_for "$s")"
  if [ ! -f "$f" ]; then echo "generating $f"; gen "$s" "${G[$((i % NG))]}" > "$OUTDIR/gen_seed$s.log" 2>&1 || { echo "  generation failed, see $OUTDIR/gen_seed$s.log"; exit 1; }; fi
  i=$((i + 1))
done

run_one() {
  local s=$1 g=$2 d="$OUTDIR/seed$1"
  if [ -f "$d/results.json" ]; then echo "seed $s: already done"; return; fi
  mkdir -p "$d"
  PYTHONUNBUFFERED=1 "$PY" "$ROOT/experiments/sde/train.py" --config "$CFG" --dataset_file "$(data_for "$s")" \
      --seed $((100 + s)) --out_dir "$d" --gpu "$g" --autotune on --deterministic-ops off > "$d/train.log" 2>&1
  echo "seed $s (gpu $g): $(grep '^Best' "$d/train.log" || echo FAILED)"
}
export -f run_one data_for; export PY ROOT CFG OUTDIR DATA_ROOT T
i=0; jobs=""
for s in $SEEDS; do jobs+="$s ${G[$((i % NG))]}"$'\n'; i=$((i + 1)); done
printf "%s" "$jobs" | xargs -P $((PAR * NG)) -L 1 bash -c 'run_one "$@"' _

"$PY" - "$EDIR" "$OUTDIR" "$SEEDS" "${BANK:-0}" <<'PYEOF'
import json, os, sys
import numpy as np
edir, out, seeds, bank = sys.argv[1], sys.argv[2], sys.argv[3].split(), sys.argv[4] == "1"
def load(root):
    vals = {}
    for s in seeds:
        p = f"{root}/seed{s}/results.json"
        if os.path.exists(p):
            b = json.load(open(p))["best"]; vals[s] = (b["val_rmse"], b["test_rmse"])
    return vals
got = load(out)
print(f"\n{'seed':>6} {'val':>12} {'test':>12}")
for s, (v, t) in got.items():
    print(f"{s:>6} {v:12.3e} {t:12.3e}")
if got:
    v = np.array([x[0] for x in got.values()]); t = np.array([x[1] for x in got.values()])
    print(f"val  {v.mean():.3e} +/- {v.std():.3e}   test {t.mean():.3e} +/- {t.std():.3e}   ({len(got)} seeds)")
if not bank:
    ref = load(edir + "/runs")
    common = [s for s in got if s in ref]
    if common:
        rt = np.array([ref[s][1] for s in common]); gt = np.array([got[s][1] for s in common])
        print(f"banked test {rt.mean():.3e} +/- {rt.std():.3e} on the same seeds; max |diff| per seed {np.abs(rt - gt).max():.2e}")
PYEOF
