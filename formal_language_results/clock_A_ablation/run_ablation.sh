#!/bin/bash
# FL clock / A-structure ablation runner (the banked protocol): every cell to 100,000 steps, no futility stop, runs 0,1,2
# (seeds 1234-1236), best validation accuracy over the run. Claim-based (claims/<cfg>_100k_r<r>), so several machines or
# cards can share the cells. Configs in configs/: <task>_uniform (champion A, uniform clocks), <task>_efm (efm diagonal A,
# champion clocks), <task>_efm_uniform (efm A, uniform clocks); the champion/champion cell is the banked task result.
#   usage: GPUS="0 1" MAX_PER_GPU=3 [CFGS="parity_uniform ..."] [PY=python] bash formal_language_results/clock_A_ablation/run_ablation.sh
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-python}; A=formal_language_results/clock_A_ablation; mkdir -p $A/runs $A/claims
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
GPUS=(${GPUS:-0}); MAX_PER_GPU=${MAX_PER_GPU:-3}
TASKS="parity even_pairs cycle_nav mod_arith_no_brack"
CFGS="${CFGS:-$(for t in $TASKS; do echo -n "${t}_uniform ${t}_efm ${t}_efm_uniform "; done)}"
L=$A/runs/lane_$(hostname -s).log
say() { echo "[$(date '+%m-%d %H:%M') $(hostname -s)] $*" | tee -a $L; }
say "ablation up: configs [$CFGS], 100k steps, no futility; GPUs ${GPUS[*]}, $MAX_PER_GPU per card"
declare -A load; for g in "${GPUS[@]}"; do load[$g]=0; done
pick_gpu() { local best="" n=999; for g in "${GPUS[@]}"; do [ "${load[$g]}" -lt "$n" ] && { best=$g; n=${load[$g]}; }; done; echo $best; }
running=0
for cfg in $CFGS; do
  [ -f $A/configs/$cfg.json ] || { say "no config $cfg.json; skipped"; continue; }
  for r in 0 1 2; do
    log=$A/runs/${cfg}_100k_r$r.log
    if grep -q "Best Validation Accuracy" $log 2>/dev/null; then say "skip $cfg r$r (done)"; continue; fi
    while [ $running -ge $(( ${#GPUS[@]} * MAX_PER_GPU )) ]; do wait -n; running=$((running - 1)); done
    mkdir $A/claims/${cfg}_100k_r$r 2>/dev/null || { say "skip $cfg r$r (claimed elsewhere)"; continue; }
    gpu=$(pick_gpu); load[$gpu]=$(( ${load[$gpu]} + 1 ))
    say "RUN $cfg run $r on GPU $gpu"
    ( CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.22 \
      $PY experiments/fl/train.py --config $A/configs/$cfg.json --run $r --gpu $gpu --autotune off --deterministic-ops off \
      --num-steps 100001 > $log 2>&1
      rc=$?; res=$(grep -o 'Best Validation Accuracy: [0-9.]*, Early stop: [A-Za-z]*' $log | tail -1)
      say "END $cfg r$r rc=$rc: $res"
      [ -z "$res" ] && rmdir $A/claims/${cfg}_100k_r$r 2>/dev/null ) &
    running=$((running + 1)); sleep 15
  done
done
wait
say "ablation lane done on $(hostname -s)"
