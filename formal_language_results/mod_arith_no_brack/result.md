# mod_arith_no_brack: EWS under Walker's protocol

**Result: 96.71 ± 0.75** (best validation accuracy %, mean ± population std over seeds 1234–1238; 0/5 seeds solved at ≥ 99.95)

| seed | 1234 | 1235 | 1236 | 1237 | 1238 |
|---|---|---|---|---|---|
| best val acc | 97.08 | 96.58 | 97.44 | 95.31 | 97.13 |

**Config:** `config.json` (fully resolved; `training.seed` 1234 is seed 0, `--run r` gives seed 1234 + r).

**Protocol** (Walker, `structured-linear-cdes`): train lengths 3–40, validate 40–256, batch 256, 100k steps, eval every 5000, metric = best validation accuracy over the run, early stop at > 0.9995, non-deterministic GPU regime.

**Deviations from Walker's training defaults:** dropout 0.01 → 0, weight decay on A 0.01 → 0 with an ‖A‖_F penalty (λ = 1e-3); block-1 init scale calibrated (`angle_target` 0.95).

**Runs:** `runs/seed<S>/` holds each banked run's `results.json` (every eval) and `sidecar.json` (resolved config + init calibration).

## Reproduce

```bash
bash formal_language_results/replicate.sh mod_arith_no_brack            # all five seeds
python experiments/fl/train.py --config formal_language_results/mod_arith_no_brack/config.json --run <r> --gpu <g>   # one seed
```
