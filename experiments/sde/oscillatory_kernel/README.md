# Oscillatory kernel: causal windowed Fourier coefficients

The third synthetic experiment of the SDE family. Three per-timestep targets of a synthetic signal, computed in closed form, are regressed from the
prefix EWS of the time-augmented integral path by three bare models: the classical signature
(`sig`, A = 0), the EFM (positive diagonal A) and the EWS with damped-rotation planes.

| file | role |
|---|---|
| `data.py` | signals, exact ZOH targets (T1 coefficients, T2 high-band energy, T3 burst flag), quadrature oracle, oscillation count |
| `generate.py` | writes `data/oscillatory_kernel/oscillatory_kernel_seed<k>.npz` + `.json` (runs the oracle check) |
| `model.py` | lift + EWS layer + linear readout, the seeded log-uniform init override of A, the planted solutions |
| `sanity.py` | planted depth-1 and depth-2 checks, oracle, oscillation count -> `sanity.json` (run first) |
| `train.py` | one run: target, model, seed -> `results.json`, `model.eqx` |
| `grid.py` | the (model, n, m) x target x seed grid, `results.csv`, `config.yaml` |
| `report.py` | `fig_T1/T2/T3.png`, `fig_T1_bands.png`, `fig_freq_recovery.png`, `table_matched_budget.md` |
| `configs/template.json` | the config every grid run is derived from |

```bash
PY=python   # an interpreter with requirements.txt installed
JAX_PLATFORMS=cpu $PY experiments/sde/oscillatory_kernel/generate.py --seed 0
JAX_PLATFORMS=cpu $PY experiments/sde/oscillatory_kernel/sanity.py                      # gate: must print ALL PASS
$PY experiments/sde/oscillatory_kernel/grid.py --out outputs/oscillatory_kernel --gpus 0,1 --par 6
$PY experiments/sde/oscillatory_kernel/report.py --out outputs/oscillatory_kernel
# one run by hand
$PY experiments/sde/oscillatory_kernel/train.py --config <run config> --seed 0 --target T1 --gpu 0 --out_dir <dir>
```

`--cpu --steps 50 --only "ews:1:8" --seeds 0` on `grid.py` is the CPU smoke test. The `experiment`
section of the config holds what the engine does not: the init ranges, the lift's learning rate,
the global gradient clip, the readout init and the kill rule.
