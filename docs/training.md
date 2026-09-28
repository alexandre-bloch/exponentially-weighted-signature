# Training and reproducibility

| family | trainer | model | protocol |
|---|---|---|---|
| UEA classification | `experiments/uea/train.py` | one layer | Walker's: one constant learning rate, the best-validation checkpoint (latest on a tie), five seeds, each on its own per-seed split |
| formal languages | `experiments/fl/train.py` | stacked blocks | Walker's: train lengths 3 to 40, validate 40 to 256, batch 256, 100,001 steps, an evaluation every 5,000, best validation accuracy, early stop above 0.9995; AdamW with parameter groups, warmup then cosine decay to `final_lr` |
| SDE regression | `experiments/sde/train.py` | one layer | path-to-path regression on generated data |
| weather | `experiments/weather/train.py` | stacked blocks | D-LinOSS's (see `weather_results/README.md`) |
| PPG-DaLiA | `experiments/ppg/train.py` | stacked blocks | LinOSS and D-LinOSS's (see `ppg_results/README.md`) |

```bash
python experiments/uea/train.py --config uea_results/scp1/config.json --seed 0 --dataset_file <split.npz> --gpu 0
python experiments/fl/train.py --config formal_language_results/parity/config.json --run 0 --gpu 0
```

Each formal-language run writes `results.json` (every evaluation), `sidecar.json` (the resolved config, seed,
dimensions and init calibrations) and the best-validation checkpoint `checkpoint.eqx`. The formal-language trainer
calibrates two things on one training batch before it builds the model: constant clock normalisers for learned
drivers, and the graded-norm floors when `graded_eps_init` is `calibrated`. Both are recorded in the sidecar.

## Reproducibility

Two XLA switches decide whether a GPU run is reproducible (`ews/determinism.py`):

| switch | effect |
|---|---|
| autotuning | XLA picks kernels by timing them: faster, and the choice can differ between runs |
| deterministic ops | deterministic reductions and scatters: bit-reproducible, slower |

The trainers take them as `--autotune {on,off}` and `--deterministic-ops {on,off}` or as `training.xla_autotune` /
`training.deterministic_ops`; the command line wins over the config, which wins over a value already in `XLA_FLAGS`,
which wins over the trainer's default. Each bank's `replicate.sh` runs in the regime the bank was produced in:
autotuning off and deterministic ops on for the six UEA datasets (recorded in each `grid.json`), autotuning off and
deterministic ops off for the formal languages, weather and PPG, autotuning on and deterministic ops off for the SDE
experiments.

## Banked results

Each results folder holds the selected `config.json`, the `grid.json` it was selected from where there is one, a
`result.md` or README with the per-seed numbers, the banked runs where they are small enough, and a `replicate.sh`
that trains the seeds again and prints them beside the banked values. The formal-language `result.md` files list the
deviations from Walker's training defaults.
