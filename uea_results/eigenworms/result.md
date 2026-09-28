# EigenWorms

The selected cell (`config.json`) has the highest 5-seed mean validation accuracy in the grid of `grid.json`.
Deterministic XLA (autotuning off, deterministic ops on).

| | mean | per seed (splits 0-4) |
|---|---|---|
| test | 93.33 ± 1.36 | 91.67, 94.44, 94.44, 91.67, 94.44 |
| validation | 96.00 | 94.29, 97.14, 91.43, 97.14, 100.00 |

`search.csv` is the TPE search run before the grid (80 trials, objective: mean validation accuracy on split seeds 2
and 4).

Depth 2, width 8: two complex Jordan blocks of size 4 on a trained gated clock, value path with a clock channel, mean readout; 446 parameters.

```bash
bash uea_results/replicate.sh eigenworms
```
