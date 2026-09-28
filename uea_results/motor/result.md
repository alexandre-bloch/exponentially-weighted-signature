# MotorImagery

The selected cell (`config.json`) has the highest 5-seed mean validation accuracy in the grid of `grid.json`.
Deterministic XLA (autotuning off, deterministic ops on).

| | mean | per seed (splits 0-4) |
|---|---|---|
| test | 61.05 ± 5.01 | 59.65, 57.89, 64.91, 54.39, 68.42 |
| validation | 65.96 | 68.42, 64.91, 68.42, 64.91, 63.16 |

Depth 3, width 16: skew-symmetric A, terminal readout; 9,887 parameters.

```bash
bash uea_results/replicate.sh motor
```
