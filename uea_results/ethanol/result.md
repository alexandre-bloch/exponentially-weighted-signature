# EthanolConcentration

The selected cell (`config.json`) has the highest 5-seed mean validation accuracy in the grid of `grid.json`.
Deterministic XLA (autotuning off, deterministic ops on).

| | mean | per seed (splits 0-4) |
|---|---|---|
| test | 71.90 ± 4.76 | 77.22, 64.56, 73.42, 75.95, 68.35 |
| validation | 75.95 | 82.28, 62.03, 74.68, 83.54, 77.22 |

Depth 2, width 24: twelve damped rotation planes on a time clock, derivative path with a basepoint, terminal readout; 2,508 parameters. The BatchNorm feature norm is evaluated with training-set statistics.

```bash
bash uea_results/replicate.sh ethanol
```
