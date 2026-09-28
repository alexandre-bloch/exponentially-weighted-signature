# Heartbeat

The selected cell (`config.json`) has the highest 5-seed mean validation accuracy in the grid of `grid.json`.
Deterministic XLA (autotuning off, deterministic ops on).

| | mean | per seed (splits 0-4) |
|---|---|---|
| test | 80.00 ± 4.52 | 72.58, 79.03, 79.03, 83.87, 85.48 |
| validation | 77.71 | 75.41, 86.89, 80.33, 75.41, 70.49 |

Depth 3, width 10: five complex 2x2 blocks (unconstrained rotation), mean readout; 2,847 parameters.

```bash
bash uea_results/replicate.sh heartbeat
```
