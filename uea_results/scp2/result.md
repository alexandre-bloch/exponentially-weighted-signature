# SelfRegulationSCP2

The selected cell (`config.json`) has the highest 5-seed mean validation accuracy in the grid of `grid.json`.
Deterministic XLA (autotuning off, deterministic ops on).

| | mean | per seed (splits 0-4) |
|---|---|---|
| test | 56.49 ± 5.81 | 66.67, 50.88, 50.88, 56.14, 57.89 |
| validation | 61.40 | 63.16, 61.40, 56.14, 63.16, 63.16 |

Depth 3, width 14: skew-symmetric A, terminal readout; 6,104 parameters.

```bash
bash uea_results/replicate.sh scp2
```
