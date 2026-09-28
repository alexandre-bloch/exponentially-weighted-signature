# SelfRegulationSCP1

The selected cell (`config.json`) has the highest 5-seed mean validation accuracy in the grid of `grid.json`.
Deterministic XLA (autotuning off, deterministic ops on).

| | mean | per seed (splits 0-4) |
|---|---|---|
| test | 87.06 ± 2.47 | 88.24, 89.41, 87.06, 82.35, 88.24 |
| validation | 90.24 | 85.71, 95.24, 88.10, 92.86, 89.29 |

Depth 2, width 17: skew-symmetric A, mean readout; 856 parameters.

```bash
bash uea_results/replicate.sh scp1
```
