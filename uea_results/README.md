# UEA classification

The six UEA datasets under Walker's protocol: one constant learning rate, the best-validation checkpoint (latest on a
tie), and five seeds, each on its own split `data/<stem>/splits/<stem>_seed{0..4}.npz`. Each selected configuration is the cell with the
highest 5-seed mean validation accuracy in the grid of its `grid.json`.

| dataset | test accuracy (mean ± sd) |
|---|---|
| EigenWorms | 93.33 ± 1.36 |
| EthanolConcentration | 71.90 ± 4.76 |
| Heartbeat | 80.00 ± 4.52 |
| MotorImagery | 61.05 ± 5.01 |
| SelfRegulationSCP1 | 87.06 ± 2.47 |
| SelfRegulationSCP2 | 56.49 ± 5.81 |

± is the population standard deviation over the five seeds. Each `<dataset>/` holds the selected configuration's
`config.json`, the `grid.json` it was selected from (axes, fixed settings, selection rule, regime, its per-seed
numbers) and a short `result.md`. `timing/` has the speed and memory of the six selected models, `ablation/` the
signature-depth ablation.

```bash
bash uea_results/replicate.sh motor    # or eigenworms, ethanol, heartbeat, scp1, scp2
```

All six are deterministic (XLA autotuning off, deterministic ops on, as recorded in each `grid.json`), so a replicate
reproduces every per-seed number exactly.
