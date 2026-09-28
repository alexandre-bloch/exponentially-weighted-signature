# Signature-depth ablation

For each dataset we draw 33 configurations once, the champion and 32 at random from a grid around it, and train each
at depths 1, 2 and 3 with everything else the same, on depth 3's width ladder, so the two members of a pair differ in
depth alone. That is 99 cells on three seeds, the five best per depth extended to five seeds, at 6,000 steps under
Walker's protocol. EigenWorms has 32 configurations on a smaller grid, all at width 8, since depth 3 does not fit
above that width (configuration 18 is left out: at width 8 it is configuration 13). Each dataset's `configs.json`
records the grid, the draw seed and the configurations, and `champion_config.json` is the champion they were drawn
around.

The statistic is the per-configuration change with depth; "better" counts the configurations where the deeper model
scored higher.

| dataset | pair | val better | val median | test better | test median | test mean |
|---|---|---|---|---|---|---|
| EigenWorms (champion depth 2, 32 configs) | n1 → n2 | 29/32 | +10.67 | 23/32 | +9.07 | +15.31 |
|  | n2 → n3 | 18/32 | +0.95 | 10/32 | -1.85 | -2.18 |
|  | n1 → n3 | 28/32 | +11.14 | 20/32 | +8.15 | +13.13 |
| Heartbeat (champion depth 3) | n1 → n2 | 25/33 | +1.09 | 23/33 | +1.08 | +0.72 |
|  | n2 → n3 | 20/33 | +0.55 | 16/33 | +0.00 | +0.44 |
|  | n1 → n3 | 28/33 | +1.64 | 23/33 | +1.08 | +1.16 |
| MotorImagery (champion depth 3) | n1 → n2 | 25/33 | +1.75 | 20/33 | +1.17 | +0.98 |
|  | n2 → n3 | 26/33 | +2.92 | 18/33 | +0.94 | +0.80 |
|  | n1 → n3 | 28/33 | +4.68 | 19/33 | +1.17 | +1.78 |
| EthanolConcentration (champion depth 2) | n1 → n2 | 22/33 | +3.80 | 27/33 | +4.22 | +3.28 |
|  | n2 → n3 | 15/33 | +0.00 | 15/33 | -0.42 | -0.36 |
|  | n1 → n3 | 22/33 | +3.38 | 26/33 | +4.64 | +2.93 |
| SelfRegulationSCP2 (champion depth 3) | n1 → n2 | 24/33 | +2.34 | 10/33 | -3.51 | -3.69 |
|  | n2 → n3 | 26/33 | +2.22 | 26/33 | +2.92 | +2.56 |
|  | n1 → n3 | 29/33 | +4.33 | 14/33 | -0.94 | -1.13 |
| SelfRegulationSCP1 (champion depth 2) | n1 → n2 | 17/33 | +0.00 | 3/33 | -1.57 | -1.66 |
|  | n2 → n3 | 5/33 | -2.38 | 12/33 | -1.57 | -0.51 |
|  | n1 → n3 | 6/33 | -2.38 | 5/33 | -2.35 | -2.17 |

Per-depth means over the cells (top-5: mean test of the five best-validation cells):

| dataset | depth 1 val / test | depth 2 val / test | depth 3 val / test | top-5 test by depth |
|---|---|---|---|---|
| EigenWorms | 50.18 / 55.86 | 69.71 / 71.17 | 69.50 / 68.99 | 76.00 → 90.67 → 88.22 |
| Heartbeat | 76.26 / 73.73 | 77.08 / 74.46 | 77.63 / 74.90 | 74.02 → 74.41 → 75.48 |
| MotorImagery | 55.21 / 48.13 | 57.12 / 49.11 | 59.88 / 49.90 | 50.04 → 50.53 → 52.05 |
| EthanolConcentration | 34.35 / 30.47 | 38.19 / 33.75 | 38.11 / 33.39 | 51.90 → 53.70 → 56.35 |
| SelfRegulationSCP2 | 53.22 / 51.21 | 55.22 / 47.53 | 57.45 / 50.08 | 54.74 → 48.07 → 48.84 |
| SelfRegulationSCP1 | 87.23 / 86.45 | 87.13 / 84.79 | 84.91 / 84.28 | 87.58 → 86.87 → 84.33 |

`champion_depths.json` is the second ablation: each banked champion trained at depths 1, 2 and 3 under its own bank's
protocol (its width, step budget and five per-seed splits), everything else identical. Ethanol's depth-3 cells use the
`van_loan` leaf, since `plane_closed` has closed forms for depths 1 and 2 only; EigenWorms' row is trained at width 8
under this ablation's protocol, by `eigenworms_champion_row.py`.

| file | contents |
|---|---|
| `uea_paired.py` | draws the configurations (`build`), trains the cells (`run`, several workers may share a dataset) and prints the tables (`report`) |
| `common.py` | the champion config, the per-cell config edit and the call to `experiments/uea/train.py` |
| `<dataset>/configs.json`, `<dataset>/matched.json` | the draw, and every cell's per-seed validation and test accuracy |
| `champion_depths.py`, `champion_depths.json` | the champion at each depth under its bank's protocol |
| `ethanol_champion_row.py`, `eigenworms_champion_row.py`, `<dataset>/champion_row*.json` | the Ethanol and EigenWorms champions at each depth on this ablation's terms (6,000 steps, width capped at depth 3's largest), with their configurations |
| `summary.json` | the tables above, as data |

```bash
python uea_results/ablation/uea_paired.py build  --dataset Heartbeat                              # reproduces Heartbeat/configs.json
python uea_results/ablation/uea_paired.py run    --dataset Heartbeat --gpu 0 --seeds 0,1,2               # every cell
python uea_results/ablation/uea_paired.py run    --dataset Heartbeat --gpu 0 --seeds 0,1,2,3,4 --top_k 5  # the top 5 per depth
python uea_results/ablation/uea_paired.py report --dataset Heartbeat
python uea_results/ablation/champion_depths.py --gpu 0
```

Every run is `experiments/uea/train.py` on the per-seed split `data/<stem>/splits/<stem>_seed<k>.npz`, with
`--autotune off --deterministic-ops on`.
