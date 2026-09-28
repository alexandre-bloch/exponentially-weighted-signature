# Clock and A ablation of the formal-language champions

The four champions use a gated clock and a rotation A. This 2 × 2 per task asks which of the two carries the result:
{champion A, a positive diagonal A (`efm`)} × {the champion clocks, the uniform `time` clock}, where the champion cell
is the banked result. `configs/<task>_uniform.json` changes only the clock, `configs/<task>_efm.json` only A (in both
blocks), and `configs/<task>_efm_uniform.json` both; everything else is the banked config. mod_arith's block-0
`angle_target` calibration is dropped in its efm cells, since a diagonal A has no rotation angle.

Walker's protocol: train lengths 3 to 40, validate 40 to 256, batch 256, an evaluation every 5,000 steps, best
validation accuracy, early stop above 0.9995, every run to 100,000 steps; runs 0, 1, 2 are seeds 1234 to 1236. A run
whose training loss becomes NaN is marked diverged: its value is the best validation accuracy before the divergence
(`*`, or `**` when every seed diverged). even_pairs "efm A + champion clocks" run 0 solved the task and stopped under
an earlier 40k-step cap and was not re-run.

Best validation accuracy (%), mean ± population std over the three seeds:

| task | arm | rule | runs | mean +- std | ended by | champion |
|---|---|---|---|---|---|---|
| parity | champion A + uniform clocks | 100k | 52.38, 54.17, 52.70 | 53.08 +- 0.78 | budget, budget, budget | 99.995 (chance 50) |
| parity | efm A + champion clocks | 100k | 55.80, 55.42, 53.49 | 54.90 +- 1.01 | budget, budget, budget | 99.995 (chance 50) |
| parity | efm A + uniform clocks | 100k | 55.36, 54.59, 53.99 | 54.65 +- 0.56 | budget, budget, budget | 99.995 (chance 50) |
| even_pairs | champion A + uniform clocks | 100k | 59.83, 56.68, 57.92 | 58.14 +- 1.30 | budget, budget, budget | 100.0 (chance 50) |
| even_pairs | efm A + champion clocks | 100k | 100.00, 99.18, 99.49 | 99.56 +- 0.34 | solved (40k rule), budget, budget | 100.0 (chance 50) |
| even_pairs | efm A + uniform clocks | 100k | 91.42, 92.97, 88.23 | 90.87 +- 1.97 | budget, budget, budget | 100.0 (chance 50) |
| cycle_nav | champion A + uniform clocks | 100k | 29.19, 26.82, 30.77 | 28.93 +- 1.62 | budget, budget, budget | 99.985 (chance 20) |
| cycle_nav | efm A + champion clocks | 100k | 51.82, 56.19, 48.41 | 52.14 +- 3.18 | budget, budget, budget | 99.985 (chance 20) |
| cycle_nav | efm A + uniform clocks | 100k | 54.72, 58.86, 36.96 | 50.18 +- 9.50 | budget, budget, budget | 99.985 (chance 20) |
| mod_arith_no_brack | champion A + uniform clocks | 100k | 20.92, 20.98, 20.17 | 20.69 +- 0.37 | budget, budget, budget | 96.71 (chance 20) |
| mod_arith_no_brack | efm A + champion clocks | 100k | 38.12, 36.84, 44.15 | 39.70 +- 3.19 | budget, budget, budget | 96.71 (chance 20) |
| mod_arith_no_brack | efm A + uniform clocks | 100k | 20.90*, 19.71*, 19.69* | 20.10** +- 0.57 (best before divergence, all 3 seeds) | diverged at 10k, diverged at 10k, diverged at 10k | 96.71 (chance 20) |

Only the champion cell solves parity, cycle_nav and mod_arith: the rotation A with a uniform clock stays at chance on
all three, and the efm cells get part of the way on cycle_nav and mod_arith. even_pairs is a clock result: efm with
the champion clocks solves it, the rotation A with a uniform clock does not.

`runs/` holds every run's training log, `collate.py` prints the table from them and `run_ablation.sh` runs the cells.

```bash
GPUS="0 1" MAX_PER_GPU=3 bash formal_language_results/clock_A_ablation/run_ablation.sh
python formal_language_results/clock_A_ablation/collate.py
```
