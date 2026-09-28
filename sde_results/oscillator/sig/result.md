# Coupled oscillator, Signature learner

**Test RMSE 1.27e-01 ± 5.08e-03** (validation 1.28e-01 ± 4.92e-03); mean ± population std over 10 seeds, normalised targets.

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| val | 1.31e-01 | 1.21e-01 | 1.22e-01 | 1.32e-01 | 1.22e-01 | 1.32e-01 | 1.24e-01 | 1.26e-01 | 1.32e-01 | 1.35e-01 |
| test | 1.29e-01 | 1.27e-01 | 1.21e-01 | 1.30e-01 | 1.22e-01 | 1.31e-01 | 1.20e-01 | 1.21e-01 | 1.34e-01 | 1.33e-01 |

**Learner:** `A` A = 0 (fixed); `config.json` holds the resolved config with the tuned hyperparameters (seed 100 is seed 0, `--seed 100 + k` with the seed-k data file gives seed k).

**Protocol:** 750 trajectories of the coupled oscillatory SDE on [0, 4] with 10,000 Euler-Maruyama steps, shuffled 70/15/15 split; inputs (t, W1, W2) scaled to [0, 1] on the training set with a basepoint (0, 0, 0) prepended, target X1 scaled to [0, 1]; depth-2 learners with M = 10 and exact matrix exponentials; AdamW with 500 warmup steps and a cosine decay, batch 32, 15,000 steps, evaluation every 500 steps, RMSE at the best validation checkpoint; ten seeds, seed k using data seed k and model seed 100 + k.

**Hyperparameters:** Optuna (TPE), 100 trials of 15,000 steps per learner, on the seed-42 data.

**Runs:** `runs/seed<k>/results.json` holds each banked run's evaluation history, best checkpoint, and the learned `A` with its eigenvalues.

## Reproduce

```bash
PY=<your-python> GPUS=0 bash sde_results/replicate.sh oscillator/sig   # all ten seeds
python experiments/sde/train.py --config sde_results/oscillator/sig/config.json --dataset_file <seed-k data> --seed <100 + k> --gpu <g>   # one seed
```
