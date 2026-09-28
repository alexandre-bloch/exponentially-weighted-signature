# Coupled oscillator, EFM learner

**Test RMSE 9.36e-02 ± 4.55e-03** (validation 9.46e-02 ± 4.51e-03); mean ± population std over 10 seeds, normalised targets.

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| val | 9.26e-02 | 9.03e-02 | 9.02e-02 | 9.92e-02 | 9.08e-02 | 9.85e-02 | 9.33e-02 | 8.94e-02 | 1.01e-01 | 1.01e-01 |
| test | 9.22e-02 | 9.29e-02 | 8.97e-02 | 9.47e-02 | 9.15e-02 | 9.85e-02 | 8.91e-02 | 8.68e-02 | 1.02e-01 | 9.92e-02 |

**Learner:** `A` positive diagonal (theta**2); `config.json` holds the resolved config with the tuned hyperparameters (seed 100 is seed 0, `--seed 100 + k` with the seed-k data file gives seed k).

**Protocol:** 750 trajectories of the coupled oscillatory SDE on [0, 4] with 10,000 Euler-Maruyama steps, shuffled 70/15/15 split; inputs (t, W1, W2) scaled to [0, 1] on the training set with a basepoint (0, 0, 0) prepended, target X1 scaled to [0, 1]; depth-2 learners with M = 10 and exact matrix exponentials; AdamW with 500 warmup steps and a cosine decay, batch 32, 15,000 steps, evaluation every 500 steps, RMSE at the best validation checkpoint; ten seeds, seed k using data seed k and model seed 100 + k.

**Hyperparameters:** Optuna (TPE), 100 trials of 15,000 steps per learner, on the seed-42 data.

**Runs:** `runs/seed<k>/results.json` holds each banked run's evaluation history, best checkpoint, and the learned `A` with its eigenvalues.

## Reproduce

```bash
PY=<your-python> GPUS=0 bash sde_results/replicate.sh oscillator/efm   # all ten seeds
python experiments/sde/train.py --config sde_results/oscillator/efm/config.json --dataset_file <seed-k data> --seed <100 + k> --gpu <g>   # one seed
```
