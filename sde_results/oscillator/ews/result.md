# Coupled oscillator, EWS learner

**Test RMSE 2.57e-02 ± 2.60e-03** (validation 2.60e-02 ± 2.02e-03); mean ± population std over 10 seeds, normalised targets.

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| val | 2.54e-02 | 2.44e-02 | 2.46e-02 | 2.88e-02 | 2.52e-02 | 2.87e-02 | 2.88e-02 | 2.66e-02 | 2.31e-02 | 2.42e-02 |
| test | 2.50e-02 | 2.76e-02 | 2.27e-02 | 2.77e-02 | 2.72e-02 | 2.93e-02 | 2.75e-02 | 2.67e-02 | 2.12e-02 | 2.23e-02 |

**Learner:** `A` full 3 x 3 matrix; `config.json` holds the resolved config with the tuned hyperparameters (seed 100 is seed 0, `--seed 100 + k` with the seed-k data file gives seed k).

**Protocol:** 750 trajectories of the coupled oscillatory SDE on [0, 4] with 10,000 Euler-Maruyama steps, shuffled 70/15/15 split; inputs (t, W1, W2) scaled to [0, 1] on the training set with a basepoint (0, 0, 0) prepended, target X1 scaled to [0, 1]; depth-2 learners with M = 10 and exact matrix exponentials; AdamW with 500 warmup steps and a cosine decay, batch 32, 15,000 steps, evaluation every 500 steps, RMSE at the best validation checkpoint; ten seeds, seed k using data seed k and model seed 100 + k.

**Hyperparameters:** Optuna (TPE), 100 trials of 15,000 steps per learner, on the seed-42 data.

**Runs:** `runs/seed<k>/results.json` holds each banked run's evaluation history, best checkpoint, and the learned `A` with its eigenvalues.

## Reproduce

```bash
PY=<your-python> GPUS=0 bash sde_results/replicate.sh oscillator/ews   # all ten seeds
python experiments/sde/train.py --config sde_results/oscillator/ews/config.json --dataset_file <seed-k data> --seed <100 + k> --gpu <g>   # one seed
```
