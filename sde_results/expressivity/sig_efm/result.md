# Signature target, EFM learner

**Test RMSE 1.99e-05 ± 7.59e-07** (validation 1.95e-05 ± 1.98e-06); mean ± population std over 10 seeds, normalised targets.

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| val | 1.76e-05 | 1.97e-05 | 2.22e-05 | 1.73e-05 | 1.96e-05 | 2.20e-05 | 1.75e-05 | 1.99e-05 | 2.21e-05 | 1.70e-05 |
| test | 1.93e-05 | 2.00e-05 | 2.07e-05 | 1.90e-05 | 2.06e-05 | 2.09e-05 | 1.87e-05 | 2.03e-05 | 2.05e-05 | 1.91e-05 |

**Learner:** `A` positive diagonal (theta**2); `config.json` holds the resolved config with the tuned hyperparameters (seed 100 is seed 0, `--seed 100 + k` with the seed-k data file gives seed k).

**Protocol:** 750 time-augmented Brownian paths on [0, 5] with 10,000 steps, split 525/112/113 in order; the target is the (W2, W1) component of the depth-2 EWS of (t, W1, W2) under the target operator, scaled to [0, 1] on the training set; raw paths; depth-2 learners with M = 10 and exact matrix exponentials; AdamW with 500 warmup steps and a cosine decay, batch 32, 30,000 steps, evaluation every 500 steps, test RMSE at the best validation checkpoint; ten seeds, seed k using Brownian paths drawn with data seed k and model seed 100 + k.

**Hyperparameters:** Optuna (TPE), 15 trials of 15,000 steps per target-learner pair (50 for the EWS learner on the EWS target), on the seed-321 data.

**Runs:** `runs/seed<k>/results.json` holds each banked run's evaluation history, best checkpoint, and the learned `A` with its eigenvalues.

## Reproduce

```bash
PY=<your-python> GPUS=0 bash sde_results/replicate.sh expressivity/sig_efm   # all ten seeds
python experiments/sde/train.py --config sde_results/expressivity/sig_efm/config.json --dataset_file <seed-k data> --seed <100 + k> --gpu <g>   # one seed
```
