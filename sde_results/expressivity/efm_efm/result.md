# EFM target, EFM learner

**Test RMSE 2.66e-05 ± 2.84e-06** (validation 2.58e-05 ± 3.14e-06); mean ± population std over 10 seeds, normalised targets.

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| val | 2.44e-05 | 2.37e-05 | 3.08e-05 | 2.42e-05 | 2.26e-05 | 3.03e-05 | 2.42e-05 | 2.29e-05 | 3.04e-05 | 2.42e-05 |
| test | 2.67e-05 | 2.34e-05 | 3.05e-05 | 2.63e-05 | 2.30e-05 | 3.04e-05 | 2.61e-05 | 2.28e-05 | 3.02e-05 | 2.65e-05 |

**Learner:** `A` positive diagonal (theta**2); `config.json` holds the resolved config with the tuned hyperparameters (seed 100 is seed 0, `--seed 100 + k` with the seed-k data file gives seed k).

**Protocol:** 750 time-augmented Brownian paths on [0, 5] with 10,000 steps, split 525/112/113 in order; the target is the (W2, W1) component of the depth-2 EWS of (t, W1, W2) under the target operator, scaled to [0, 1] on the training set; raw paths; depth-2 learners with M = 10 and exact matrix exponentials; AdamW with 500 warmup steps and a cosine decay, batch 32, 30,000 steps, evaluation every 500 steps, test RMSE at the best validation checkpoint; ten seeds, seed k using Brownian paths drawn with data seed k and model seed 100 + k.

**Hyperparameters:** Optuna (TPE), 15 trials of 15,000 steps per target-learner pair (50 for the EWS learner on the EWS target), on the seed-321 data.

**Runs:** `runs/seed<k>/results.json` holds each banked run's evaluation history, best checkpoint, and the learned `A` with its eigenvalues.

## Reproduce

```bash
PY=<your-python> GPUS=0 bash sde_results/replicate.sh expressivity/efm_efm   # all ten seeds
python experiments/sde/train.py --config sde_results/expressivity/efm_efm/config.json --dataset_file <seed-k data> --seed <100 + k> --gpu <g>   # one seed
```
