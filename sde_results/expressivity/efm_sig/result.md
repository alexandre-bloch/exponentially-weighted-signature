# EFM target, Signature learner

**Test RMSE 4.28e-02 ± 5.26e-03** (validation 4.41e-02 ± 4.97e-03); mean ± population std over 10 seeds, normalised targets.

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| val | 4.30e-02 | 3.86e-02 | 5.16e-02 | 4.28e-02 | 3.91e-02 | 5.15e-02 | 4.23e-02 | 3.92e-02 | 5.09e-02 | 4.17e-02 |
| test | 4.04e-02 | 3.82e-02 | 5.07e-02 | 4.06e-02 | 3.81e-02 | 5.03e-02 | 4.05e-02 | 3.75e-02 | 5.10e-02 | 4.06e-02 |

**Learner:** `A` A = 0 (fixed); `config.json` holds the resolved config with the tuned hyperparameters (seed 100 is seed 0, `--seed 100 + k` with the seed-k data file gives seed k).

**Protocol:** 750 time-augmented Brownian paths on [0, 5] with 10,000 steps, split 525/112/113 in order; the target is the (W2, W1) component of the depth-2 EWS of (t, W1, W2) under the target operator, scaled to [0, 1] on the training set; raw paths; depth-2 learners with M = 10 and exact matrix exponentials; AdamW with 500 warmup steps and a cosine decay, batch 32, 30,000 steps, evaluation every 500 steps, test RMSE at the best validation checkpoint; ten seeds, seed k using Brownian paths drawn with data seed k and model seed 100 + k.

**Hyperparameters:** Optuna (TPE), 15 trials of 15,000 steps per target-learner pair (50 for the EWS learner on the EWS target), on the seed-321 data.

**Runs:** `runs/seed<k>/results.json` holds each banked run's evaluation history, best checkpoint, and the learned `A` with its eigenvalues.

## Reproduce

```bash
PY=<your-python> GPUS=0 bash sde_results/replicate.sh expressivity/efm_sig   # all ten seeds
python experiments/sde/train.py --config sde_results/expressivity/efm_sig/config.json --dataset_file <seed-k data> --seed <100 + k> --gpu <g>   # one seed
```
