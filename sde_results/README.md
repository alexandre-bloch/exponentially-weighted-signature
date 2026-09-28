# sde_results: EWS, EFM and the classical signature on two synthetic regression tasks

Banked runs of the two synthetic experiments. Each `<experiment>/<entry>/` holds:
- `config.json`: the exact config (tuned hyperparameters; seed 100 is seed 0)
- `result.md`: the banked numbers, protocol, reproduce command
- `runs/seed<k>/results.json`: the banked run's evaluation history, best checkpoint, learned `A` and eigenvalues

All numbers are RMSE on normalised targets at the best validation checkpoint, mean ± population std over ten seeds.

## Expressivity gap (depth 2): test RMSE per target (rows) and learner (columns)

| target \ learner | EWS | EFM | Signature |
|---|---|---|---|
| **EWS** | [9.18e-04 ± 8.80e-04](expressivity/ews_ews/result.md) | [2.43e-02 ± 2.57e-03](expressivity/ews_efm/result.md) | [2.41e-02 ± 2.55e-03](expressivity/ews_sig/result.md) |
| **EFM** | [2.77e-05 ± 2.78e-06](expressivity/efm_ews/result.md) | [2.66e-05 ± 2.84e-06](expressivity/efm_efm/result.md) | [4.28e-02 ± 5.26e-03](expressivity/efm_sig/result.md) |
| **Signature** | [2.02e-05 ± 1.73e-06](expressivity/sig_ews/result.md) | [1.99e-05 ± 7.59e-07](expressivity/sig_efm/result.md) | [2.00e-05 ± 7.68e-07](expressivity/sig_sig/result.md) |

The target is the (W2, W1) component of the depth-2 EWS of a time-augmented Brownian path under a
fixed operator: a full matrix (EWS; built as the real part of a unitary similarity of diag(−0.5 ± 5.2i, 0.8),
which gives the eigenvalues −0.44 ± 3.22i and 0.68), diag(0.5, 0.3, 0.8) (EFM) or zero (Signature).
Each learner is trained at the same depth with an unconstrained matrix (EWS), a
positive diagonal (EFM) or a fixed zero (Signature).

## Coupled oscillator: validation and test RMSE per learner

| learner | validation | test |
|---|---|---|
| [EWS](oscillator/ews/result.md) | 2.60e-02 ± 2.02e-03 | 2.57e-02 ± 2.60e-03 |
| [EFM](oscillator/efm/result.md) | 9.46e-02 ± 4.51e-03 | 9.36e-02 ± 4.55e-03 |
| [Signature](oscillator/sig/result.md) | 1.28e-01 ± 4.92e-03 | 1.27e-01 ± 5.08e-03 |

Same-time regression of X1 from the driving path (t, W1, W2) of dX1 = (3 sin X2 − 0.5 X1) dt + 0.4 dW1,
dX2 = (3 cos X1 − 0.5 X2) dt + 0.4 dW2.

## Learned spectra on the oscillator (mean ± std over seeds)

| | EWS | EFM |
|---|---|---|
| real part (complex pair) | 1.86 ± 0.15 | – |
| imaginary magnitude | 10.06 ± 0.84 | – |
| real eigenvalues | 0.06 ± 0.23 | 7.65 ± 1.83, 2.78 ± 0.21, 0.17 ± 0.37 |

## Oscillatory kernel: causal windowed Fourier coefficients (test metric, mean ± std over 3 seeds)

Three per-timestep targets of a synthetic signal (T1: the eight windowed Fourier coefficients at
{2, 4, 22, 27} Hz with damping 5; T2: the high-band energy; T3: the burst flag), read from the
prefix EWS of the integral path by a dense lift, one layer at depth n and width m and a linear
readout. Matched budget (n, m) = (2, 8) is 72 features for every model; best is each model's validation pick over its
(n, m) grid (D <= 5000). Normalised MSE for T1 and T2, accuracy for T3.

| target | metric | EWS (2,8) | EFM (2,8) | sig (2,8) | best EWS | best EFM | best sig |
|---|---|---|---|---|---|---|---|
| T1 | normalised MSE (test) | 1.17e-01 ± 1.41e-01 | 4.96e-01 ± 2.40e-02 | 8.81e-01 ± 2.42e-04 | 3.49e-05 ± 1.24e-05 at (n,m)=(1,32) | 4.19e-01 ± 1.04e-02 at (n,m)=(3,16) | 7.32e-01 ± 1.20e-04 at (n,m)=(4,8) |
| T2 | normalised MSE (test) | 2.85e-01 ± 1.73e-01 | 9.21e-01 ± 6.66e-03 | 9.06e-01 ± 9.34e-04 | 7.30e-04 ± 3.43e-04 at (n,m)=(2,32) | 9.09e-01 ± 3.97e-03 at (n,m)=(2,64) | 9.06e-01 ± 9.34e-04 at (n,m)=(2,8) |
| T3 | accuracy (test) | 0.772 ± 0.104 | 0.691 ± 0.000 | 0.691 ± 0.000 | 0.947 ± 0.002 at (n,m)=(2,32) | 0.694 ± 0.001 at (n,m)=(3,16) | 0.692 ± 0.000 at (n,m)=(2,2) |

The per-band split of T1, the frequency-recovery figure and the sanity checks are in
[oscillatory_kernel/](oscillatory_kernel/README.md) (figures, `results.csv`, per-run `results.json`).

## Reproduce

```bash
PY=<your-python> GPUS=<list> bash sde_results/replicate.sh <experiment>/<entry>   # e.g. expressivity/ews_efm, oscillator/ews
```

The runs used XLA autotuning on and deterministic ops off.
`python experiments/sde/report.py` rewrites these tables from `runs/`.
