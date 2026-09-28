# Matched budget and the best cell of each model (chosen on validation; test metric, mean ± std over seeds)

Matched budget: (n, m) = (2, 8), D = 72.

| target | metric | EWS (2,8) | EFM (2,8) | sig (2,8) | best EWS | best EFM | best sig |
|---|---|---|---|---|---|---|---|
| T1 | normalised MSE (test) | 1.17e-01 ± 1.41e-01 | 4.96e-01 ± 2.40e-02 | 8.81e-01 ± 2.42e-04 | 3.49e-05 ± 1.24e-05 at (n,m)=(1,32) | 4.19e-01 ± 1.04e-02 at (n,m)=(3,16) | 7.32e-01 ± 1.20e-04 at (n,m)=(4,8) |
| T2 | normalised MSE (test) | 2.85e-01 ± 1.73e-01 | 9.21e-01 ± 6.66e-03 | 9.06e-01 ± 9.34e-04 | 7.30e-04 ± 3.43e-04 at (n,m)=(2,32) | 9.09e-01 ± 3.97e-03 at (n,m)=(2,64) | 9.06e-01 ± 9.34e-04 at (n,m)=(2,8) |
| T3 | accuracy (test) | 0.772 ± 0.104 | 0.691 ± 0.000 | 0.691 ± 0.000 | 0.947 ± 0.002 at (n,m)=(2,32) | 0.694 ± 0.001 at (n,m)=(3,16) | 0.692 ± 0.000 at (n,m)=(2,2) |

T3 F1 (test) at the matched budget and at each model's best-accuracy configuration:

| | EWS (2,8) | EFM (2,8) | sig (2,8) | best EWS | best EFM | best sig |
|---|---|---|---|---|---|---|
| F1 | 0.403 ± 0.326 | 0.417 ± 0.003 | 0.404 ± 0.004 | 0.909 ± 0.004 | 0.416 ± 0.002 | 0.393 ± 0.002 |

T1 by band (normalised MSE over the band's four outputs, test), matched budget (2,8) and each model's best cell:

| band | EWS (2,8) | EFM (2,8) | sig (2,8) | best EWS | best EFM | best sig |
|---|---|---|---|---|---|---|
| low, 2 and 4 Hz | 2.35e-02 ± 1.90e-02 | 9.22e-02 ± 1.55e-02 | 7.65e-01 ± 4.34e-04 | 1.16e-05 ± 7.63e-06 at (1,32) | 4.06e-02 ± 3.92e-03 at (3,16) | 4.86e-01 ± 1.85e-04 at (4,8) |
| high, 22 and 27 Hz | 2.10e-01 ± 2.87e-01 | 8.99e-01 ± 3.30e-02 | 9.97e-01 ± 1.23e-04 | 5.82e-05 ± 1.79e-05 at (1,32) | 7.98e-01 ± 1.68e-02 at (3,16) | 9.78e-01 ± 6.19e-05 at (4,8) |
