# SSM → EWS conversion

Each linear state-space model family below is an EWS layer at depth one with explicit weights. This folder holds the
conversions (`families/`), the two converters they use (`method_a.py` reads the continuous generator off the
parameters, `method_b.py` inverts the family's own discrete transition) and the checks.

`python -m ews.conversion.verify` routes each family's converted parameters through the real EWS layer and compares
the layer's level-one state, at every step and coordinate, with the state the family's own implementation produces:
inputs N(0, 1) of length 128, the family's own initialisation, seeds 0, 1 and 2 (the worst is reported), float64 on
both sides, CPU.

| family | conversion | size | max abs error | max rel error | worst seed |
|---|---|---|---|---|---|
| S5 | read-off, lift Δt·B | P = 8 pairs, H = 4 (m = 16) | 1.069e-15 | 4.549e-15 | 1 |
| LRU | eq. (8), unit clock | N = 16, H = 4 (m = 32) | 4.996e-14 | 1.621e-14 | 0 |
| LinOSS-IM | eq. (8), per 2 × 2 block | P = 8, H = 4 (m = 32) | 2.731e-14 | 4.614e-15 | 0 |
| LinOSS-IMEX | eq. (8), per 2 × 2 block | P = 8, H = 4 (m = 32) | 7.905e-14 | 5.831e-15 | 2 |
| D-LinOSS | eq. (8), per 2 × 2 block | P = 8, H = 4 (m = 32) | 1.572e-08 | 5.176e-10 | 0 |
| S4D | read-off (ZOH), 2 × 2 rotation blocks | H = 4, N = 64 (m = 256) | 2.873e-15 | 9.940e-15 | 2 |
| S4 | eq. (8) on the dense (Ā, B̄) | H = 2, N = 8 (m = 32) | 4.519e-15 | 2.087e-14 | 2 |
| Mamba-1 | per channel: clock ΣΔ, identity lift | 4 channels, N = 16 | 6.314e-16 | 1.498e-14 | 0 |
| Mamba-2 | per head: clock ΣΔ, identity lift | 2 heads, P = 4, N = 16 | 2.220e-14 | 7.318e-15 | 2 |

D-LinOSS is the one row above 1e-13 because its lift G(A) = ∫₀^Δθ exp(−sA) ds is ill-conditioned at its own
initialisation (cond G(A) up to 1.8e5 over the three seeds), which amplifies float64 rounding.

The references are read-only clones named by environment variables: `DLINOSS_REPO` (jaredbmit/damped-linoss at
a396997), `S4_REPO` (state-spaces/s4), `MAMBA_SSM_DIR` (an installed `mamba_ssm` 1.2.2) and `MAMBA_REPO`
(state-spaces/mamba). S5, LRU and LinOSS are vendored in `reference/vendored/`.

```bash
DLINOSS_REPO=<dir> S4_REPO=<dir> MAMBA_SSM_DIR=<dir> MAMBA_REPO=<dir> python -m ews.conversion.verify
pytest tests/test_conversion.py
```

`trained_row.py` checks a trained LinOSS-IM Heartbeat checkpoint on all five seeds: the test accuracy is identical
before and after conversion and the logits agree to 1e-13 (it needs `LINOSS_HEARTBEAT_CKPT` and `LNCDE_DATA_DIR`).
