# Vendored from data_dir/generate_coeffs.py of github.com/Benjamin-Walker/log-neural-cdes, in the version used with
# signax 0.2.1 (tk-rusch/linoss's copy also imports diffrax; the function is the same).
# Vendored for the trained-checkpoint check (trained_row.py). Modifications: NONE.
"""
This module contains a function for generating the coefficients for a Hermite cubic spline with backwards differences.
"""

import jax
import jax.numpy as jnp


def calc_coeffs(data, include_time, T):
    import diffrax  # lazy: Hermite coeffs are only for the NCDE data path, not linear_ncde (log-sigs)
    if include_time:
        ts = data[:, :, 0]
    else:
        ts = (T / data.shape[1]) * jnp.repeat(
            jnp.arange(data.shape[1])[None, :], data.shape[0], axis=0
        )
    coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(ts, data)
    return coeffs
