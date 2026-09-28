# Vendored from data_dir/generate_paths.py of github.com/Benjamin-Walker/log-neural-cdes, with its log-signature
# call moved to the signax 0.2.1 API (tk-rusch/linoss's copy uses the older signax<=0.1 API).
# Vendored for the trained-checkpoint check (trained_row.py).
# Modifications: ONE import line -- from data_dir.hall_set import HallSet -> from .hall_set import HallSet
"""
This module contains functions for generating log-signature of paths over intervals of length stepsize.
"""

import jax
import jax.numpy as jnp
import signax

from .hall_set import HallSet


def hall_basis_logsig(x, depth, t2l):
    # signax 0.2.x port: the native `logsignature` reproduces the original
    #   flatten(log(signature(x, depth)))  [depth 1]  and
    #   t2l[:, 1:] @ flatten(log(signature(x, depth)))  [depth 2, Hall-basis projection]
    # exactly (verified max|Δ|~1.5e-8), once the level-0 constant is prepended as before.
    # `t2l` retained in the signature for call-site compatibility (now unused).
    return jnp.concatenate((jnp.array([0.0]), signax.logsignature(x, depth)))


def calc_paths(data, stepsize, depth):
    """
    Generate log-signature objects from data.

    In the future, this function will use RoughPy, and return path objects,
    which can be queried over any interval for the log-signature. Right now,
    it is necessary to specify the stepsize and depth ahead of time.
    """
    data = jnp.concatenate(
        (jnp.zeros((data.shape[0], 1, data.shape[-1])), data), axis=1
    )

    if depth == 2:
        hs = HallSet(data.shape[-1], depth)
        t2l = hs.t2l_matrix(depth)
    else:
        t2l = None

    prepend = lambda x: jnp.concatenate(
        (
            jnp.concatenate((jnp.zeros((1, data.shape[-1])), x[:-1, -1, :]))[
                :, None, :
            ],
            x,
        ),
        axis=1,
    )

    if stepsize > data.shape[1]:
        stepsize = data.shape[1]

    if data.shape[1] % stepsize != 0:
        final_data = data[:, -(data.shape[1] % stepsize) - 1 :, ...]
        data = data[:, : -(data.shape[1] % stepsize), ...].reshape(
            data.shape[0], -1, stepsize, data.shape[-1]
        )
        data = jax.vmap(prepend)(data)
        final_data = prepend(final_data)
    else:
        data = data.reshape(data.shape[0], -1, stepsize, data.shape[-1])
        data = jax.vmap(prepend)(data)
        final_data = None

    vmap_calc_logsig = jax.vmap(hall_basis_logsig, in_axes=(0, None, None))
    logsigs = jax.vmap(vmap_calc_logsig, in_axes=(0, None, None))(data, depth, t2l)
    if final_data is not None:
        final_logsigs = vmap_calc_logsig(final_data, depth, t2l)[:, None, :]
        logsigs = jnp.concatenate(
            (
                logsigs,
                final_logsigs,
            ),
            axis=1,
        )

    return logsigs
