"""
Batched signature computation for reweighted paths using signax.

This module provides JIT-compiled, batched signature computation for
reweighted paths produced by the Van Loan method.
"""

import jax

jax.config.update("jax_enable_x64", True)

from functools import partial

import jax.numpy as jnp
import signax
from jax import vmap


@partial(jax.jit, static_argnums=(1, 2))
def compute_reweighted_signatures(reweighted_paths, N, dtype=jnp.float64):
    """Signatures of all the reweighted paths, vmapped.

    Args:
        reweighted_paths: (n_intervals, M+1, d). N: the truncation depth (static). dtype: the compute dtype.

    Returns a tuple of N arrays, level k of shape (n_intervals,) + (d,)*k (signax omits level 0, as the EWS does here).
    """
    reweighted_paths = jnp.asarray(reweighted_paths, dtype=dtype)

    compute_sig_vmapped = vmap(
        lambda path: signax.signature(path, depth=N, stream=False, flatten=False), in_axes=0
    )

    signatures_batched = compute_sig_vmapped(reweighted_paths)

    return tuple(signatures_batched)
