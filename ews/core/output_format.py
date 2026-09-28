"""
Utilities for EWS output format: list-of-tensors <-> flat vectors.
"""

import jax.numpy as jnp
import signax
from jax import flatten_util

from ews.core.chen_scan import pack_signatures


def flatten_signature_list(signature_list, include_level0: bool = True):
    """
    Flatten a global signature list into a 1D vector.
    """
    flat, _ = flatten_util.ravel_pytree(signature_list)
    if not include_level0:
        return flat
    return jnp.concatenate([jnp.ones((1,), dtype=flat.dtype), flat], axis=0)


def flatten_signature_prefix(signature_list, dim, depth, include_level0: bool = True):
    """
    Flatten a prefix signature list into (n_intervals, D).
    """
    flat = pack_signatures(signature_list, dim, depth)
    if not include_level0:
        return flat
    ones = jnp.ones((flat.shape[0], 1), dtype=flat.dtype)
    return jnp.concatenate([ones, flat], axis=1)


def unflatten_signature_vector(flat, dim, depth):
    """
    Unflatten a 1D signature vector into list-of-tensors.
    """
    return signax.utils.unravel_signature(flat, dim, depth)
