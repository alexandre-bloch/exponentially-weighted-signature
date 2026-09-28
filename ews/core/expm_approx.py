"""
Exponential matrix computation helpers.

Supports exact expm and low-order approximations used in benchmarks.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _approx_expm_matrix(X: jnp.ndarray, method: str) -> jnp.ndarray:
    """Approximate expm(X) with a low-order scheme."""
    eye = jnp.eye(X.shape[0], dtype=X.dtype)

    if method == "euler":
        return eye + X

    if method == "cayley":
        left = eye - 0.5 * X
        right = eye + 0.5 * X
        return jnp.linalg.solve(left, right)

    if method == "pade":
        X2 = X @ X
        num = eye + 0.5 * X + (1.0 / 12.0) * X2
        den = eye - 0.5 * X + (1.0 / 12.0) * X2
        return jnp.linalg.solve(den, num)

    raise ValueError(f"Unknown expm method: {method}")


def expm_matrix(X: jnp.ndarray, method: str) -> jnp.ndarray:
    """Compute expm(X) with the selected method."""
    method = method.lower()
    if method in {"pade_lite", "pade-lite"}:
        method = "pade"
    if method == "exact":
        return jax.scipy.linalg.expm(X)
    return _approx_expm_matrix(X, method)
