"""
Series leaf: the closed-form alternative to the Van Loan leaf.

Unlike Van Loan, this produces the per-interval signature directly: there is no separate
signax step (ews/core/signature.py), because the level tensors are assembled from the series
letters. Van Loan produces knots which signature.py then converts.

For one linear segment with increment ΔX and clock span Δθ, the depth-k term
of the interval EWS (apex at the right endpoint) is the exact series

    S^(k) = Σ_{m ∈ Z_{>=0}^k}  c_m · b_{m1} ⊗ ... ⊗ b_{mk},

    letters       b_p = ((−Δθ)^p / p!) · A^p ΔX        (b_0 = ΔX),
    coefficients  c_m = Π_{j=1..k} (M_j + k − j + 1)^{-1},   M_j = m_j+...+m_k

(simplex integrals of the monomials w^{m_j}; derived by expanding e^{−Δθ w A}
under the ordered integral). Truncated at total order |m| ≤ q, the error at
level k is bounded by (k·x)^{q+1}/(q+1)! · e^{k·x} with x = ‖A‖_op·Δθ:
factorial decay in q, versus the Van Loan leaf's O(1/M²).

It needs no sub-knots, no per-interval expm and no matrix inverse, and the Chen combine downstream is unchanged.

Validity regime: the series needs x = ‖A‖_op·Δθ_i ≲ 1 per interval, and A is
learned: use `series_regime` as a runtime diagnostic.
"""

import string
from fractions import Fraction
from functools import lru_cache
from itertools import product as _iproduct

import jax

# Match the convention in the sibling ews/core modules (van_loan.py:20,
# chen_scan.py:9, signature.py:10): x64 is enabled at import. Several modules depend on
# this firing as an import side effect.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

# ------------------------------------------------------------- coefficients


def coeff_exact(m) -> Fraction:
    """c_m as an exact rational, from the closed form."""
    k = len(m)
    out = Fraction(1)
    for j in range(1, k + 1):
        M_j = sum(m[j - 1 :])
        out /= M_j + k - j + 1
    return out


@lru_cache(maxsize=None)
def series_coeffs(depth: int, q: int):
    """[C^1, ..., C^depth]: level-k coefficient tensor of shape (q+1,)*k,
    C^k[m] = c_m if |m| <= q else 0. Exact rationals cast to float64."""
    out = []
    for k in range(1, depth + 1):
        C = np.zeros((q + 1,) * k, dtype=np.float64)
        for m in _iproduct(range(q + 1), repeat=k):
            if sum(m) <= q:
                C[m] = float(coeff_exact(m))
        out.append(C)
    return tuple(out)


# ------------------------------------------------------------------ letters


def series_letters(delta_x, delta_theta, A, q: int):
    """(N, q+1, d) letters b_p per interval via the stable recursion
    b_0 = ΔX;  b_p = (−Δθ/p) · A b_{p−1}."""
    A = jnp.asarray(A, dtype=delta_x.dtype)

    def per_interval(dx, dth):
        def step(b_prev, p):
            b = (-dth / p) * (A @ b_prev)
            return b, b

        ps = jnp.arange(1, q + 1, dtype=delta_x.dtype)
        _, bs = jax.lax.scan(step, dx, ps)
        return jnp.concatenate([dx[None], bs], axis=0)  # (q+1, d)

    return jax.vmap(per_interval)(delta_x, delta_theta)


# ----------------------------------------------------------------- assembly


def _level_einsum(k: int) -> str:
    """e.g. k=2 -> 'pq,npi,nqj->nij' (coefficient axes p..; letter axes i..)."""
    coeff_idx = string.ascii_lowercase[:k]  # p-like axes (reuse a..)
    out_idx = string.ascii_uppercase[:k]  # tensor axes
    factors = [f"n{coeff_idx[j]}{out_idx[j]}" for j in range(k)]
    return coeff_idx + "," + ",".join(factors) + "->n" + out_idx


def assemble_from_letters(b, depth: int, q: int, dtype=jnp.float64):
    """Per-interval EWS level tensors from letters b (N, q+1, d):
    level k = Σ_m c_m · b_{m1} ⊗ ... ⊗ b_{mk}. Used both by the standalone
    series pipelines and by the M1×M2 composition (which twists the letters
    by the block suffix factors before assembly; valid because those factors
    are functions of A and commute with it, so G·b_p is exactly the letter
    of the G-twisted increment)."""
    coeffs = series_coeffs(depth, q)
    levels = []
    for k in range(1, depth + 1):
        C = jnp.asarray(coeffs[k - 1], dtype=dtype)
        operands = [C] + [b] * k
        levels.append(jnp.einsum(_level_einsum(k), *operands))
    return levels


def series_local_signatures(path, clock, A, depth: int, q: int, dtype=jnp.float64):
    """Per-interval EWS signatures as a level list [(N,d), (N,d,d), ...]:
    the drop-in replacement for the Van Loan leaf + per-interval signax."""
    path = jnp.asarray(path, dtype=dtype)
    clock = jnp.asarray(clock, dtype=dtype)
    A = jnp.asarray(A, dtype=dtype)

    delta_x = path[1:] - path[:-1]
    delta_theta = clock[1:] - clock[:-1]
    b = series_letters(delta_x, delta_theta, A, q)  # (N, q+1, d)
    return assemble_from_letters(b, depth, q, dtype)


def series_regime(A, clock):
    """max_i ‖A‖_op · Δθ_i: the series' convergence parameter x. Keep ≲ 0.5
    (error at level k ~ (k·x)^{q+1}/(q+1)!·e^{kx}); A is learned, so check
    periodically during training, not just at build time."""
    A = jnp.asarray(A)
    clock = jnp.asarray(clock, dtype=A.dtype)
    op = jnp.linalg.norm(A, 2)
    return float(op * jnp.max(clock[1:] - clock[:-1]))


def series_error_bound(x: float, k: int, q: int) -> float:
    """Per-interval relative error bound at level k."""
    import math

    return (k * x) ** (q + 1) / math.factorial(q + 1) * math.exp(k * x)
