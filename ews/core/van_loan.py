"""Van Loan reweighted-path computation.

Each path increment is reweighted by the learnable drift ``A`` using the augmented-matrix
("Van Loan") trick: ``expm([[A·δ, ΔX/M], [0, 0]])`` whose top-right block yields the
reweighted signature increments. One function covers every mode via flags:

    reweighted_segments_van_loan(path, interval_times, A, M, ...,
                                 uniform=False, basepoint=False)

    uniform=False                  displacement form: one expm per interval (any Δt, Δt=0 safe)
    uniform=True                   fast path: a single shared expm reused across uniform intervals
    uniform=True, basepoint=True   as uniform, but interval 0 is a Δt=0 basepoint (pure jump)

``reweighted_segment_van_loan`` is the single-interval displacement kernel (vmapped by the
displacement path and reused by incremental prediction).
"""

import jax

jax.config.update("jax_enable_x64", True)

from functools import partial

import jax.numpy as jnp
from jax import vmap

from ews.core.expm_approx import expm_matrix


@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def reweighted_segment_van_loan(delta_x, delta_t, A, M, d, dtype=jnp.float64, expm_method="exact"):
    """M-point reweighted path for one interval [tᵢ, tᵢ₊₁] (displacement form, Δt=0 safe).

    Args:
        delta_x: Displacement ΔX for this interval, shape (d,)
        delta_t: Time interval Δt (scalar)
        A: Drift matrix, shape (d, d)
        M, d: Discretization points / path dimension (static)
        dtype, expm_method: Compute dtype and matrix-exponential method (static)

    Returns:
        reweighted_path: shape (M+1, d)
    """
    delta_x = jnp.asarray(delta_x, dtype=dtype)
    delta_t = jnp.asarray(delta_t, dtype=dtype)
    A = jnp.asarray(A, dtype=dtype)

    delta = delta_t / M
    delta_x_step = delta_x / M

    # expm([[A*delta, delta_x/M], [0, 0]])
    M_step = jnp.zeros((d + 1, d + 1), dtype=dtype)
    M_step = M_step.at[:d, :d].set(A * delta)
    M_step = M_step.at[:d, d].set(delta_x_step)

    E = expm_matrix(M_step, expm_method)

    def compute_powers(E_power, _):
        next_power = E_power @ E
        return next_power, next_power

    E_power_init = jnp.eye(d + 1, dtype=dtype)
    _, E_powers_1_to_M = jax.lax.scan(compute_powers, E_power_init, None, length=M)
    E_powers = jnp.concatenate([E_power_init[None, :, :], E_powers_1_to_M], axis=0)

    K_values = E_powers[: M + 1, :d, d]
    exp_delta_t_A = E_powers[-1, :d, :d]
    W = jnp.linalg.inv(exp_delta_t_A)

    reweighted_path = jax.vmap(lambda k: W @ k)(K_values)
    return reweighted_path


def _displacement(path, interval_times, A, M, dtype, expm_method):
    """Displacement form: one expm per interval (numerically stable at any Δt, incl. 0)."""
    path = jnp.asarray(path, dtype=dtype)
    A = jnp.asarray(A, dtype=dtype)
    d = path.shape[1]

    interval_times = jnp.asarray(interval_times, dtype=dtype)
    delta_t = interval_times
    delta_x = path[1:] - path[:-1]

    compute_segment_vmapped = vmap(
        lambda dx, dt: reweighted_segment_van_loan(dx, dt, A, M, d, dtype, expm_method),
        in_axes=(0, 0),
    )
    return compute_segment_vmapped(delta_x, delta_t)


def _uniform_core(path, interval_times, A, M, dtype, expm_method, eps=1e-12):
    """Uniform-Δt fast path: one shared expm of the block matrix, reused across intervals."""
    path = jnp.asarray(path, dtype=dtype)
    interval_times = jnp.asarray(interval_times, dtype=dtype)
    A = jnp.asarray(A, dtype=dtype)

    d = path.shape[1]
    delta_x = path[1:] - path[:-1]
    delta_t0 = interval_times[0]

    def jump_case(_):
        frac = jnp.arange(M + 1, dtype=dtype) / M
        return frac[None, :, None] * delta_x[:, None, :]

    def normal_case(_):
        delta = delta_t0 / M
        delta_x_step = delta_x / M

        A_delta = A * delta

        # Block matrix -> E_A and F = ∫_0^1 exp(s A_delta) ds in one expm
        eye = jnp.eye(d, dtype=dtype)
        Z = jnp.zeros((d, d), dtype=dtype)
        top = jnp.concatenate([A_delta, eye], axis=1)
        bottom = jnp.concatenate([Z, Z], axis=1)
        M_block = jnp.concatenate([top, bottom], axis=0)

        E_block = expm_matrix(M_block, expm_method)
        E_A = E_block[:d, :d]
        F = E_block[:d, d:]

        K = (F @ delta_x_step.T).T  # (n_intervals, d)

        def compute_powers(E_power, _):
            next_power = E_power @ E_A
            return next_power, next_power

        E_power_init = jnp.eye(d, dtype=dtype)
        _, E_powers_1_to_M = jax.lax.scan(compute_powers, E_power_init, None, length=M)
        E_powers = jnp.concatenate([E_power_init[None, :, :], E_powers_1_to_M], axis=0)

        # S_j = sum_{i=0}^{j-1} E_A^i  (S_0 = 0)
        E_powers_excl_last = E_powers[:-1]
        S_partial = jnp.cumsum(E_powers_excl_last, axis=0)
        S = jnp.concatenate([jnp.zeros((1, d, d), dtype=dtype), S_partial], axis=0)

        K_values = jnp.einsum("mij,tj->tmi", S, K)

        exp_delta_t_A = E_powers[-1]
        W = jnp.linalg.inv(exp_delta_t_A)

        return jnp.einsum("ij,tmj->tmi", W, K_values)

    return jax.lax.cond(delta_t0 > eps, normal_case, jump_case, operand=None)


def _uniform_basepoint(path, interval_times, A, M, dtype, expm_method):
    """Uniform-Δt with a Δt=0 basepoint at interval 0 (pure jump), fast path for the rest."""
    path = jnp.asarray(path, dtype=dtype)
    A = jnp.asarray(A, dtype=dtype)

    # Interval 0: basepoint (Δt=0) -> exp(-0·A)=I, integral reduces to linear interpolation.
    delta_x_0 = path[1] - path[0]
    frac = jnp.arange(M + 1, dtype=dtype) / M
    reweighted_0 = frac[:, None] * delta_x_0[None, :]

    # Intervals 1..n-1: real uniform Δt via the shared-expm fast path.
    reweighted_rest = _uniform_core(path[1:], interval_times[1:], A, M, dtype, expm_method)

    return jnp.concatenate([reweighted_0[None, :, :], reweighted_rest], axis=0)


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def reweighted_segments_van_loan(
    path,
    interval_times,
    A,
    M,
    dtype=jnp.float64,
    expm_method="exact",
    uniform=False,
    basepoint=False,
):
    """Reweighted paths for all intervals. Returns (n_intervals, M+1, d).

    uniform=False -> displacement (one expm/interval);
    uniform=True  -> shared-expm fast path (requires uniform Δt);
    uniform=True, basepoint=True -> interval 0 is a Δt=0 basepoint jump.
    """
    if not uniform:
        return _displacement(path, interval_times, A, M, dtype, expm_method)
    if basepoint:
        return _uniform_basepoint(path, interval_times, A, M, dtype, expm_method)
    return _uniform_core(path, interval_times, A, M, dtype, expm_method)
