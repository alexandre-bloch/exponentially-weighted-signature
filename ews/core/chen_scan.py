"""
Chen composition using packed signatures with M precomputation.

Minimal implementation retained for the clean pipeline.
"""

import jax

jax.config.update("jax_enable_x64", True)

import math
from functools import partial

import jax.numpy as jnp
import jax.scipy.linalg
import signax

from ews.core.expm_approx import expm_matrix

# -----------------------------
# Packed scan (flat vectors)
# -----------------------------


def _level_sizes(dim, trunc):
    return [dim**k for k in range(1, trunc + 1)]


def _level_shapes(dim, trunc):
    return [tuple([dim] * k) for k in range(1, trunc + 1)]


@partial(jax.jit, static_argnums=(1, 2))
def pack_signatures(signature_list, dim, trunc):
    """
    Pack a signature list into a flat vector per interval.

    Returns:
        packed: array of shape (n_intervals, sum_{k=1..N} d^k)
    """
    flat_levels = []
    for k in range(1, trunc + 1):
        level = signature_list[k - 1]
        n = level.shape[0]
        flat_levels.append(jnp.reshape(level, (n, dim**k)))
    return jnp.concatenate(flat_levels, axis=1)


@partial(jax.jit, static_argnums=(1, 2))
def pack_signature_vector(signature_list, dim, trunc):
    """
    Pack a single signature list into a flat vector.

    Returns:
        packed_vec: array of shape (sum_{k=1..N} d^k,)
    """
    flat_levels = []
    for k in range(1, trunc + 1):
        level = signature_list[k - 1]
        flat_levels.append(jnp.reshape(level, (dim**k,)))
    return jnp.concatenate(flat_levels, axis=0)


@partial(jax.jit, static_argnums=(1, 2))
def unpack_signature_vector(packed_vec, dim, trunc):
    """
    Unpack a single packed signature vector into a list of tensors.
    """
    sizes = _level_sizes(dim, trunc)
    shapes = _level_shapes(dim, trunc)
    out = []
    start = 0
    for size, shape in zip(sizes, shapes):
        end = start + size
        level_flat = packed_vec[start:end]
        out.append(jnp.reshape(level_flat, shape))
        start = end
    return out


@partial(jax.jit, static_argnums=(1, 2))
def unpack_signature_batch(packed, dim, trunc):
    """
    Unpack packed signatures for all intervals into list-of-tensors format.
    """
    sizes = _level_sizes(dim, trunc)
    shapes = _level_shapes(dim, trunc)
    out = []
    start = 0
    for size, shape in zip(sizes, shapes):
        end = start + size
        level_flat = packed[:, start:end]
        out.append(jnp.reshape(level_flat, (packed.shape[0],) + shape))
        start = end
    return out


def _einsum_apply_M(M, tensor):
    """Apply matrix M to each mode of a tensor using einsum."""
    L = tensor.ndim
    letters = "abcdefghijklmnopqrstuvwxyz"
    if 2 * L > len(letters):
        raise ValueError("Tensor rank too large for einsum indices.")

    out_idx = letters[:L]
    in_idx = letters[L : 2 * L]
    mats = [f"{out_idx[k]}{in_idx[k]}" for k in range(L)]
    eq = ",".join(mats + ["".join(in_idx)]) + "->" + "".join(out_idx)

    operands = [M] * L + [tensor]
    return jnp.einsum(eq, *operands)


@partial(jax.jit, static_argnums=(2, 3))
def _apply_M_to_signature(signature_list, M, dim, trunc):
    """Apply precomputed matrix M to each level of a signature list."""
    result_list = []
    for L_idx in range(len(signature_list)):
        tensor = signature_list[L_idx]
        result_list.append(_einsum_apply_M(M, tensor))
    return result_list


@partial(jax.jit, static_argnums=(3, 4))
def chen_binary_op_packed_with_M(left_pair, right_pair, A, dim, trunc):
    """
    Chen operator on packed signatures carrying M = exp(-h A) in the state.

    Args:
        left_pair: (packed_left, h_left, M_left)
        right_pair: (packed_right, h_right, M_right)
        A: drift matrix
        dim: dimension
        trunc: truncation level
    Returns:
        (packed_result, h_result, M_result)
    """
    packed_L, h_L, M_L = left_pair
    packed_R, h_R, M_R = right_pair

    S_L = unpack_signature_vector(packed_L, dim, trunc)
    S_R = unpack_signature_vector(packed_R, dim, trunc)

    # Apply flow using precomputed M_R (no expm here)
    S_L_flowed = _apply_M_to_signature(S_L, M_R, dim, trunc)
    S_result = signax.tensor_ops.mult(S_L_flowed, S_R)

    packed_result = pack_signature_vector(S_result, dim, trunc)
    h_result = h_L + h_R
    M_result = M_L @ M_R

    return packed_result, h_result, M_result


def _chen_M_per_interval(interval_times, A, expm_method, cached, basepoint):
    """Per-interval M = exp(-hA): one expm/interval (cached=False); a single shared expm tiled
    across uniform Δt (cached=True); or shared-expm with M[0]=I for a Δt=0 basepoint."""
    if not cached:
        return jax.vmap(lambda h: expm_matrix(-h * A, expm_method))(interval_times)
    if basepoint:
        n = interval_times.shape[0]
        M_single = expm_matrix(-interval_times[1] * A, expm_method)  # first real Δθ
        I_d = jnp.eye(A.shape[0], dtype=A.dtype)  # interval 0: exp(-0·A)=I
        return jnp.concatenate(
            [I_d[None, :, :], jnp.tile(M_single[None, :, :], (n - 1, 1, 1))], axis=0
        )
    M_single = expm_matrix(-interval_times[0] * A, expm_method)
    return jnp.tile(M_single[None, :, :], (interval_times.shape[0], 1, 1))


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def chen_associative_scan_prefix_packed_with_M(
    local_signatures,
    interval_times,
    A,
    dim,
    trunc,
    expm_method="exact",
    cached=False,
    basepoint=False,
):
    """
    Packed prefix associative scan carrying M = exp(-h A) in state.

    interval_times are clock increments (Δθ) per interval.

    Returns:
        packed_prefix: (n_intervals, sum d^k)
        cumulative_times: (n_intervals,)
    """

    packed = pack_signatures(local_signatures, dim, trunc)
    interval_times = jnp.asarray(interval_times, dtype=A.dtype)

    M_per_interval = _chen_M_per_interval(interval_times, A, expm_method, cached, basepoint)

    def op(left, right):
        left_vecs, left_times, left_M = left
        right_vecs, right_times, right_M = right

        def single(left_vec, left_time, left_M, right_vec, right_time, right_M):
            return chen_binary_op_packed_with_M(
                (left_vec, left_time, left_M), (right_vec, right_time, right_M), A, dim, trunc
            )

        out_vecs, out_times, out_M = jax.vmap(single, in_axes=(0, 0, 0, 0, 0, 0))(
            left_vecs, left_times, left_M, right_vecs, right_times, right_M
        )
        return out_vecs, out_times, out_M

    packed_prefix, prefix_times, _ = jax.lax.associative_scan(
        op, (packed, interval_times, M_per_interval)
    )
    return packed_prefix, prefix_times


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def chen_associative_scan_final_packed_with_M(
    local_signatures,
    interval_times,
    A,
    dim,
    trunc,
    expm_method="exact",
    cached=False,
    basepoint=False,
):
    """
    Packed final (global) signature using packed+M scan.

    interval_times are clock increments (Δθ) per interval. cached/basepoint select the
    uniform-Δt fast path (single shared expm; M[0]=I for a Δt=0 basepoint).

    Returns:
        packed_final: (sum d^k,)
        total_time: scalar
    """
    packed_prefix, prefix_times = chen_associative_scan_prefix_packed_with_M(
        local_signatures, interval_times, A, dim, trunc, expm_method, cached, basepoint
    )
    return packed_prefix[-1], prefix_times[-1]


def _pad_to_pow2(packed, times, M_mats, dim):
    n = packed.shape[0]
    if n <= 1:
        return packed, times, M_mats
    target = 1 << int(math.ceil(math.log2(n)))
    if target == n:
        return packed, times, M_mats

    pad = target - n
    packed_pad = jnp.zeros((pad, packed.shape[1]), dtype=packed.dtype)
    times_pad = jnp.zeros((pad,), dtype=times.dtype)
    eye = jnp.eye(dim, dtype=M_mats.dtype)
    M_pad = jnp.tile(eye[None, :, :], (pad, 1, 1))

    packed = jnp.concatenate([packed, packed_pad], axis=0)
    times = jnp.concatenate([times, times_pad], axis=0)
    M_mats = jnp.concatenate([M_mats, M_pad], axis=0)
    return packed, times, M_mats


def _tree_reduce_padded(packed, times, M_mats, A, dim, trunc):
    n = packed.shape[0]
    while n > 1:
        packed_2 = packed.reshape((n // 2, 2, packed.shape[1]))
        times_2 = times.reshape((n // 2, 2))
        M_2 = M_mats.reshape((n // 2, 2, dim, dim))

        def combine(p_vecs, p_times, p_Ms):
            return chen_binary_op_packed_with_M(
                (p_vecs[0], p_times[0], p_Ms[0]),
                (p_vecs[1], p_times[1], p_Ms[1]),
                A,
                dim,
                trunc,
            )

        packed, times, M_mats = jax.vmap(combine)(packed_2, times_2, M_2)
        n = packed.shape[0]

    return packed[0], times[0], M_mats[0]


@partial(jax.jit, static_argnums=(3, 4, 5))
def chen_tree_reduce_final_packed_with_M(
    local_signatures, interval_times, A, dim, trunc, expm_method="exact"
):
    """
    Packed final (global) signature using tree reduction and expm(-h A).
    """
    interval_times = jnp.asarray(interval_times, dtype=A.dtype)
    M_per_interval = jax.vmap(lambda h: expm_matrix(-h * A, expm_method))(interval_times)
    packed = pack_signatures(local_signatures, dim, trunc)
    packed, times, M_mats = _pad_to_pow2(packed, interval_times, M_per_interval, dim)
    packed_final, total_time, _ = _tree_reduce_padded(packed, times, M_mats, A, dim, trunc)
    return packed_final, total_time
