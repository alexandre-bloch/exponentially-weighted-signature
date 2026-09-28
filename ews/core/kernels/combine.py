"""The combine engines behind `EWS(combine=...)`: the stage after the leaf, the modified Chen recursion.

    "assoc"         (default) the associative scan of ews.core.chen_scan; the terminal readout follows
                    `terminal_reduce`.
    "lean"          the same Chen operator on precomputed flows M_t = exp(-dtheta_t A), so no exponential is taken
                    inside the scan; any A and depth, f32 or f64; the fastest for dense A.
    "block_kernel"  a fused Pallas/Triton kernel for block-diagonal A, padded exactly to a power-of-two layout, with a
                    custom VJP; the fastest for block-diagonal A (1.4x at m 32 to 9x at m 128 over "lean", with 3-10x
                    less memory). Tested for blocks of 2 and 4 at depths 1-2. GPU only.
    "dense_kernel"  a buffered Pallas/Triton kernel for any A (f32, m >= 16, prefix only): slower than "lean", with the
                    smallest memory footprint. GPU only.

Every combine takes the local signatures level by level and returns the packed layout of ews.core.chen_scan, so
everything after the combine is the same.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ews.core.chen_scan import _chen_M_per_interval
from ews.core.expm_approx import expm_matrix
from ews.core.kernels import pallas_combine as pc

COMBINE_KINDS = ("assoc", "lean", "block_kernel", "dense_kernel")
ROTATION_CLASSES = (
    "AParamBoundedRotation",
    "AParamDampedRotation",
    "AParamReluDampedRotation",
    "AParamClampedRotation",
    "AParamUnboundedRotation",
)


def block_sizes_of(a_param):
    """Block structure of an A parametrisation, decided from its class (static, known at build):
    a tuple of diagonal-block sizes, or None when A is dense. Rotation-plane parametrisations are
    2x2 blocks [[d, w], [-w, d]]; ``AParamUnconstrainedRotation`` / ``AParamJordanRepeated`` / ``AParamBlockDiag`` carry their own sizes;
    ``AParamStableBlock`` is 2x2 block-diagonal with a shear, so it is blocked but not rotation planes."""
    from ews import params as P

    if isinstance(a_param, P.AParamBlockDiag):
        return tuple(int(s) for s in a_param.sizes)
    if isinstance(a_param, (P.AParamUnconstrainedRotation, P.AParamJordanRepeated)):
        return tuple(int(s) for s in a_param.block_sizes)
    if isinstance(a_param, P.AParamStableBlock):
        # 2x2 blocks, but not canonical rotation planes: the shear breaks the [[d, w], [-w, d]] form, so this is
        # block-diagonal for the combine and the blockwise series while `is_rotation_planes` stays False and the
        # closed-form plane leaf correctly refuses it.
        return (2,) * (int(a_param.dim) // 2)
    if type(a_param).__name__ in ROTATION_CLASSES:
        d = int(a_param.dim) if hasattr(a_param, "dim") else int(a_param().shape[0])
        if d % 2:
            return None
        return (2,) * (d // 2)
    return None


def is_rotation_planes(a_param):
    """True when A() is block-diagonal with canonical 2x2 planes [[d, w], [-w, d]] (the closed-form leaf applies)."""
    return type(a_param).__name__ in ROTATION_CLASSES


def pick_chunk(L, target=1024):
    """Largest divisor of L that is <= target (falls back to L): the kernels' chunk length."""
    best = 1
    for c in range(1, min(L, target) + 1):
        if L % c == 0:
            best = c
    return best if best > 1 else L


def pad_block_problem(A, levels, sizes):
    """Pad a block-diagonal problem to Kp = 2^ceil(log2 K) blocks of the uniform size b = max(sizes).

    Padding is exact: the padded coordinates carry zero increments, so their local terms are zero
    and their (identity) flows mix nothing. Returns (blocks (Kp, b, b), padded flat levels
    [(L, mp^k)], keep (padded index of every real coordinate), mp)."""
    L = levels[0].shape[0]
    m = int(sum(sizes))
    K, b = len(sizes), int(max(sizes))
    Kp = 1 << (K - 1).bit_length()
    mp = Kp * b
    offs = np.cumsum([0] + list(sizes))
    keep = np.concatenate([j * b + np.arange(sz) for j, sz in enumerate(sizes)])
    blocks = jnp.zeros((Kp, b, b), A.dtype)
    for j, sz in enumerate(sizes):
        blocks = blocks.at[j, :sz, :sz].set(A[offs[j] : offs[j] + sz, offs[j] : offs[j] + sz])
    padded = []
    for k, lv in enumerate(levels, 1):
        if mp == m:
            padded.append(lv.reshape(L, m**k))
            continue
        t = lv.reshape((L,) + (m,) * k)
        tp = jnp.zeros((L,) + (mp,) * k, lv.dtype)
        tp = tp.at[(slice(None),) + np.ix_(*([keep] * k))].set(t)
        padded.append(tp.reshape(L, mp**k))
    return blocks, padded, keep, mp


def unpad_packed(packed, keep, mp, m, N):
    """(L, sum mp^k) padded packed levels -> (L, sum m^k) on the real coordinates (also for a single (sum mp^k,) row)."""
    if mp == m:
        return packed
    single = packed.ndim == 1
    if single:
        packed = packed[None]
    L = packed.shape[0]
    outs, start = [], 0
    for k in range(1, N + 1):
        t = packed[:, start : start + mp**k].reshape((L,) + (mp,) * k)
        outs.append(t[(slice(None),) + np.ix_(*([keep] * k))].reshape(L, m**k))
        start += mp**k
    out = jnp.concatenate(outs, axis=1)
    return out[0] if single else out


def block_operators(blocks, dtheta, expm_method, cached=False, basepoint=False):
    """Per-interval flows of the diagonal blocks: E[t, j] = exp(-dtheta_t A_j), shape (L, Kp, b, b).

    cached / basepoint: the uniform-clock fast path, the same contract as `chen_scan._chen_M_per_interval`; every
    interval has the same dtheta, so one exponential per block is computed and tiled over the intervals, with the
    identity for a dtheta = 0 basepoint interval (the first real dtheta is then dtheta[1]). The flows are the same
    arrays the per-interval path computes (forward bit-identical, checked on three weather models); only the
    gradient summation order differs (1e-6 relative).
    """
    if not cached:
        return jax.vmap(lambda h: jax.vmap(lambda Bj: expm_matrix(-h * Bj, expm_method))(blocks))(
            dtheta
        )
    L = dtheta.shape[0]
    h = dtheta[1] if basepoint else dtheta[0]
    E1 = jax.vmap(lambda Bj: expm_matrix(-h * Bj, expm_method))(blocks)  # (Kp, b, b)
    E = jnp.broadcast_to(E1[None], (L,) + E1.shape)
    if basepoint:
        eye = jnp.broadcast_to(jnp.eye(blocks.shape[-1], dtype=blocks.dtype), E1.shape)
        E = jnp.concatenate([eye[None], E[1:]], axis=0)
    return E


def jordan_block_operators(blocks, dtheta, chain):
    """Per-interval flows exp(-dtheta_t J_j) of uniform Jordan chains, in closed form: shape (L, Kp, b, b).

    A chain is J = D + eps Nil with D and Nil commuting and Nil^n = 0, so

        exp(-h J) = exp(-h D) sum_{p < n} (-h eps)^p / p! Nil^p          (exact for any h)

      real chain, size n       D = lam I:  E[i, i+p] = e^{-h lam} (-h eps)^p / p!
      complex chain, size 2k   D = diag(C, ..., C) with C = [[a, -b], [b, a]] and
                               exp(-h C) = e^{-h a} [[cos hb, sin hb], [-sin hb, cos hb]]:
                               the 2x2 block (i, i+p) is (-h eps)^p / p! exp(-h C)

    lam, a, b and eps are read from the blocks (a padded zero block gives E = I), so the layer's a_scale is
    included. Under a clock whose increments all differ this replaces L x Kp matrix exponentials, which is what
    makes a gated clock affordable on long paths. chain: (kind, size) from
    ``ews.core.kernels.jordan_engine.jordan_chains_of``.
    """
    kind, size = chain
    h = dtheta[:, None]  # (L, 1)
    if kind == "real":
        n = size
        lam, eps = blocks[:, 0, 0], blocks[:, 0, 1]
        coef = jnp.exp(-h * lam[None])  # (L, Kp): the p = 0 term
        x = -h * eps[None]
        E = jnp.zeros(dtheta.shape + blocks.shape, blocks.dtype)
        for p in range(n):
            if p:
                coef = coef * x / p  # e^{-h lam} x^p / p!, with no 0 ** 0 in the gradient
            E = E + coef[..., None, None] * jnp.asarray(np.eye(n, k=p), blocks.dtype)
        return E
    k = size // 2
    a, b = blocks[:, 0, 0], blocks[:, 1, 0]
    eps = blocks[:, 0, 2] if k > 1 else jnp.zeros_like(a)
    ea = jnp.exp(-h * a[None])  # (L, Kp)
    c, s = jnp.cos(h * b[None]), jnp.sin(h * b[None])
    R = ea[..., None, None] * jnp.stack(
        [jnp.stack([c, s], -1), jnp.stack([-s, c], -1)], -2
    )  # (L, Kp, 2, 2)
    E = jnp.zeros(dtheta.shape + blocks.shape, blocks.dtype)
    x = -h * eps[None]
    coef = jnp.ones_like(x)  # (L, Kp): x^p / p!, built by products
    for p in range(k):
        if p:
            coef = coef * x / p
        shift = jnp.asarray(np.eye(k, k=p), blocks.dtype)  # chain positions i -> i + p
        blk = jnp.einsum("ij,...ab->...iajb", shift, coef[..., None, None] * R)
        E = E + blk.reshape(dtheta.shape + blocks.shape)
    return E


def combine_packed(
    kind,
    local_signatures,
    A,
    interval_times,
    N,
    chen_expm="exact",
    cached=False,
    basepoint=False,
    *,
    terminal=False,
    block_sizes=None,
    chunk_target=1024,
    tile=4,
    jordan_chain=None,
):
    """Combine per-interval local signatures with the selected engine.

    Args:
        kind: one of COMBINE_KINDS except "assoc" (that is the layer's own path).
        local_signatures: level k of shape (L, d, ..., d) (the leaf output).
        A: (d, d) effective drift. interval_times: (L,) clock increments dtheta.
        chen_expm / cached / basepoint: exactly the original scan's operator options
            (M_t = expm(-dtheta_t A) by ``chen_expm``; one shared exponential for uniform clocks).
        terminal: return the final signature (sum d^k,) instead of the prefix stream (L, sum d^k).
        block_sizes: diagonal-block sizes of A (required by "block_kernel"; see ``block_sizes_of``).
        chunk_target / tile: kernel chunk length (largest divisor of L below it) and block tile.
        jordan_chain: (kind, size) of A's uniform Jordan chains: "block_kernel" then builds its flows in closed
            form (``jordan_block_operators``) instead of by matrix exponentials. None (default) keeps the
            exponentials.
    """
    if kind not in COMBINE_KINDS or kind == "assoc":
        raise ValueError(f"combine kind must be one of {COMBINE_KINDS[1:]} here, got {kind!r}")
    if jordan_chain is not None and kind != "block_kernel":
        raise ValueError(f"jordan_chain applies to combine='block_kernel' only, got {kind!r}")
    L = local_signatures[0].shape[0]
    d = local_signatures[0].shape[1]
    levels = [lv.reshape(L, d**k) for k, lv in enumerate(local_signatures, 1)]
    dtheta = jnp.asarray(interval_times, dtype=A.dtype)

    if kind == "block_kernel":
        if block_sizes is None:
            raise ValueError(
                "combine='block_kernel' needs a block-diagonal A (rotation planes, AParamUnconstrainedRotation, AParamJordanRepeated or AParamBlockDiag)"
            )
        if int(sum(block_sizes)) != d:
            raise ValueError(f"block sizes {block_sizes} do not sum to d={d}")
        blocks, padded, keep, mp = pad_block_problem(A, levels, block_sizes)
        if jordan_chain is None:
            E = block_operators(blocks, dtheta, chen_expm, cached, basepoint)  # (L, Kp, b, b)
        else:
            if set(block_sizes) != {int(jordan_chain[1])}:
                raise ValueError(
                    f"block sizes {block_sizes} are not chains of size {jordan_chain[1]}"
                )
            E = jordan_block_operators(blocks, dtheta, jordan_chain)
        Kp = E.shape[1]
        chunk = pick_chunk(L, chunk_target)
        t = min(int(tile), Kp)
        if terminal:
            out = pc.block_terminal_vjp(E[None], [lv[None] for lv in padded], tile=t, chunk=chunk)[
                0
            ]
        else:
            out = pc.block_prefix_vjp(E[None], [lv[None] for lv in padded], tile=t, chunk=chunk)[0]
        return unpad_packed(out, keep, mp, d, N)

    # dense operators: bit-for-bit the original scan's M_t (same expm method, same uniform-clock caching)
    M = _chen_M_per_interval(dtheta, A, chen_expm, cached, basepoint)  # (L, d, d)
    if kind == "lean":
        if terminal:
            return pc.dense_tree_lean(M[None], [lv[None] for lv in levels])[0]
        return pc.dense_assoc_general_ref(M[None], [lv[None] for lv in levels])[0]
    # dense_kernel: buffered any-depth kernel with a custom VJP (prefix only; the terminal is its last row)
    chunk = pick_chunk(L, chunk_target)
    out = pc.dense_general_prefix_vjp(M[None], [lv[None] for lv in levels], chunk=chunk)[0]
    return out[-1] if terminal else out
