"""EWS engine for general block-diagonal A (free n x n blocks, no closed form): the plane engine's structure with
(K, n, n) blocks and batched small matmuls in place of complex scalars.

For each interval (increment v_t, clock step h_t) and block j, E = exp(-h A_j) and G = (1/h) int_0^h exp(-s A_j) ds come
from one Van Loan exponential, exp([[-h A, I], [0, 0]]) = [[E, G], [0, I]], computed by a pluggable `expm_fn`. Level 1
is s_t = E_t s_{t-1} + G_t v_t; the level-2 local term is the series leaf's, block by block, exact to
O((2x)^{Q+1}/(Q+1)!) with x = ||A|| h; the prefix is the plane engine's modified Chen recursion. The output is the EWS
flat layout without level 0.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg

from ews.core.series_leaf import series_coeffs

# ----------------------------------------------------------------- plug-ins
# The engine has two plug-in points (the plane closed form is its 2x2 special case):
#   eg_fn(blocks, dth) -> (E, G): the per-interval exponential exp(-h A_j) and the
#       level-1 integral G_j = (1/h) int_0^h exp(-s A_j) ds, both (L', K, n, n).
#       None = build them from `expm_fn` (Pade or the kernel) via the augmented
#       exponential or the solve (see g_mode). `plane_eg` = the closed form for
#       canonical 2x2 planes [[d, w], [-w, d]] (exact for any h, no expm).
#   local2 = 'series' (blockwise series letters, any blocks) | 'plane' (closed-form
#       double integrals, canonical 2x2 planes only; from ews.core.kernels.plane_engine).


def _planes_from_blocks(blocks, tol=1e-12):
    """(K, 2, 2) canonical planes -> (K, 2) [d, w]; asserts the canonical form."""
    if blocks.shape[-1] != 2:
        raise ValueError("plane plug-ins need 2x2 blocks")
    d, w = blocks[:, 0, 0], blocks[:, 0, 1]
    dev = jnp.max(jnp.abs(blocks[:, 1, 0] + w)) + jnp.max(jnp.abs(blocks[:, 1, 1] - d))
    return jnp.stack([d, w], -1), dev


def plane_eg(blocks, dth):
    """Closed-form (E, G) for canonical planes: E = real form of exp(-h z),
    G = real form of phi(-h z), phi(x) = (e^x - 1)/x, z = d + i w."""
    from ews.core.kernels.plane_engine import phi

    def complex_to_blocks(z):  # (..., K) complex -> (..., K, 2, 2) [[Re, Im], [-Im, Re]]
        re, im = jnp.real(z), jnp.imag(z)
        return jnp.stack([jnp.stack([re, im], -1), jnp.stack([-im, re], -1)], -2)

    dw, _ = _planes_from_blocks(blocks)
    z = jax.lax.complex(dw[:, 0], dw[:, 1])
    x = -dth[:, None] * z[None, :]  # (L', K)
    return complex_to_blocks(jnp.exp(x)), complex_to_blocks(phi(x))


def expm_pade(X):
    """Default exponential stage: jax's batched Pade on a (..., n, n) stack."""
    return jax.scipy.linalg.expm(X)


def scaled_squared(expm_small, X, x_max, thresh=0.5):
    """Scaling-and-squaring around an exponential that is only accurate for
    small arguments (the kernel): exp(X) = exp(X / 2^s)^{2^s}, s from x_max."""
    import math

    s = max(0, int(math.ceil(math.log2(max(x_max, 1e-9) / thresh))))
    Y = expm_small(X / (2.0**s))
    for _ in range(s):
        Y = Y @ Y
    return Y, s


def augmented(blocks, dtheta):
    """(K, n, n) blocks, (L,) steps -> (L, K, 2n, 2n) [[-h A, I], [0, 0]]."""
    K, n, _ = blocks.shape
    L = dtheta.shape[0]
    top = jnp.concatenate(
        [
            -dtheta[:, None, None, None] * blocks[None],
            jnp.broadcast_to(jnp.eye(n, dtype=blocks.dtype), (L, K, n, n)),
        ],
        axis=-1,
    )
    bottom = jnp.zeros((L, K, n, 2 * n), blocks.dtype)
    return jnp.concatenate([top, bottom], axis=-2)


def eg_from_augmented(Eaug, n):
    return Eaug[..., :n, :n], Eaug[..., :n, n:]


def local_terms(
    v,
    blocks,
    dtheta,
    N,
    expm_fn=expm_pade,
    series_q=None,
    x_max=None,
    g_mode="augmented",
    uniform=False,
    eg_fn=None,
    local2="series",
):
    """v (L, m), blocks (K, n, n), dtheta (L,). Returns E (L, K, n, n),
    s1 (L, K, n), s2 (L, K, K, n, n) or None.

    g_mode: 'augmented' -> E and G from one (2n x 2n) exponential (any A);
            'solve'     -> E = exp(-h A) at n x n and G = A^{-1}(I - E)/h by a
                           batched solve (requires invertible blocks and h > 0;
                           lets an n-wide exponential kernel serve n-wide blocks)."""
    K, n, _ = blocks.shape
    L = v.shape[0]
    # uniform clock: one exponential per block, tiled (what the dense engine's
    # displacement_uniform mode does): the fair uniform-clock comparison
    dth = dtheta[:1] if uniform else dtheta
    if eg_fn is not None:
        E, G = eg_fn(blocks, dth)
    elif g_mode == "augmented":
        Eaug = expm_fn(augmented(blocks, dth))
        E, G = eg_from_augmented(Eaug, n)
    elif g_mode == "solve":
        E = expm_fn(-dth[:, None, None, None] * blocks[None])  # (L or 1, K, n, n)
        eye = jnp.eye(n, dtype=blocks.dtype)
        pos = dth > 0
        h_safe = jnp.where(pos, dth, jnp.ones_like(dth))[:, None, None, None]
        rhs = (eye - E) / h_safe
        G = jnp.linalg.solve(jnp.broadcast_to(blocks[None], E.shape), rhs)
        G = jnp.where(pos[:, None, None, None], G, jnp.broadcast_to(eye, E.shape))  # h -> 0: G = I
    else:
        raise ValueError("g_mode must be 'augmented' or 'solve'")
    if uniform:
        E = jnp.broadcast_to(E, (L,) + E.shape[1:])
        G = jnp.broadcast_to(G, (L,) + G.shape[1:])
    vb = v.reshape(L, K, n)
    s1 = jnp.einsum("lkab,lkb->lka", G, vb)
    if N == 1:
        return E, s1, None
    if local2 == "plane":
        from ews.core.kernels.plane_engine import local_terms as plane_local_terms

        dw, _ = _planes_from_blocks(blocks)
        _, _, s2 = plane_local_terms(v, dw, dtheta, 2)  # (L, K, K, 2, 2), same layout
        return E, s1, s2
    if series_q is None:
        raise ValueError("N=2 needs series_q (size it with ews.core.leaf.suggest_series_q)")

    # letters b_p = (-h/p) A b_{p-1}, blockwise; (L, q+1, K, n)
    def step(b_prev, p):
        b = (-dtheta[:, None, None] / p) * jnp.einsum("kab,lkb->lka", blocks, b_prev)
        return b, b

    ps = jnp.arange(1, series_q + 1, dtype=v.dtype)
    _, bs = jax.lax.scan(step, vb, ps)
    letters = jnp.concatenate([vb[None], bs], axis=0)  # (q+1, L, K, n)
    C2 = jnp.asarray(series_coeffs(2, series_q)[1], v.dtype)  # (q+1, q+1)
    # S2[l, j, a, k, b] = sum_{p,q} C[p,q] letters[p, l, j, a] letters[q, l, k, b]
    s2 = jnp.einsum("pq,plja,qlkb->ljkab", C2, letters, letters)
    return E, s1, s2


def _combine_n1(left, right):
    E_l, s_l = left
    E_r, s_r = right
    return jnp.einsum("...kab,...kbc->...kac", E_r, E_l), jnp.einsum(
        "...kab,...kb->...ka", E_r, s_l
    ) + s_r


def _combine_n2(left, right):
    E_l, s1_l, s2_l = left
    E_r, s1_r, s2_r = right
    E = jnp.einsum("...kab,...kbc->...kac", E_r, E_l)
    flowed_s1 = jnp.einsum("...kab,...kb->...ka", E_r, s1_l)
    flowed_s2 = jnp.einsum("...jab,...jkbc,...kdc->...jkad", E_r, s2_l, E_r)
    cross = flowed_s1[..., :, None, :, None] * s1_r[..., None, :, None, :]
    return E, flowed_s1 + s1_r, flowed_s2 + cross + s2_r


def blocks_to_flat(Q):
    """(..., K, K, n, n) -> (..., m*m) row-major of the (m, m) tensor."""
    K, n = Q.shape[-3], Q.shape[-1]
    return jnp.swapaxes(Q, -3, -2).reshape(Q.shape[:-4] + (K * n * K * n,))


def blockdiag_prefix(
    v,
    blocks,
    dtheta,
    N=1,
    expm_fn=expm_pade,
    series_q=None,
    combine="assoc",
    g_mode="augmented",
    uniform=False,
    eg_fn=None,
    local2="series",
):
    """Prefix EWS levels 1..N (one row per interval end) for block-diagonal A.
    v: (L, m) increments (block j = channels j*n .. j*n+n-1), blocks: (K, n, n),
    dtheta: (L,). Returns (L, m) for N=1, (L, m + m^2) for N=2.
    Plug-ins: eg_fn (None -> expm_fn via g_mode; plane_eg for canonical planes),
    local2 ('series' | 'plane')."""
    if N not in (1, 2):
        raise ValueError("N in {1, 2}")
    E, s1, s2 = local_terms(
        v,
        blocks,
        dtheta,
        N,
        expm_fn,
        series_q,
        g_mode=g_mode,
        uniform=uniform,
        eg_fn=eg_fn,
        local2=local2,
    )
    elems = (E, s1) if N == 1 else (E, s1, s2)
    op = _combine_n1 if N == 1 else _combine_n2
    if combine == "assoc":
        out = jax.lax.associative_scan(op, elems)
    else:
        K, n = blocks.shape[0], blocks.shape[1]
        init = (jnp.broadcast_to(jnp.eye(n, dtype=v.dtype), (K, n, n)),) + tuple(
            jnp.zeros_like(e[0]) for e in elems[1:]
        )
        _, out = jax.lax.scan(lambda c, e: (op(c, e),) * 2, init, elems)
    lvl1 = out[1].reshape(v.shape[0], -1)
    if N == 1:
        return lvl1
    return jnp.concatenate([lvl1, blocks_to_flat(out[2])], axis=-1)


def blocks_to_dense(blocks):
    return jax.scipy.linalg.block_diag(*blocks)
