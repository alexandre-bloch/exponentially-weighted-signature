"""Closed-form EWS for rotation-plane A at depths 1 and 2, with a combine that never forms an m x m matrix.

Plane j holds channels (2j, 2j+1) with A_j = [[d_j, w_j], [-w_j, d_j]], which acts on p = x + i y as multiplication by
conj(z_j), z_j = d_j + i w_j. Over an interval with clock step h:

    level 1   s_t = rho_t s_{t-1} + phi(-h conj z) v_t,    rho = exp(-h conj z),    phi(x) = (e^x - 1)/x
    level 2   the local term of planes (j, k) is a real 2x2 block built from I(conj z_j, z_k) v_j conj(v_k) and
              I(conj z_j, conj z_k) v_j v_k, where I(a, b) = D(-ha, -hb) is the divided difference of phi

and the prefix is the modified Chen recursion over (rho, s1, s2), at a cost of O(m) at depth 1 and O(m^2) at depth 2.
The output is the EWS flat layout without level 0: level 1 (L, m) with plane j at (2j, 2j+1), level 2 (L, m*m).
"""

import jax
import jax.numpy as jnp

# ----------------------------------------------------------------- phi and friends
# phi(x) = (e^x - 1)/x and its derivatives, entire functions; closed forms with
# small-|x| Taylor guards. Guards are set for f64; in f32 the closed forms are
# used only where the cancellation is harmless (|x| >~ 1e-2 keeps ~5 digits).


def _guarded(x, closed, series, tol):
    small = jnp.abs(x) < tol
    xs = jnp.where(small, jnp.ones_like(x), x)
    return jnp.where(small, series(x), closed(xs))


def phi(x):
    return _guarded(
        x,
        lambda x: jnp.expm1(x) / x,
        lambda x: 1 + x * (1 / 2 + x * (1 / 6 + x * (1 / 24 + x * (1 / 120 + x / 720)))),
        1e-3,
    )


def dphi(x):
    return _guarded(
        x,
        lambda x: (jnp.exp(x) * (x - 1) + 1) / x**2,
        lambda x: 1 / 2 + x * (1 / 3 + x * (1 / 8 + x * (1 / 30 + x * (1 / 144 + x / 840)))),
        3e-2,
    )


def d2phi(x):
    return _guarded(
        x,
        lambda x: (jnp.exp(x) * (x**2 - 2 * x + 2) - 2) / x**3,
        lambda x: 1 / 3 + x * (1 / 4 + x * (1 / 10 + x * (1 / 36 + x * (1 / 168 + x / 960)))),
        1e-1,
    )


def d3phi(x):
    return _guarded(
        x,
        lambda x: (jnp.exp(x) * (x**3 - 3 * x**2 + 6 * x - 6) + 6) / x**4,
        lambda x: 1 / 4 + x * (1 / 5 + x * (1 / 12 + x * (1 / 42 + x * (1 / 192 + x / 1080)))),
        2.5e-1,
    )


def divdiff_phi(x, delta, tol=None):
    """D(x, delta) = (phi(x + delta) - phi(x)) / delta, entire in both arguments.

    Below |delta| < tol the third-order Taylor expansion around x is used
    (error ~ tol^4 * phi''''/24); above it the direct quotient (error ~
    eps/tol). tol balances the two per dtype: 4e-4 (f64), 5e-2 (f32).
    """
    if tol is None:
        tol = 4e-4 if jnp.finfo(jnp.real(x).dtype).bits == 64 else 5e-2
    small = jnp.abs(delta) < tol
    ds = jnp.where(small, jnp.ones_like(delta), delta)
    direct = (phi(x + ds) - phi(x)) / ds
    series = dphi(x) + delta * (d2phi(x) / 2 + delta * d3phi(x) / 6)
    return jnp.where(small, series, direct)


# ----------------------------------------------------------------- layouts


def to_complex_planes(v):
    """(..., m) real, plane j at (2j, 2j+1) -> (..., K) complex x + i y."""
    v = v.reshape(v.shape[:-1] + (-1, 2))
    return jax.lax.complex(v[..., 0], v[..., 1])


def from_complex_planes(p):
    """(..., K) complex -> (..., m) real interleaved (Re, Im)."""
    return jnp.stack([jnp.real(p), jnp.imag(p)], -1).reshape(p.shape[:-1] + (-1,))


def rot_blocks(rho_conj):
    """Real 2x2 form of the reweighting e^{-hA_j} from rho_j = e^{-h conj z_j}:
    e^{-h z_j} = conj(rho_j) -> [[Re, Im], [-Im, Re]] of conj(rho)."""
    c = jnp.conj(rho_conj)
    re, im = jnp.real(c), jnp.imag(c)
    return jnp.stack([jnp.stack([re, im], -1), jnp.stack([-im, re], -1)], -2)


def outer_blocks(u, w):
    """Real (..., K, K, 2, 2) outer product u w^T of two plane-vector stacks
    given in complex form (..., K): from A = u_j conj(w_k), B = u_j w_k."""
    A = u[..., :, None] * jnp.conj(w)[..., None, :]
    B = u[..., :, None] * w[..., None, :]
    return _blocks_from_AB(A, B)


def _blocks_from_AB(A, B):
    q00 = (jnp.real(A) + jnp.real(B)) / 2
    q11 = (jnp.real(A) - jnp.real(B)) / 2
    q01 = (jnp.imag(B) - jnp.imag(A)) / 2
    q10 = (jnp.imag(A) + jnp.imag(B)) / 2
    return jnp.stack([jnp.stack([q00, q01], -1), jnp.stack([q10, q11], -1)], -2)


def blocks_to_flat(Q):
    """(..., K, K, 2, 2) -> (..., m*m) row-major of the (m, m) tensor."""
    K = Q.shape[-3]
    return jnp.swapaxes(Q, -3, -2).reshape(Q.shape[:-4] + (2 * K * 2 * K,))


def flat_to_blocks(T, K):
    """(..., m*m) -> (..., K, K, 2, 2)."""
    T = T.reshape(T.shape[:-1] + (K, 2, K, 2))
    return jnp.swapaxes(T, -3, -2)


# ----------------------------------------------------------------- local terms


def local_terms(v, dw, dtheta, N):
    """Per-interval pieces. v: (L, m) increments; dw: (K, 2); dtheta: (L,).
    Returns rho (L, K) complex, s1 (L, K) complex, and s2 (L, K, K, 2, 2) real
    (N=2) or None."""
    z = jax.lax.complex(dw[:, 0], dw[:, 1])  # (K,)
    zc = jnp.conj(z)
    h = dtheta[:, None]  # (L, 1)
    rho = jnp.exp(-h * zc[None, :])  # (L, K)
    p = to_complex_planes(v)  # (L, K)
    s1 = phi(-h * zc[None, :]) * p
    if N == 1:
        return rho, s1, None
    # I(a, b) = D(-h a, -h b): a = conj z_j (earlier), b = z_k (for A) / conj z_k (for B)
    x = -h[:, :, None] * zc[None, :, None]  # (L, K, 1)
    IA = divdiff_phi(x, -h[:, :, None] * z[None, None, :])  # (L, K, K)
    IB = divdiff_phi(x, -h[:, :, None] * zc[None, None, :])
    A = IA * p[:, :, None] * jnp.conj(p)[:, None, :]
    B = IB * p[:, :, None] * p[:, None, :]
    return rho, s1, _blocks_from_AB(A, B)


# ----------------------------------------------------------------- combine


def _combine_n1(left, right):
    rho_l, s_l = left
    rho_r, s_r = right
    return rho_r * rho_l, rho_r * s_l + s_r


def _combine_n2(left, right):
    rho_l, s1_l, s2_l = left
    rho_r, s1_r, s2_r = right
    R = rot_blocks(rho_r)  # (..., K, 2, 2)
    flowed = jnp.einsum("...jab,...jkbc,...kdc->...jkad", R, s2_l, R)
    cross = outer_blocks(rho_r * s1_l, s1_r)
    return rho_r * rho_l, rho_r * s1_l + s1_r, flowed + cross + s2_r


def plane_prefix(v, dw, dtheta, N=1, combine="assoc"):
    """Prefix EWS levels 1..N over the intervals, one row per interval end.

    v: (L, m) increments of the driving path (plane layout), dw: (K, 2) planes,
    dtheta: (L,) clock increments. Returns (L, m + m^2) for N=2, (L, m) for N=1,
    in the EWS flat layout minus the level-0 constant. combine: 'assoc'
    (jax.lax.associative_scan, O(log L) depth) or 'seq' (lax.scan, memory-light).
    """
    if N not in (1, 2):
        raise ValueError("plane_prefix supports N in {1, 2}")
    rho, s1, s2 = local_terms(v, dw, dtheta, N)
    elems = (rho, s1) if N == 1 else (rho, s1, s2)
    op = _combine_n1 if N == 1 else _combine_n2
    if combine == "assoc":
        out = jax.lax.associative_scan(op, elems)
    elif combine == "seq":

        def step(carry, e):
            new = op(carry, e)
            return new, new

        init = tuple(jnp.zeros_like(e[0]) for e in elems)
        init = (jnp.ones_like(init[0]),) + init[1:]  # identity: rho = 1, s = 0
        _, out = jax.lax.scan(step, init, elems)
    else:
        raise ValueError("combine must be 'assoc' or 'seq'")
    lvl1 = from_complex_planes(out[1])
    if N == 1:
        return lvl1
    return jnp.concatenate([lvl1, blocks_to_flat(out[2])], axis=-1)


def planes_from_jordan_tied(a_param):
    """(K, 2) [d, w] stack in channel order from an ews.conversion._mixer.JordanTiedA."""
    d = jnp.repeat(a_param.d, a_param.copies)
    w = jnp.repeat(a_param.w, a_param.copies)
    return jnp.stack([d, w], -1)
