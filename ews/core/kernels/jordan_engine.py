"""Closed-form local EWS terms for uniform Jordan-chain A (`AParamJordanRepeated`), depths 1 and 2: the leaves
`jordan_closed` and `jordan_closed_tables`.

A chain is J = lam I + eps Nil with Nil^n = 0, so an analytic g gives g(J) = sum_{p<n} g^(p)(lam) / p! (eps Nil)^p.
The plane engine's per-interval quantities (phi(x) = (e^x - 1)/x at level 1, its divided differences at level 2) are
analytic in the eigenvalue, so they carry over with their first n - 1 derivatives; a complex chain is the same over
the coordinates v_{2i} + i v_{2i+1}. Two forms compute them:

    fast    real chains of 2 and complex chains of 4: forward-mode derivatives of the plane engine's functions, in the
            layer's precision, exact at any x = |lam| h
    tables  any supported chain (real 2-4, complex 4, 6, 8): float64 tables exact at any argument with Re <= 0

`jordan_closed` uses the fast form where it exists and the tables otherwise; `jordan_closed_tables` always uses the
tables. The output is the EWS flat layout: level 1 (L, m), level 2 (L, m, m).
"""

import math

import jax
import jax.numpy as jnp

from ews.core.kernels.plane_engine import blocks_to_flat, divdiff_phi, dphi, phi

JORDAN_LEAVES = ("jordan_closed", "jordan_closed_tables")
FAST_CHAINS = (("real", 2), ("complex", 4))
REAL_SIZES, COMPLEX_SIZES = (2, 3, 4), (4, 6, 8)


def jordan_chains_of(a_param):
    """(kind, size) when A is ``AParamJordanRepeated`` with uniform chains of a supported size (real 2-4, complex 4 / 6 / 8),
    else None. Decided from the class and its static fields, like ``combine.block_sizes_of``."""
    from ews import params as P

    if not isinstance(a_param, P.AParamJordanRepeated) or not a_param.block_sizes:
        return None
    sizes, kinds = set(a_param.block_sizes), set(a_param.block_kinds)
    if len(sizes) != 1 or len(kinds) != 1:
        return None
    kind, size = kinds.pop(), int(sizes.pop())
    if size not in (REAL_SIZES if kind == "real" else COMPLEX_SIZES):
        return None
    return kind, size


# ----------------------------------------------------------------- fast forms (forward-mode derivatives of the plane engine)
def real2_local_terms(v, lam, eps, dtheta, N):
    """Real chains of length 2, the dual-number twin of the plane engine: g(J) = g(lam) I + g'(lam) eps Nil.
    v: (L, m), chain j at (2j, 2j+1) with 2j+1 the driven coordinate; lam, eps: (K,); dtheta: (L,)."""
    L, m = v.shape
    h = dtheta[:, None]  # (L, 1)
    x = -h * lam[None, :]  # (L, K)
    g = phi(x)
    gp = -h * dphi(x)
    v0, v1 = v[:, 0::2], v[:, 1::2]  # (L, K)
    n1 = eps[None, :] * v1  # (eps Nil v)_0; (eps Nil v)_1 = 0
    s1 = jnp.stack([g * v0 + gp * n1, g * v1], -1).reshape(L, m)
    if N == 1:
        return [s1]
    hh = dtheta[:, None, None]

    def I_fn(a, b):  # (K,), (K,) -> (L, K, K): I(a_j, b_k)
        return divdiff_phi(-hh * a[None, :, None], -hh * b[None, None, :])

    ones = jnp.ones_like(lam)
    I0 = I_fn(lam, lam)
    Ia = jax.jvp(lambda a: I_fn(a, lam), (lam,), (ones,))[1]
    Ib = jax.jvp(lambda b: I_fn(lam, b), (lam,), (ones,))[1]
    Iab = jax.jvp(lambda a: jax.jvp(lambda b: I_fn(a, b), (lam,), (ones,))[1], (lam,), (ones,))[1]
    x0, x1, xn = v0[:, :, None], v1[:, :, None], n1[:, :, None]  # earlier index j
    y0, y1, yn = v0[:, None, :], v1[:, None, :], n1[:, None, :]  # later index k
    q00 = I0 * x0 * y0 + Ia * xn * y0 + Ib * x0 * yn + Iab * xn * yn
    q01 = I0 * x0 * y1 + Ia * xn * y1
    q10 = I0 * x1 * y0 + Ib * x1 * yn
    q11 = I0 * x1 * y1
    Q = jnp.stack([jnp.stack([q00, q01], -1), jnp.stack([q10, q11], -1)], -2)  # (L, K, K, 2, 2)
    return [s1, blocks_to_flat(Q).reshape(L, m, m)]


def _real_blocks(Ab, Bb):
    """Real 2x2 blocks of complex coordinate pairs from A = U conj(W), B = U W (plane_engine._blocks_from_AB), then the EWS
    flat layout. Ab, Bb: (L, K_j, K_k, k_i, k_l) complex -> (L, m, m) real, m = 2 K k."""
    q00 = (jnp.real(Ab) + jnp.real(Bb)) / 2
    q11 = (jnp.real(Ab) - jnp.real(Bb)) / 2
    q01 = (jnp.imag(Bb) - jnp.imag(Ab)) / 2
    q10 = (jnp.imag(Ab) + jnp.imag(Bb)) / 2
    Q = jnp.stack(
        [jnp.stack([q00, q01], -1), jnp.stack([q10, q11], -1)], -2
    )  # (L, K_j, K_k, k_i, k_l, 2, 2)
    Q = jnp.transpose(Q, (0, 1, 3, 5, 2, 4, 6))  # (L, K_j, k_i, 2, K_k, k_l, 2)
    m = Q.shape[1] * Q.shape[2] * 2
    return Q.reshape(Q.shape[0], m, m)


def complex4_local_terms(v, c, eps, dtheta, N):
    """Complex chains of size 4 (two complex coordinates), the complex twin of ``real2_local_terms``: the plane engine's
    functions and their first holomorphic derivatives, in v's precision (complex64 for a float32 layer).
    v: (L, m) real; c: (K,) complex, c = a + ib; eps: (K,) real; dtheta: (L,)."""
    L, m = v.shape
    K = m // 4
    rdt = v.dtype
    cdt = jnp.complex64 if rdt == jnp.float32 else jnp.complex128
    c = jnp.asarray(c).astype(cdt)
    h = dtheta.astype(rdt)[:, None]
    x = -h * c[None, :]  # (L, K)
    vv = v.reshape(L, K, 2, 2)
    P = jax.lax.complex(vv[..., 0], vv[..., 1])  # (L, K, 2)
    P0, P1 = P[..., 0], P[..., 1]
    N1 = eps.astype(rdt)[None, :] * P1  # (eps Nil P)_0
    g = phi(x)
    gp = -h * dphi(x)
    s1c = jnp.stack([g * P0 + gp * N1, g * P1], -1)  # (L, K, 2)
    s1 = jnp.stack([jnp.real(s1c), jnp.imag(s1c)], -1).reshape(L, m)
    if N == 1:
        return [s1]
    hh = dtheta.astype(rdt)[:, None, None]

    def derivs(second):
        def I_fn(a, b):
            return divdiff_phi(-hh * a[None, :, None], -hh * b[None, None, :])  # (L, K_j, K_k)

        ones = jnp.ones_like(c)
        I0 = I_fn(c, second)
        Ia = jax.jvp(lambda a: I_fn(a, second), (c,), (ones,))[1]
        Ib = jax.jvp(lambda b: I_fn(c, b), (second,), (ones,))[1]
        Iab = jax.jvp(
            lambda a: jax.jvp(lambda b: I_fn(a, b), (second,), (ones,))[1], (c,), (ones,)
        )[1]
        return I0, Ia, Ib, Iab

    zero = jnp.zeros_like(P0)
    Pj = [
        [P0[:, :, None], N1[:, :, None]],
        [P1[:, :, None], zero[:, :, None]],
    ]  # Pj[i][p] = ((eps Nil)^p P_j)_i
    Pk = [[P0[:, None, :], N1[:, None, :]], [P1[:, None, :], zero[:, None, :]]]

    def block(co, conj_k):
        c0, ca, cb, cab = co
        rows = []
        for i in range(2):
            cols = []
            for l_ in range(2):
                w = [jnp.conj(Pk[l_][q]) if conj_k else Pk[l_][q] for q in range(2)]
                cols.append(
                    c0 * Pj[i][0] * w[0]
                    + ca * Pj[i][1] * w[0]
                    + cb * Pj[i][0] * w[1]
                    + cab * Pj[i][1] * w[1]
                )
            rows.append(jnp.stack(cols, -1))
        return jnp.stack(rows, -2)  # (L, K_j, K_k, 2_i, 2_l)

    Ab = block(derivs(jnp.conj(c)), True)
    Bb = block(derivs(c), False)
    return [s1, _real_blocks(Ab, Bb).astype(rdt)]


# ----------------------------------------------------------------- tables: phi^(r) and D^(p,q), exact at any argument with Re <= 0
PHI_J = 64  # series terms for |y| <= PHI_SWITCH
PHI_SWITCH = (
    8.0  # above it the recurrence phi^(r) = (e^y - r phi^(r-1)) / y (error gain r/|y| per step)
)
PAIR_J = 24  # chain-pair grid: series terms for |y| <= PAIR_SWITCH (1.5^24 / 24! is about 3e-20)
PAIR_SWITCH = 1.5  # above it the recurrence; only r <= 6 is needed on the grid, total error gain <= 6!/1.5^6, about 63
PAIR_S = 10  # terms of the delta-expansion of D, used only for |delta| < PAIR_DELTA0
PAIR_DELTA0 = 0.25  # 0.25^10 / 10! is about 3e-13


def _wide(x):
    """float64 or complex128, by the input's kind."""
    x = jnp.asarray(x)
    return x.astype(jnp.complex128) if jnp.iscomplexobj(x) else x.astype(jnp.float64)


def phi_derivative_table(y, r_max, n_terms=PHI_J, switch=PHI_SWITCH):
    """phi^(r)(y), r = 0..r_max, for phi(y) = (e^y - 1)/y = int_0^1 e^{ty} dt, at any real or complex y with Re y <= 0.

    |y| <= switch: the entire series phi^(r)(y) = sum_j y^j / ((j + r + 1) j!) (at |y| = 8 the truncation is far below
    float64 and the cancellation costs about 3 digits). |y| > switch: y phi = e^y - 1 differentiated r times,
    phi^(r) = (e^y - r phi^(r-1)) / y, started at expm1(y)/y; its error gain per step is r/|y|, so it is stable for the orders
    used here (r <= 17 at switch 8, r <= 6 at the pair grid's switch 1.5). Both branches are evaluated on sanitised
    arguments, so no NaN can leak through the select, forward or backward. y: any shape -> (..., r_max + 1)."""
    y = _wide(y)
    small = jnp.abs(y) <= switch
    ys = jnp.where(small, y, jnp.zeros_like(y))  # series branch argument
    yl = jnp.where(
        small, jnp.full_like(y, -2.0 * switch), y
    )  # recurrence branch argument (never near 0)
    j = jnp.arange(int(n_terms), dtype=jnp.float64)
    ratios = jnp.where(j == 0, jnp.ones_like(ys[..., None]), ys[..., None] / jnp.maximum(j, 1.0))
    ypow = jnp.cumprod(ratios, axis=-1)  # y^j / j!
    r = jnp.arange(r_max + 1, dtype=jnp.float64)
    C = (1.0 / (j[None, :] + r[:, None] + 1.0)).astype(ypow.dtype)  # (R, J)
    series = ypow @ C.T  # (..., R)
    ey = jnp.exp(yl)
    cols = [jnp.expm1(yl) / yl]
    for rr in range(1, r_max + 1):
        cols.append((ey - rr * cols[-1]) / yl)
    return jnp.where(small[..., None], series, jnp.stack(cols, axis=-1))


def pair_integral_derivatives(x, delta, p_max, q_max):
    """D^(p,q)(x_j, delta_k) = d_x^p d_delta^q of D(x, delta) = (phi(x + delta) - phi(x)) / delta = int_0^1 phi'(x + t delta) dt
    for p <= p_max, q <= q_max on the chain-pair grid: x (L, K), delta (L, K) -> dict[(p, q)] of (L, K, K), float64 or
    complex128. Any real or complex arguments with Re <= 0.

    |delta| < PAIR_DELTA0: the expansion in delta, D^(p,q) = sum_s delta^s phi^(p+q+1+s)(x) / (s! (q+s+1)).
    otherwise: Leibniz on F(delta) / delta with F = phi^(p)(x + delta) - phi^(p)(x):
        D^(p,q) = sum_{s<=q} C(q,s) (-1)^(q-s) (q-s)! delta^-(q-s+1) F_s,   F_0 = F,  F_s = phi^(p+s)(x + delta),
    which has no small divisor there (at |delta| = 0.25 and q = 3 it costs about 4 digits of float64)."""
    if p_max + q_max > 6:
        raise ValueError("pair grid: orders above 6 would need a wider recurrence margin")
    x, delta = _wide(x), _wide(delta)
    if jnp.iscomplexobj(x) != jnp.iscomplexobj(delta):
        x, delta = x.astype(jnp.complex128), delta.astype(jnp.complex128)
    small = jnp.abs(delta) < PAIR_DELTA0  # (L, K_k)
    ds = jnp.where(small, delta, jnp.zeros_like(delta))
    dl = jnp.where(small, jnp.full_like(delta, -1.0), delta)
    tab_x = phi_derivative_table(x, p_max + q_max + 1 + PAIR_S)  # (L, K_j, R)
    s_idx = jnp.arange(PAIR_S, dtype=jnp.float64)
    ratios = jnp.where(
        s_idx == 0, jnp.ones_like(ds[..., None]), ds[..., None] / jnp.maximum(s_idx, 1.0)
    )
    dpow = jnp.cumprod(ratios, axis=-1)  # delta^s / s!, (L, K_k, S)
    # the pair grid is the large array (L, K, K, terms) that the backward pass keeps: a short series and the recurrence
    tab_xd = phi_derivative_table(
        x[:, :, None] + dl[:, None, :], p_max + q_max, PAIR_J, PAIR_SWITCH
    )  # (L, K_j, K_k, p+q+1)
    inv = 1.0 / dl  # (L, K_k)
    out = {}
    for p in range(p_max + 1):
        for q in range(q_max + 1):
            base = p + q + 1
            ser = jnp.einsum(
                "njs,nks->njk", tab_x[:, :, base : base + PAIR_S], dpow / (q + s_idx + 1.0)
            )
            lei = 0.0
            for s in range(q + 1):
                F = tab_xd[..., p + s] - (tab_x[:, :, None, p] if s == 0 else 0.0)
                coef = math.comb(q, s) * (-1.0) ** (q - s) * math.factorial(q - s)
                lei = lei + coef * F * inv[:, None, :] ** (q - s + 1)
            out[(p, q)] = jnp.where(small[:, None, :], ser, lei)
    return out


def _shifted(P, eps, n):
    """[(eps Nil)^p P for p < n]; P: (L, K, n), eps: (K,)."""
    L, K, _ = P.shape
    e = eps.astype(P.real.dtype)[None, :, None]
    return [
        jnp.concatenate([P[:, :, p:], jnp.zeros((L, K, p), P.dtype)], -1) * e**p for p in range(n)
    ]


def real_local_terms_tables(v, lam, eps, dtheta, N, n):
    """Real chains of length n (2 <= n <= 4) from the float64 tables; the tensor assembly is in v's dtype."""
    L, m = v.shape
    n = int(n)
    K = m // n
    dtype = v.dtype
    h = jnp.asarray(dtheta, jnp.float64)[:, None]  # (L, 1)
    x = -h * jnp.asarray(lam, jnp.float64)[None, :]  # (L, K)
    shifted = _shifted(v.reshape(L, K, n), eps, n)  # chain j, coordinate i
    tab = phi_derivative_table(x, n - 1)  # (L, K, n)
    s1 = jnp.zeros((L, K, n), dtype)
    for p in range(n):
        coef = ((-h) ** p * tab[:, :, p] / math.factorial(p)).astype(dtype)  # (L, K)
        s1 = s1 + coef[:, :, None] * shifted[p]
    s1 = s1.reshape(L, m)
    if N == 1:
        return [s1]
    D = pair_integral_derivatives(x, x, n - 1, n - 1)  # delta_k = -h lam_k = x_k
    Q = jnp.zeros((L, K, K, n, n), dtype)
    for p in range(n):
        for q in range(n):
            coef = (
                (-h[:, :, None]) ** (p + q) * D[(p, q)] / (math.factorial(p) * math.factorial(q))
            ).astype(dtype)  # (L, K, K)
            Q = (
                Q
                + coef[..., None, None]
                * shifted[p][:, :, None, :, None]
                * shifted[q][:, None, :, None, :]
            )
    return [s1, jnp.swapaxes(Q, 2, 3).reshape(L, m, m)]  # chain pair (j, k) at rows jn.., cols kn..


def complex_local_terms_tables(v, c, eps, dtheta, N, k):
    """Complex chains of k complex coordinates (size 2k = 4, 6, 8) from the complex128 tables:
    A_il = sum_{p,q} (-h)^(p+q) D^(p,q)(-h c_j, -h conj(c_k)) / (p! q!) ((eps Nil)^p P_j)_i conj(((eps Nil)^q P_k)_l)
    B_il = the same with D^(p,q)(-h c_j, -h c_k) and no conjugate."""
    L, m = v.shape
    k = int(k)
    K = m // (2 * k)
    rdt = v.dtype
    cdt = jnp.complex64 if rdt == jnp.float32 else jnp.complex128
    h = jnp.asarray(dtheta, jnp.float64)[:, None]  # (L, 1)
    x = -h * jnp.asarray(c, jnp.complex128)[None, :]  # (L, K)
    vv = v.reshape(L, K, k, 2)
    shifted = _shifted(jax.lax.complex(vv[..., 0], vv[..., 1]).astype(cdt), eps, k)
    tab = phi_derivative_table(x, k - 1)  # (L, K, k)
    s1c = jnp.zeros((L, K, k), cdt)
    for p in range(k):
        coef = ((-h) ** p * tab[:, :, p] / math.factorial(p)).astype(cdt)
        s1c = s1c + coef[:, :, None] * shifted[p]
    s1 = jnp.stack([jnp.real(s1c), jnp.imag(s1c)], -1).reshape(L, m).astype(rdt)
    if N == 1:
        return [s1]
    DA = pair_integral_derivatives(x, jnp.conj(x), k - 1, k - 1)  # delta_k = -h conj(c_k)
    DB = pair_integral_derivatives(x, x, k - 1, k - 1)  # delta_k = -h c_k
    Ab = jnp.zeros((L, K, K, k, k), cdt)
    Bb = jnp.zeros((L, K, K, k, k), cdt)
    for p in range(k):
        for q in range(k):
            scale = (-h[:, :, None]) ** (p + q) / (math.factorial(p) * math.factorial(q))
            Pj = shifted[p][:, :, None, :, None]  # (L, K_j, 1, k_i, 1)
            Pk = shifted[q][:, None, :, None, :]  # (L, 1, K_k, 1, k_l)
            Ab = Ab + (scale * DA[(p, q)]).astype(cdt)[..., None, None] * Pj * jnp.conj(Pk)
            Bb = Bb + (scale * DB[(p, q)]).astype(cdt)[..., None, None] * Pj * Pk
    return [s1, _real_blocks(Ab, Bb).astype(rdt)]


# ----------------------------------------------------------------- the leaf
def local_terms(v, A, dtheta, N, kind, size, tables=False):
    """Local EWS levels [s1 (L, m)] or [s1, s2 (L, m, m)] of uniform Jordan chains, read off the effective generator A
    (m, m): eigenvalue and coupling of chain j from its first rows, so the layer's ``a_scale`` is honoured.
    ``tables=False``: the fast form where it exists (FAST_CHAINS), the tables otherwise; ``tables=True``: tables always."""
    if N > 2:
        raise NotImplementedError("the closed-form Jordan leaf exists for depths 1 and 2")
    size = int(size)
    if size not in (REAL_SIZES if kind == "real" else COMPLEX_SIZES) or A.shape[0] % size:
        raise ValueError(
            f"the closed-form Jordan leaf needs real chains of 2-4 or complex chains of size 4, 6, 8 that tile A; "
            f"got kind {kind!r}, size {size}, dimension {A.shape[0]}"
        )
    first = jnp.arange(0, A.shape[0], size)
    fast = not tables and (kind, size) in FAST_CHAINS
    if kind == "real":
        lam, eps = A[first, first], A[first, first + 1]
        return (
            real2_local_terms(v, lam, eps, dtheta, N)
            if fast
            else real_local_terms_tables(v, lam, eps, dtheta, N, size)
        )
    eps = A[first, first + 2]
    if fast:
        return complex4_local_terms(
            v, jax.lax.complex(A[first, first], A[first + 1, first]), eps, dtheta, N
        )
    c = jax.lax.complex(
        A[first, first].astype(jnp.float64), A[first + 1, first].astype(jnp.float64)
    )  # a + ib
    return complex_local_terms_tables(v, c, eps, dtheta, N, size // 2)
