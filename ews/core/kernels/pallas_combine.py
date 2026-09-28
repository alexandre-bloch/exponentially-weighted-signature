"""Fused Chen-combine kernels in Pallas / Triton, chunked over the sequence.

Block mode (2x2 blocks, planes or free, depths 1 and 2), per interval t and block j with flow E_t,j:
    S1 <- E_t S1 + s1_t
    S2 <- E_t S2 E_t^T + (E_t S1_prev) (x) s1_t + s2_t
Level 2 is carried as four (K, K) parity tiles; a row-plane of it depends only on its own rows, so the grid is
(batch, row tiles, chunks). Long sequences use two levels: the chunks run in parallel from a zero carry, their end
states are combined in JAX, and each position then receives its chunk's incoming carry, so the memory is the output
stream alone. The module also holds the dense kernels, the lean JAX scans and the custom VJPs.
"""

import functools

# jax 0.9 takes pallas_call(backend="triton"); jax 0.10 dropped the keyword (Triton is the GPU backend by default), so
# the keyword is passed only when the installed pallas_call accepts it.
import inspect

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

_PALLAS_CALL_HAS_BACKEND = "backend" in inspect.signature(pl.pallas_call).parameters


def _backend_kw(backend):
    return {"backend": backend} if _PALLAS_CALL_HAS_BACKEND else {}


from ews.core.kernels import plane_engine as pe

HI = jax.lax.Precision.HIGHEST  # chunk-carry scans run in XLA: keep TF32 out of them


def _flow2(e00, e01, e10, e11, c00, c01, c10, c11, P00, P01, P10, P11):
    """S' = E S E^T blockwise, with row factors e** (tile, 1) and column factors c** (1, K)."""
    T00 = e00 * P00 + e01 * P10
    T01 = e00 * P01 + e01 * P11
    T10 = e10 * P00 + e11 * P10
    T11 = e10 * P01 + e11 * P11
    N00 = T00 * c00 + T01 * c01
    N01 = T00 * c10 + T01 * c11
    N10 = T10 * c00 + T11 * c01
    N11 = T10 * c10 + T11 * c11
    return N00, N01, N10, N11


# Layouts (all native, zero-copy; strided ref indexing):
#   E   (BC, Lc, K, 2, 2)      tile block (None, Lc, tile, 2, 2)  / full-column block (None, Lc, K, 2, 2)
#   s1  (BC, Lc, K, 2)         tile block (None, Lc, tile, 2)     / full (None, Lc, K, 2)
#   s2  (BC, Lc, K, K, 2, 2)   tile block (None, Lc, tile, K, 2, 2)
#   out (BC, Lc, m+1, m)       full block; row 0 = level 1 (plane j at columns 2j, 2j+1), rows 1..m = level-2
#                              row-major (m, m) -> reshape (B, L, m + m^2) is the plane engine's flat layout.
#   Each row-tile program writes its own rows 1 + 2j + a (j in its tile) with stride-2 stores.


def _rows(o_ref, t, j0, tile, a):
    return (t, pl.ds(1 + 2 * j0 + a, tile, stride=2))


def _kernel_n2(et, s1t, ef, s1f, s2t, o_ref, *, tile, K):
    L = et.shape[0]
    i = pl.program_id(1)
    j0 = i * tile
    z = s1t[0, :, 0] * 0.0
    z2 = s2t[0, :, :, 0, 0] * 0.0
    cols = [pl.ds(b, K, stride=2) for b in (0, 1)]

    def body(t, carry):
        P1a, P1b, P00, P01, P10, P11 = carry
        e00 = et[t, :, 0, 0]
        e01 = et[t, :, 0, 1]
        e10 = et[t, :, 1, 0]
        e11 = et[t, :, 1, 1]
        f1a = e00 * P1a + e01 * P1b  # E P1 (tile planes)
        f1b = e10 * P1a + e11 * P1b
        N00, N01, N10, N11 = _flow2(
            e00[:, None],
            e01[:, None],
            e10[:, None],
            e11[:, None],
            ef[t, :, 0, 0][None, :],
            ef[t, :, 0, 1][None, :],
            ef[t, :, 1, 0][None, :],
            ef[t, :, 1, 1][None, :],
            P00,
            P01,
            P10,
            P11,
        )
        wa = s1f[t, :, 0][None, :]
        wb = s1f[t, :, 1][None, :]  # s1_t, all columns
        ua = f1a[:, None]
        ub = f1b[:, None]
        N00 = N00 + ua * wa + s2t[t, :, :, 0, 0]
        N01 = N01 + ua * wb + s2t[t, :, :, 0, 1]
        N10 = N10 + ub * wa + s2t[t, :, :, 1, 0]
        N11 = N11 + ub * wb + s2t[t, :, :, 1, 1]
        P1a = f1a + s1t[t, :, 0]
        P1b = f1b + s1t[t, :, 1]
        o_ref[t, 0, pl.ds(2 * j0, tile, stride=2)] = P1a
        o_ref[t, 0, pl.ds(2 * j0 + 1, tile, stride=2)] = P1b
        o_ref[_rows(o_ref, t, j0, tile, 0) + (cols[0],)] = N00
        o_ref[_rows(o_ref, t, j0, tile, 0) + (cols[1],)] = N01
        o_ref[_rows(o_ref, t, j0, tile, 1) + (cols[0],)] = N10
        o_ref[_rows(o_ref, t, j0, tile, 1) + (cols[1],)] = N11
        return (P1a, P1b, N00, N01, N10, N11)

    jax.lax.fori_loop(0, L, body, (z, z, z2, z2, z2, z2))


def _kernel_n1(et, s1t, o_ref, *, tile):
    L = et.shape[0]
    j0 = pl.program_id(1) * tile
    z = s1t[0, :, 0] * 0.0

    def body(t, carry):
        P1a, P1b = carry
        e00 = et[t, :, 0, 0]
        e01 = et[t, :, 0, 1]
        e10 = et[t, :, 1, 0]
        e11 = et[t, :, 1, 1]
        P1a, P1b = e00 * P1a + e01 * P1b + s1t[t, :, 0], e10 * P1a + e11 * P1b + s1t[t, :, 1]
        o_ref[t, pl.ds(2 * j0, tile, stride=2)] = P1a
        o_ref[t, pl.ds(2 * j0 + 1, tile, stride=2)] = P1b
        return (P1a, P1b)

    jax.lax.fori_loop(0, L, body, (z, z))


def _fixup_kernel_n2(gt, gf, p_ref, c1, c2t, o_ref, *, tile, K):
    """S1_t = Ecum_t C1 + P1_t ; S2_t = Ecum_t C2 Ecum_t^T + (Ecum_t C1) (x) P1_t + P2_t, per position.
    gt/gf: Ecum_t (tile / full-column blocks, (.., K, 2, 2)); p_ref: pass-1 prefix in the output layout (full);
    c1: chunk carry level 1 (K, 2) full; c2t: chunk carry level 2, tile rows (tile, K, 2, 2)."""
    L = gt.shape[0]
    j0 = pl.program_id(1) * tile
    cols = [pl.ds(b, K, stride=2) for b in (0, 1)]
    rt = [
        pl.ds(2 * j0 + a, tile, stride=2) for a in (0, 1)
    ]  # level-1 columns of this tile's planes
    c1a_t = c1[pl.ds(j0, tile), 0]
    c1b_t = c1[pl.ds(j0, tile), 1]
    C00 = c2t[:, :, 0, 0]
    C01 = c2t[:, :, 0, 1]
    C10 = c2t[:, :, 1, 0]
    C11 = c2t[:, :, 1, 1]

    def body(t, carry):
        e00 = gt[t, :, 0, 0]
        e01 = gt[t, :, 0, 1]
        e10 = gt[t, :, 1, 0]
        e11 = gt[t, :, 1, 1]
        fa = e00 * c1a_t + e01 * c1b_t  # Ecum C1, tile planes
        fb = e10 * c1a_t + e11 * c1b_t
        N00, N01, N10, N11 = _flow2(
            e00[:, None],
            e01[:, None],
            e10[:, None],
            e11[:, None],
            gf[t, :, 0, 0][None, :],
            gf[t, :, 0, 1][None, :],
            gf[t, :, 1, 0][None, :],
            gf[t, :, 1, 1][None, :],
            C00,
            C01,
            C10,
            C11,
        )
        wa = p_ref[t, 0, cols[0]][None, :]
        wb = p_ref[t, 0, cols[1]][None, :]  # local prefix P1_t, all columns
        o_ref[_rows(o_ref, t, j0, tile, 0) + (cols[0],)] = (
            N00 + fa[:, None] * wa + p_ref[_rows(o_ref, t, j0, tile, 0) + (cols[0],)]
        )
        o_ref[_rows(o_ref, t, j0, tile, 0) + (cols[1],)] = (
            N01 + fa[:, None] * wb + p_ref[_rows(o_ref, t, j0, tile, 0) + (cols[1],)]
        )
        o_ref[_rows(o_ref, t, j0, tile, 1) + (cols[0],)] = (
            N10 + fb[:, None] * wa + p_ref[_rows(o_ref, t, j0, tile, 1) + (cols[0],)]
        )
        o_ref[_rows(o_ref, t, j0, tile, 1) + (cols[1],)] = (
            N11 + fb[:, None] * wb + p_ref[_rows(o_ref, t, j0, tile, 1) + (cols[1],)]
        )
        o_ref[t, 0, rt[0]] = fa + p_ref[t, 0, rt[0]]
        o_ref[t, 0, rt[1]] = fb + p_ref[t, 0, rt[1]]
        return carry

    jax.lax.fori_loop(0, L, body, 0)


def _cum_ops(E):
    """Cumulative operator inside each chunk: Ecum_t = E_t ... E_1 (2x2 per block). E: (B, C, Lc, K, 2, 2)."""
    return jax.lax.associative_scan(
        lambda l, r: jnp.einsum("...ab,...bc->...ac", r, l, precision=HI), E, axis=2
    )


def combine_pallas(E, s1, s2, N=2, tile=4, chunk=None, backend="triton"):
    """E (B, L, K, 2, 2) block operators; s1 (B, L, m) plane-interleaved; s2 (B, L, K, K, 2, 2) or None.
    Returns (B, L, m) for N=1 / (B, L, m + m^2) for N=2 in the plane engine's flat layout. Zero layout copies."""
    B, L, K = E.shape[:3]
    m = 2 * K
    dt = E.dtype
    tile = min(tile, K)
    if K % tile:
        raise ValueError("tile must divide K")
    nt = K // tile
    chunk = L if chunk is None else min(chunk, L)
    if L % chunk:
        raise ValueError("chunk must divide L")
    C = L // chunk
    Ec = E.reshape(B * C, chunk, K, 2, 2)
    s1c = s1.reshape(B * C, chunk, K, 2)
    spec_et = pl.BlockSpec((None, chunk, tile, 2, 2), lambda b, i: (b, 0, i, 0, 0))
    spec_ef = pl.BlockSpec((None, chunk, K, 2, 2), lambda b, i: (b, 0, 0, 0, 0))
    spec_1t = pl.BlockSpec((None, chunk, tile, 2), lambda b, i: (b, 0, i, 0))
    spec_1f = pl.BlockSpec((None, chunk, K, 2), lambda b, i: (b, 0, 0, 0))
    spec_2t = pl.BlockSpec((None, chunk, tile, K, 2, 2), lambda b, i: (b, 0, i, 0, 0, 0))
    if N == 1:
        spec_o = pl.BlockSpec((None, chunk, m), lambda b, i: (b, 0, 0))
        P1 = pl.pallas_call(
            functools.partial(_kernel_n1, tile=tile),
            out_shape=jax.ShapeDtypeStruct((B * C, chunk, m), dt),
            grid=(B * C, nt),
            in_specs=[spec_et, spec_1t],
            out_specs=spec_o,
            **_backend_kw(backend),
        )(Ec, s1c)
        if C > 1:
            Ecum = _cum_ops(E.reshape(B, C, chunk, K, 2, 2))
            ends = Ecum[:, :, -1]  # chunk operators (B, C, K, 2, 2)
            P1c = P1.reshape(B, C, chunk, K, 2)

            def op(l, r):
                El, pl_ = l
                Er, pr = r
                return (
                    jnp.einsum("...ab,...bc->...ac", Er, El, precision=HI),
                    jnp.einsum("...ab,...b->...a", Er, pl_, precision=HI) + pr,
                )

            _, P1car = jax.lax.associative_scan(op, (ends, P1c[:, :, -1]), axis=1)
            carry_in = jnp.concatenate(
                [jnp.zeros_like(P1car[:, :1]), P1car[:, :-1]], axis=1
            )  # (B, C, K, 2)
            P1c = P1c + jnp.einsum("nclkxy,ncky->nclkx", Ecum, carry_in, precision=HI)
            return P1c.reshape(B, L, m)
        return P1.reshape(B, L, m)
    s2c = s2.reshape(B * C, chunk, K, K, 2, 2)
    spec_o = pl.BlockSpec((None, chunk, m + 1, m), lambda b, i: (b, 0, 0, 0))
    P = pl.pallas_call(
        functools.partial(_kernel_n2, tile=tile, K=K),
        out_shape=jax.ShapeDtypeStruct((B * C, chunk, m + 1, m), dt),
        grid=(B * C, nt),
        in_specs=[spec_et, spec_1t, spec_ef, spec_1f, spec_2t],
        out_specs=spec_o,
        **_backend_kw(backend),
    )(Ec, s1c, Ec, s1c, s2c)
    if C > 1:
        Ecum = _cum_ops(E.reshape(B, C, chunk, K, 2, 2))  # (B, C, Lc, K, 2, 2)
        ends = Ecum[:, :, -1]
        Pe = P.reshape(B, C, chunk, m + 1, m)[:, :, -1]  # chunk end states (B, C, m+1, m)
        P1e = Pe[:, :, 0].reshape(B, C, K, 2)
        Q2e = pe.flat_to_blocks(Pe[:, :, 1:].reshape(B, C, m * m), K)  # (B, C, K, K, 2, 2)

        def op(l, r):
            El, p1l, q2l = l
            Er, p1r, q2r = r
            E_ = jnp.einsum("...kab,...kbc->...kac", Er, El, precision=HI)
            f1 = jnp.einsum("...kab,...kb->...ka", Er, p1l, precision=HI)
            flowed = jnp.einsum("...jab,...jkbc,...kdc->...jkad", Er, q2l, Er, precision=HI)
            cross = f1[..., :, None, :, None] * p1r[..., None, :, None, :]
            return E_, f1 + p1r, flowed + cross + q2r

        _, C1, C2 = jax.lax.associative_scan(op, (ends, P1e, Q2e), axis=1)
        C1in = jnp.concatenate([jnp.zeros_like(C1[:, :1]), C1[:, :-1]], 1).reshape(B * C, K, 2)
        C2in = jnp.concatenate([jnp.zeros_like(C2[:, :1]), C2[:, :-1]], 1).reshape(
            B * C, K, K, 2, 2
        )
        Gc = Ecum.reshape(B * C, chunk, K, 2, 2)
        spec_c1 = pl.BlockSpec((None, K, 2), lambda b, i: (b, 0, 0))
        spec_c2t = pl.BlockSpec((None, tile, K, 2, 2), lambda b, i: (b, i, 0, 0, 0))
        P = pl.pallas_call(
            functools.partial(_fixup_kernel_n2, tile=tile, K=K),
            out_shape=jax.ShapeDtypeStruct((B * C, chunk, m + 1, m), dt),
            grid=(B * C, nt),
            in_specs=[spec_et, spec_ef, spec_o, spec_c1, spec_c2t],
            out_specs=spec_o,
            **_backend_kw(backend),
        )(Gc, Gc, P, C1in, C2in)
    return P.reshape(B, L, m + m * m)


def planes_to_E(rho):
    """rho (..., K) complex = e^{-h conj z} -> E (..., K, 2, 2) = real form of e^{-h z} = [[c, s], [-s, c]]
    with c + i s = conj(rho) (plane_engine.rot_blocks)."""
    return pe.rot_blocks(rho)


# =========================================================================== dense mode (any A)
# Operators M_t = exp(-h_t A) (m x m, any A) are inputs (from stage 1 / one cached matrix under a uniform
# clock); the kernel only does the Chen recursion:  S1 <- M_t S1 + s1_t ;  S2 <- M_t S2 M_t^T + (M_t S1) s1_t^T + s2_t.
# Rows of S2 are mixed by M_t, so there is no row tiling; parallelism = samples x chunks (two-level scan):
#   pass 1: every chunk from a zero carry, writes only its end state (S1e, S2e) and its operator M_chunk = M_Lc..M_1
#   pass 2: associative scan over the C chunk summaries (JAX, C small)
#   pass 3: every chunk again from its true incoming carry, writing the output stream.
# Memory = output stream + chunk summaries; compute = 2x one sequential pass (pass 1 carries M_chunk: 3 matmuls/step,
# pass 3: 2 matmuls/step). Block mode uses the cheaper fix-up instead (row-tiled, no recompute).


def _dense_kernel(*refs, N, uniform, write, carry_M, rank1, refstate=False):
    """rank1=False: S2 carried in registers, matmuls via pl.dot (tensor-core FMA path, TF32 off); f32 only,
    Triton has no f64 MMA. rank1=True (f64): S2 lives in e2_ref, products as m rank-1 updates with rows/columns
    loaded from refs (tmp_ref holds the intermediate T = M S2)."""
    if N == 2:
        m_ref, s1_ref, s2_ref, c1_ref, c2_ref, o_ref, e1_ref, e2_ref, em_ref, tmp_ref = (
            refs  # o_ref (Lo, m+1, m)
        )
    else:
        m_ref, s1_ref, c1_ref, o_ref, e1_ref, em_ref = refs  # o_ref (Lo, m)
        s2_ref = c2_ref = e2_ref = tmp_ref = None
    L = s1_ref.shape[0]
    m = s1_ref.shape[1]
    dt = s1_ref.dtype

    def mcol(t, k):
        return m_ref[0, :, k] if uniform else m_ref[t, :, k]

    def Mt(t):
        return m_ref[0, :, :] if uniform else m_ref[t, :, :]

    if rank1 or refstate:
        if N == 2:
            e2_ref[...] = c2_ref[...]
        if carry_M and rank1:
            em_ref[...] = jnp.eye(m, dtype=dt)
        plgpu.debug_barrier()  # ref state is shared by the program's threads: sync after every store

    def body(t, carry):
        S1, S2, Mc = carry
        M = Mt(t)
        f1 = jnp.sum(M * S1[None, :], axis=1)
        S1n = f1 + s1_ref[t, :]
        if N == 2:
            if rank1:
                T = jax.lax.fori_loop(
                    0,
                    m,
                    lambda k, acc: acc + mcol(t, k)[:, None] * e2_ref[k, :][None, :],
                    jnp.zeros((m, m), dt),
                )
                tmp_ref[...] = T
                plgpu.debug_barrier()
                S2n = jax.lax.fori_loop(
                    0,
                    m,
                    lambda k, acc: acc + tmp_ref[:, k][:, None] * mcol(t, k)[None, :],
                    jnp.zeros((m, m), dt),
                )
                S2n = S2n + f1[:, None] * s1_ref[t, :][None, :] + s2_ref[t, :, :]
                plgpu.debug_barrier()  # all reads of the old state are done before it is overwritten
                e2_ref[...] = S2n
                plgpu.debug_barrier()
                if write:
                    o_ref[t, pl.ds(1, m), :] = S2n
                S2n = S2  # placeholder carry in rank-1 mode
            elif refstate:
                # S2 lives in e2_ref (L2-resident), reloaded each step: keeps the register footprint at one m x m tile
                T = pl.dot(M, e2_ref[...], allow_tf32=False)
                S2n = (
                    pl.dot(T, M, trans_b=True, allow_tf32=False)
                    + f1[:, None] * s1_ref[t, :][None, :]
                    + s2_ref[t, :, :]
                )
                plgpu.debug_barrier()
                e2_ref[...] = S2n
                plgpu.debug_barrier()
                if write:
                    o_ref[t, pl.ds(1, m), :] = S2n
                S2n = S2
            else:
                T = pl.dot(M, S2, allow_tf32=False)
                S2n = (
                    pl.dot(T, M, trans_b=True, allow_tf32=False)
                    + f1[:, None] * s1_ref[t, :][None, :]
                    + s2_ref[t, :, :]
                )
        else:
            S2n = S2
        if carry_M:
            if rank1:
                Mn = jax.lax.fori_loop(
                    0,
                    m,
                    lambda k, acc: acc + mcol(t, k)[:, None] * em_ref[k, :][None, :],
                    jnp.zeros((m, m), dt),
                )
                plgpu.debug_barrier()
                em_ref[...] = Mn
                plgpu.debug_barrier()
            else:
                Mc = pl.dot(M, Mc, allow_tf32=False)
        if write:
            if N == 2:
                o_ref[t, 0, :] = S1n
                if not rank1 and not refstate:
                    o_ref[t, pl.ds(1, m), :] = S2n
            else:
                o_ref[t, :] = S1n
        return (S1n, S2n, Mc)

    S1 = c1_ref[...]
    S2 = c2_ref[...] if (N == 2 and not rank1 and not refstate) else S1
    Mc = jnp.eye(m, dtype=dt) if (carry_M and not rank1) else S1
    S1, S2, Mc = jax.lax.fori_loop(0, L, body, (S1, S2, Mc))
    e1_ref[...] = S1
    if N == 2 and not rank1 and not refstate:
        e2_ref[...] = S2
    if carry_M and not rank1:
        em_ref[...] = Mc
    if not carry_M:
        em_ref[...] = jnp.zeros_like(em_ref)


def _dense_pass(
    Mc_, s1c, s2c, C1, C2, N, uniform, write, carry_M, backend, num_warps=None, refstate=False
):
    BC, Lc, m = s1c.shape
    dt = s1c.dtype
    rank1 = dt == jnp.float64
    Lm = 1 if uniform else Lc
    spec_M = pl.BlockSpec((None, Lm, m, m), lambda b: (b, 0, 0, 0))
    spec_v = pl.BlockSpec((None, Lc, m), lambda b: (b, 0, 0))
    spec_t = pl.BlockSpec((None, Lc, m, m), lambda b: (b, 0, 0, 0))
    spec_c1 = pl.BlockSpec((None, m), lambda b: (b, 0))
    spec_c2 = pl.BlockSpec((None, m, m), lambda b: (b, 0, 0))
    Lo = Lc if write else 1
    spec_o1 = pl.BlockSpec((None, Lo, m), lambda b: (b, 0, 0))
    spec_o = pl.BlockSpec((None, Lo, m + 1, m), lambda b: (b, 0, 0, 0))
    kern = functools.partial(
        _dense_kernel,
        N=N,
        uniform=uniform,
        write=write,
        carry_M=carry_M,
        rank1=rank1,
        refstate=refstate and not rank1,
    )
    if N == 2:
        ins = (Mc_, s1c, s2c, C1, C2)
        in_specs = [spec_M, spec_v, spec_t, spec_c1, spec_c2]
        out_shape = [
            jax.ShapeDtypeStruct((BC, Lo, m + 1, m), dt),
            jax.ShapeDtypeStruct((BC, m), dt),
            jax.ShapeDtypeStruct((BC, m, m), dt),
            jax.ShapeDtypeStruct((BC, m, m), dt),
            jax.ShapeDtypeStruct((BC, m, m), dt),
        ]
        out_specs = [spec_o, spec_c1, spec_c2, spec_c2, spec_c2]
    else:
        ins = (Mc_, s1c, C1)
        in_specs = [spec_M, spec_v, spec_c1]
        out_shape = [
            jax.ShapeDtypeStruct((BC, Lo, m), dt),
            jax.ShapeDtypeStruct((BC, m), dt),
            jax.ShapeDtypeStruct((BC, m, m), dt),
        ]
        out_specs = [spec_o1, spec_c1, spec_c2]
    cp = plgpu.CompilerParams(num_warps=num_warps) if num_warps else None
    outs = pl.pallas_call(
        kern,
        out_shape=out_shape,
        grid=(BC,),
        in_specs=in_specs,
        out_specs=out_specs,
        **_backend_kw(backend),
        compiler_params=cp,
    )(*ins)
    return outs[:4] if N == 2 else outs


def combine_pallas_dense(M, s1, s2, N=2, chunk=None, backend="triton", num_warps=8, refstate=False):
    """num_warps=8: Triton's default 4 warps spills the m x m carry at m=64 (8x slower)."""
    """M: (B, L, m, m) per-interval operators, or (B, 1, m, m) / (m, m) under a uniform clock.
    s1: (B, L, m); s2: (B, L, m, m) or None. Returns (B, L, m) for N=1 / (B, L, m + m^2) for N=2 (dense flat)."""
    B, L, m = s1.shape
    dt = s1.dtype
    if M.ndim == 2:
        M = jnp.broadcast_to(M[None, None], (B, 1, m, m))
    uniform = M.shape[1] == 1
    chunk = L if chunk is None else min(chunk, L)
    if L % chunk:
        raise ValueError("chunk must divide L")
    C = L // chunk
    Mc_ = (
        jnp.broadcast_to(M[:, None], (B, C, 1, m, m)).reshape(B * C, 1, m, m)
        if uniform
        else M.reshape(B * C, chunk, m, m)
    )
    s1c = s1.reshape(B * C, chunk, m)
    s2c = None if N == 1 else s2.reshape(B * C, chunk, m, m)
    z1 = jnp.zeros((B * C, m), dt)
    z2 = jnp.zeros((B * C, m, m), dt)
    if C == 1:
        outs = _dense_pass(
            Mc_, s1c, s2c, z1, z2, N, uniform, True, False, backend, num_warps, refstate
        )
        return outs[0].reshape(B, L, m) if N == 1 else outs[0].reshape(B, L, (m + 1) * m)
    outs = _dense_pass(Mc_, s1c, s2c, z1, z2, N, uniform, False, True, backend, num_warps, refstate)
    if N == 2:
        _, e1, e2, em = outs
    else:
        _, e1, em = outs
        e2 = None
    em = em.reshape(B, C, m, m)
    e1 = e1.reshape(B, C, m)

    def op(l, r):
        Ml, p1l, q2l = l
        Mr, p1r, q2r = r
        Mn = jnp.einsum("...ab,...bc->...ac", Mr, Ml, precision=HI)
        f1 = jnp.einsum("...ab,...b->...a", Mr, p1l, precision=HI)
        if q2l is None:
            return Mn, f1 + p1r, None
        flowed = jnp.einsum("...ab,...bc,...dc->...ad", Mr, q2l, Mr, precision=HI)
        return Mn, f1 + p1r, flowed + f1[..., :, None] * p1r[..., None, :] + q2r

    if N == 2:
        e2 = e2.reshape(B, C, m, m)
        _, C1, C2 = jax.lax.associative_scan(op, (em, e1, e2), axis=1)
        C2in = jnp.concatenate([jnp.zeros_like(C2[:, :1]), C2[:, :-1]], 1).reshape(B * C, m, m)
    else:

        def op1(l, r):
            Ml, p1l = l
            Mr, p1r = r
            return jnp.einsum("...ab,...bc->...ac", Mr, Ml, precision=HI), jnp.einsum(
                "...ab,...b->...a", Mr, p1l, precision=HI
            ) + p1r

        _, C1 = jax.lax.associative_scan(op1, (em, e1), axis=1)
        C2in = z2
    C1in = jnp.concatenate([jnp.zeros_like(C1[:, :1]), C1[:, :-1]], 1).reshape(B * C, m)
    outs = _dense_pass(
        Mc_, s1c, s2c, C1in, C2in, N, uniform, True, False, backend, num_warps, refstate
    )
    return outs[0].reshape(B, L, m) if N == 1 else outs[0].reshape(B, L, (m + 1) * m)


def dense_assoc_ref(M, s1, s2, N):
    """Pure-JAX associative scan with dense operators, same inputs as combine_pallas_dense (reference)."""
    B, L, m = s1.shape
    if M.ndim == 2:
        M = jnp.broadcast_to(M[None, None], (B, 1, m, m))
    if M.shape[1] == 1:
        M = jnp.broadcast_to(M, (B, L, m, m))

    def op(l, r):
        Ml, p1l, q2l = l
        Mr, p1r, q2r = r
        Mn = jnp.einsum("...ab,...bc->...ac", Mr, Ml, precision=HI)
        f1 = jnp.einsum("...ab,...b->...a", Mr, p1l, precision=HI)
        if N == 1:
            return Mn, f1 + p1r, q2r
        flowed = jnp.einsum("...ab,...bc,...dc->...ad", Mr, q2l, Mr, precision=HI)
        return Mn, f1 + p1r, flowed + f1[..., :, None] * p1r[..., None, :] + q2r

    s2_ = jnp.zeros((B, L, 1, 1), s1.dtype) if N == 1 else s2
    _, P1, Q = jax.lax.associative_scan(op, (M, s1, s2_), axis=1)
    return P1 if N == 1 else jnp.concatenate([P1, Q.reshape(B, L, m * m)], -1)


def block_to_dense_M(E):
    """(..., K, b, b) block operators -> (..., m, m) block-diagonal dense (m = b K)."""
    K, bsz = E.shape[-3], E.shape[-1]
    eye = jnp.eye(K, dtype=E.dtype)
    D = eye[:, :, None, None] * E[..., :, None, :, :]  # (..., K, K, b, b)
    return jnp.swapaxes(D, -3, -2).reshape(E.shape[:-3] + (bsz * K, bsz * K))


# =========================================================================== dense mode, general depth n
# Level k (k >= 2) is held as a 2-D ref view (m^{k-1}, m) = the row-major flatten of the m^k tensor (the
# original engine's pack_signatures layout). Per step, with M = M_t:
#   F_k = M^{(x)k} S_k        (apply M along every axis: slab-by-slab (m, m) matmuls, strided slabs for inner axes)
#   S_k' = F_k + sum_{j=1}^{k-1} F_j (x) s_{k-j} + s_k       (outer products in (m, m) tiles)
# f32: pl.dot (TF32 off).  f64: rank-1 loops from refs (Triton has no f64 MMA).  All state in refs (L2-resident),
# barriers after every store (the program's threads share the refs).  Chunking: pass 1 (ends + M_chunk only),
# pass 2 = the original engine's packed Chen op over the C chunk summaries, pass 3 recompute with the true carry.


def _mm_left(M, rows, m, rank1, mcol):
    """M @ S for an (m, m) slab S given as a value (dot path) or as a row loader k -> S[k, :] (rank-1 path)."""
    if rank1:
        return jax.lax.fori_loop(
            0,
            m,
            lambda k, acc: acc + mcol(k)[:, None] * rows(k)[None, :],
            jnp.zeros((m, m), M.dtype),
        )
    return pl.dot(M, rows, allow_tf32=False)


def _dense_kernel_general(*refs, N, m, uniform, write, carry_M, rank1):
    # refs: m_ref, s_1..s_N (local), c_1..c_N (carry), [o_1..o_N if write], e_1..e_N (ends/state), em, tmp_1..tmp_N
    it = iter(refs)
    m_ref = next(it)
    s_refs = [next(it) for _ in range(N)]
    c_refs = [next(it) for _ in range(N)]
    o_refs = [next(it) for _ in range(N)] if write else [None] * N
    e_refs = [next(it) for _ in range(N)]
    em_ref = next(it)
    tmp = [next(it) for _ in range(N)]
    tmp2 = [next(it) for _ in range(N)]
    L = s_refs[0].shape[0]
    dt = s_refs[0].dtype
    bar = plgpu.debug_barrier

    def Mt(t):
        return m_ref[0, :, :] if uniform else m_ref[t, :, :]

    def mcol_fn(t):
        return (lambda k: m_ref[0, :, k]) if uniform else (lambda k: m_ref[t, :, k])

    # init state: e_k <- c_k ; em <- I
    for k in range(N):
        e_refs[k][...] = c_refs[k][...]
    if carry_M:
        em_ref[...] = jnp.eye(m, dtype=dt)
    bar()

    def flow_level(t, k):
        """F <- M^{(x)(k+1)} e[k] for level index k (0-based). Ping-pong between tmp[k] and tmp2[k]: every axis reads
        one buffer and writes the other (no intra-slab hazards), one barrier per axis. Returns the buffer holding F."""
        M = Mt(t)
        mcol = mcol_fn(t)
        nrow = m**k
        src = e_refs[k]
        if k == 0:  # level 1: vector
            tmp[0][0, :] = jnp.sum(M * src[0, :][None, :], axis=1)
            bar()
            return tmp[0]
        bufs = [tmp[k], tmp2[k]]
        # last axis: slab @ M^T, src -> bufs[0]
        nslab = nrow // m

        def last_axis(q, c):
            r0 = pl.multiple_of(q * m, m)
            if rank1:
                acc = jax.lax.fori_loop(
                    0,
                    m,
                    lambda c_, acc: acc + src[pl.ds(r0, m), c_][:, None] * mcol(c_)[None, :],
                    jnp.zeros((m, m), dt),
                )
            else:
                acc = pl.dot(src[pl.ds(r0, m), :], M, trans_b=True, allow_tf32=False)
            bufs[0][pl.ds(r0, m), :] = acc
            return c

        jax.lax.fori_loop(0, nslab, last_axis, 0)
        bar()
        cur = 0
        # inner axes i = 0..k-1 of the row multi-index: rows base + v * stride, stride = m^{k-1-i}; read bufs[cur], write bufs[1-cur]
        for i in range(k):
            stride = m ** (k - 1 - i)
            ncomb = m ** (k - 1)
            rd, wr = bufs[cur], bufs[1 - cur]

            def inner(c, carry, stride=stride, rd=rd, wr=wr):
                high = c // stride
                low = c - high * stride
                base = high * (stride * m) + low
                if rank1:
                    acc = jax.lax.fori_loop(
                        0,
                        m,
                        lambda v, acc: acc + mcol(v)[:, None] * rd[base + v * stride, :][None, :],
                        jnp.zeros((m, m), dt),
                    )
                else:
                    acc = pl.dot(M, rd[pl.ds(base, m, stride=stride), :], allow_tf32=False)
                wr[pl.ds(base, m, stride=stride), :] = acc
                return carry

            jax.lax.fori_loop(0, ncomb, inner, 0)
            bar()
            cur = 1 - cur
        return bufs[cur]

    def body(t, carry):
        F = [flow_level(t, k) for k in range(N)]
        if carry_M:
            M = Mt(t)
            mcol = mcol_fn(t)
            Mn = _mm_left(M, (lambda k_: em_ref[k_, :]) if rank1 else em_ref[...], m, rank1, mcol)
            bar()
            em_ref[...] = Mn
            bar()
        # assemble S_k' = F_k + sum_j F_j (x) s_{k-j} + s_k   (levels 1-based: k = 1..N)
        for k in range(1, N + 1):
            kk = k - 1
            nrow = m**kk
            e_k, o_k, s_k, F_k = e_refs[kk], o_refs[kk], s_refs[kk], F[kk]
            if k == 1:
                v = F_k[0, :] + s_k[t, 0, :]
                e_k[0, :] = v
                if write:
                    o_k[t, 0, :] = v
                continue
            nslab = nrow // m

            def slab(q, c, k=k, kk=kk, e_k=e_k, o_k=o_k, s_k=s_k, F_k=F_k):
                r0 = pl.multiple_of(q * m, m)
                acc = F_k[pl.ds(r0, m), :] + s_k[t, pl.ds(r0, m), :]
                # cross terms F_j (x) s_{k-j}: flat row r = a * nb + b_row with a in [0, m^j), b_row in [0, nb = m^{k-j-1})
                late = []
                for j in range(1, k):
                    Fj = F[j - 1]  # (m^{j-1}, m) view of F_j
                    s_kj = s_refs[k - j - 1]  # (Lc, nb, m)
                    nb = m ** (k - j - 1)
                    if nb >= m:  # slab inside one a
                        a = r0 // nb
                        b0 = r0 - a * nb
                        fa = Fj[a // m, a - (a // m) * m]
                        acc = acc + fa * s_kj[t, pl.ds(b0, m), :]
                    else:  # slab spans m // nb values of a
                        late.append((Fj, s_kj, nb))
                e_k[pl.ds(r0, m), :] = acc
                bar()
                for Fj, s_kj, nb in late:
                    na = m // nb
                    a0 = r0 // nb
                    for u in range(na):
                        a = a0 + u
                        fa = Fj[a // m, a - (a // m) * m]
                        rows = pl.ds(r0 + u * nb, nb)
                        e_k[rows, :] = e_k[rows, :] + fa * s_kj[t, :, :]
                    bar()
                if write:
                    o_k[t, pl.ds(r0, m), :] = e_k[pl.ds(r0, m), :]
                return c

            jax.lax.fori_loop(0, nslab, slab, 0)
        bar()
        return carry

    jax.lax.fori_loop(0, L, body, 0)
    if not carry_M:
        em_ref[...] = jnp.zeros_like(em_ref)


def _dense_pass_general(Mc_, locals_, carries, N, uniform, write, carry_M, backend, num_warps=8):
    BC, Lc = locals_[0].shape[:2]
    m = locals_[0].shape[-1]
    dt = locals_[0].dtype
    rank1 = dt == jnp.float64 or m < 16  # Triton dot needs every dim >= 16
    Lm = 1 if uniform else Lc
    Lo = Lc if write else 1

    def spec3(n1, n2):
        return pl.BlockSpec((None, n1, n2, m), lambda b: (b, 0, 0, 0))

    def spec2(n1):
        return pl.BlockSpec((None, n1, m), lambda b: (b, 0, 0))

    in_specs = [pl.BlockSpec((None, Lm, m, m), lambda b: (b, 0, 0, 0))]
    in_specs += [spec3(Lc, m ** (k - 1)) for k in range(1, N + 1)]
    in_specs += [spec2(m ** (k - 1)) for k in range(1, N + 1)]
    out_shape, out_specs = [], []
    if write:
        for k in range(1, N + 1):
            out_shape.append(jax.ShapeDtypeStruct((BC, Lo, m ** (k - 1), m), dt))
            out_specs.append(spec3(Lo, m ** (k - 1)))
    for k in range(1, N + 1):
        out_shape.append(jax.ShapeDtypeStruct((BC, m ** (k - 1), m), dt))
        out_specs.append(spec2(m ** (k - 1)))
    out_shape.append(jax.ShapeDtypeStruct((BC, m, m), dt))
    out_specs.append(spec2(m))  # em
    for _ in range(2):  # tmp, tmp2
        for k in range(1, N + 1):
            out_shape.append(jax.ShapeDtypeStruct((BC, m ** (k - 1), m), dt))
            out_specs.append(spec2(m ** (k - 1)))
    kern = functools.partial(
        _dense_kernel_general, N=N, m=m, uniform=uniform, write=write, carry_M=carry_M, rank1=rank1
    )
    cp = plgpu.CompilerParams(num_warps=num_warps) if num_warps else None
    outs = pl.pallas_call(
        kern,
        out_shape=out_shape,
        grid=(BC,),
        in_specs=in_specs,
        out_specs=out_specs,
        **_backend_kw(backend),
        compiler_params=cp,
    )(Mc_, *locals_, *carries)
    outs = list(outs)
    o = outs[:N] if write else []
    e = outs[N : 2 * N] if write else outs[:N]
    em = outs[2 * N] if write else outs[N]
    return o, e, em


def combine_pallas_dense_general(
    M, levels, chunk=None, backend="triton", num_warps=8, final_only=False
):
    """Any depth N = len(levels). M: (B, L, m, m) or (B, 1, m, m) / (m, m) (uniform). levels[k-1]: local level k as
    (B, L, m^k) flat (pack_signatures layout) or any shape with leading (B, L). Returns (B, L, sum_k m^k) packed."""
    from ews.core.chen_scan import chen_binary_op_packed_with_M

    N = len(levels)
    B, L = levels[0].shape[:2]
    m = levels[0].shape[-1] if levels[0].ndim == 3 else int(round(np.prod(levels[0].shape[2:])))
    dt = levels[0].dtype
    if M.ndim == 2:
        M = jnp.broadcast_to(M[None, None], (B, 1, m, m))
    uniform = M.shape[1] == 1
    chunk = L if chunk is None else min(chunk, L)
    if L % chunk:
        raise ValueError("chunk must divide L")
    C = L // chunk
    Mc_ = (
        jnp.broadcast_to(M[:, None], (B, C, 1, m, m)).reshape(B * C, 1, m, m)
        if uniform
        else M.reshape(B * C, chunk, m, m)
    )
    loc = [levels[k - 1].reshape(B * C, chunk, m ** (k - 1), m) for k in range(1, N + 1)]
    zeros = [jnp.zeros((B * C, m ** (k - 1), m), dt) for k in range(1, N + 1)]

    def pack(o):
        return jnp.concatenate([x.reshape(B, L, -1) for x in o], -1)

    if C == 1 and not final_only:
        o, _, _ = _dense_pass_general(Mc_, loc, zeros, N, uniform, True, False, backend, num_warps)
        return pack(o)
    _, e, em = _dense_pass_general(Mc_, loc, zeros, N, uniform, False, C > 1, backend, num_warps)
    packed = jnp.concatenate([x.reshape(B, C, -1) for x in e], -1)  # (B, C, D)
    if final_only and C == 1:
        return packed[:, 0]  # (B, D): no output stream
    em = em.reshape(B, C, m, m)
    h = jnp.zeros((B, C), dt)

    def op(l, r):
        f = lambda pl_, hl, Ml, pr, hr, Mr: chen_binary_op_packed_with_M(
            (pl_, hl, Ml), (pr, hr, Mr), None, m, N
        )
        return jax.vmap(jax.vmap(f))(*l, *r)

    with jax.default_matmul_precision("highest"):
        Cpacked, _, _ = jax.lax.associative_scan(op, (packed, h, em), axis=1)
    if final_only:
        return Cpacked[:, -1]  # (B, D): pass 1 + reduce only
    Cin = jnp.concatenate([jnp.zeros_like(Cpacked[:, :1]), Cpacked[:, :-1]], 1).reshape(B * C, -1)
    carries, start = [], 0
    for k in range(1, N + 1):
        carries.append(Cin[:, start : start + m**k].reshape(B * C, m ** (k - 1), m))
        start += m**k
    o, _, _ = _dense_pass_general(Mc_, loc, carries, N, uniform, True, False, backend, num_warps)
    return pack(o)


def dense_assoc_general_ref(M, levels):
    """Lean parallel prefix scan (jax.lax.associative_scan) for any depth on given operators M, using the original
    engine's packed Chen op (signax) per node: the like-for-like JAX opponent of combine_pallas_dense_general."""
    from ews.core.chen_scan import chen_binary_op_packed_with_M

    N = len(levels)
    B, L = levels[0].shape[:2]
    m = levels[0].shape[-1] if levels[0].ndim == 3 else levels[0].shape[2]
    if M.ndim == 2:
        M = jnp.broadcast_to(M[None, None], (B, 1, m, m))
    if M.shape[1] == 1:
        M = jnp.broadcast_to(M, (B, L, m, m))
    packed = jnp.concatenate([l.reshape(B, L, -1) for l in levels], -1)
    h = jnp.zeros((B, L), packed.dtype)

    def op(l, r):
        f = lambda pl_, hl, Ml, pr, hr, Mr: chen_binary_op_packed_with_M(
            (pl_, hl, Ml), (pr, hr, Mr), None, m, N
        )
        return jax.vmap(jax.vmap(f))(*l, *r)

    out, _, _ = jax.lax.associative_scan(op, (packed, h, M), axis=1)
    return out


# =========================================================================== block mode, general depth n
# Block-diagonal M (K 2x2 blocks E_j) acts on each tensor index separately, so flowing level k costs K m^k
# elementwise ops instead of m^{k+1} matmul flops, needs no matmul at all (one code path for f32 and f64), and the
# first index of every level is never mixed across planes: each program owns first-index planes j0..j0+tile of
# every level (row tile), grid = (sample x chunk, K / tile). Level k (k >= 2) state view: (2 tile m^{k-2}, m).
# Cross terms are the same outer-product tiles as in the dense general kernel (restricted to the tile's rows).


def _block_kernel_general(*refs, N, m, tile, uniform, write, carry_M):
    K = m // 2
    it = iter(refs)
    e_ref = next(it)  # (Lm, K, 2, 2)
    s_refs = [next(it) for _ in range(N)]  # level k: (Lc, m^{k-1}, m) full (level 1: (Lc, 1, m))
    c_refs = [
        next(it) for _ in range(N)
    ]  # carries, tile rows: level 1 (1, 2 tile), k>=2 (rows_k, m)
    o_refs = [next(it) for _ in range(N)] if write else [None] * N
    st = [next(it) for _ in range(N)]  # state / chunk ends (tile rows)
    tmp = [next(it) for _ in range(N)]  # flowed levels (tile rows), ping-pong buffer A
    tmp2 = [next(it) for _ in range(N)]  # ping-pong buffer B
    L = s_refs[0].shape[0]
    j0 = pl.program_id(1) * tile
    bar = plgpu.debug_barrier

    def rows_of(k):  # rows of the (., m) view of level k (k >= 2) in this tile
        return 2 * tile * m ** (k - 2)

    def row0_of(k):  # first global row of this tile in level k's full view
        return 2 * j0 * m ** (k - 2)

    def Et(t, a, b, sl):
        return e_ref[0 if uniform else t, sl, a, b]

    for k in range(N):
        st[k][...] = c_refs[k][...]
    bar()

    def pair_mix(R0, R1, e00, e01, e10, e11):
        return e00 * R0 + e01 * R1, e10 * R0 + e11 * R1

    def flow_level(t, k):
        """F <- (E^{(x)k}) st[k-1] for level k (1-based); ping-pong between tmp and tmp2, one barrier per axis.
        Returns the buffer holding F."""
        src = st[k - 1]
        tl_ = pl.ds(j0, tile)
        if k == 1:
            v0 = src[0, pl.ds(0, tile, stride=2)]
            v1 = src[0, pl.ds(1, tile, stride=2)]
            n0, n1 = pair_mix(
                v0, v1, Et(t, 0, 0, tl_), Et(t, 0, 1, tl_), Et(t, 1, 0, tl_), Et(t, 1, 1, tl_)
            )
            tmp[0][0, pl.ds(0, tile, stride=2)] = n0
            tmp[0][0, pl.ds(1, tile, stride=2)] = n1
            bar()
            return tmp[0]
        nrows = rows_of(k)
        bufs = [tmp[k - 1], tmp2[k - 1]]
        full = slice(None)
        c00 = Et(t, 0, 0, full)[None, :]
        c01 = Et(t, 0, 1, full)[None, :]
        c10 = Et(t, 1, 0, full)[None, :]
        c11 = Et(t, 1, 1, full)[None, :]  # (1, K) column coefficients
        # last axis (columns): src -> bufs[0], slabs of sr rows
        sr = min(4 * m, nrows)

        def cols(q, c):
            r0 = pl.multiple_of(q * sr, sr)
            C0 = src[pl.ds(r0, sr), pl.ds(0, K, stride=2)]
            C1 = src[pl.ds(r0, sr), pl.ds(1, K, stride=2)]
            N0, N1 = pair_mix(C0, C1, c00, c01, c10, c11)
            bufs[0][pl.ds(r0, sr), pl.ds(0, K, stride=2)] = N0
            bufs[0][pl.ds(r0, sr), pl.ds(1, K, stride=2)] = N1
            return c

        jax.lax.fori_loop(0, nrows // sr, cols, 0)
        bar()
        cur = 0
        # axis 0 (first index, tile-local planes): pairs (2u, 2u+1) u < tile, stride m^{k-2}
        stride0 = m ** (k - 2)
        t00 = Et(t, 0, 0, tl_)[:, None]
        t01 = Et(t, 0, 1, tl_)[:, None]
        t10 = Et(t, 1, 0, tl_)[:, None]
        t11 = Et(t, 1, 1, tl_)[:, None]
        rd, wr = bufs[cur], bufs[1 - cur]

        def ax0(base, c, rd=rd, wr=wr):
            R0 = rd[pl.ds(base, tile, stride=2 * stride0), :]
            R1 = rd[pl.ds(base + stride0, tile, stride=2 * stride0), :]
            N0, N1 = pair_mix(R0, R1, t00, t01, t10, t11)
            wr[pl.ds(base, tile, stride=2 * stride0), :] = N0
            wr[pl.ds(base + stride0, tile, stride=2 * stride0), :] = N1
            return c

        jax.lax.fori_loop(0, stride0, ax0, 0)
        bar()
        cur = 1 - cur
        # inner row axes a = 1..k-2 (full K planes): stride_a = m^{k-2-a}; other-index combos = nrows / m
        f00 = Et(t, 0, 0, full)[:, None]
        f01 = Et(t, 0, 1, full)[:, None]
        f10 = Et(t, 1, 0, full)[:, None]
        f11 = Et(t, 1, 1, full)[:, None]
        for a in range(1, k - 1):
            stride = m ** (k - 2 - a)
            rd, wr = bufs[cur], bufs[1 - cur]

            def axa(c, carry, stride=stride, rd=rd, wr=wr):
                high = c // stride
                low = c - high * stride
                base = high * (stride * m) + low
                R0 = rd[pl.ds(base, K, stride=2 * stride), :]
                R1 = rd[pl.ds(base + stride, K, stride=2 * stride), :]
                N0, N1 = pair_mix(R0, R1, f00, f01, f10, f11)
                wr[pl.ds(base, K, stride=2 * stride), :] = N0
                wr[pl.ds(base + stride, K, stride=2 * stride), :] = N1
                return carry

            jax.lax.fori_loop(0, nrows // m, axa, 0)
            bar()
            cur = 1 - cur
        return bufs[cur]

    def body(t, carry):
        F = [flow_level(t, k) for k in range(1, N + 1)]

        def Fj_scalar(j, a):
            """F_j[a] for the global flat index a of level j (its first index lies in this tile)."""
            if j == 1:
                return F[0][0, a - 2 * j0]
            a_loc = a - 2 * j0 * m ** (j - 1)
            return F[j - 1][a_loc // m, a_loc - (a_loc // m) * m]

        # level 1
        v = F[0][0, :] + s_refs[0][t, 0, pl.ds(2 * j0, 2 * tile)]
        st[0][0, :] = v
        if write:
            o_refs[0][t, pl.ds(2 * j0, 2 * tile)] = v
        for k in range(2, N + 1):
            nrows = rows_of(k)
            R0g = row0_of(k)
            sr = min(m, nrows)
            e_k, o_k, s_k, F_k = st[k - 1], o_refs[k - 1], s_refs[k - 1], F[k - 1]

            def slab(
                q,
                c,
                k=k,
                e_k=e_k,
                o_k=o_k,
                s_k=s_k,
                F_k=F_k,
                nrows=nrows,
                R0g=R0g,
                sr=sr,
                Fj_scalar=Fj_scalar,
            ):
                r0 = pl.multiple_of(q * sr, sr)  # tile-local row
                g0 = R0g + r0  # global row
                acc = F_k[pl.ds(r0, sr), :] + s_k[t, pl.ds(g0, sr), :]
                late = []
                for j in range(1, k):
                    s_kj = s_refs[k - j - 1]
                    nb = m ** (k - j - 1)
                    if nb >= sr:
                        a = g0 // nb
                        b0 = g0 - a * nb
                        acc = acc + Fj_scalar(j, a) * s_kj[t, pl.ds(b0, sr), :]
                    else:
                        late.append((j, s_kj, nb))
                e_k[pl.ds(r0, sr), :] = acc
                bar()
                for j, s_kj, nb in late:
                    na = sr // nb
                    a0 = g0 // nb
                    for u in range(na):
                        rows = pl.ds(r0 + u * nb, nb)
                        e_k[rows, :] = e_k[rows, :] + Fj_scalar(j, a0 + u) * s_kj[t, :, :]
                    bar()
                if write:
                    o_k[t, pl.ds(g0, sr), :] = e_k[pl.ds(r0, sr), :]
                return c

            jax.lax.fori_loop(0, nrows // sr, slab, 0)
        bar()
        return carry

    jax.lax.fori_loop(0, L, body, 0)


def _block_pass_general(Ec, locals_, carries, N, tile, uniform, write, backend, num_warps=4):
    BC, Lc = locals_[0].shape[:2]
    m = locals_[0].shape[-1]
    K = m // 2
    nt = K // tile
    dt = locals_[0].dtype
    Lm = 1 if uniform else Lc
    Lo = Lc if write else 1
    rows = [2 * tile] + [2 * tile * m ** (k - 2) for k in range(2, N + 1)]  # tile rows per level
    in_specs = [pl.BlockSpec((None, Lm, K, 2, 2), lambda b, i: (b, 0, 0, 0, 0))]
    in_specs += [
        pl.BlockSpec((None, Lc, m ** (k - 1), m), lambda b, i: (b, 0, 0, 0))
        for k in range(1, N + 1)
    ]

    def tspec(k):
        return (
            pl.BlockSpec((None, None, 1, rows[0]), lambda b, i: (b, i, 0, 0))
            if k == 1
            else pl.BlockSpec((None, None, rows[k - 1], m), lambda b, i: (b, i, 0, 0))
        )

    def tshape(k):
        return (
            jax.ShapeDtypeStruct((BC, nt, 1, rows[0]), dt)
            if k == 1
            else jax.ShapeDtypeStruct((BC, nt, rows[k - 1], m), dt)
        )

    in_specs += [tspec(k) for k in range(1, N + 1)]
    out_shape, out_specs = [], []
    if write:
        out_shape.append(jax.ShapeDtypeStruct((BC, Lo, m), dt))
        out_specs.append(pl.BlockSpec((None, Lo, m), lambda b, i: (b, 0, 0)))
        for k in range(2, N + 1):
            out_shape.append(jax.ShapeDtypeStruct((BC, Lo, m ** (k - 1), m), dt))
            out_specs.append(pl.BlockSpec((None, Lo, m ** (k - 1), m), lambda b, i: (b, 0, 0, 0)))
    for _ in range(3):  # state/ends, tmp, tmp2
        for k in range(1, N + 1):
            out_shape.append(tshape(k))
            out_specs.append(tspec(k))
    kern = functools.partial(
        _block_kernel_general, N=N, m=m, tile=tile, uniform=uniform, write=write, carry_M=False
    )
    cp = plgpu.CompilerParams(num_warps=num_warps) if num_warps else None
    outs = list(
        pl.pallas_call(
            kern,
            out_shape=out_shape,
            grid=(BC, nt),
            in_specs=in_specs,
            out_specs=out_specs,
            **_backend_kw(backend),
            compiler_params=cp,
        )(Ec, *locals_, *carries)
    )
    o = outs[:N] if write else []
    e = outs[N : 2 * N] if write else outs[:N]
    return o, e


def combine_pallas_block_general(
    E, levels, tile=4, chunk=None, backend="triton", num_warps=4, final_only=False
):
    """Any depth with 2x2-block operators E (B, L, K, 2, 2) (or (B, 1, K, 2, 2) uniform). levels[k-1] = local
    level k, (B, L, m^k) flat or any shape with leading (B, L). Returns (B, L, sum_k m^k) packed (pack_signatures)."""
    from ews.core.chen_scan import chen_binary_op_packed_with_M

    N = len(levels)
    B, L = levels[0].shape[:2]
    K = E.shape[-3]
    m = 2 * K
    dt = levels[0].dtype
    tile = min(tile, K)
    if K % tile:
        raise ValueError("tile must divide K")
    nt = K // tile
    uniform = E.shape[1] == 1
    chunk = L if chunk is None else min(chunk, L)
    if L % chunk:
        raise ValueError("chunk must divide L")
    C = L // chunk
    Ec = (
        jnp.broadcast_to(E[:, None], (B, C, 1, K, 2, 2)).reshape(B * C, 1, K, 2, 2)
        if uniform
        else E.reshape(B * C, chunk, K, 2, 2)
    )
    loc = [levels[k - 1].reshape(B * C, chunk, m ** (k - 1), m) for k in range(1, N + 1)]

    def tiled(packed_rows):
        """(BC, D) packed carry -> per-level tile-row arrays."""
        outs, start = [], 0
        for k in range(1, N + 1):
            lv = packed_rows[:, start : start + m**k]
            start += m**k
            if k == 1:
                outs.append(lv.reshape(B * C, nt, 1, 2 * tile))
            else:
                outs.append(lv.reshape(B * C, nt, 2 * tile * m ** (k - 2), m))
        return outs

    zeros = tiled(jnp.zeros((B * C, sum(m**k for k in range(1, N + 1))), dt))

    def pack(o):
        return jnp.concatenate([x.reshape(B, L, -1) for x in o], -1)

    if C == 1 and not final_only:
        o, _ = _block_pass_general(Ec, loc, zeros, N, tile, uniform, True, backend, num_warps)
        return pack(o)
    _, e = _block_pass_general(Ec, loc, zeros, N, tile, uniform, False, backend, num_warps)
    packed = jnp.concatenate([x.reshape(B, C, -1) for x in e], -1)  # (B, C, D)
    if final_only and C == 1:
        return packed[:, 0]
    Efull = (
        jnp.broadcast_to(E[:, None], (B, C, chunk, K, 2, 2))
        if uniform
        else E.reshape(B, C, chunk, K, 2, 2)
    )
    Echunk = _cum_ops(Efull)[:, :, -1]  # (B, C, K, 2, 2)
    Mchunk = block_to_dense_M(Echunk)  # (B, C, m, m) small
    h = jnp.zeros((B, C), dt)

    def op(l, r):
        f = lambda pl_, hl, Ml, pr, hr, Mr: chen_binary_op_packed_with_M(
            (pl_, hl, Ml), (pr, hr, Mr), None, m, N
        )
        return jax.vmap(jax.vmap(f))(*l, *r)

    with jax.default_matmul_precision("highest"):
        Cpacked, _, _ = jax.lax.associative_scan(op, (packed, h, Mchunk), axis=1)
    if final_only:
        return Cpacked[:, -1]
    Cin = jnp.concatenate([jnp.zeros_like(Cpacked[:, :1]), Cpacked[:, :-1]], 1).reshape(B * C, -1)
    o, _ = _block_pass_general(Ec, loc, tiled(Cin), N, tile, uniform, True, backend, num_warps)
    return pack(o)


# =========================================================================== block mode, any depth, register state
# The same algorithm as _block_kernel_general but with the state held in registers as parity tiles: level k is the
# 2^k tiles T_k[p] (p in {0,1}^k) of shape (tile, K, ..., K) [k-1 K's], T_k[p][j, i_1, .., i_{k-1}] =
# S_k[2j+p_0, 2i_1+p_1, ..]. Flowing along axis a is a pair-mix over the a-th parity bit with the E coefficients
# broadcast along that axis; cross terms are outer products of tiles; local terms are read with one stride-2 ds per
# axis; outputs written the same way. No buffers, no barriers. Chosen automatically when the tiles fit in
# registers (2^k tile K^{k-1} per level); otherwise the buffered kernel above is used.
import itertools


def _block_kernel_parity(*refs, N, m, tile, uniform, write, b=2):
    K = m // b
    it = iter(refs)
    e_ref = next(it)  # (Lm, K, b, b)
    s_refs = [next(it) for _ in range(N)]  # level k: (Lc, m, ..., m) k-dim
    c_refs = [next(it) for _ in range(N)]  # carries: (m, ..., m) k-dim
    o_refs = [next(it) for _ in range(N)] if write else [None] * N
    e_out = [next(it) for _ in range(N)]  # ends: (m, ..., m) k-dim
    L = s_refs[0].shape[0]
    j0 = pl.program_id(1) * tile
    tl_ = pl.ds(j0, tile)
    parities = {k: list(itertools.product(range(b), repeat=k)) for k in range(1, N + 1)}

    def idx(p, first_tile=True):
        """strided index tuple for residue class p: tile blocks on axis 0 (or full K), full K blocks on the others."""
        first = pl.ds(b * j0 + p[0], tile, stride=b) if first_tile else pl.ds(p[0], K, stride=b)
        return (first,) + tuple(pl.ds(q, K, stride=b) for q in p[1:])

    def coef(t, a, x, y, k):
        """E entry (x, y) along axis a of a level-k tile, broadcast to that tile's shape."""
        v = e_ref[0 if uniform else t, tl_ if a == 0 else slice(None), x, y]
        shape = [1] * k
        shape[a] = tile if a == 0 else K
        return v.reshape(shape) if k > 1 else v

    def flow(t, k, T):
        """T (dict residue class -> tile) of level k flowed along every axis: b-term mix over that axis's residue."""
        for a in range(k):
            new = {}
            for p in parities[k]:
                acc = None
                for y in range(b):
                    q = p[:a] + (y,) + p[a + 1 :]
                    term = coef(t, a, p[a], y, k) * T[q]
                    acc = term if acc is None else acc + term
                new[p] = acc
            T = new
        return T

    def local(t, k, p):
        return s_refs[k - 1][(t,) + idx(p)]

    def body(t, S):
        F = [flow(t, k, S[k - 1]) for k in range(1, N + 1)]
        newS = []
        for k in range(1, N + 1):
            T = {}
            for p in parities[k]:
                acc = F[k - 1][p] + local(t, k, p)
                for j in range(1, k):
                    pj, pr = p[:j], p[j:]
                    fj = F[j - 1][pj]  # (tile, K^{j-1})
                    sr = s_refs[k - j - 1][
                        (t,) + idx(pr, first_tile=False)
                    ]  # (K^{k-j}) full planes
                    acc = acc + fj.reshape(fj.shape + (1,) * (k - j)) * sr.reshape(
                        (1,) * j + sr.shape
                    )
                T[p] = acc
                if write:
                    o_refs[k - 1][(t,) + idx(p)] = acc
            newS.append(T)
        return newS

    S0 = [{p: c_refs[k - 1][idx(p)] for p in parities[k]} for k in range(1, N + 1)]
    S = jax.lax.fori_loop(0, L, body, S0)
    for k in range(1, N + 1):
        for p in parities[k]:
            e_out[k - 1][idx(p)] = S[k - 1][p]


def _parity_fits(N, m, tile, budget=6144, b=2):
    K = m // b
    return all(b**k * tile * K ** (k - 1) <= budget for k in range(1, N + 1))


def _block_pass_parity(Ec, locals_, carries, N, tile, uniform, write, backend, num_warps=4):
    """locals_[k-1]: (BC, Lc, m, .., m); carries[k-1]: (BC, m, .., m). Returns (outputs or [], ends)."""
    BC, Lc = locals_[0].shape[:2]
    m = locals_[0].shape[2]
    bsz = Ec.shape[-1]
    K = m // bsz
    nt = K // tile
    dt = locals_[0].dtype
    Lm = 1 if uniform else Lc
    Lo = Lc if write else 1

    def full(nd, lead):  # block = whole array minus the batch axis
        return pl.BlockSpec(
            (None,) + tuple(lead) + (m,) * nd, lambda b, i, n=len(lead) + nd: (b,) + (0,) * n
        )

    in_specs = [pl.BlockSpec((None, Lm, K, bsz, bsz), lambda b, i: (b, 0, 0, 0, 0))]
    in_specs += [full(k, (Lc,)) for k in range(1, N + 1)]
    in_specs += [full(k, ()) for k in range(1, N + 1)]
    out_shape, out_specs = [], []
    if write:
        for k in range(1, N + 1):
            out_shape.append(jax.ShapeDtypeStruct((BC, Lo) + (m,) * k, dt))
            out_specs.append(full(k, (Lo,)))
    for k in range(1, N + 1):
        out_shape.append(jax.ShapeDtypeStruct((BC,) + (m,) * k, dt))
        out_specs.append(full(k, ()))
    kern = functools.partial(
        _block_kernel_parity, N=N, m=m, tile=tile, uniform=uniform, write=write, b=bsz
    )
    cp = plgpu.CompilerParams(num_warps=num_warps) if num_warps else None
    outs = list(
        pl.pallas_call(
            kern,
            out_shape=out_shape,
            grid=(BC, nt),
            in_specs=in_specs,
            out_specs=out_specs,
            **_backend_kw(backend),
            compiler_params=cp,
        )(Ec, *locals_, *carries)
    )
    return (outs[:N], outs[N:]) if write else ([], outs)


def combine_pallas_block(
    E, levels, tile=4, chunk=None, backend="triton", final_only=False, mode="auto"
):
    """One entry point for block-diagonal A at any depth. mode: 'auto' (registers when the parity tiles fit, else
    buffered), 'parity', 'buffered'. Same contract as combine_pallas_block_general."""
    N = len(levels)
    K = E.shape[-3]
    bsz = E.shape[-1]
    m = bsz * K
    tile = min(tile, K)
    # auto: shrink the tile before giving up on the register path; the buffered path has no custom VJP, so a state that
    # fits at a smaller tile (e.g. 16 planes at depth 3 fit at tile 2, not 4) should still train
    while mode == "auto" and tile > 1 and not _parity_fits(N, m, tile, b=bsz):
        tile //= 2
    if mode == "buffered" or (mode == "auto" and not _parity_fits(N, m, tile, b=bsz)):
        if bsz != 2:
            raise ValueError(
                "buffered block kernel supports 2x2 blocks only; use the dense kernel for larger blocks that do not fit registers"
            )
        return combine_pallas_block_general(
            E, levels, tile=tile, chunk=chunk, backend=backend, final_only=final_only
        )
    from ews.core.chen_scan import chen_binary_op_packed_with_M

    B, L = levels[0].shape[:2]
    dt = levels[0].dtype
    if K % tile:
        raise ValueError("tile must divide K")
    uniform = E.shape[1] == 1
    chunk = L if chunk is None else min(chunk, L)
    if L % chunk:
        raise ValueError("chunk must divide L")
    C = L // chunk
    Ec = (
        jnp.broadcast_to(E[:, None], (B, C, 1, K, bsz, bsz)).reshape(B * C, 1, K, bsz, bsz)
        if uniform
        else E.reshape(B * C, chunk, K, bsz, bsz)
    )
    loc = [levels[k - 1].reshape((B * C, chunk) + (m,) * k) for k in range(1, N + 1)]

    def carries_from(packed_rows):
        outs, start = [], 0
        for k in range(1, N + 1):
            outs.append(packed_rows[:, start : start + m**k].reshape((B * C,) + (m,) * k))
            start += m**k
        return outs

    D = sum(m**k for k in range(1, N + 1))
    zeros = carries_from(jnp.zeros((B * C, D), dt))

    def pack(o):
        return jnp.concatenate([x.reshape(B, L, -1) for x in o], -1)

    if C == 1 and not final_only:
        o, _ = _block_pass_parity(Ec, loc, zeros, N, tile, uniform, True, backend)
        return pack(o)
    _, e = _block_pass_parity(Ec, loc, zeros, N, tile, uniform, False, backend)
    packed = jnp.concatenate([x.reshape(B, C, -1) for x in e], -1)
    if final_only and C == 1:
        return packed[:, 0]
    Efull = (
        jnp.broadcast_to(E[:, None], (B, C, chunk, K, bsz, bsz))
        if uniform
        else E.reshape(B, C, chunk, K, bsz, bsz)
    )
    Mchunk = block_to_dense_M(_cum_ops(Efull)[:, :, -1])
    h = jnp.zeros((B, C), dt)

    def op(l, r):
        f = lambda pl_, hl, Ml, pr, hr, Mr: chen_binary_op_packed_with_M(
            (pl_, hl, Ml), (pr, hr, Mr), None, m, N
        )
        return jax.vmap(jax.vmap(f))(*l, *r)

    with jax.default_matmul_precision("highest"):
        Cpacked, _, _ = jax.lax.associative_scan(op, (packed, h, Mchunk), axis=1)
    if final_only:
        return Cpacked[:, -1]
    Cin = jnp.concatenate([jnp.zeros_like(Cpacked[:, :1]), Cpacked[:, :-1]], 1).reshape(B * C, -1)
    o, _ = _block_pass_parity(Ec, loc, carries_from(Cin), N, tile, uniform, True, backend)
    return pack(o)


# =========================================================================== backward (block mode, register kernel)
# The per-step map S_t = step(Ec_t, loc_t, S_{t-1}) is a pure function on values (coefficient vectors, local parity
# tiles, state tiles), so its adjoint is obtained inside the kernel with jax.vjp (elementwise ops only -> lowers to
# Triton). Reverse kernel per (sample-chunk, row tile): lam <- 0 at the chunk end (or the incoming cotangent),
# for t = Lc-1 .. 0: S_{t-1} from the stored forward stream (prefix mode), lam += g_t, (dS_{t-1}, dEc, dloc) = vjp.
# Cross-program sums (coefficients on the full axes, local tiles with a full first axis) go to per-row-tile partial
# buffers (nt copies of the lower-level streams: nt/m of the top-level stream) and are reduced in JAX.
# Chunking = the forward's two-level scheme transposed: B1 (all chunks, zero incoming lam, in parallel) gives each
# chunk's start-cotangent; B2 (JAX, C small) propagates cotangents across chunk boundaries with jax.vjp of the
# packed Chen op on the saved chunk summaries; B3 (all chunks in parallel, incoming lam, no g) adds the rest.


def _parity_step(Ec, loc, S, N, k_axes):
    """Pure per-step map. Ec: dict (kind, x, y) -> coefficient vector (kind 'tile': (tile,), 'full': (K,)).
    loc: dict level k -> {'tile': {p: tile}, 'full': {q: tile with full first axis}} (full only for k < N).
    S: list over levels of {p: tile}. Returns the new S. Residue classes p in {0..b-1}^k."""
    parities = k_axes["parities"]
    tile = k_axes["tile"]
    K = k_axes["K"]
    bsz = k_axes["b"]

    def coef(a, x, y, k):
        v = Ec[("tile" if a == 0 else "full", x, y)]
        shape = [1] * k
        shape[a] = tile if a == 0 else K
        return v.reshape(shape) if k > 1 else v

    def flow(k, T):
        for a in range(k):
            new = {}
            for p in parities[k]:
                acc = None
                for y in range(bsz):
                    term = coef(a, p[a], y, k) * T[p[:a] + (y,) + p[a + 1 :]]
                    acc = term if acc is None else acc + term
                new[p] = acc
            T = new
        return T

    F = [flow(k, S[k - 1]) for k in range(1, N + 1)]
    newS = []
    for k in range(1, N + 1):
        T = {}
        for p in parities[k]:
            acc = F[k - 1][p] + loc[k]["tile"][p]
            for j in range(1, k):
                fj = F[j - 1][p[:j]]
                sr = loc[k - j]["full"][p[j:]]
                acc = acc + fj.reshape(fj.shape + (1,) * (k - j)) * sr.reshape((1,) * j + sr.shape)
            T[p] = acc
        newS.append(T)
    return newS


def _parity_axes(N, m, tile, b=2):
    return dict(
        parities={k: list(itertools.product(range(b), repeat=k)) for k in range(1, N + 1)},
        tile=tile,
        K=m // b,
        b=b,
    )


def _block_kernel_parity_bwd(*refs, N, m, tile, uniform, has_g, has_lam, b=2):
    K = m // b
    it = iter(refs)
    e_ref = next(it)  # (Lm, K, b, b)
    s_refs = [next(it) for _ in range(N)]  # locals (Lc, m^k)
    c_refs = [next(it) for _ in range(N)]  # carry-in (m^k) = S_{-1}
    o_refs = [next(it) for _ in range(N)]  # forward outputs (Lc, m^k)
    g_refs = [next(it) for _ in range(N)] if has_g else [None] * N  # output cotangents (Lc, m^k)
    l_refs = (
        [next(it) for _ in range(N)] if has_lam else [None] * N
    )  # incoming cotangent at the chunk end (m^k)
    de_tile = next(it)  # (Lm, tile, 2, 2) this program's own planes (tile-axis coefficients)
    de_full = next(it)  # (Lm, K, 2, 2) this program's partial for the full-axis coefficients
    ds_tile = [next(it) for _ in range(N)]  # (Lc, m^k) tile-own cotangent of the locals (direct)
    ds_full = [
        next(it) for _ in range(N - 1)
    ]  # (Lc, m^k) this program's partial for the full-first-axis use
    dc_refs = [next(it) for _ in range(N)]  # cotangent wrt the carry-in (m^k), tile planes
    L = s_refs[0].shape[0]
    dt = s_refs[0].dtype
    j0 = pl.program_id(1) * tile
    tl_ = pl.ds(j0, tile)
    ax = _parity_axes(N, m, tile, b)
    parities = ax["parities"]
    kinds = [("tile", x, y) for x in range(b) for y in range(b)] + [
        ("full", x, y) for x in range(b) for y in range(b)
    ]

    def idx(p, first_tile=True):
        first = pl.ds(b * j0 + p[0], tile, stride=b) if first_tile else pl.ds(p[0], K, stride=b)
        return (first,) + tuple(pl.ds(q, K, stride=b) for q in p[1:])

    def load_E(t):
        tt = 0 if uniform else t
        return {
            (kind, x, y): e_ref[tt, tl_ if kind == "tile" else slice(None), x, y]
            for (kind, x, y) in kinds
        }

    def load_loc(t):
        loc = {}
        for k in range(1, N + 1):
            d = {"tile": {p: s_refs[k - 1][(t,) + idx(p)] for p in parities[k]}}
            if k < N:
                d["full"] = {q: s_refs[k - 1][(t,) + idx(q, first_tile=False)] for q in parities[k]}
            loc[k] = d
        return loc

    def load_state(refs, t):
        return [
            {
                p: (refs[k - 1][(t,) + idx(p)] if t is not None else refs[k - 1][idx(p)])
                for p in parities[k]
            }
            for k in range(1, N + 1)
        ]

    if uniform:
        dE_acc = {key: jnp.zeros_like(v) for key, v in load_E(0).items()}
    lam0 = (
        load_state(l_refs, None)
        if has_lam
        else [
            {p: jnp.zeros((tile,) + (K,) * (k - 1), dt) for p in parities[k]}
            for k in range(1, N + 1)
        ]
    )

    def body(i, carry):
        t = L - 1 - i
        lam = carry[0]
        Ec = load_E(t)
        loc = load_loc(t)
        S_o = load_state(o_refs, jnp.maximum(t - 1, 0))
        S_c = load_state(c_refs, None)
        S_prev = [{p: jnp.where(t > 0, S_o[k][p], S_c[k][p]) for p in S_o[k]} for k in range(N)]
        if has_g:
            g = load_state(g_refs, t)
            lam = [{p: lam[k][p] + g[k][p] for p in lam[k]} for k in range(N)]
        _, vjp = jax.vjp(lambda Ec_, loc_, S_: _parity_step(Ec_, loc_, S_, N, ax), Ec, loc, S_prev)
        dEc, dloc, dS = vjp(lam)
        for k in range(1, N + 1):
            for p in parities[k]:
                ds_tile[k - 1][(t,) + idx(p)] = dloc[k]["tile"][p]
            if k < N:
                for q in parities[k]:
                    ds_full[k - 1][(t,) + idx(q, first_tile=False)] = dloc[k]["full"][q]
        if uniform:
            return (dS, {key: carry[1][key] + dEc[key] for key in dEc})
        for (kind, x, y), v in dEc.items():
            if kind == "tile":
                de_tile[t, :, x, y] = v
            else:
                de_full[t, :, x, y] = v
        return (dS, carry[1])

    init = (lam0, dE_acc if uniform else 0)
    lam_final, dE_acc = jax.lax.fori_loop(0, L, body, init)
    if uniform:
        for (kind, x, y), v in dE_acc.items():
            if kind == "tile":
                de_tile[0, :, x, y] = v
            else:
                de_full[0, :, x, y] = v
    for k in range(1, N + 1):
        for p in parities[k]:
            dc_refs[k - 1][idx(p)] = lam_final[k - 1][p]


def _block_pass_parity_bwd(
    Ec, locals_, carries, outs, gs, lams, N, tile, uniform, backend, num_warps=4
):
    """Returns (dE partial (BC, nt, Lm, K, 2, 2), ds_tile [N], ds_full partial [N-1] (BC, nt, Lc, m^k), dcarry [N])."""
    BC, Lc = locals_[0].shape[:2]
    m = locals_[0].shape[2]
    bsz = Ec.shape[-1]
    K = m // bsz
    nt = K // tile
    dt = locals_[0].dtype
    Lm = 1 if uniform else Lc

    def full(nd, lead, tiled=False):
        if tiled:
            return pl.BlockSpec(
                (None, None) + tuple(lead) + (m,) * nd,
                lambda b, i, n=len(lead) + nd: (b, i) + (0,) * n,
            )
        return pl.BlockSpec(
            (None,) + tuple(lead) + (m,) * nd, lambda b, i, n=len(lead) + nd: (b,) + (0,) * n
        )

    in_specs = [pl.BlockSpec((None, Lm, K, bsz, bsz), lambda b, i: (b, 0, 0, 0, 0))]
    ins = [Ec]
    for group in (locals_, carries, outs, gs, lams):
        if group is None:
            continue
        for k, x in enumerate(group, 1):
            ins.append(x)
            in_specs.append(full(k, x.shape[1:-k]))
    out_shape = [
        jax.ShapeDtypeStruct((BC, Lm, K, bsz, bsz), dt),
        jax.ShapeDtypeStruct((BC, nt, Lm, K, bsz, bsz), dt),
    ]
    out_specs = [
        pl.BlockSpec((None, Lm, tile, bsz, bsz), lambda b, i: (b, 0, i, 0, 0)),
        pl.BlockSpec((None, None, Lm, K, bsz, bsz), lambda b, i: (b, i, 0, 0, 0, 0)),
    ]
    for k in range(1, N + 1):
        out_shape.append(jax.ShapeDtypeStruct((BC, Lc) + (m,) * k, dt))
        out_specs.append(full(k, (Lc,)))
    for k in range(1, N):
        out_shape.append(jax.ShapeDtypeStruct((BC, nt, Lc) + (m,) * k, dt))
        out_specs.append(full(k, (Lc,), tiled=True))
    for k in range(1, N + 1):
        out_shape.append(jax.ShapeDtypeStruct((BC,) + (m,) * k, dt))
        out_specs.append(full(k, ()))
    kern = functools.partial(
        _block_kernel_parity_bwd,
        N=N,
        m=m,
        tile=tile,
        uniform=uniform,
        has_g=gs is not None,
        has_lam=lams is not None,
        b=bsz,
    )
    cp = plgpu.CompilerParams(num_warps=num_warps) if num_warps else None
    outs_ = list(
        pl.pallas_call(
            kern,
            out_shape=out_shape,
            grid=(BC, nt),
            in_specs=in_specs,
            out_specs=out_specs,
            **_backend_kw(backend),
            compiler_params=cp,
        )(*ins)
    )
    dE = (outs_[0], outs_[1])
    ds_tile = outs_[2 : 2 + N]
    ds_full = outs_[2 + N : 2 * N + 1]
    dc = outs_[2 * N + 1 :]
    return dE, ds_tile, ds_full, dc


def block_prefix_vjp(E, levels, tile=4, chunk=None, backend="triton"):
    """Prefix EWS combine (block mode, register kernel) with a custom VJP. Same contract as combine_pallas_block.
    Fails fast when the parity tiles do not fit registers (the register kernel is the only one with a backward)."""
    K, bsz = E.shape[-3], E.shape[-1]
    if not _parity_fits(len(levels), bsz * K, min(tile, K), b=bsz):
        raise NotImplementedError(
            f"block kernel backward: state does not fit registers (b={bsz}, m={bsz * K}, depth={len(levels)})"
        )
    return _block_prefix_cvjp(E, tuple(levels), tile, chunk, backend)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def _block_prefix_cvjp(E, levels, tile, chunk, backend):
    return combine_pallas_block(
        E, list(levels), tile=tile, chunk=chunk, backend=backend, mode="parity"
    )


def _cvjp_fwd(E, levels, tile, chunk, backend):
    out = combine_pallas_block(
        E, list(levels), tile=tile, chunk=chunk, backend=backend, mode="parity"
    )
    return out, (E, levels, out)


def _cvjp_bwd(tile, chunk, backend, res, g):
    from ews.core.chen_scan import chen_binary_op_packed_with_M

    E, levels, out = res
    N = len(levels)
    B, L = levels[0].shape[:2]
    K = E.shape[-3]
    bsz = E.shape[-1]
    m = bsz * K
    tile = min(tile, K)
    dt = levels[0].dtype
    uniform = E.shape[1] == 1
    chunk_ = L if chunk is None else min(chunk, L)
    C = L // chunk_
    Ec = (
        jnp.broadcast_to(E[:, None], (B, C, 1, K, bsz, bsz)).reshape(B * C, 1, K, bsz, bsz)
        if uniform
        else E.reshape(B * C, chunk_, K, bsz, bsz)
    )
    sizes = [m**k for k in range(1, N + 1)]
    offs = np.cumsum([0] + sizes)

    def split(x, lead):  # packed (..., D) -> per-level (..., m, .., m)
        return [x[..., offs[k] : offs[k + 1]].reshape(lead + (m,) * (k + 1)) for k in range(N)]

    loc = [levels[k].reshape((B * C, chunk_) + (m,) * (k + 1)) for k in range(N)]
    outs = split(out.reshape(B * C, chunk_, -1), (B * C, chunk_))
    gs = split(g.reshape(B * C, chunk_, -1), (B * C, chunk_))
    # carry-in of every chunk = the stored output at the previous chunk's end (zero for the first chunk)
    prev = (
        jnp.concatenate([jnp.zeros_like(out[:, :1]), out[:, chunk_ - 1 :: chunk_][:, :-1]], 1)
        if C > 1
        else jnp.zeros_like(out[:, :1])
    )
    carries = split(prev.reshape(B * C, -1), (B * C,))
    dE1, ds_t1, ds_f1, dc1 = _block_pass_parity_bwd(
        Ec, loc, carries, outs, gs, None, N, tile, uniform, backend
    )

    def reduce_partials(dE, ds_t, ds_f):
        dEs = dE[0] + dE[1].sum(1)  # (BC, Lm, K, 2, 2)
        ds = [ds_t[k] + (ds_f[k].sum(1) if k < N - 1 else 0) for k in range(N)]
        return dEs, ds

    dEs, ds = reduce_partials(dE1, ds_t1, ds_f1)
    if C > 1:
        # B2: cotangents across chunk boundaries. Chunk c+1 maps its carry-in S_end(c) -> its outputs; the carry-in
        # cotangent of chunk c+1 (dc1, from B1) is the cotangent of S_end(c), which is an output of chunk c (already
        # in g via the stored stream? no: g is the loss cotangent; S_end(c) also feeds chunk c+1) -> add it as an
        # extra output cotangent at the last position of chunk c, and propagate: lam_end(c) = dc(c+1) + J^T ...
        # Since chunk c's outputs are what chunk c+1 reads, the full chain is handled by adding dc(c+1) to g at the
        # end position of chunk c and rerunning the reverse pass for chunk c; sequential over chunks in reverse.
        dcp = jnp.concatenate([x.reshape(B * C, -1) for x in dc1], -1).reshape(
            B, C, -1
        )  # (B, C, D)
        lam_end = jnp.zeros((B, C, dcp.shape[-1]), dt)
        # reverse sequential over chunks (C small): lam_end(c) = dc(c+1) where dc(c+1) itself depends on lam_end(c+1)
        # via chunk c+1's reverse pass. Linear in lam_end: dc(c+1) = dc1(c+1) + J_{c+1}^T lam_end(c+1), with
        # J_{c+1}^T lam = cotangent of the carry-in of chunk c+1 under output-cotangent lam at its last position
        # = jax.vjp of the packed Chen op (carry -> chunk end) with the chunk's summary as right operand.
        # The chunk summary (zero-carry end state P_c, operator M_c) is recomputed with a forward pass-1.
        zeros = [jnp.zeros((B * C,) + (m,) * k, dt) for k in range(1, N + 1)]
        _, e = _block_pass_parity(Ec, loc, zeros, N, tile, uniform, False, backend)
        P = jnp.concatenate([x.reshape(B, C, -1) for x in e], -1)
        Efull = (
            jnp.broadcast_to(E[:, None], (B, C, chunk_, K, bsz, bsz))
            if uniform
            else E.reshape(B, C, chunk_, K, bsz, bsz)
        )
        Mchunk = block_to_dense_M(_cum_ops(Efull)[:, :, -1])

        def JT(c_idx, lam):  # lam: (B, D) cotangent at chunk c_idx's end
            Pc, Mc = P[:, c_idx], Mchunk[:, c_idx]
            f = lambda carry: jax.vmap(
                lambda cr, p_, M_: chen_binary_op_packed_with_M(
                    (cr, 0.0, jnp.eye(m, dtype=dt)), (p_, 0.0, M_), None, m, N
                )[0]
            )(carry, Pc, Mc)
            _, vjp = jax.vjp(f, jnp.zeros((B, P.shape[-1]), dt))
            return vjp(lam)[0]

        def step(lam_next, c_idx):  # lam_next = lam_end(c_idx + 1) -> lam_end(c_idx)
            with jax.default_matmul_precision("highest"):
                lam_c = dcp[:, c_idx + 1] + JT(c_idx + 1, lam_next)
            return lam_c, lam_c

        _, lam_hist = jax.lax.scan(step, jnp.zeros((B, P.shape[-1]), dt), jnp.arange(C - 2, -1, -1))
        lam_end = jnp.concatenate(
            [lam_hist[::-1], jnp.zeros((1, B, P.shape[-1]), dt)], 0
        ).transpose(1, 0, 2)  # (B, C, D)
        lams = split(lam_end.reshape(B * C, -1), (B * C,))
        dE3, ds_t3, ds_f3, _ = _block_pass_parity_bwd(
            Ec, loc, carries, outs, None, lams, N, tile, uniform, backend
        )
        dEs3, ds3 = reduce_partials(dE3, ds_t3, ds_f3)
        dEs = dEs + dEs3
        ds = [a + b for a, b in zip(ds, ds3)]
        # the carry-in cotangents dc (B1 + B3) of chunk c+1 were fed to chunk c as lam_end -> already included
    dE_out = dEs.reshape(B, C, 1, K, bsz, bsz).sum(1) if uniform else dEs.reshape(B, L, K, bsz, bsz)
    dlevels = tuple(ds[k].reshape(levels[k].shape) for k in range(N))
    return dE_out, dlevels


_block_prefix_cvjp.defvjp(_cvjp_fwd, _cvjp_bwd)


def sequential_ref(M, levels):
    """Plain sequential Chen recursion in JAX (lax.scan) with per-step operators: the reference for gradients wrt the
    individual M_t. (The packed op's composite M_L @ M_R only equals the true M_R M_L for commuting operators, so
    jax.grad of the parallel scans wrt a single M_t is not the true derivative; wrt A it is.)"""
    from ews.core.chen_scan import chen_binary_op_packed_with_M

    N = len(levels)
    B, L = levels[0].shape[:2]
    m = levels[0].shape[2]
    if M.ndim == 2:
        M = jnp.broadcast_to(M[None, None], (B, 1, m, m))
    if M.shape[1] == 1:
        M = jnp.broadcast_to(M, (B, L, m, m))
    packed = jnp.concatenate([l.reshape(B, L, -1) for l in levels], -1)
    I = jnp.eye(m, dtype=packed.dtype)

    def step(S, x):
        p_t, M_t = x
        Sn = jax.vmap(
            lambda S_, p_, M_: chen_binary_op_packed_with_M(
                (S_, 0.0, I), (p_, 0.0, M_), None, m, N
            )[0]
        )(S, p_t, M_t)
        return Sn, Sn

    _, out = jax.lax.scan(
        step,
        jnp.zeros((B, packed.shape[-1]), packed.dtype),
        (packed.transpose(1, 0, 2), M.transpose(1, 0, 2, 3)),
    )
    return out.transpose(1, 0, 2)


def _tree_reduce(op, elems, identity, axis=1):
    """Log-depth pairwise reduction with a binary associative op over `axis` (terminal readout, no prefix stream).
    Pads to a power of two with identity elements: identity(n) -> pytree of n identity elements along `axis`."""
    L = jax.tree_util.tree_leaves(elems)[0].shape[axis]
    P = 1 << (L - 1).bit_length()
    xs = elems
    if P > L:
        xs = jax.tree_util.tree_map(
            lambda x, i: jnp.concatenate([x, i], axis), elems, identity(P - L)
        )
    while P > 1:
        pairs = (
            jax.tree_util.tree_map(lambda x: jax.lax.slice_in_dim(x, 0, P, 2, axis=axis), xs),
            jax.tree_util.tree_map(lambda x: jax.lax.slice_in_dim(x, 1, P, 2, axis=axis), xs),
        )
        xs = op(*pairs)
        P //= 2
    return jax.tree_util.tree_map(lambda x: jax.lax.index_in_dim(x, 0, axis, keepdims=False), xs)


def dense_tree_lean(M, levels):
    """Lean terminal combine: tree reduction with the packed Chen op on given operators (any depth)."""
    from ews.core.chen_scan import chen_binary_op_packed_with_M

    N = len(levels)
    B, L = levels[0].shape[:2]
    m = levels[0].shape[2]
    if M.ndim == 2:
        M = jnp.broadcast_to(M[None, None], (B, 1, m, m))
    if M.shape[1] == 1:
        M = jnp.broadcast_to(M, (B, L, m, m))
    packed = jnp.concatenate([l.reshape(B, L, -1) for l in levels], -1)
    h = jnp.zeros((B, L), packed.dtype)

    def op(l, r):
        f = lambda pl_, hl, Ml, pr, hr, Mr: chen_binary_op_packed_with_M(
            (pl_, hl, Ml), (pr, hr, Mr), None, m, N
        )
        return jax.vmap(jax.vmap(f))(*l, *r)

    D = packed.shape[-1]
    ident = lambda n: (
        jnp.zeros((B, n, D), packed.dtype),
        jnp.zeros((B, n), packed.dtype),
        jnp.broadcast_to(jnp.eye(m, dtype=packed.dtype), (B, n, m, m)),
    )
    out, _, _ = _tree_reduce(op, (packed, h, M), ident, axis=1)
    return out


def plane_tree(rho, s1, s2, N):
    """Plane engine (JAX, complex scalars) terminal combine: tree reduction with _combine_n1/_combine_n2."""
    op = pe._combine_n1 if N == 1 else pe._combine_n2
    elems = (rho, s1) if N == 1 else (rho, s1, s2)
    B, K = rho.shape[0], rho.shape[-1]

    def ident(n):
        one = jnp.ones((B, n, K), rho.dtype)
        z1 = jnp.zeros((B, n, K), s1.dtype)
        return (one, z1) if N == 1 else (one, z1, jnp.zeros((B, n, K, K, 2, 2), s2.dtype))

    out = _tree_reduce(op, elems, ident, axis=1)
    lvl1 = pe.from_complex_planes(out[1])
    return lvl1 if N == 1 else jnp.concatenate([lvl1, pe.blocks_to_flat(out[2])], -1)


# =========================================================================== terminal-mode training (block kernel)
# Forward = final EWS only (chunk ends + chunk scan, no stream). Backward = chunk-checkpointed: the forward saves the
# C chunk-start states (C x D per sample); the backward walks the chunks in reverse, recomputes one chunk's prefix
# stream from its saved start (register kernel, write=True) and runs the reverse kernel on it with the incoming
# cotangent of that chunk's end state (the carry-in cotangent of the next chunk). Training memory = one chunk's
# stream (B x chunk x D) instead of the whole prefix (B x L x D); C sequential launches of (B x row tiles) programs.


def block_terminal_vjp(E, levels, tile=4, chunk=None, backend="triton"):
    return _block_terminal_cvjp(E, tuple(levels), tile, chunk, backend)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def _block_terminal_cvjp(E, levels, tile, chunk, backend):
    return combine_pallas_block(
        E, list(levels), tile=tile, chunk=chunk, backend=backend, mode="parity", final_only=True
    )


def _terminal_fwd(E, levels, tile, chunk, backend):
    from ews.core.chen_scan import chen_binary_op_packed_with_M

    N = len(levels)
    B, L = levels[0].shape[:2]
    K, bsz = E.shape[-3], E.shape[-1]
    m = bsz * K
    tile = min(tile, K)
    dt = levels[0].dtype
    uniform = E.shape[1] == 1
    chunk_ = L if chunk is None else min(chunk, L)
    C = L // chunk_
    Ec = (
        jnp.broadcast_to(E[:, None], (B, C, 1, K, bsz, bsz)).reshape(B * C, 1, K, bsz, bsz)
        if uniform
        else E.reshape(B * C, chunk_, K, bsz, bsz)
    )
    loc = [levels[k - 1].reshape((B * C, chunk_) + (m,) * k) for k in range(1, N + 1)]
    D = sum(m**k for k in range(1, N + 1))

    def carries_from(packed_rows):
        outs, start = [], 0
        for k in range(1, N + 1):
            outs.append(packed_rows[:, start : start + m**k].reshape((B * C,) + (m,) * k))
            start += m**k
        return outs

    _, e = _block_pass_parity(
        Ec, loc, carries_from(jnp.zeros((B * C, D), dt)), N, tile, uniform, False, backend
    )
    packed = jnp.concatenate([x.reshape(B, C, -1) for x in e], -1)
    if C == 1:
        final = packed[:, 0]
        starts = jnp.zeros((B, 1, D), dt)
    else:
        Efull = (
            jnp.broadcast_to(E[:, None], (B, C, chunk_, K, bsz, bsz))
            if uniform
            else E.reshape(B, C, chunk_, K, bsz, bsz)
        )
        Mchunk = block_to_dense_M(_cum_ops(Efull)[:, :, -1])
        h = jnp.zeros((B, C), dt)

        def op(l, r):
            f = lambda pl_, hl, Ml, pr, hr, Mr: chen_binary_op_packed_with_M(
                (pl_, hl, Ml), (pr, hr, Mr), None, m, N
            )
            return jax.vmap(jax.vmap(f))(*l, *r)

        with jax.default_matmul_precision("highest"):
            Cpacked, _, _ = jax.lax.associative_scan(op, (packed, h, Mchunk), axis=1)
        final = Cpacked[:, -1]
        starts = jnp.concatenate(
            [jnp.zeros_like(Cpacked[:, :1]), Cpacked[:, :-1]], 1
        )  # (B, C, D) chunk-start states
    return final, (E, levels, starts)


def _terminal_bwd(tile, chunk, backend, res, g_final):
    E, levels, starts = res
    N = len(levels)
    B, L = levels[0].shape[:2]
    K, bsz = E.shape[-3], E.shape[-1]
    m = bsz * K
    tile = min(tile, K)
    uniform = E.shape[1] == 1
    chunk_ = L if chunk is None else min(chunk, L)
    C = L // chunk_
    sizes = [m**k for k in range(1, N + 1)]
    offs = np.cumsum([0] + sizes)

    def split(x, lead):
        return [x[..., offs[k] : offs[k + 1]].reshape(lead + (m,) * (k + 1)) for k in range(N)]

    lam = g_final  # (B, D) cotangent of the current chunk's end state
    dE_chunks, ds_chunks = [None] * C, [None] * C
    for c in range(C - 1, -1, -1):
        sl = slice(c * chunk_, (c + 1) * chunk_)
        Ec = E[:, :1] if uniform else E[:, sl]  # (B, Lm, K, b, b)
        loc = [levels[k].reshape((B, L) + (m,) * (k + 1))[:, sl] for k in range(N)]
        carries = split(starts[:, c], (B,))
        o, _ = _block_pass_parity(
            Ec, loc, carries, N, tile, uniform, True, backend
        )  # one chunk's stream
        outs = [x for x in o]
        lams = split(lam, (B,))
        dE, ds_t, ds_f, dc = _block_pass_parity_bwd(
            Ec, loc, carries, outs, None, lams, N, tile, uniform, backend
        )
        dE_chunks[c] = dE[0] + dE[1].sum(1)  # (B, Lm, K, b, b)
        ds_chunks[c] = [ds_t[k] + (ds_f[k].sum(1) if k < N - 1 else 0) for k in range(N)]
        lam = jnp.concatenate(
            [x.reshape(B, -1) for x in dc], -1
        )  # carry-in cotangent -> previous chunk's end
    if uniform:
        dE_out = sum(dE_chunks)  # (B, 1, K, b, b)
    else:
        dE_out = jnp.concatenate(dE_chunks, 1)
    dlevels = tuple(
        jnp.concatenate([ds_chunks[c][k] for c in range(C)], 1).reshape(levels[k].shape)
        for k in range(N)
    )
    return dE_out, dlevels


_block_terminal_cvjp.defvjp(_terminal_fwd, _terminal_bwd)


# =========================================================================== dense kernel backward (depths 1-2, f32)
# Same construction as the block kernel's backward: the per-step map on values (M_t, s_t, S_{t-1}) -> S_t is pure, so
# its adjoint is jax.vjp traced inside the reverse kernel (matmul transposes lower to Triton dots). State in
# registers (depths 1-2, m <= 64 with 8 warps). Chunking: B1 / B2 (jax.vjp of the packed Chen op on chunk
# summaries) / B3, as for the block kernel. f32 only (Triton has no f64 MMA).


def _dense_step(M, s1, s2, S1, S2, N):
    f1 = jnp.sum(M * S1[None, :], axis=1)
    S1n = f1 + s1
    if N == 1:
        return S1n, S2
    T = pl.dot(M, S2, allow_tf32=False)
    S2n = pl.dot(T, M, trans_b=True, allow_tf32=False) + f1[:, None] * s1[None, :] + s2
    return S1n, S2n


def _dense_kernel_bwd(*refs, N, m, uniform, has_g, has_lam):
    it = iter(refs)
    m_ref = next(it)  # (Lm, m, m)
    s_ref = next(it)  # (Lc, m+1, m): row 0 = s1, rows 1..m = s2
    c_ref = next(it)  # (m+1, m) carry-in
    o_ref = next(it)  # (Lc, m+1, m) forward stream
    g_ref = next(it) if has_g else None  # (Lc, m+1, m)
    l_ref = next(it) if has_lam else None  # (m+1, m) incoming cotangent at the chunk end
    dm_ref = next(it)  # (Lm, m, m)
    ds_ref = next(it)  # (Lc, m+1, m)
    dc_ref = next(it)  # (m+1, m) cotangent wrt the carry-in
    L = s_ref.shape[0]
    dt = s_ref.dtype

    def load(ref, t):
        if t is None:
            return ref[0, :], (ref[pl.ds(1, m), :] if N == 2 else None)
        return ref[t, 0, :], (ref[t, pl.ds(1, m), :] if N == 2 else None)

    def step(M, s1, s2, S1, S2):
        return _dense_step(M, s1, s2, S1, S2, N)

    lam1, lam2 = (
        load(l_ref, None)
        if has_lam
        else (jnp.zeros((m,), dt), jnp.zeros((m, m), dt) if N == 2 else None)
    )
    dM_acc = jnp.zeros((m, m), dt)

    def body(i, carry):
        t = L - 1 - i
        lam1, lam2, dM_acc = carry
        M = m_ref[0, :, :] if uniform else m_ref[t, :, :]
        s1, s2 = load(s_ref, t)
        So1, So2 = load(o_ref, jnp.maximum(t - 1, 0))
        Sc1, Sc2 = load(c_ref, None)
        S1p = jnp.where(t > 0, So1, Sc1)
        S2p = jnp.where(t > 0, So2, Sc2) if N == 2 else None
        if has_g:
            g1, g2 = load(g_ref, t)
            lam1 = lam1 + g1
            lam2 = lam2 + g2 if N == 2 else None
        if N == 2:
            _, vjp = jax.vjp(
                lambda M_, a_, b_, x_, y_: step(M_, a_, b_, x_, y_), M, s1, s2, S1p, S2p
            )
            dM, ds1, ds2, dS1, dS2 = vjp((lam1, lam2))
        else:
            _, vjp = jax.vjp(lambda M_, a_, x_: step(M_, a_, None, x_, None)[0], M, s1, S1p)
            dM, ds1, dS1 = vjp(lam1)
            ds2 = dS2 = None
        ds_ref[t, 0, :] = ds1
        if N == 2:
            ds_ref[t, pl.ds(1, m), :] = ds2
        if uniform:
            dM_acc = dM_acc + dM
        else:
            dm_ref[t, :, :] = dM
        return (dS1, dS2, dM_acc)

    lam1, lam2, dM_acc = jax.lax.fori_loop(0, L, body, (lam1, lam2, dM_acc))
    if uniform:
        dm_ref[0, :, :] = dM_acc
    dc_ref[0, :] = lam1
    if N == 2:
        dc_ref[pl.ds(1, m), :] = lam2


def _dense_pass_bwd(Mc_, sc, carries, outs, gs, lams, N, uniform, backend, num_warps=8):
    """sc/outs/gs: (BC, Lc, m+1, m); carries/lams: (BC, m+1, m). Returns dM (BC, Lm, m, m), ds (BC, Lc, m+1, m), dc."""
    BC, Lc = sc.shape[:2]
    m = sc.shape[-1]
    R = sc.shape[2]  # m + 1 rows at depth 2, 1 row at depth 1
    dt = sc.dtype
    Lm = 1 if uniform else Lc
    spec_M = pl.BlockSpec((None, Lm, m, m), lambda b: (b, 0, 0, 0))
    spec_s = pl.BlockSpec((None, Lc, R, m), lambda b: (b, 0, 0, 0))
    spec_c = pl.BlockSpec((None, R, m), lambda b: (b, 0, 0))
    ins, in_specs = [Mc_, sc, carries, outs], [spec_M, spec_s, spec_c, spec_s]
    if gs is not None:
        ins.append(gs)
        in_specs.append(spec_s)
    if lams is not None:
        ins.append(lams)
        in_specs.append(spec_c)
    out_shape = [
        jax.ShapeDtypeStruct((BC, Lm, m, m), dt),
        jax.ShapeDtypeStruct((BC, Lc, R, m), dt),
        jax.ShapeDtypeStruct((BC, R, m), dt),
    ]
    out_specs = [spec_M, spec_s, spec_c]
    kern = functools.partial(
        _dense_kernel_bwd, N=N, m=m, uniform=uniform, has_g=gs is not None, has_lam=lams is not None
    )
    cp = plgpu.CompilerParams(num_warps=num_warps)
    return pl.pallas_call(
        kern,
        out_shape=out_shape,
        grid=(BC,),
        in_specs=in_specs,
        out_specs=out_specs,
        **_backend_kw(backend),
        compiler_params=cp,
    )(*ins)


def dense_prefix_vjp(M, levels, chunk=None, backend="triton"):
    """Prefix combine, dense (any A) kernel, depths 1-2, with a custom VJP wrt (M, levels). M: (B, L, m, m) or (B, 1, m, m)."""
    return _dense_prefix_cvjp(M, tuple(levels), chunk, backend)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _dense_prefix_cvjp(M, levels, chunk, backend):
    N = len(levels)
    return combine_pallas_dense(
        M, levels[0], levels[1] if N == 2 else None, N=N, chunk=chunk, backend=backend
    )


def _dense_cvjp_fwd(M, levels, chunk, backend):
    out = _dense_prefix_cvjp(M, levels, chunk, backend)
    return out, (M, levels, out)


def _dense_cvjp_bwd(chunk, backend, res, g):
    from ews.core.chen_scan import chen_binary_op_packed_with_M

    M, levels, out = res
    N = len(levels)
    B, L, m = levels[0].shape
    dt = levels[0].dtype
    uniform = M.shape[1] == 1
    chunk_ = L if chunk is None else min(chunk, L)
    C = L // chunk_
    D = m + (m * m if N == 2 else 0)
    Mc_ = (
        jnp.broadcast_to(M[:, None], (B, C, 1, m, m)).reshape(B * C, 1, m, m)
        if uniform
        else M.reshape(B * C, chunk_, m, m)
    )

    def to_rows(x):  # (..., D) packed -> (..., m+1, m) rows layout (N=1: (.., 1, m))
        lead = x.shape[:-1]
        if N == 1:
            return x.reshape(lead + (1, m))
        return jnp.concatenate(
            [x[..., :m].reshape(lead + (1, m)), x[..., m:].reshape(lead + (m, m))], -2
        )

    def from_rows(x):
        return x.reshape(x.shape[:-2] + (-1,))

    R = m + 1 if N == 2 else 1
    sc = to_rows(jnp.concatenate([l.reshape(B, L, -1) for l in levels], -1)).reshape(
        B * C, chunk_, R, m
    )
    outs = out.reshape(B * C, chunk_, R, m)
    gs = g.reshape(B * C, chunk_, R, m)
    prev = (
        jnp.concatenate([jnp.zeros_like(out[:, :1]), out[:, chunk_ - 1 :: chunk_][:, :-1]], 1)
        if C > 1
        else jnp.zeros_like(out[:, :1])
    )
    carries = prev.reshape(B * C, R, m)
    dM1, ds1, dc1 = _dense_pass_bwd(Mc_, sc, carries, outs, gs, None, N, uniform, backend)
    dM, ds = dM1, ds1
    if C > 1:
        dcp = dc1.reshape(B, C, -1)
        # chunk summaries (zero-carry ends + M_chunk) via forward pass 1
        z1 = jnp.zeros((B * C, m), dt)
        z2 = jnp.zeros((B * C, m, m), dt)
        s1c = levels[0].reshape(B * C, chunk_, m)
        s2c = levels[1].reshape(B * C, chunk_, m, m) if N == 2 else None
        o = _dense_pass(Mc_, s1c, s2c, z1, z2, N, uniform, False, True, backend)
        if N == 2:
            _, e1, e2, em = o
            P = jnp.concatenate([e1.reshape(B, C, m), e2.reshape(B, C, m * m)], -1)
        else:
            _, e1, em = o
            P = e1.reshape(B, C, m)
        Mchunk = em.reshape(B, C, m, m)

        def JT(c_idx, lam):
            Pc, Mc = P[:, c_idx], Mchunk[:, c_idx]
            f = lambda carry: jax.vmap(
                lambda cr, p_, M_: chen_binary_op_packed_with_M(
                    (cr, 0.0, jnp.eye(m, dtype=dt)), (p_, 0.0, M_), None, m, N
                )[0]
            )(carry, Pc, Mc)
            _, vjp = jax.vjp(f, jnp.zeros((B, D), dt))
            return vjp(lam)[0]

        def step(lam_next, c_idx):
            with jax.default_matmul_precision("highest"):
                lam_c = dcp[:, c_idx + 1] + JT(c_idx + 1, lam_next)
            return lam_c, lam_c

        _, lam_hist = jax.lax.scan(step, jnp.zeros((B, D), dt), jnp.arange(C - 2, -1, -1))
        lam_end = jnp.concatenate([lam_hist[::-1], jnp.zeros((1, B, D), dt)], 0).transpose(1, 0, 2)
        lams = to_rows(lam_end).reshape(B * C, R, m)
        dM3, ds3, _ = _dense_pass_bwd(Mc_, sc, carries, outs, None, lams, N, uniform, backend)
        dM = dM + dM3
        ds = ds + ds3
    dM_out = dM.reshape(B, C, 1, m, m).sum(1) if uniform else dM.reshape(B, L, m, m)
    ds = ds.reshape(B, L, R, m)
    dlevels = (ds[:, :, 0, :],) + ((ds[:, :, 1:, :].reshape(levels[1].shape),) if N == 2 else ())
    return dM_out, dlevels


_dense_prefix_cvjp.defvjp(_dense_cvjp_fwd, _dense_cvjp_bwd)


# =========================================================================== plane / block engine in JAX, any depth
# The block kernel's algorithm (residue-class tiles, b-term mixes per axis, tile outer products) as a plain JAX
# associative scan: the "plane engine at any depth" (rotation planes are b = 2). Element per interval:
# (E (K, b, b), tiles level k: (b,)*k + (K,)*k). Prefix via jax.lax.associative_scan, terminal via _tree_reduce.
import string


def _packed_to_tiles(x, m, b, k):
    """(..., m^k) row-major -> (..., b,..,b, K,..,K) with index i_a = b * j_a + p_a."""
    K = m // b
    lead = x.shape[:-1]
    t = x.reshape(lead + (K, b) * k)  # (.., K1, p1, K2, p2, ..)
    n = len(lead)
    perm = list(range(n)) + [n + 2 * a + 1 for a in range(k)] + [n + 2 * a for a in range(k)]
    return jnp.transpose(t, perm)


def _tiles_to_packed(t, m, b, k):
    n = t.ndim - 2 * k
    perm = list(range(n)) + [
        v for a in range(k) for v in (n + k + a, n + a)
    ]  # (.., K1, p1, K2, p2, ..)
    return jnp.transpose(t, perm).reshape(t.shape[:n] + (m**k,))


def _flow_tiles(E, T, k, b):
    """T (.., b^k (as k axes), K^k (as k axes)) flowed by E (.., K, b, b) along every axis."""
    n = T.ndim - 2 * k
    L = string.ascii_lowercase
    lead = L[:n]
    p = [L[n + a] for a in range(k)]
    i = [L[n + k + a] for a in range(k)]
    for a in range(k):
        y = "Y"
        src = lead + "".join(p[:a] + [y] + p[a + 1 :]) + "".join(i)
        e = lead + i[a] + p[a] + y
        out = lead + "".join(p) + "".join(i)
        T = jnp.einsum(f"{e},{src}->{out}", E, T, precision=HI)
    return T


def _tile_outer(F, S, j, kj, n):
    """F (.., b^j, K^j) (x) S (.., b^kj, K^kj) -> (.., b^k, K^k) with axes ordered (p_1..p_k, i_1..i_k)."""
    L = string.ascii_lowercase
    lead = L[:n]
    pf = L[n : n + j]
    jf = L[n + j : n + 2 * j]
    ps = L[n + 2 * j : n + 2 * j + kj]
    js = L[n + 2 * j + kj : n + 2 * j + 2 * kj]
    return jnp.einsum(f"{lead}{pf}{jf},{lead}{ps}{js}->{lead}{pf}{ps}{jf}{js}", F, S, precision=HI)


def _block_op_jax(N, b):
    def op(l, r):
        El, Tl = l
        Er, Tr = r
        E = jnp.einsum("...kab,...kbc->...kac", Er, El, precision=HI)
        n = El.ndim - 3
        F = [_flow_tiles(Er, Tl[k - 1], k, b) for k in range(1, N + 1)]
        out = []
        for k in range(1, N + 1):
            acc = F[k - 1] + Tr[k - 1]
            for j in range(1, k):
                acc = acc + _tile_outer(F[j - 1], Tr[k - j - 1], j, k - j, n)
            out.append(acc)
        return E, out

    return op


def block_assoc_jax(E, levels, readout="prefix"):
    """E (B, L, K, b, b) or (B, 1, K, b, b); levels[k-1] (B, L, m^k). -> (B, L, D) prefix or (B, D) terminal."""
    N = len(levels)
    B, L = levels[0].shape[:2]
    K, b = E.shape[-3], E.shape[-1]
    m = b * K
    if E.shape[1] == 1:
        E = jnp.broadcast_to(E, (B, L, K, b, b))
    T = [_packed_to_tiles(levels[k - 1].reshape(B, L, -1), m, b, k) for k in range(1, N + 1)]
    op = _block_op_jax(N, b)
    if readout == "prefix":
        _, out = jax.lax.associative_scan(op, (E, T), axis=1)
        return jnp.concatenate([_tiles_to_packed(out[k - 1], m, b, k) for k in range(1, N + 1)], -1)

    def ident(n):
        Ei = jnp.broadcast_to(jnp.eye(b, dtype=E.dtype), (B, n, K, b, b))
        return Ei, [jnp.zeros((B, n) + (b,) * k + (K,) * k, E.dtype) for k in range(1, N + 1)]

    _, out = _tree_reduce(op, (E, T), ident, axis=1)
    return jnp.concatenate([_tiles_to_packed(out[k - 1], m, b, k) for k in range(1, N + 1)], -1)


# =========================================================================== dense kernel backward, any depth (buffered, f32)
# Hand-written adjoint of _dense_kernel_general (state in buffers, so no in-kernel jax.vjp). Per step t (descending):
#   recompute the partial flows P^(0..k) of S_{t-1} for every level (forward slab loops);
#   lam_k <- lam_k + g_k[t];  ds_k[t] += lam_k;  cross-term adjoints: dF_j += <lam_k, s_{k-j}>, ds_{k-j}[t] += F_j lam_k;
#   dF_k += lam_k; transposed flows in reverse axis order: cotangent slab <- M^T @ slab (inner axes) / slab @ M (last
#   axis), dM += slab-wise outer products with the matching partial flow; the result is lam_k for step t-1.
# f32 (pl.dot); chunking reuses the B1 / B2 / B3 scheme.


def _dense_kernel_general_bwd(*refs, N, m, uniform, has_g, has_lam):
    it = iter(refs)
    m_ref = next(it)
    s_refs = [next(it) for _ in range(N)]
    c_refs = [next(it) for _ in range(N)]
    o_refs = [next(it) for _ in range(N)]
    g_refs = [next(it) for _ in range(N)] if has_g else [None] * N
    l_refs = [next(it) for _ in range(N)] if has_lam else [None] * N
    dm_ref = next(it)
    ds_refs = [next(it) for _ in range(N)]
    dc_refs = [next(it) for _ in range(N)]
    lamA = [next(it) for _ in range(N)]  # cotangent buffers (ping-pong A/B)
    lamB = [next(it) for _ in range(N)]
    P = [
        [next(it) for _ in range(k + 2)] for k in range(N)
    ]  # partial flows: level index k has k+2 buffers
    dF = [next(it) for _ in range(N)]
    L = s_refs[0].shape[0]
    dt = s_refs[0].dtype
    bar = plgpu.debug_barrier

    def Mt(t):
        return m_ref[0, :, :] if uniform else m_ref[t, :, :]

    # init lam
    for k in range(N):
        lamA[k][...] = l_refs[k][...] if has_lam else jnp.zeros_like(lamA[k])
    bar()

    def flow_partials(t, k):
        """P[k][0] = S_{t-1} (level index k); P[k][1] = last-axis flow; P[k][2+i] = inner-axis-i flow."""
        M = Mt(t)
        nrow = m**k
        src0, cref, oref = P[k][0], c_refs[k], o_refs[k]
        # S_{t-1}
        sr = m if k > 0 else 1

        def cp(q, c):
            r0 = pl.multiple_of(q * sr, sr)
            src0[pl.ds(r0, sr), :] = jnp.where(
                t > 0, oref[jnp.maximum(t - 1, 0), pl.ds(r0, sr), :], cref[pl.ds(r0, sr), :]
            )
            return c

        jax.lax.fori_loop(0, max(nrow // sr, 1), cp, 0)
        bar()
        if k == 0:
            P[0][1][0, :] = jnp.sum(M * src0[0, :][None, :], axis=1)
            bar()
            return
        nslab = nrow // m

        def last_axis(q, c):
            r0 = pl.multiple_of(q * m, m)
            P[k][1][pl.ds(r0, m), :] = pl.dot(
                src0[pl.ds(r0, m), :], M, trans_b=True, allow_tf32=False
            )
            return c

        jax.lax.fori_loop(0, nslab, last_axis, 0)
        bar()
        for i in range(k):
            stride = m ** (k - 1 - i)
            rd, wr = P[k][1 + i], P[k][2 + i]

            def inner(c, carry, stride=stride, rd=rd, wr=wr):
                high = c // stride
                low = c - high * stride
                base = high * (stride * m) + low
                wr[pl.ds(base, m, stride=stride), :] = pl.dot(
                    M, rd[pl.ds(base, m, stride=stride), :], allow_tf32=False
                )
                return carry

            jax.lax.fori_loop(0, m ** (k - 1), inner, 0)
            bar()

    def body(i, dM_acc):
        t = L - 1 - i
        M = Mt(t)
        for k in range(N):
            flow_partials(t, k)
        F = [P[k][k + 1] for k in range(N)]  # fully flowed S_{t-1}, level index k
        # lam_k <- lam_k + g ; dF_k = lam_k ; ds_k[t] = lam_k
        for k in range(N):
            nrow = m**k
            sr = m if k > 0 else 1

            def upd(q, c, k=k, sr=sr):
                r0 = pl.multiple_of(q * sr, sr)
                v = lamA[k][pl.ds(r0, sr), :]
                if has_g:
                    v = v + g_refs[k][t, pl.ds(r0, sr), :]
                lamA[k][pl.ds(r0, sr), :] = v
                dF[k][pl.ds(r0, sr), :] = v
                ds_refs[k][t, pl.ds(r0, sr), :] = v
                return c

            jax.lax.fori_loop(0, max(nrow // sr, 1), upd, 0)
        bar()
        # cross-term adjoints: S_k' = ... + F_j (x) s_{k-j}  (1-based k, j)
        for k in range(2, N + 1):
            kk = k - 1
            nslab = m**kk // m
            for j in range(1, k):
                Fj, dFj = F[j - 1], dF[j - 1]  # (m^{j-1}, m)
                s_kj, ds_kj = s_refs[k - j - 1], ds_refs[k - j - 1]  # (Lc, nb, m)
                nb = m ** (k - j - 1)

                def cross(q, c, kk=kk, nb=nb, Fj=Fj, dFj=dFj, s_kj=s_kj, ds_kj=ds_kj):
                    r0 = pl.multiple_of(q * m, m)
                    lam = lamA[kk][pl.ds(r0, m), :]  # (m, m) slab of lam_k
                    if nb >= m:  # slab inside one a
                        a = r0 // nb
                        b0 = r0 - a * nb
                        fa = Fj[a // m, a - (a // m) * m]
                        dFj[a // m, a - (a // m) * m] = dFj[a // m, a - (a // m) * m] + jnp.sum(
                            lam * s_kj[t, pl.ds(b0, m), :]
                        )
                        ds_kj[t, pl.ds(b0, m), :] = ds_kj[t, pl.ds(b0, m), :] + fa * lam
                    else:  # nb == 1 (j = k-1): rows of the slab are consecutive a
                        s1 = s_kj[t, 0, :]
                        fv = Fj[r0 // m, :]  # F_{k-1}[a] for a = r0..r0+m
                        dFj[r0 // m, :] = dFj[r0 // m, :] + jnp.sum(lam * s1[None, :], axis=1)
                        ds_kj[t, 0, :] = ds_kj[t, 0, :] + jnp.sum(fv[:, None] * lam, axis=0)
                    return c

                jax.lax.fori_loop(0, nslab, cross, 0)
                bar()
        # transposed flows: cotangent of F_k (in dF[k]) -> cotangent of S_{t-1} level k (into lamB[k]); dM accumulates
        for k in range(N):
            nrow = m**k
            if k == 0:
                lam1 = dF[0][0, :]
                v = P[0][0][0, :]
                dM_acc = dM_acc + lam1[:, None] * v[None, :]
                lamB[0][0, :] = jnp.sum(
                    M * lam1[:, None], axis=0
                )  # (M^T lam)_j = sum_i M[i, j] lam[i]
                continue
            # reverse of: last axis (P0 -> P1), inner axes i = 0..k-1 (P[1+i] -> P[2+i]); cotangent buffers: dF[k] -> lamB/lamA'
            bufs = [dF[k], lamB[k]]
            cur = 0
            for i in range(k - 1, -1, -1):
                stride = m ** (k - 1 - i)
                rd, wr, Pin = bufs[cur], bufs[1 - cur], P[k][1 + i]

                def inner_T(c, acc, stride=stride, rd=rd, wr=wr, Pin=Pin):
                    high = c // stride
                    low = c - high * stride
                    base = high * (stride * m) + low
                    lam = rd[pl.ds(base, m, stride=stride), :]
                    acc = acc + pl.dot(
                        lam, Pin[pl.ds(base, m, stride=stride), :], trans_b=True, allow_tf32=False
                    )
                    wr[pl.ds(base, m, stride=stride), :] = pl.dot(
                        M, lam, trans_a=True, allow_tf32=False
                    )
                    return acc

                dM_acc = jax.lax.fori_loop(0, m ** (k - 1), inner_T, dM_acc)
                bar()
                cur = 1 - cur
            rd, wr, Pin = bufs[cur], bufs[1 - cur], P[k][0]

            def last_T(q, acc, rd=rd, wr=wr, Pin=Pin):
                r0 = pl.multiple_of(q * m, m)
                lam = rd[pl.ds(r0, m), :]
                acc = acc + pl.dot(lam, Pin[pl.ds(r0, m), :], trans_a=True, allow_tf32=False)
                wr[pl.ds(r0, m), :] = pl.dot(lam, M, allow_tf32=False)
                return acc

            dM_acc = jax.lax.fori_loop(0, nrow // m, last_T, dM_acc)
            bar()
            cur = 1 - cur
            if bufs[cur] is not lamB[k]:  # result landed in dF[k]: copy to lamB[k]

                def cpb(q, c, k=k):
                    r0 = pl.multiple_of(q * m, m)
                    lamB[k][pl.ds(r0, m), :] = dF[k][pl.ds(r0, m), :]
                    return c

                jax.lax.fori_loop(0, nrow // m, cpb, 0)
                bar()
        # lamA <- lamB for the next (earlier) step
        for k in range(N):
            nrow = m**k
            sr = m if k > 0 else 1

            def cpa(q, c, k=k, sr=sr):
                r0 = pl.multiple_of(q * sr, sr)
                lamA[k][pl.ds(r0, sr), :] = lamB[k][pl.ds(r0, sr), :]
                return c

            jax.lax.fori_loop(0, max(nrow // sr, 1), cpa, 0)
        bar()
        if not uniform:
            dm_ref[t, :, :] = dM_acc
            dM_acc = jnp.zeros_like(dM_acc)
        return dM_acc

    dM_acc = jax.lax.fori_loop(0, L, body, jnp.zeros((m, m), dt))
    if uniform:
        dm_ref[0, :, :] = dM_acc
    for k in range(N):
        dc_refs[k][...] = lamA[k][...]


def _dense_pass_general_bwd(
    Mc_, locals_, carries, outs, gs, lams, N, uniform, backend, num_warps=8
):
    BC, Lc = locals_[0].shape[:2]
    m = locals_[0].shape[-1]
    dt = locals_[0].dtype
    Lm = 1 if uniform else Lc

    def spec3(n1, n2):
        return pl.BlockSpec((None, n1, n2, m), lambda b: (b, 0, 0, 0))

    def spec2(n1):
        return pl.BlockSpec((None, n1, m), lambda b: (b, 0, 0))

    ins = [Mc_]
    in_specs = [pl.BlockSpec((None, Lm, m, m), lambda b: (b, 0, 0, 0))]
    for group, lead in ((locals_, True), (carries, False), (outs, True), (gs, True), (lams, False)):
        if group is None:
            continue
        for k, x in enumerate(group, 1):
            ins.append(x)
            in_specs.append(spec3(Lc, m ** (k - 1)) if lead else spec2(m ** (k - 1)))
    out_shape = [jax.ShapeDtypeStruct((BC, Lm, m, m), dt)]
    out_specs = [pl.BlockSpec((None, Lm, m, m), lambda b: (b, 0, 0, 0))]
    for k in range(1, N + 1):
        out_shape.append(jax.ShapeDtypeStruct((BC, Lc, m ** (k - 1), m), dt))
        out_specs.append(spec3(Lc, m ** (k - 1)))
    for group_count in [N, N, N]:  # dc, lamA, lamB
        for k in range(1, N + 1):
            out_shape.append(jax.ShapeDtypeStruct((BC, m ** (k - 1), m), dt))
            out_specs.append(spec2(m ** (k - 1)))
    for k in range(1, N + 1):  # partial flows: k + 1 buffers for level k (index k-1 -> k-1+2)
        for _ in range(k + 1):
            out_shape.append(jax.ShapeDtypeStruct((BC, m ** (k - 1), m), dt))
            out_specs.append(spec2(m ** (k - 1)))
    for k in range(1, N + 1):  # dF
        out_shape.append(jax.ShapeDtypeStruct((BC, m ** (k - 1), m), dt))
        out_specs.append(spec2(m ** (k - 1)))
    kern = functools.partial(
        _dense_kernel_general_bwd,
        N=N,
        m=m,
        uniform=uniform,
        has_g=gs is not None,
        has_lam=lams is not None,
    )
    outs_ = list(
        pl.pallas_call(
            kern,
            out_shape=out_shape,
            grid=(BC,),
            in_specs=in_specs,
            out_specs=out_specs,
            **_backend_kw(backend),
            compiler_params=plgpu.CompilerParams(num_warps=num_warps),
        )(*ins)
    )
    return outs_[0], outs_[1 : 1 + N], outs_[1 + N : 1 + 2 * N]


def dense_general_prefix_vjp(M, levels, chunk=None, backend="triton"):
    """Any depth, any A (f32): prefix combine by the buffered dense kernel with a custom VJP wrt (M, levels)."""
    return _dense_general_cvjp(M, tuple(levels), chunk, backend)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _dense_general_cvjp(M, levels, chunk, backend):
    return combine_pallas_dense_general(M, list(levels), chunk=chunk, backend=backend)


def _dense_general_fwd(M, levels, chunk, backend):
    out = _dense_general_cvjp(M, levels, chunk, backend)
    return out, (M, levels, out)


def _dense_general_bwd(chunk, backend, res, g):
    from ews.core.chen_scan import chen_binary_op_packed_with_M

    M, levels, out = res
    N = len(levels)
    B, L = levels[0].shape[:2]
    m = levels[0].shape[-1] if levels[0].ndim == 3 else levels[0].shape[2]
    dt = levels[0].dtype
    uniform = M.shape[1] == 1
    chunk_ = L if chunk is None else min(chunk, L)
    C = L // chunk_
    sizes = [m**k for k in range(1, N + 1)]
    offs = np.cumsum([0] + sizes)
    D = int(offs[-1])

    def split(x, lead):
        return [x[..., offs[k] : offs[k + 1]].reshape(lead + (m**k, m)) for k in range(N)]

    Mc_ = (
        jnp.broadcast_to(M[:, None], (B, C, 1, m, m)).reshape(B * C, 1, m, m)
        if uniform
        else M.reshape(B * C, chunk_, m, m)
    )
    loc = [levels[k].reshape(B * C, chunk_, m**k, m) for k in range(N)]
    outs = split(out.reshape(B * C, chunk_, D), (B * C, chunk_))
    gs = split(g.reshape(B * C, chunk_, D), (B * C, chunk_))
    prev = (
        jnp.concatenate([jnp.zeros_like(out[:, :1]), out[:, chunk_ - 1 :: chunk_][:, :-1]], 1)
        if C > 1
        else jnp.zeros_like(out[:, :1])
    )
    carries = split(prev.reshape(B * C, D), (B * C,))
    dM, ds, dc = _dense_pass_general_bwd(Mc_, loc, carries, outs, gs, None, N, uniform, backend)
    if C > 1:
        dcp = jnp.concatenate([x.reshape(B * C, -1) for x in dc], -1).reshape(B, C, D)
        zeros = [jnp.zeros((B * C, m**k, m), dt) for k in range(N)]
        _, e, em = _dense_pass_general(Mc_, loc, zeros, N, uniform, False, True, backend)
        Pk = jnp.concatenate([x.reshape(B, C, -1) for x in e], -1)
        Mchunk = em.reshape(B, C, m, m)

        def JT(c_idx, lam):
            Pc, Mc = Pk[:, c_idx], Mchunk[:, c_idx]
            f = lambda carry: jax.vmap(
                lambda cr, p_, M_: chen_binary_op_packed_with_M(
                    (cr, 0.0, jnp.eye(m, dtype=dt)), (p_, 0.0, M_), None, m, N
                )[0]
            )(carry, Pc, Mc)
            _, vjp = jax.vjp(f, jnp.zeros((B, D), dt))
            return vjp(lam)[0]

        def step(lam_next, c_idx):
            with jax.default_matmul_precision("highest"):
                lam_c = dcp[:, c_idx + 1] + JT(c_idx + 1, lam_next)
            return lam_c, lam_c

        _, lam_hist = jax.lax.scan(step, jnp.zeros((B, D), dt), jnp.arange(C - 2, -1, -1))
        lam_end = jnp.concatenate([lam_hist[::-1], jnp.zeros((1, B, D), dt)], 0).transpose(1, 0, 2)
        lams = split(lam_end.reshape(B * C, D), (B * C,))
        dM3, ds3, _ = _dense_pass_general_bwd(
            Mc_, loc, carries, outs, None, lams, N, uniform, backend
        )
        dM = dM + dM3
        ds = [a + b for a, b in zip(ds, ds3)]
    dM_out = dM.reshape(B, C, 1, m, m).sum(1) if uniform else dM.reshape(B, L, m, m)
    dlevels = tuple(ds[k].reshape(levels[k].shape) for k in range(N))
    return dM_out, dlevels


_dense_general_cvjp.defvjp(_dense_general_fwd, _dense_general_bwd)
