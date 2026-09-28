"""The EWS translation mixer: a drop-in replacement for an SSM layer, (L, H) -> (L, H).

It is the EWS block's mixer with the converted values pinned:

    path   = [0; cumsum(u @ B_ews^T)]           the integral path with a full-step basepoint
    theta  = (0, 1, ..., L)                     the uniform clock
    levels = GradedWarmNorm(EWS(A_ews, N)[:, 1:])
    y      = sum_k levels_k @ C_k^T + D * u     C_1 the converted C, C_k (k >= 2) initialised at 0

The read-down is stored per level so the blocks can be initialised, frozen and trained separately. GradedWarmNorm is
GradedRMSNorm with level 1 passed through untouched, which keeps the converted pathway exact; levels >= 2 get a live
RMS with a learnable gain. combine is left at the EWS default "assoc.build" (seq.direct.remat does not work
correctly).
"""

import jax

jax.config.update("jax_enable_x64", True)

from typing import Optional, Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from ews.block import graded_level_slices
from ews.layer import EWS


class GradedWarmNorm(eqx.Module):
    """Per-level norm that spares level 1: identity on lvl1, live-RMS on >= 2.

    Input/output: (..., D) with D = sum_{k=1..N} m^k (level-0-free layout).
    At N=1 the module is the identity and carries no parameters. For k >= 2:

        out_k = g_k * lvl_k / (RMS_k + eps_k)

    with RMS over the m^k level coordinates per (sample, position) (causal)
    computed with the zero-safe sqrt (value and gradient exactly 0 on an
    all-zero level block, e.g. the trivial prefix row).
    """

    m: int = eqx.field(static=True)
    N: int = eqx.field(static=True)
    level_slices: Tuple[Tuple[int, int], ...] = eqx.field(static=True)
    eps: Tuple[float, ...] = eqx.field(static=True)
    gains: Optional[jnp.ndarray]  # (N-1,) for levels 2..N; None at N=1

    def __init__(self, m: int, N: int, eps=1e-3, dtype=jnp.float64, level_dims=None):
        self.m = int(m)
        self.N = int(N)
        if level_dims is None:
            self.level_slices = graded_level_slices(self.m, self.N)
        else:  # explicit per-level widths, e.g. [m, (2K')^2] for the K'-plane level 2
            slices, start = [], 0
            for d in level_dims:
                slices.append((start, start + int(d)))
                start += int(d)
            self.level_slices = tuple(slices)
        if isinstance(eps, (int, float)):
            eps_tuple = tuple(float(eps) for _ in range(max(self.N - 1, 0)))
        else:
            eps_tuple = tuple(float(e) for e in eps)
            if len(eps_tuple) != max(self.N - 1, 0):
                raise ValueError("eps must be a scalar or a sequence of length N-1")
        self.eps = eps_tuple
        self.gains = jnp.ones((self.N - 1,), dtype=dtype) if self.N >= 2 else None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.shape[-1] != self.level_slices[-1][1]:
            raise ValueError(f"last axis must be D = {self.level_slices[-1][1]}, got {x.shape[-1]}")
        if self.N == 1:
            return x
        s1, e1 = self.level_slices[0]
        blocks = [x[..., s1:e1]]  # level 1: untouched, structurally
        for k in range(2, self.N + 1):
            s, e = self.level_slices[k - 1]
            block = x[..., s:e]
            ms = jnp.mean(block * block, axis=-1, keepdims=True)
            safe = jnp.where(ms > 0, ms, 1.0)
            rms = jnp.where(ms > 0, jnp.sqrt(safe), 0.0)
            blocks.append(self.gains[k - 2] * block / (rms + self.eps[k - 2]))
        return jnp.concatenate(blocks, axis=-1)


class EWSMixer(eqx.Module):
    """Drop-in SSM-layer replacement built from converted EWS weights.

    __call__(u: (L, H)) -> (L, H). At N=1 (or with zero C_{>=2}) the output
    equals the source SSM layer to machine precision; the level-1 pathway is
    the converted SSM at every N.
    """

    ewms: EWS
    B: jnp.ndarray  # (m, H) lift
    C_levels: Tuple[jnp.ndarray, ...]  # level k: (H, m^k); C_1 = converted C
    D: jnp.ndarray  # (H,) feedthrough
    lvl2_proj: Optional[jnp.ndarray]  # (r, m^2) frozen projection (PCA-r arm)
    norm: GradedWarmNorm
    # Options: a learnable relu clock (increment relu(u.w - b), initialised to the uniform clock LinOSS was trained
    # under), a bilinear lowrank(r) level-2 readout U^T S V -> C2r (zero init), an antisymmetrised level 2 (log form);
    # engine="kernel" = closed forms / blockwise series + fused block kernel
    clock_w: Optional[jnp.ndarray]  # (H,) or None (uniform clock)
    clock_b: Optional[jnp.ndarray]  # () or None
    clock_alpha: Optional[jnp.ndarray]  # () learnable alpha of the blended-variation clock, or None
    clock_c: float = eqx.field(
        static=True, default=1.0
    )  # variation normalisation constant of that clock
    U: Optional[jnp.ndarray]  # (m, r) or None (full C_2)
    V: Optional[jnp.ndarray]  # (m, r)
    C2r: Optional[jnp.ndarray]  # (H, r^2)
    log2: bool = eqx.field(static=True, default=False)
    kernel_chunk: int = eqx.field(static=True, default=1024)
    # level 2 on a subset of planes J (indices of 2x2 blocks): level 1 stays on all K planes,
    # level 2 is the exact level-2 EWS of the sub-path of the J planes ((2K')^2 coordinates instead of m^2)
    planes_J: Optional[tuple] = eqx.field(static=True, default=None)
    m2: int = eqx.field(static=True, default=0)  # width of the level-2 "path": 2K' (or m)
    m: int = eqx.field(static=True)
    N: int = eqx.field(static=True)
    H: int = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)
    engine: str = eqx.field(static=True, default="dense")
    plane_combine: str = eqx.field(static=True, default="assoc")
    series_q_bd: int = eqx.field(
        static=True, default=24
    )  # series order for engine='blockdiag' level 2

    def __init__(
        self,
        weights: dict,
        N: int = 1,
        *,
        leaf: str = "van_loan",
        van_loan_expm: str = "exact",
        chen_expm: str = "exact",
        M: int = 4,
        series_q=None,
        graded_eps=1e-3,
        a_param=None,
        lvl2_proj=None,
        dtype=jnp.float64,
        engine: str = "dense",
        plane_combine: str = "assoc",
        series_q_bd: int = 24,
        relu_clock: bool = False,
        lowrank_r=None,
        log2: bool = False,
        kernel_chunk: int = 1024,
        key=None,
        planes_J=None,
        alpha_clock=False,
        alpha_init=1.0,
        alpha_c=1.0,
    ):
        # engine="plane": the block-aware engine of
        # ews.core.kernels.plane_engine: per-plane closed forms, no m x m matrix in
        # the scan. Requires a_param to be a JordanTiedA (canonical planes).
        # Same flat layout as the dense EWS, so everything downstream (norm,
        # read-down, level-2 projection) is shared. N <= 2.
        if engine not in ("dense", "plane", "blockdiag", "kernel"):
            raise ValueError("engine must be 'dense', 'plane', 'blockdiag' or 'kernel'")
        if engine in ("plane", "blockdiag"):
            if a_param is None or not isinstance(a_param, JordanTiedA):
                raise ValueError("engine='plane' requires a JordanTiedA a_param")
            if int(N) > 2:
                raise ValueError("engine='plane' supports N <= 2")
        if engine == "kernel":
            if a_param is None or not (
                isinstance(a_param, JordanTiedA) or hasattr(a_param, "sizes")
            ):
                raise ValueError(
                    "engine='kernel' requires a JordanTiedA or a block-diagonal a_param"
                )
            if int(N) > 2:
                raise ValueError("engine='kernel' supports N <= 2")
        self.log2 = bool(log2)
        self.kernel_chunk = int(kernel_chunk)
        if planes_J is not None and engine != "kernel":
            raise ValueError("planes_J needs engine='kernel'")
        self.planes_J = None if planes_J is None else tuple(sorted(int(j) for j in planes_J))
        self.engine = engine
        self.plane_combine = plane_combine
        self.series_q_bd = int(series_q_bd)
        A = jnp.asarray(weights["A"], dtype)
        if a_param is not None and a_param.dim != A.shape[0]:
            raise ValueError(f"a_param.dim {a_param.dim} != weights A dim {A.shape[0]}")
        self.m = int(A.shape[0])
        self.N = int(N)
        self.dtype = dtype
        self.m2 = 2 * len(self.planes_J) if self.planes_J is not None else self.m
        self.B = jnp.asarray(weights["B"], dtype)
        self.H = int(self.B.shape[1])
        C1 = jnp.asarray(weights["C"], dtype)
        self.lvl2_proj = None if lvl2_proj is None else jnp.asarray(lvl2_proj, dtype)
        if self.lvl2_proj is not None and self.N != 2:
            raise ValueError("lvl2_proj requires N=2")
        levels = [C1]
        for k in range(2, self.N + 1):
            width = (
                self.lvl2_proj.shape[0]
                if (k == 2 and self.lvl2_proj is not None)
                else (self.m2**2 if k == 2 else self.m**k)
            )
            levels.append(jnp.zeros((self.H, width), dtype))
        self.C_levels = tuple(levels)
        self.D = jnp.asarray(weights["D"], dtype)
        self.norm = GradedWarmNorm(
            self.m,
            self.N,
            eps=graded_eps,
            dtype=dtype,
            level_dims=None if self.planes_J is None else [self.m, self.m2**2],
        )
        if (
            relu_clock and not alpha_clock
        ):  # w = 0, b = -1: relu(0 + 1) = 1 per token = the uniform clock
            self.clock_w = jnp.zeros((self.H,), dtype)
            self.clock_b = jnp.asarray(-1.0, dtype)
        else:
            self.clock_w = None
            self.clock_b = None
        if alpha_clock:  # increment = alpha + (1 - alpha) ||du|| / c; alpha = 1 = uniform
            self.clock_alpha = jnp.asarray(float(alpha_init), dtype)
            self.clock_c = float(alpha_c)
        else:
            self.clock_alpha = None
            self.clock_c = 1.0
        if lowrank_r is not None:
            if self.N != 2:
                raise ValueError("lowrank_r needs N=2")
            r = int(lowrank_r)
            k1, k2 = jr.split(jr.PRNGKey(0) if key is None else key)
            self.U = (jr.normal(k1, (self.m2, r)) / jnp.sqrt(self.m2)).astype(dtype)
            self.V = (jr.normal(k2, (self.m2, r)) / jnp.sqrt(self.m2)).astype(dtype)
            self.C2r = jnp.zeros((self.H, r * r), dtype)
        else:
            self.U = self.V = self.C2r = None
        self.ewms = EWS(
            A if a_param is None else a_param,
            N=self.N,
            M=int(M),
            dtype=dtype,
            reweighting_mode="displacement_uniform",
            terminal_reduce="tree",
            van_loan_expm=van_loan_expm,
            chen_expm=chen_expm,
            leaf=leaf,
            series_q=series_q,
            # combine left at default "assoc.build": seq.direct.remat banned.
        )

    def __call__(self, u: jnp.ndarray) -> jnp.ndarray:
        u = jnp.asarray(u, self.dtype)
        L = u.shape[0]
        v = u @ self.B.T
        path = jnp.concatenate([jnp.zeros((1, self.m), self.dtype), jnp.cumsum(v, axis=0)])
        if self.engine == "kernel":
            from ews.conversion.kernel_ews import blocks_prefix, rotation_prefix

            if self.clock_w is not None:
                dtheta = jax.nn.relu(u @ self.clock_w - self.clock_b)
            elif self.clock_alpha is not None:
                a = jnp.clip(self.clock_alpha, 0.0, 1.0)
                du = jnp.concatenate(
                    [jnp.zeros((1, self.H), self.dtype), jnp.diff(u, axis=0)], axis=0
                )
                dtheta = a + (1.0 - a) * jnp.sqrt((du * du).sum(axis=1) + 1e-8) / self.clock_c
            else:
                dtheta = jnp.ones((L,), self.dtype)
            A = self.ewms.A_param().astype(self.dtype)
            sizes = getattr(self.ewms.A_param, "sizes", None)

            def _prefix(vv, AA, N, sz):
                if sz is None:
                    return rotation_prefix(vv, dtheta, AA, N, tile=4, chunk=self.kernel_chunk)
                return blocks_prefix(
                    vv,
                    dtheta,
                    AA,
                    sz,
                    N,
                    tile=4,
                    chunk=self.kernel_chunk,
                    series_q=self.series_q_bd,
                )

            if self.N == 2 and self.planes_J is not None:
                import numpy as _np

                cols = _np.asarray([[2 * j, 2 * j + 1] for j in self.planes_J]).reshape(-1)
                raw1 = _prefix(v, A, 1, sizes)  # level 1 on all planes
                A_J = A[_np.ix_(cols, cols)]
                raw2 = _prefix(
                    v[:, cols], A_J, 2, None if sizes is None else [2] * len(self.planes_J)
                )
                raw = jnp.concatenate(
                    [raw1, raw2[:, self.m2 :]], axis=1
                )  # [level 1 (m) | level 2 of the J sub-path]
            else:
                raw = _prefix(v, A, self.N, sizes)
        elif self.engine == "plane":
            from ews.conversion.plane_engine import plane_prefix, planes_from_jordan_tied

            dw = planes_from_jordan_tied(self.ewms.A_param).astype(self.dtype)
            raw = plane_prefix(
                v, dw, jnp.ones((L,), self.dtype), N=self.N, combine=self.plane_combine
            )
        elif self.engine == "blockdiag":
            # the general block-diagonal engine on the planes as free 2x2 blocks: batched Pade
            # exponential (no closed form used), series level-2 local terms, one cached exponential
            # under the uniform clock: what a non-Jordan block_diagonal A would get
            from ews.conversion.blockdiag_engine import blockdiag_prefix, expm_pade
            from ews.conversion.plane_engine import planes_from_jordan_tied

            dw = planes_from_jordan_tied(self.ewms.A_param).astype(self.dtype)
            d, w = dw[:, 0], dw[:, 1]
            blocks = jnp.stack([jnp.stack([d, w], -1), jnp.stack([-w, d], -1)], -2)
            raw = blockdiag_prefix(
                v,
                blocks,
                jnp.ones((L,), self.dtype),
                N=self.N,
                expm_fn=expm_pade,
                series_q=self.series_q_bd,
                local2="series",
                uniform=True,
            )
        else:
            theta = jnp.arange(L + 1, dtype=self.dtype)
            flat = self.ewms(path, theta, readout="prefix", output_format="flat")
            raw = flat[:, 1:]  # drop the level-0 constant
        levels = self.norm(raw)
        y = u * self.D
        start = 0
        for k, Ck in enumerate(self.C_levels, start=1):
            size = self.m2**2 if k == 2 else self.m**k
            feats = levels[:, start : start + size]
            if (
                k == 2 and self.U is not None
            ):  # bilinear lowrank(r): U^T S V (antisymmetrised = log form) -> C2r
                S = feats.reshape(L, self.m2, self.m2)
                if self.log2:
                    S = 0.5 * (S - jnp.swapaxes(S, 1, 2))
                Z = jnp.einsum("ma,lmn,nb->lab", self.U, S, self.V).reshape(L, -1)
                y = y + Z @ self.C2r.T
                start += size
                continue
            if k == 2 and self.lvl2_proj is not None:
                feats = feats @ self.lvl2_proj.T  # frozen r-dim projection
            y = y + feats @ Ck.T
            start += size
        return y


class JordanTiedA(eqx.Module):
    """Canonical damped-rotation A with tied copies.

    Stores only the generating parameters:
    one (decay d_k, frequency w_k) pair per source channel, materialised as the
    (m, m) block-diagonal matrix with each channel's canonical 2x2 block
    [[d, w], [-w, d]] repeated `copies` times (2 for LinOSS: the re/im halves
    share one oscillator: the tied choice; 1 for S5/LRU). Trainable leaves:
    d and w only (32 per LinOSS block), never the dense store: training the
    dense constant would free its 3,968 structural zeros, which is the dense-A
    ablation.
    """

    d: jnp.ndarray  # (P,) decay
    w: jnp.ndarray  # (P,) rotation per unit clock
    copies: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)

    def __init__(self, d, w, copies=2):
        self.d = jnp.asarray(d)
        self.w = jnp.asarray(w)
        self.copies = int(copies)
        self.dim = 2 * self.copies * self.d.shape[0]

    def __call__(self) -> jnp.ndarray:
        P = self.d.shape[0]
        blk = jnp.zeros((P, 2, 2), self.d.dtype)
        blk = blk.at[:, 0, 0].set(self.d).at[:, 1, 1].set(self.d)
        blk = blk.at[:, 0, 1].set(self.w).at[:, 1, 0].set(-self.w)
        blk = jnp.repeat(blk, self.copies, axis=0)
        return jax.scipy.linalg.block_diag(*blk)
