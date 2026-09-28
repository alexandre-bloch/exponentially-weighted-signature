"""The weather model: stacked EWS blocks (`ews.block.StackedEWS`) with a block-diagonal A on the fused block kernel,
and the evaluation helpers.

A is rotation planes (`damped_rotation` and the other rotation families, on the plane closed forms) or uniform Jordan
chains (`jordan_repeated`, on the closed-form Jordan leaf; the banked model uses complex chains of size 4 on the time
clock). Each block has a general lift (identity-block init), the integral path with a basepoint, a live-RMS graded
norm, a gelu-GLU channel stage and a pre-norm residual, and is rematerialised in the backward pass. The encoder is
linear and the head acts per position; its last `pred_len` positions are the forecast.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ews.block import StackedEWS
from ews.core.kernels import pallas_combine as pc
from ews.core.kernels import plane_engine as pe

PRED_LEN = 720


class RematEWSBlock(eqx.Module):
    """jax.checkpoint around an EWSBlock call, same signature; attribute access (clock, ewms, ...) is forwarded to the
    wrapped block so StackedEWS's clock-source / a_scale logic keeps working."""

    block: eqx.Module

    def __getattr__(self, name):
        if name == "block":
            raise AttributeError(name)
        return getattr(self.block, name)

    def __call__(
        self, x, *, t=None, mask=None, clock_driver=None, key=None, inference=False, a_scale=1.0
    ):
        fn = lambda x, t, mask, cd, key: self.block(
            x, t=t, mask=mask, clock_driver=cd, key=key, inference=inference, a_scale=a_scale
        )
        return eqx.filter_checkpoint(fn)(x, t, mask, clock_driver, key)


def block_kwargs(
    m,
    depth,
    norm_constant,
    dropout,
    omega_scale=1.0,
    damp_init_mean=-2.0,
    leaf="series",
    series_q=None,
    clock_mode="time",
    a_param="damped_rotation",
    sigma_range=0.25,
    path_mode="increment",
    basepoint=True,
    jordan=None,
    init_variance=1.0,
    jordan_flows=False,
):
    """EWSBlock keyword arguments. clock_mode: 'time' (unit increments; the banked runs), 'token_gated'
    (relu(w.dx - b) / c) or 'clamped_gate' (sum of 4 clipped units / c), c = norm_constant."""
    compute = (
        None
        if a_param == "jordan_repeated"  # set in the Jordan branch below
        else dict(leaf="series", series_q=int(series_q))
        if leaf == "series"
        else dict(leaf="van_loan", M=4, van_loan_expm="exact")
    )
    if clock_mode == "token_gated":
        clock = {
            "mode": "token_gated",
            "normalize": "constant",
            "scaling": "none",
            "norm_constant": float(norm_constant),
            "gate_init_bias": -2.0,
            "source": "block_input",
        }
    elif clock_mode == "clamped_gate":
        clock = {
            "mode": "clamped_gate",
            "normalize": "constant",
            "scaling": "none",
            "gate_K": 4,
            "norm_constant": float(norm_constant),
            "gate_init_bias": -2.0,
            "source": "block_input",
        }
    elif clock_mode == "time":
        clock = {"mode": "time", "scaling": "none"}
    else:
        raise ValueError(
            f"clock_mode must be time, token_gated or clamped_gate, got {clock_mode!r}"
        )
    if a_param == "jordan_repeated":
        # Uniform Jordan chains, `jordan` = "complex4", "real2", ...: m / size blocks of one kind. A complex block is
        # [[a, -b], [b, a]] on the diagonal with a = softplus(v) and b free, 2x2 identity above; v and b are both drawn
        # from N(damp_init_mean, init_variance) (the jordan_repeated init), so the damping matches the rotation-plane
        # model's softplus(N(damp_init_mean, 1)) when init_variance = 1. The plane kernel engine cannot run a Jordan
        # A, so these blocks keep the generic EWS path, on the closed-form Jordan leaf and the block-kernel combine.
        kind, size = (
            ("complex", int(jordan[len("complex") :]))
            if jordan.startswith("complex")
            else ("real", int(jordan[len("real") :]))
        )
        if m % size:
            raise ValueError(f"m = {m} is not a multiple of the Jordan chain size {size}")
        a_config = {
            "parametrization": "jordan_repeated",
            "learn": True,
            "blocks": [size] * (m // size),
            "kinds": [kind] * (m // size),
            "init": {
                "type": "jordan_repeated",
                "mean": float(damp_init_mean),
                "variance": float(init_variance),
            },
        }
        compute = dict(leaf="jordan_closed", combine="block_kernel")
        # The time clock is uniform, so the block kernel may build one flow exp(-A) per block and reuse it for
        # every step (the uniform-clock fast path, `reweighting_mode` "displacement_uniform[_basepoint]"; the
        # basepoint interval has dtheta = 0 and gets the identity). Same arrays, 1.17x faster per step on the
        # banked model. A gated clock is not uniform and keeps the per-interval path.
        # jordan_flows: the block kernel writes each interval's flow in closed form instead (see
        # ews.core.kernels.combine.jordan_block_operators), which is what a gated clock needs on long paths.
        if jordan_flows:
            compute["jordan_flows"] = True
        if clock_mode == "time":
            compute["reweighting_mode"] = (
                "displacement_uniform_basepoint" if basepoint else "displacement_uniform"
            )
    else:
        a_config = {
            "parametrization": a_param,
            "init": {"type": "sig", "sigma_range": float(sigma_range)},
            "damp_init_mean": float(damp_init_mean),
            "damp_init_std": 1.0,
            "omega_scale": float(omega_scale),
        }
    return dict(
        m=int(m),
        n=int(depth),
        a_config=a_config,
        b_config={
            "enabled": True,
            "out_dim": int(m),
            "parametrization": "general",
            "init": {"type": "identity_block", "mean": 0.0, "variance": 1.0},
        },
        **{"reweighting_mode": "displacement", **compute},
        chen_expm="exact",
        basepoint=bool(basepoint),
        path_mode=path_mode,
        clock=clock,
        graded_norm="live_rms",
        channel_stage="gelu_glu",
        ff_mult=4,
        dropout_rate=float(dropout),
        norm_placement="pre",
    )


def build(
    d_in,
    d_out,
    H,
    num_blocks,
    m,
    depth,
    norm_constant,
    dropout,
    key,
    remat=True,
    leaf="van_loan",
    series_q=None,
    clock_mode="time",
    a_param="damped_rotation",
    omega_scale=1.0,
    damp_init_mean=-2.0,
    sigma_range=0.25,
    path_mode="increment",
    basepoint=True,
    jordan=None,
    init_variance=1.0,
    omega_log_uniform_hi=0.0,
    jordan_flows=False,
):
    kw = block_kwargs(
        m,
        depth,
        norm_constant,
        dropout,
        leaf=leaf,
        series_q=series_q,
        clock_mode=clock_mode,
        a_param=a_param,
        omega_scale=omega_scale,
        damp_init_mean=damp_init_mean,
        sigma_range=sigma_range,
        path_mode=path_mode,
        basepoint=basepoint,
        jordan=jordan,
        init_variance=init_variance,
        jordan_flows=jordan_flows,
    )
    model = StackedEWS(
        num_blocks,
        d_in,
        H,
        d_out,
        tokens=False,
        head_mode="seq2seq",
        block_kwargs=kw,
        per_block_kwargs=None,
        dtype=jnp.float32,
        key=key,
    )
    if a_param == "jordan_repeated" and float(omega_log_uniform_hi) > 0.0:
        # The jordan_repeated init draws a complex block's damping and its rotation from one Gaussian, so it cannot set
        # a rotation scale on its own. `omega_log_uniform_hi` redraws each complex block's rotation log-uniformly in
        # [hi/100, hi] rad/step (the rotation-plane model spreads its frequencies over (0, 1.8)); the damping is kept.
        # Block i uses jr.fold_in(key, 7000 + i), the same convention as experiments/uea/train.py.
        hi = float(omega_log_uniform_hi)
        new_blocks = []
        for i, blk in enumerate(model.blocks):
            bp = blk.ewms.A_param.block_params
            u = jr.uniform(
                jr.fold_in(key, 7000 + i),
                (len(bp),),
                minval=math.log(hi / 100.0),
                maxval=math.log(hi),
            )
            nb = tuple(
                jnp.stack([p[0], jnp.exp(u[j])]).astype(p.dtype) if np.size(p) == 2 else p
                for j, p in enumerate(bp)
            )
            new_blocks.append(eqx.tree_at(lambda b_: b_.ewms.A_param.block_params, blk, nb))
        model = eqx.tree_at(lambda mm: mm.blocks, model, tuple(new_blocks))
    if remat:
        model = eqx.tree_at(
            lambda mm: mm.blocks, model, tuple(RematEWSBlock(b) for b in model.blocks)
        )
    return model


def calibrate_norm_constant(d_in, d_out, H, num_blocks, m, depth, X, key, **model_kw):
    """norm_constant such that block 1's gated clock has mean increment 1 at init on a batch (for the 'time' clock the
    increments are 1 already), plus the series-leaf regime x = max plane |z| x max clock step (no eigendecomposition:
    |z| = sqrt(d^2 + w^2) is read off the 2x2 planes)."""
    model = build(d_in, d_out, H, num_blocks, m, depth, 1.0, 0.0, key, remat=False, **model_kw)
    enc = jax.vmap(jax.vmap(model.embedding))(jnp.asarray(X, jnp.float32))  # (B, L, H)
    blk = model.blocks[0]
    L = X.shape[1]
    t = jnp.arange(L, dtype=jnp.float32)
    theta = jax.vmap(lambda drv: blk.clock(t, drv, None))(enc)  # (B, L)
    inc = jnp.diff(theta, axis=1)
    c = float(jnp.mean(inc))
    A = np.asarray(blk.ewms.A_param(), np.float64)
    ii = np.arange(0, A.shape[0], 2)
    a_norm = float(np.max(np.sqrt(A[ii, ii] ** 2 + A[ii, ii + 1] ** 2)))
    x_max = a_norm * float(inc.max()) / max(c, 1e-6)
    return max(c, 1e-6), dict(
        mean=c,
        min=float(inc.min()),
        max=float(inc.max()),
        zero_frac=float(jnp.mean(inc == 0)),
        a_spectral_radius=a_norm,
        x_max_init=x_max,
    )


@eqx.filter_jit
def forward(model, X, key, inference):
    out = model(X, key=key, inference=inference)  # (B, L, d_out)
    return out[:, -PRED_LEN:]


def eval_split(model, split, batch, key, max_batches=None):
    """MAE and MSE over a split (ordered windows, once), in the scaled space; returns the advanced key."""
    abs_sum = sq_sum = 0.0
    count = 0
    for bi, (X, y) in enumerate(split.loop_epoch(batch)):
        if max_batches is not None and bi >= max_batches:
            break
        key, sk = jr.split(key)
        pred = forward(model, jnp.asarray(X), sk, True)
        d = np.asarray(pred, np.float64) - y.astype(np.float64)
        abs_sum += np.abs(d).sum()
        sq_sum += (d**2).sum()
        count += d.size
    return abs_sum / count, sq_sum / count, key


# ------------------------------------------------------------------------------------------- kernel engine (rotation A)


def pick_chunk(L, target=1024):
    """Largest divisor of L that is <= target (falls back to L)."""
    best = 1
    for c in range(1, min(L, target) + 1):
        if L % c == 0:
            best = c
    return best if best > 1 else L


def planes_from_A(A):
    """(m, m) block-diagonal plane matrix [[d, w], [-w, d]] blocks -> (K, 2) [d, w]."""
    ii = jnp.arange(0, A.shape[0], 2)
    return jnp.stack([A[ii, ii], A[ii, ii + 1]], -1)


class KernelEWS(eqx.Module):
    """Drop-in for the block's ``ews.layer.EWS`` (same call contract, same trainable A_param): the prefix EWS of a
    rotation-plane A by the plane closed forms (stage 1) and the fused block kernel with its custom VJP (stage 2),
    depths 1 and 2. The plane count is padded to a power of two with zero planes (exact)."""

    base: eqx.Module
    chunk_target: int = eqx.field(static=True, default=1024)
    tile: int = eqx.field(static=True, default=4)

    def __getattr__(self, name):
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    def _single_path(self, path, clock, readout, output_format, a_scale=1.0):
        if readout != "prefix" or output_format != "flat":
            raise NotImplementedError("KernelEWS: readout='prefix', output_format='flat' only")
        N = int(self.base.N)
        if N > 2:
            raise NotImplementedError("closed-form plane local terms exist for depths 1 and 2")
        dtype = self.base.dtype
        path = path.astype(dtype)
        clock = clock.astype(dtype)
        A = self.base.A_param().astype(dtype) * jnp.asarray(a_scale, dtype=dtype)
        v = path[1:] - path[:-1]  # (L, m) increments
        dtheta = clock[1:] - clock[:-1]  # (L,)
        L, m = v.shape
        chunk = pick_chunk(L, self.chunk_target)
        dw = planes_from_A(A)  # (K, 2)
        K = dw.shape[0]
        Kp = 1 << (K - 1).bit_length()
        if Kp != K:
            dw = jnp.concatenate([dw, jnp.zeros((Kp - K, 2), dw.dtype)], 0)
            v = jnp.concatenate([v, jnp.zeros((L, 2 * (Kp - K)), v.dtype)], 1)
        mp = 2 * Kp
        rho, s1, s2 = pe.local_terms(v, dw, dtheta, N)  # stage 1: closed forms (autodiff)
        E = pe.rot_blocks(rho)  # (L, Kp, 2, 2)
        levels = [pe.from_complex_planes(s1)] + ([pe.blocks_to_flat(s2)] if N == 2 else [])
        out = pc.block_prefix_vjp(
            E[None], [l[None] for l in levels], tile=min(self.tile, Kp), chunk=chunk
        )[0]
        if mp != m:  # unpad: level-1 columns, level-2 rows x cols
            lvl1 = out[:, :m]
            out = (
                lvl1
                if N == 1
                else jnp.concatenate(
                    [lvl1, out[:, mp:].reshape(L, mp, mp)[:, :m, :m].reshape(L, m * m)], 1
                )
            )
        return jnp.concatenate(
            [jnp.ones((L, 1), dtype), out], axis=1
        )  # flat with the level-0 ones column

    def __call__(
        self, path, clock, return_terminal=True, output_format="flat", readout=None, a_scale=1.0
    ):
        if readout is None:
            readout = "terminal" if return_terminal else "prefix"
        if path.ndim == 2:
            return self._single_path(path, clock, readout, output_format, a_scale)
        return jax.vmap(lambda p, c: self._single_path(p, c, readout, output_format, a_scale))(
            path, clock
        )


def swap_in_kernel_ews(model, chunk_target=1024, tile=4):
    """Replace every block's ``ewms`` by a KernelEWS wrapping it (works through RematEWSBlock wrappers)."""
    from ews.core.kernels.combine import is_rotation_planes

    def inner(b):
        return b.block if hasattr(b, "block") else b

    for b in model.blocks:
        if not is_rotation_planes(inner(b).ewms.A_param):
            raise ValueError("the weather kernel engine needs a rotation-plane A parametrisation")
    new = [KernelEWS(inner(b).ewms, chunk_target=chunk_target, tile=tile) for b in model.blocks]
    return eqx.tree_at(lambda mm: [inner(b).ewms for b in mm.blocks], model, new)
