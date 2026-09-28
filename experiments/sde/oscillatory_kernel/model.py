"""The bare model of the oscillatory-kernel experiment: dense lift, one EWS layer, linear readout.

Also: the seeded log-uniform init override of the A parameters (applied after the engine's own
init, so the engine is untouched), the planted-solution constructors of the sanity checks, and
the extraction of the learned generator parameters.
"""

from __future__ import annotations

from typing import Dict

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ews.config import upgrade_layer_model
from ews.layer import EWS, B
from ews.params import (
    AParamDampedRotation,
    AParamEFM,
    AParamGeneral,
    AParamSig,
    build_ab_from_config,
    parse_a_config,
)
from experiments.sde.oscillatory_kernel.data import BANK_HZ, DAMPING, HIGH_PLANES, feature_count

MODEL_OF_PARAM = {"sig": "sig", "efm": "efm", "damped_rotation": "ews", "general": "dense"}


class OscModel(eqx.Module):
    """lift (dense m x 2) -> EWS prefix signature -> levels 1..n -> linear readout per step."""

    lift: B
    ews: EWS
    head: eqx.nn.Linear
    dtype: jnp.dtype = eqx.field(static=True)

    def features(self, paths: jnp.ndarray, clocks: jnp.ndarray) -> jnp.ndarray:
        """(batch, L, 2) paths and (batch, L) clocks -> (batch, L - 1, D) flat levels 1..n."""
        lifted = jax.vmap(self.lift)(paths.astype(self.dtype))
        sig = self.ews(
            lifted, clocks.astype(self.dtype), return_terminal=False, output_format="flat"
        )
        return sig[..., 1:]  # drop the level-0 constant

    def __call__(self, paths: jnp.ndarray, clocks: jnp.ndarray) -> jnp.ndarray:
        feats = self.features(paths, clocks)
        return jax.vmap(jax.vmap(self.head))(feats)  # (batch, L - 1, k)

    @property
    def feature_dim(self) -> int:
        return self.ews.signature_dim


def resolve_dtype(name: str):
    return jnp.float64 if name == "float64" else jnp.float32


def build_model(model_cfg: dict, n_out: int, key, dtype, readout_init: str = "zero") -> OscModel:
    """`readout_init`: "zero" (weights and bias zero, so predictions start at 0 and the first
    normalised loss is 1 for every model) or "default" (equinox's uniform init)."""
    model_cfg = upgrade_layer_model(model_cfg)
    comp = model_cfg.get("compute", {})
    k_ab, k_head = jr.split(key)
    a_param, b_layer, w, a_cfg, b_cfg = build_ab_from_config(model_cfg, 2, k_ab, dtype)
    if b_layer is None:
        raise ValueError("this experiment needs the lift: set model.B.enabled = true")
    if b_cfg.preserve_first_channel:
        raise ValueError("the lift must be dense: set model.B._preserve_first_channel = false")
    ews = EWS(
        A=a_param,
        N=int(model_cfg.get("depth", 1)),
        M=int(comp.get("M", 20)),
        dtype=dtype,
        reweighting_mode=comp.get("reweighting_mode", "displacement"),
        terminal_reduce=comp.get("terminal_reduce", "tree"),
        van_loan_expm=comp.get("van_loan_expm", "cayley"),
        chen_expm=comp.get("chen_expm", "exact"),
        leaf=comp.get("leaf", "van_loan"),
        series_q=comp.get("series_q"),
    )
    head = eqx.nn.Linear(ews.signature_dim, n_out, key=k_head)
    if readout_init == "zero":
        head = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x) if eqx.is_array(x) else x, head)
    elif readout_init != "default":
        raise ValueError("readout_init must be 'zero' or 'default'")
    head = jax.tree_util.tree_map(lambda x: x.astype(dtype) if eqx.is_array(x) else x, head)
    return OscModel(lift=b_layer, ews=ews, head=head, dtype=dtype)


# ── the init override ─────────────────────────────────────────────────────────


def _softplus_inv(y):
    return jnp.log(jnp.expm1(y))


def _log_uniform(key, shape, low, high, dtype):
    u = jr.uniform(key, shape, dtype=dtype)
    return jnp.exp(jnp.log(low) + u * (jnp.log(high) - jnp.log(low)))


def _draw_planes(key, P: int, init_cfg: dict, dtype):
    """One (damping, omega) draw per plane: damping log-uniform over `rate_range`; omega over
    `freq_range_hz` (Hz, times 2 pi), log-uniform (`freq_dist` = "log_uniform", default) or
    uniform in Hz ("uniform"); sorted by frequency."""
    lo, hi = (float(v) for v in init_cfg["rate_range"])
    f_lo, f_hi = (float(v) for v in init_cfg["freq_range_hz"])
    dist = init_cfg.get("freq_dist", "log_uniform")
    k_a, k_w = jr.split(key)
    damp = _log_uniform(k_a, (P,), lo, hi, dtype)
    if dist == "log_uniform":
        omega = _log_uniform(k_w, (P,), 2 * np.pi * f_lo, 2 * np.pi * f_hi, dtype)
    elif dist == "uniform":
        omega = 2 * np.pi * jr.uniform(k_w, (P,), dtype=dtype, minval=f_lo, maxval=f_hi)
    else:
        raise ValueError("freq_dist must be 'log_uniform' or 'uniform'")
    return damp, jnp.sort(omega)


def override_a_init(model: OscModel, init_cfg: dict, key) -> OscModel:
    """Replace the engine's A init by the experiment's: every rate (EFM lambda, EWS damping) is
    log-uniform over `rate_range`, every EWS frequency is drawn over `freq_range_hz` (in Hz, then
    times 2 pi; log-uniform or uniform per `freq_dist`) and sorted. `sig` has no A parameters. Raw
    parameters are set by inverting the parametrisation: theta = softplus^-1(lambda),
    v = softplus^-1(a), u = logit(omega / omega_scale). A dense `general` A (the control) starts at
    the block-diagonal matrix of the same (damping, omega) draw and is then free to leave it."""
    a = model.ews.A_param
    dtype = model.dtype
    lo, hi = (float(v) for v in init_cfg["rate_range"])
    if isinstance(a, AParamSig):
        return model
    if isinstance(a, AParamEFM):
        if a.positivity != "softplus":
            raise ValueError("the init override assumes efm positivity 'softplus'")
        lam = _log_uniform(key, (a.dim,), lo, hi, dtype)
        return eqx.tree_at(lambda m: m.ews.A_param.theta, model, _softplus_inv(lam))
    if isinstance(a, AParamDampedRotation):
        P = a.v.shape[0]
        damp, omega = _draw_planes(key, P, init_cfg, dtype)
        if float(omega.max()) >= a.omega_scale:
            raise ValueError("freq_range_hz exceeds omega_scale / 2 pi")
        p = omega / a.omega_scale
        u = jnp.log(p) - jnp.log1p(-p)
        model = eqx.tree_at(lambda m: m.ews.A_param.v, model, _softplus_inv(damp))
        return eqx.tree_at(lambda m: m.ews.A_param.u, model, u)
    if isinstance(a, AParamGeneral):
        if a.dim % 2 != 0:
            raise ValueError("the dense control needs an even width (block-rotation init)")
        P = a.dim // 2
        damp, omega = _draw_planes(key, P, init_cfg, dtype)
        A = jnp.zeros((a.dim, a.dim), dtype=dtype)
        idx = jnp.arange(P)
        A = A.at[2 * idx, 2 * idx].set(damp).at[2 * idx + 1, 2 * idx + 1].set(damp)
        A = A.at[2 * idx, 2 * idx + 1].set(-omega).at[2 * idx + 1, 2 * idx].set(omega)
        return eqx.tree_at(lambda m: m.ews.A_param.matrix, model, A)
    raise ValueError(f"no init override for {type(a).__name__}")


def a_params_summary(model: OscModel) -> Dict:
    """The generator's interpretable parameters, read directly from the parametrisation."""
    a = model.ews.A_param
    if isinstance(a, AParamSig):
        return {"kind": "sig"}
    if isinstance(a, AParamEFM):
        return {
            "kind": "efm",
            "lambda": np.asarray(jax.nn.softplus(a.theta), dtype=np.float64).tolist(),
        }
    if isinstance(a, AParamDampedRotation):
        omega = np.asarray(a.sigmoid_omega(), dtype=np.float64)
        return {
            "kind": "damped_rotation",
            "damping": np.asarray(a.damping(), dtype=np.float64).tolist(),
            "omega": omega.tolist(),
            "freq_hz": (omega / (2 * np.pi)).tolist(),
        }
    if isinstance(a, AParamGeneral):
        # no eigendecomposition: the norm, and how far the matrix moved off the block-diagonal
        # structure it was initialised at (fraction of the Frobenius norm outside the 2 x 2 blocks)
        A = np.asarray(a.matrix, dtype=np.float64)
        P = A.shape[0] // 2
        blocks = np.zeros_like(A)
        for j in range(P):
            blocks[2 * j : 2 * j + 2, 2 * j : 2 * j + 2] = A[2 * j : 2 * j + 2, 2 * j : 2 * j + 2]
        fro = float(np.linalg.norm(A))
        return {
            "kind": "general",
            "fro_norm": fro,
            "offblock_fraction": float(np.linalg.norm(A - blocks) / max(fro, 1e-12)),
            "sym_part_min_diag": float(np.min(np.diag((A + A.T) / 2))),
            "matrix": A.tolist(),
        }
    return {"kind": type(a).__name__}


def model_kind(model_cfg: dict) -> str:
    return MODEL_OF_PARAM[parse_a_config(model_cfg).parametrization]


def series_regime_x(model: OscModel, dtheta: float) -> float:
    """x = ||A||_2 * dtheta, the series leaf's convergence parameter (a 2-norm, no eigendecomposition)."""
    A = np.asarray(model.ews.A_param(), dtype=np.float64)
    if A.size == 0:
        return 0.0
    return float(np.linalg.norm(A, 2) * dtheta)


# ── planted solutions (sanity checks) ─────────────────────────────────────────


def bank_generator(dtype) -> jnp.ndarray:
    """Block-diagonal A with one damped-rotation block [[a, -omega], [omega, a]] per bank frequency."""
    J = len(BANK_HZ)
    A = np.zeros((2 * J, 2 * J), dtype=np.float64)
    for j, f in enumerate(BANK_HZ):
        w = 2 * np.pi * f
        A[2 * j : 2 * j + 2, 2 * j : 2 * j + 2] = [[DAMPING, -w], [w, DAMPING]]
    return jnp.asarray(A, dtype=dtype)


def planted_model(depth: int, compute: dict, dtype) -> OscModel:
    """EWS at m = 2J with A = the bank, B sending the signal channel to the first coordinate of
    every plane and the time channel to zero, and the readout that reproduces T1 (depth 1) or T2
    (depth 2).

    Under exp(-s A_j) the plane state is (Re z_j, -Im z_j), so the depth-1 readout is diag(1, -1)
    per plane. At depth 2 the energy c^2 + s^2 = 2 (S^(cc) + S^(ss)) by the shuffle identity, read
    on the level-2 diagonal coordinates of the high planes with weight 2."""
    J = len(BANK_HZ)
    m = 2 * J
    Bm = np.zeros((m, 2), dtype=np.float64)
    Bm[0::2, 1] = 1.0
    lift = B(jnp.asarray(Bm, dtype=dtype), preserve_first_channel=False)
    ews = EWS(
        A=bank_generator(dtype),
        N=depth,
        M=int(compute.get("M", 20)),
        dtype=dtype,
        reweighting_mode=compute.get("reweighting_mode", "displacement"),
        terminal_reduce=compute.get("terminal_reduce", "tree"),
        van_loan_expm=compute.get("van_loan_expm", "cayley"),
        chen_expm=compute.get("chen_expm", "exact"),
        leaf=compute.get("leaf", "van_loan"),
        series_q=compute.get("series_q"),
    )
    D = feature_count(m, depth)
    if depth == 1:
        W = np.zeros((m, D))
        for i in range(m):
            W[i, i] = 1.0 if i % 2 == 0 else -1.0
    elif depth == 2:
        W = np.zeros((1, D))
        for j in HIGH_PLANES:
            for c in (2 * j, 2 * j + 1):
                W[0, m + c * m + c] = 2.0  # level-2 flat index of (c, c), row-major after level 1
    else:
        raise ValueError("planted models exist at depth 1 and 2 only")
    head = eqx.nn.Linear(D, W.shape[0], key=jr.PRNGKey(0))
    head = eqx.tree_at(lambda h: h.weight, head, jnp.asarray(W, dtype=dtype))
    head = eqx.tree_at(lambda h: h.bias, head, jnp.zeros((W.shape[0],), dtype=dtype))
    return OscModel(lift=lift, ews=ews, head=head, dtype=dtype)


def normalised_mse(pred: np.ndarray, target: np.ndarray) -> float:
    """MSE / Var(target) averaged over outputs, variances taken on `target` itself."""
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, target.shape[-1])
    target = np.asarray(target, dtype=np.float64).reshape(-1, target.shape[-1])
    mse = ((pred - target) ** 2).mean(axis=0)
    var = target.var(axis=0)
    return float((mse / var).mean())


def predict_in_batches(model: OscModel, paths, clocks, batch: int = 250) -> np.ndarray:
    fn = eqx.filter_jit(lambda m, p, c: m(p, c))
    out = []
    for i in range(0, paths.shape[0], batch):
        out.append(
            np.asarray(
                fn(model, jnp.asarray(paths[i : i + batch]), jnp.asarray(clocks[i : i + batch]))
            )
        )
    return np.concatenate(out, axis=0)


def default_compute(q: int = 12) -> dict:
    return {
        "leaf": "series",
        "series_q": int(q),
        "chen_expm": "exact",
        "reweighting_mode": "displacement_uniform",
        "terminal_reduce": "assoc",
    }


def van_loan_compute(M: int = 8) -> dict:
    return {
        "leaf": "van_loan",
        "van_loan_expm": "exact",
        "chen_expm": "exact",
        "M": int(M),
        "reweighting_mode": "displacement_uniform",
        "terminal_reduce": "assoc",
    }
