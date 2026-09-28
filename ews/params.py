"""Parametrisations of the drift A and the lift B, and their initialisation.

`A` families are the classes below (see docs/params.md for the table); `B` is the linear lift applied
before the signature. `build_ab_from_config` builds both from their config sections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


@dataclass
class InitConfig:
    type: str
    mean: float = 0.0
    variance: float = 1.0
    sigma_range: float = 0.0


@dataclass
class AConfig:
    parametrization: str
    init: InitConfig
    learn: bool = True
    blocks: Optional[List[int]] = None
    kinds: Optional[List[str]] = None
    epsilon: float = 0.0
    # rotation families: omega = omega_scale * sigmoid(u), so the realised per-token angle is
    # capped at omega_scale * 2*pi instead of 2*pi. Raising it widens the reachable frequency
    # range; it does not make any particular angle attractive.
    omega_scale: float = 1.0
    # damped_rotation only: init of the per-plane damping, a_j = softplus(v_j), v ~ N(mean, std^2).
    # mean -2 puts most planes near a ~ 0 (rotators) with a positive tail that can grow into
    # dampers; it must not force every plane to damp, because the `acc` planes need a ~ 0.
    damp_init_mean: float = -2.0
    damp_init_std: float = 1.0
    # stable_block only: init of the per-plane shear fraction, s_j = a_j * sigmoid(z_j), z ~ N(mean, std^2).
    # 0 / 1 puts the shear anywhere in (0, a_j) at init; a very negative mean recovers damped_rotation.
    shear_init_mean: float = 0.0
    shear_init_std: float = 1.0
    # efm only: how the diagonal is kept positive, softplus(theta) or theta**2.
    positivity: str = "softplus"


@dataclass
class BConfig:
    enabled: bool = False
    out_dim: Optional[int] = None
    preserve_first_channel: bool = True
    parametrization: str = "general"
    init: InitConfig = field(default_factory=lambda: InitConfig(type="general"))
    learn: bool = True


A_PARAM_NAMES = {
    "sig",
    "efm",
    "general",
    "skew_symmetric",
    "symmetric",
    "positive_definite",
    "definite_plus_skew",
    "triangular_upper",
    "triangular_lower",
    "block_diagonal",
    "unconstrained_rotation",
    "jordan_repeated",
    "bounded_rotation",
    "unbounded_rotation",
    "clamped_rotation",
    "damped_rotation",
    "relu_damped_rotation",
    "stable_block",
}


B_PARAM_NAMES = {"general"}

EFM_POSITIVITY = ("softplus", "square")


B_INIT_NAMES = {"general", "identity_block", "sig"}


def parse_init_config(cfg: Optional[dict], default_type: str) -> InitConfig:
    if cfg is None:
        variance = 0.0 if default_type == "sig" else 1.0
        return InitConfig(type=default_type, mean=0.0, variance=variance)
    init_type = str(cfg.get("type", default_type))
    variance = cfg.get("variance", 0.0 if init_type == "sig" else 1.0)
    return InitConfig(
        type=init_type,
        mean=float(cfg.get("mean", 0.0)),
        variance=float(variance),
        sigma_range=float(cfg.get("sigma_range", 0.0)),
    )


def parse_a_config(model_cfg: dict) -> AConfig:
    a_cfg = model_cfg.get("A", {})
    return AConfig(
        parametrization=str(a_cfg.get("parametrization", "general")),
        init=parse_init_config(a_cfg.get("init", None), default_type="sig"),
        learn=bool(a_cfg.get("learn", True)),
        blocks=list(a_cfg.get("blocks", [])) or None,
        kinds=list(a_cfg.get("kinds", [])) or None,
        epsilon=float(a_cfg.get("epsilon", 0.0)),
        omega_scale=float(a_cfg.get("omega_scale", 1.0)),
        damp_init_mean=float(a_cfg.get("damp_init_mean", -2.0)),
        damp_init_std=float(a_cfg.get("damp_init_std", 1.0)),
        shear_init_mean=float(a_cfg.get("shear_init_mean", 0.0)),
        shear_init_std=float(a_cfg.get("shear_init_std", 1.0)),
        positivity=str(a_cfg.get("positivity", "softplus")),
    )


def parse_b_config(model_cfg: dict) -> BConfig:
    b_cfg = model_cfg.get("B", {})
    return BConfig(
        enabled=bool(b_cfg.get("enabled", False)),
        out_dim=b_cfg.get("out_dim", None),
        preserve_first_channel=bool(
            b_cfg.get("_preserve_first_channel", b_cfg.get("_time_handling", True))
        ),
        parametrization=str(b_cfg.get("parametrization", "general")),
        init=parse_init_config(b_cfg.get("init", None), default_type="general"),
        learn=bool(b_cfg.get("learn", True)),
    )


def _std_from_variance(variance: float) -> float:
    return float(jnp.sqrt(jnp.asarray(variance)))


def _gaussian(
    key: jax.random.PRNGKey,
    shape: Sequence[int],
    mean: float,
    variance: float,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    std = _std_from_variance(variance)
    return mean + std * jr.normal(key, shape, dtype=dtype)


def _upper_triangular_from_vec(vec: jnp.ndarray, n: int, include_diag: bool) -> jnp.ndarray:
    out = jnp.zeros((n, n), dtype=vec.dtype)
    k = 0 if include_diag else 1
    idx = jnp.triu_indices(n, k=k)
    return out.at[idx].set(vec)


def _lower_triangular_from_vec(vec: jnp.ndarray, n: int, include_diag: bool) -> jnp.ndarray:
    out = jnp.zeros((n, n), dtype=vec.dtype)
    k = 0 if include_diag else -1
    idx = jnp.tril_indices(n, k=k)
    return out.at[idx].set(vec)


def _build_block_diag(blocks: Sequence[jnp.ndarray], dtype: jnp.dtype) -> jnp.ndarray:
    total = sum(int(b.shape[0]) for b in blocks)
    out = jnp.zeros((total, total), dtype=dtype)
    offset = 0
    for block in blocks:
        size = int(block.shape[0])
        out = out.at[offset : offset + size, offset : offset + size].set(block)
        offset += size
    return out


def _build_real_jordan_repeated(v: jnp.ndarray, size: int) -> jnp.ndarray:
    """A true real Jordan block J_n(lambda) with lambda = softplus(v) >= 0: one eigenvalue on the whole diagonal,
    ones directly above it. The bound is structural, not a safety net: the theory needs Re(spec(A)) >= 0."""
    n = int(size)
    lam = jax.nn.softplus(v)
    out = jnp.zeros((n, n), dtype=v.dtype)
    idx = jnp.arange(n)
    out = out.at[idx, idx].set(jnp.broadcast_to(lam, (n,)))
    if n > 1:
        out = out.at[jnp.arange(n - 1), jnp.arange(1, n)].set(1.0)
    return out


def _build_complex_jordan_repeated(v: jnp.ndarray, b: jnp.ndarray, size: int) -> jnp.ndarray:
    """The real Jordan form of a repeated complex pair, with the real part bounded: R = [[a, -b], [b, a]] with
    a = softplus(v) >= 0 in every diagonal slot, 2x2 identity above. b (the frequency) is free: a rotation is
    norm-preserving, so only the real part decides stability."""
    if int(size) % 2 != 0:
        raise ValueError(f"Complex Jordan block size must be even, got {size}")
    k = int(size) // 2
    a = jax.nn.softplus(v)
    out = jnp.zeros((2 * k, 2 * k), dtype=v.dtype)
    R = jnp.stack([jnp.stack([a, -b]), jnp.stack([b, a])])
    for i in range(k):
        out = out.at[2 * i : 2 * i + 2, 2 * i : 2 * i + 2].set(R)
        if i < k - 1:
            out = out.at[2 * i : 2 * i + 2, 2 * i + 2 : 2 * i + 4].set(jnp.eye(2, dtype=a.dtype))
    return out


def _build_rotation_planes(params: jnp.ndarray) -> jnp.ndarray:
    k = int(params.shape[0])
    out = jnp.zeros((2 * k, 2 * k), dtype=params.dtype)
    for i in range(k):
        a = params[i, 0]
        b = params[i, 1]
        block = jnp.array([[a, -b], [b, a]], dtype=params.dtype)
        out = out.at[2 * i : 2 * i + 2, 2 * i : 2 * i + 2].set(block)
        if i < k - 1:
            out = out.at[2 * i : 2 * i + 2, 2 * i + 2 : 2 * i + 4].set(
                jnp.eye(2, dtype=params.dtype)
            )
    return out


def _identity_block(rows: int, cols: int, dtype: jnp.dtype) -> jnp.ndarray:
    out = jnp.zeros((rows, cols), dtype=dtype)
    diag = min(rows, cols)
    return out.at[jnp.arange(diag), jnp.arange(diag)].set(1.0)


def _require_blocks(cfg: AConfig, require_kinds: bool) -> None:
    if not cfg.blocks:
        raise ValueError("blocks must be provided for this parametrization")
    if require_kinds and not cfg.kinds:
        raise ValueError("kinds must be provided for jordan parametrization")
    if cfg.kinds is not None and cfg.blocks is not None:
        if len(cfg.kinds) != len(cfg.blocks):
            raise ValueError("blocks and kinds must have the same length")


def _allowed_a_inits(param_name: str) -> Iterable[str]:
    if param_name == "general":
        return A_PARAM_NAMES
    return {param_name, "sig"}


class AParamConstant(eqx.Module):
    matrix: jnp.ndarray
    dim: int = eqx.field(static=True)

    def __init__(self, matrix: jnp.ndarray):
        self.matrix = matrix
        self.dim = int(matrix.shape[0])

    def __call__(self) -> jnp.ndarray:
        return self.matrix


class AParamSig(eqx.Module):
    dim: int = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(self, dim: int, dtype: jnp.dtype):
        self.dim = int(dim)
        self.dtype = dtype

    def __call__(self) -> jnp.ndarray:
        return jnp.zeros((self.dim, self.dim), dtype=self.dtype)


class AParamGeneral(eqx.Module):
    matrix: jnp.ndarray
    dim: int = eqx.field(static=True)

    def __init__(self, matrix: jnp.ndarray):
        self.matrix = matrix
        self.dim = int(matrix.shape[0])

    def __call__(self) -> jnp.ndarray:
        return self.matrix


class AParamEFM(eqx.Module):
    """Positive diagonal: softplus(theta) or theta**2 per channel."""

    theta: jnp.ndarray
    dim: int = eqx.field(static=True)
    positivity: str = eqx.field(static=True, default="softplus")

    def __init__(self, theta: jnp.ndarray, positivity: str = "softplus"):
        if positivity not in EFM_POSITIVITY:
            raise ValueError(f"efm positivity must be one of {EFM_POSITIVITY}, got {positivity!r}")
        self.theta = theta
        self.dim = int(theta.shape[0])
        self.positivity = positivity

    def __call__(self) -> jnp.ndarray:
        return jnp.diag(_positive_diagonal(self.theta, self.positivity))


def _positive_diagonal(theta: jnp.ndarray, positivity: str) -> jnp.ndarray:
    if positivity == "square":
        return theta**2
    return jax.nn.softplus(theta)


class AParamSkewSym(eqx.Module):
    upper: jnp.ndarray
    dim: int = eqx.field(static=True)

    def __init__(self, upper: jnp.ndarray, dim: int):
        self.upper = upper
        self.dim = int(dim)

    def __call__(self) -> jnp.ndarray:
        upper_mat = _upper_triangular_from_vec(self.upper, self.dim, include_diag=False)
        return upper_mat - upper_mat.T


class AParamSym(eqx.Module):
    upper: jnp.ndarray
    dim: int = eqx.field(static=True)

    def __init__(self, upper: jnp.ndarray, dim: int):
        self.upper = upper
        self.dim = int(dim)

    def __call__(self) -> jnp.ndarray:
        upper_mat = _upper_triangular_from_vec(self.upper, self.dim, include_diag=True)
        return upper_mat + upper_mat.T - jnp.diag(jnp.diag(upper_mat))


class AParamTriUpper(eqx.Module):
    upper: jnp.ndarray
    dim: int = eqx.field(static=True)

    def __init__(self, upper: jnp.ndarray, dim: int):
        self.upper = upper
        self.dim = int(dim)

    def __call__(self) -> jnp.ndarray:
        return _upper_triangular_from_vec(self.upper, self.dim, include_diag=True)


class AParamTriLower(eqx.Module):
    lower: jnp.ndarray
    dim: int = eqx.field(static=True)

    def __init__(self, lower: jnp.ndarray, dim: int):
        self.lower = lower
        self.dim = int(dim)

    def __call__(self) -> jnp.ndarray:
        return _lower_triangular_from_vec(self.lower, self.dim, include_diag=True)


class AParamPositiveDefinite(eqx.Module):
    lower: jnp.ndarray
    dim: int = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)

    def __init__(self, lower: jnp.ndarray, dim: int, epsilon: float):
        self.lower = lower
        self.dim = int(dim)
        self.epsilon = float(epsilon)

    def __call__(self) -> jnp.ndarray:
        L = _lower_triangular_from_vec(self.lower, self.dim, include_diag=True)
        A = L @ L.T
        if self.epsilon > 0.0:
            A = A + self.epsilon * jnp.eye(self.dim, dtype=A.dtype)
        return A


class AParamDefinitePlusSkew(eqx.Module):
    lower: jnp.ndarray
    upper: jnp.ndarray
    dim: int = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)

    def __init__(self, lower: jnp.ndarray, upper: jnp.ndarray, dim: int, epsilon: float):
        self.lower = lower
        self.upper = upper
        self.dim = int(dim)
        self.epsilon = float(epsilon)

    def __call__(self) -> jnp.ndarray:
        L = _lower_triangular_from_vec(self.lower, self.dim, include_diag=True)
        pd = L @ L.T
        if self.epsilon > 0.0:
            pd = pd + self.epsilon * jnp.eye(self.dim, dtype=pd.dtype)
        upper_mat = _upper_triangular_from_vec(self.upper, self.dim, include_diag=False)
        skew = upper_mat - upper_mat.T
        return pd + skew


class AParamBlockDiag(eqx.Module):
    blocks: Tuple[jnp.ndarray, ...]
    sizes: Tuple[int, ...] = eqx.field(static=True)

    def __init__(self, blocks: Sequence[jnp.ndarray], sizes: Sequence[int]):
        self.blocks = tuple(blocks)
        self.sizes = tuple(int(x) for x in sizes)

    def __call__(self) -> jnp.ndarray:
        if not self.blocks:
            return jnp.zeros((0, 0), dtype=jnp.float32)
        return _build_block_diag(self.blocks, dtype=self.blocks[0].dtype)

    @property
    def dim(self) -> int:
        return int(sum(self.sizes))


class AParamJordanRepeated(eqx.Module):
    """Block-diagonal A whose blocks are Jordan blocks: one eigenvalue per block, repeated down its diagonal.

        real kind, size n:      lambda on the diagonal, 1 directly above
        complex kind, size 2k:  R = [[a, -b], [b, a]] in every diagonal slot, the 2x2 identity above

    One parameter (real) or two (complex) per block, whatever its size. A defective block's response is exp(-t lambda)
    times a polynomial in t, so one coordinate accumulates another's output over time, which no diagonal or rotation state
    space can do. Different eigenvalues come from different blocks (blocks=[2, 2] is J_2(l1) + J_2(l2)). There is no
    stability constraint; `stable_block` is the damped family.
    """

    block_sizes: Tuple[int, ...] = eqx.field(static=True)
    block_kinds: Tuple[str, ...] = eqx.field(static=True)
    block_params: Tuple[jnp.ndarray, ...]

    def __init__(self, block_sizes, block_kinds, block_params):
        self.block_sizes = tuple(int(x) for x in block_sizes)
        self.block_kinds = tuple(str(x) for x in block_kinds)
        self.block_params = tuple(block_params)
        for size, kind in zip(self.block_sizes, self.block_kinds):
            if kind not in ("real", "complex"):
                raise ValueError(f"Unknown Jordan block kind '{kind}'")
            if kind == "real" and size < 2:
                raise ValueError(
                    "a real Jordan block needs size >= 2 for the eigenvalue to repeat; "
                    "size 1 is a plain diagonal entry"
                )
            if kind == "complex" and (size % 2 != 0 or size < 4):
                raise ValueError(
                    f"a complex Jordan block needs an even size >= 4, got {size}: a conjugate pair "
                    "already occupies a 2x2 real block, so size 2 is one pair with nothing to chain "
                    "to -- that is `unconstrained_rotation`, not a Jordan block"
                )

    def __call__(self) -> jnp.ndarray:
        blocks: List[jnp.ndarray] = []
        for size, kind, p in zip(self.block_sizes, self.block_kinds, self.block_params):
            if kind == "real":
                blocks.append(_build_real_jordan_repeated(p[0], size))
            else:
                blocks.append(_build_complex_jordan_repeated(p[0], p[1], size))
        if not blocks:
            return jnp.zeros((0, 0), dtype=jnp.float32)
        return _build_block_diag(blocks, dtype=blocks[0].dtype)

    @property
    def dim(self) -> int:
        return int(sum(self.block_sizes))


class AParamUnconstrainedRotation(eqx.Module):
    """Block-diagonal 2x2 rotation planes [[a, -b], [b, a]] with both the scale and the frequency free and unsigned.

    The one rotation family with no stability guarantee: `a` takes either sign, so roughly half the planes expand
    rather than decay. That is survivable on a short path (Heartbeat, 405 tokens, init variance 0.004, where the
    selected model uses it) and fatal on a long one: on MotorImagery's 3,000 tokens it overflows.

    Blocks are fixed at 2x2 complex, and anything else is rejected; `jordan_repeated` is the Jordan family.
    """

    block_sizes: Tuple[int, ...] = eqx.field(static=True)
    block_kinds: Tuple[str, ...] = eqx.field(static=True)
    block_params: Tuple[jnp.ndarray, ...]

    def __init__(self, block_sizes, block_kinds, block_params):
        self.block_sizes = tuple(int(x) for x in block_sizes)
        self.block_kinds = tuple(str(x) for x in block_kinds)
        self.block_params = tuple(block_params)
        if any(s != 2 for s in self.block_sizes) or any(k != "complex" for k in self.block_kinds):
            raise ValueError(
                "unconstrained_rotation is 2x2 complex rotation planes: every block must be size 2 and "
                f"kind 'complex', got sizes {self.block_sizes} kinds {self.block_kinds}"
            )

    def __call__(self) -> jnp.ndarray:
        blocks = [_build_rotation_planes(p) for p in self.block_params]
        if not blocks:
            return jnp.zeros((0, 0), dtype=jnp.float32)
        return _build_block_diag(blocks, dtype=blocks[0].dtype)

    @property
    def dim(self) -> int:
        return int(sum(self.block_sizes))


class AParamStableBlock(eqx.Module):
    """The free 2x2 block, restricted to the region that cannot expand on a long path.

        A_j = a_j I + b_j J + s_j S(t_j)       J = [[0, -1], [1, 0]]   S(t) = [[cos t, sin t], [sin t, -cos t]]
        a_j = softplus(v_j) >= 0               b_j = omega_scale * sigmoid(u_j)       s_j = a_j * sigmoid(z_j) < a_j

    Four parameters per plane, as many as a free real 2x2 block, but the symmetric part a_j I + s_j S(t_j) has
    eigenvalues a_j +- s_j >= 0, so exp(-delta A) expands no direction at any path length. A free Gaussian 2x2 block
    expands with probability about 1/2, which overflows float32 over MotorImagery's 3,000 steps. z -> -inf gives
    damped_rotation, and v -> -inf as well gives bounded_rotation.
    """

    v: jnp.ndarray
    u: jnp.ndarray
    z: jnp.ndarray
    t: jnp.ndarray
    dim: int = eqx.field(static=True)
    omega_scale: float = eqx.field(static=True, default=1.0)

    def __init__(self, v, u, z, t, dim: int, omega_scale: float = 1.0):
        if int(dim) % 2 != 0:
            raise ValueError("stable_block requires even A width (2 rows per plane)")
        self.v, self.u, self.z, self.t = v, u, z, t
        self.dim = int(dim)
        self.omega_scale = float(omega_scale)

    def damping(self) -> jnp.ndarray:
        """Per-plane contraction rate a_j = softplus(v_j) >= 0 (the damped_rotation interface)."""
        return jax.nn.softplus(self.v)

    def sigmoid_omega(self) -> jnp.ndarray:
        """Per-plane frequency b_j (the damped_rotation interface)."""
        return self.omega_scale * jax.nn.sigmoid(self.u)

    def shear(self) -> jnp.ndarray:
        """Per-plane shear magnitude s_j = a_j * sigmoid(z_j), strictly less than the damping."""
        return self.damping() * jax.nn.sigmoid(self.z)

    def __call__(self) -> jnp.ndarray:
        a = self.damping()
        b = self.sigmoid_omega()
        s = self.shear()
        c, sn = jnp.cos(self.t), jnp.sin(self.t)
        P = self.v.shape[0]
        A = jnp.zeros((2 * P, 2 * P), dtype=self.v.dtype)
        idx = jnp.arange(P)
        A = A.at[2 * idx, 2 * idx].set(a + s * c)
        A = A.at[2 * idx + 1, 2 * idx + 1].set(a - s * c)
        A = A.at[2 * idx, 2 * idx + 1].set(-b + s * sn)
        A = A.at[2 * idx + 1, 2 * idx].set(b + s * sn)
        return A


class AParamBoundedRotation(eqx.Module):
    """Axis-aligned block-diagonal skew rotation with sigmoid-bounded frequencies.

    Stores raw per-plane parameters u_j; __call__ returns the unit skew generator with
    per-plane frequency sigmoid(u_j) in (0, 1). The enclosing block rescales the whole
    matrix by 2*pi / dtheta_max at forward time, so the realised per-token angle lands in
    (0, 2*pi): one period, no pi-vs-3pi duplicate-solution basins. u_j = 0 -> sigmoid = 1/2
    -> the maximal-dtheta token rotates by exactly pi.
    """

    u: jnp.ndarray
    dim: int = eqx.field(static=True)
    omega_scale: float = eqx.field(static=True, default=1.0)

    def __init__(self, u: jnp.ndarray, dim: int, omega_scale: float = 1.0):
        if int(dim) % 2 != 0:
            raise ValueError("bounded_rotation requires even A width (2 rows per plane)")
        self.u = u
        self.dim = int(dim)
        self.omega_scale = float(omega_scale)

    def sigmoid_omega(self) -> jnp.ndarray:
        """Per-plane frequency in (0, omega_scale); the block scales by 2*pi/dtheta_max."""
        return self.omega_scale * jax.nn.sigmoid(self.u)

    def __call__(self) -> jnp.ndarray:
        omega = self.omega_scale * jax.nn.sigmoid(self.u)  # (P,) in (0, omega_scale)
        P = self.u.shape[0]
        A = jnp.zeros((2 * P, 2 * P), dtype=self.u.dtype)
        idx = jnp.arange(P)
        A = A.at[2 * idx, 2 * idx + 1].set(-omega)
        A = A.at[2 * idx + 1, 2 * idx].set(omega)
        return A


class AParamDampedRotation(eqx.Module):
    """Per-plane damping and rotation: A_j = [[a_j, -b_j], [b_j, a_j]], block-diagonal.

        a_j = softplus(v_j) >= 0        b_j = omega_scale * sigmoid(u_j)

    so exp(-delta A_j) = exp(-delta a_j) R(-delta b_j): a_j is the plane's contraction rate and b_j its frequency. The
    enclosing block rescales by 2 pi / dtheta_max, as for bounded_rotation. a_j = 0 gives bounded_rotation (the relation
    between LinOSS and D-LinOSS); per-plane damping lets one plane reset while another keeps its state, and a_j >= 0 means
    no direction can expand.
    """

    v: jnp.ndarray
    u: jnp.ndarray
    dim: int = eqx.field(static=True)
    omega_scale: float = eqx.field(static=True, default=1.0)

    def __init__(self, v: jnp.ndarray, u: jnp.ndarray, dim: int, omega_scale: float = 1.0):
        if int(dim) % 2 != 0:
            raise ValueError("damped_rotation requires even A width (2 rows per plane)")
        self.v = v
        self.u = u
        self.dim = int(dim)
        self.omega_scale = float(omega_scale)

    def damping(self) -> jnp.ndarray:
        """Per-plane contraction rate a_j = softplus(v_j) >= 0."""
        return jax.nn.softplus(self.v)

    def sigmoid_omega(self) -> jnp.ndarray:
        """Per-plane frequency b_j; same parametrisation as bounded_rotation."""
        return self.omega_scale * jax.nn.sigmoid(self.u)

    def __call__(self) -> jnp.ndarray:
        a = jax.nn.softplus(self.v)
        b = self.omega_scale * jax.nn.sigmoid(self.u)
        P = self.v.shape[0]
        A = jnp.zeros((2 * P, 2 * P), dtype=self.v.dtype)
        idx = jnp.arange(P)
        A = A.at[2 * idx, 2 * idx].set(a)
        A = A.at[2 * idx + 1, 2 * idx + 1].set(a)
        A = A.at[2 * idx, 2 * idx + 1].set(-b)
        A = A.at[2 * idx + 1, 2 * idx].set(b)
        return A


class AParamReluDampedRotation(eqx.Module):
    """damped_rotation with a = relu(v) instead of softplus(v).

        a_j = relu(v_j) >= 0            b_j = omega_scale * sigmoid(u_j)

    Under softplus a = 0 is unattainable (a ~ e^v for v << 0, with a vanishing gradient back
    towards it). Under relu, a = 0 is attained exactly on the open set v < 0, and weight decay's
    v -> 0 attractor coincides with a = 0 instead of with softplus(0) = 0.693. The trade-off is a
    zero gradient on a for v < 0: a plane whose damping has reached zero is only revived by weight
    decay or noise pushing v past 0.
    Init: the damped_rotation convention (u via sigma_range, v ~ N(damp_init_mean,
    damp_init_std^2)).
    """

    v: jnp.ndarray
    u: jnp.ndarray
    dim: int = eqx.field(static=True)
    omega_scale: float = eqx.field(static=True, default=1.0)

    def __init__(self, v: jnp.ndarray, u: jnp.ndarray, dim: int, omega_scale: float = 1.0):
        if int(dim) % 2 != 0:
            raise ValueError("relu_damped_rotation requires even A width (2 rows per plane)")
        self.v = v
        self.u = u
        self.dim = int(dim)
        self.omega_scale = float(omega_scale)

    def damping(self) -> jnp.ndarray:
        """Per-plane contraction rate a_j = relu(v_j) >= 0; a = 0 attained for v <= 0."""
        return jax.nn.relu(self.v)

    def sigmoid_omega(self) -> jnp.ndarray:
        """Per-plane frequency b_j; same parametrisation as damped_rotation."""
        return self.omega_scale * jax.nn.sigmoid(self.u)

    def __call__(self) -> jnp.ndarray:
        a = jax.nn.relu(self.v)
        b = self.omega_scale * jax.nn.sigmoid(self.u)
        P = self.v.shape[0]
        A = jnp.zeros((2 * P, 2 * P), dtype=self.v.dtype)
        idx = jnp.arange(P)
        A = A.at[2 * idx, 2 * idx].set(a)
        A = A.at[2 * idx + 1, 2 * idx + 1].set(a)
        A = A.at[2 * idx, 2 * idx + 1].set(-b)
        A = A.at[2 * idx + 1, 2 * idx].set(b)
        return A


class AParamClampedRotation(eqx.Module):
    """Damping and one closed period of rotation: a_j = softplus(v_j), b_j = omega_scale * clip(u_j, 0, 1).

    sigmoid only approaches a frozen plane (omega = 0) and a full turn (omega = 1); clip attains both on open sets of u, so
    a pure-integrator plane (omega = 0, a ~ 0) is an ordinary configuration. Init: u_j ~ U(0, 1) and
    v ~ N(damp_init_mean, damp_init_std^2). clip has no gradient outside [0, 1], so a plane pushed past either end stays
    there.
    """

    v: jnp.ndarray
    u: jnp.ndarray
    dim: int = eqx.field(static=True)
    omega_scale: float = eqx.field(static=True, default=1.0)

    def __init__(self, v: jnp.ndarray, u: jnp.ndarray, dim: int, omega_scale: float = 1.0):
        if int(dim) % 2 != 0:
            raise ValueError("clamped_rotation requires even A width (2 rows per plane)")
        self.v = v
        self.u = u
        self.dim = int(dim)
        self.omega_scale = float(omega_scale)

    def damping(self) -> jnp.ndarray:
        """Per-plane contraction rate a_j = softplus(v_j) >= 0 (same interface as damped_rotation)."""
        return jax.nn.softplus(self.v)

    def sigmoid_omega(self) -> jnp.ndarray:
        """Named for interface compatibility with the other rotation variants; clip, not sigmoid."""
        return self.omega_scale * jnp.clip(self.u, 0.0, 1.0)

    def __call__(self) -> jnp.ndarray:
        a = jax.nn.softplus(self.v)
        omega = self.omega_scale * jnp.clip(self.u, 0.0, 1.0)
        P = self.u.shape[0]
        A = jnp.zeros((2 * P, 2 * P), dtype=self.u.dtype)
        idx = jnp.arange(P)
        A = A.at[2 * idx, 2 * idx].set(a)
        A = A.at[2 * idx + 1, 2 * idx + 1].set(a)
        A = A.at[2 * idx, 2 * idx + 1].set(-omega)
        A = A.at[2 * idx + 1, 2 * idx].set(omega)
        return A


class AParamUnboundedRotation(eqx.Module):
    """bounded_rotation with the cap removed: omega_j = softplus(u_j), unbounded above.

    The ablation for "is the sigmoid doing anything beyond setting the range?". The skew arm
    (effectively unbounded) failed 0/10 on cycle_nav with the b2 automaton never forming and b1's
    token separation destroyed through the backward pass, but skew also changes the matrix
    structure (dense skew vs axis-aligned planes). This variant changes only the squashing
    function, so it isolates the cap from the structure.

    Init is chosen to reproduce bounded_rotation's omega exactly at step 0
    (u = softplus^-1(sigmoid(v)) for the same uniform draw v), so the arms start identical and
    differ only in what they can reach.
    """

    u: jnp.ndarray
    dim: int = eqx.field(static=True)

    def __init__(self, u: jnp.ndarray, dim: int):
        if int(dim) % 2 != 0:
            raise ValueError("unbounded_rotation requires even A width (2 rows per plane)")
        self.u = u
        self.dim = int(dim)

    def sigmoid_omega(self) -> jnp.ndarray:
        """Named for interface compatibility with the bounded variants; softplus, not sigmoid."""
        return jax.nn.softplus(self.u)

    def __call__(self) -> jnp.ndarray:
        omega = jax.nn.softplus(self.u)
        P = self.u.shape[0]
        A = jnp.zeros((2 * P, 2 * P), dtype=self.u.dtype)
        idx = jnp.arange(P)
        A = A.at[2 * idx, 2 * idx + 1].set(-omega)
        A = A.at[2 * idx + 1, 2 * idx].set(omega)
        return A


def _zeros_like_params(param_name: str, w: int, cfg: AConfig, dtype: jnp.dtype):
    if param_name == "sig":
        return None
    if param_name == "general":
        return jnp.zeros((w, w), dtype=dtype)
    if param_name == "efm":
        return jnp.zeros((w,), dtype=dtype)
    if param_name == "skew_symmetric":
        return jnp.zeros((w * (w - 1) // 2,), dtype=dtype)
    if param_name == "symmetric":
        return jnp.zeros((w * (w + 1) // 2,), dtype=dtype)
    if param_name in {"triangular_upper", "triangular_lower", "positive_definite"}:
        return jnp.zeros((w * (w + 1) // 2,), dtype=dtype)
    if param_name == "definite_plus_skew":
        lower = jnp.zeros((w * (w + 1) // 2,), dtype=dtype)
        upper = jnp.zeros((w * (w - 1) // 2,), dtype=dtype)
        return (lower, upper)
    if param_name == "block_diagonal":
        _require_blocks(cfg, require_kinds=False)
        return [jnp.zeros((b, b), dtype=dtype) for b in cfg.blocks]
    if param_name == "jordan_repeated":
        _require_blocks(cfg, require_kinds=True)
        return [jnp.zeros((1 if k == "real" else 2,), dtype=dtype) for k in (cfg.kinds or [])]
    if param_name == "unconstrained_rotation":
        _require_blocks(cfg, require_kinds=True)
        params: List[jnp.ndarray] = []
        for size, kind in zip(cfg.blocks or [], cfg.kinds or []):
            if kind == "real":
                params.append(jnp.zeros((size,), dtype=dtype))
            elif kind == "complex":
                if size % 2 != 0:
                    raise ValueError(f"Complex Jordan block size must be even, got {size}")
                params.append(jnp.zeros((size // 2, 2), dtype=dtype))
            else:
                raise ValueError(f"Unknown Jordan block kind '{kind}'")
        return params
    raise ValueError(f"Unknown parametrization '{param_name}'")


def _sample_params_for_param(
    param_name: str,
    w: int,
    cfg: AConfig,
    key: jax.random.PRNGKey,
    dtype: jnp.dtype,
    init_name: str,
):
    if init_name == "sig":
        return _zeros_like_params(param_name, w, cfg, dtype)
    mean = cfg.init.mean
    variance = cfg.init.variance
    if param_name == "general":
        return _gaussian(key, (w, w), mean, variance, dtype)
    if param_name == "efm":
        return _gaussian(key, (w,), mean, variance, dtype)
    if param_name == "skew_symmetric":
        return _gaussian(key, (w * (w - 1) // 2,), mean, variance, dtype)
    if param_name == "symmetric":
        return _gaussian(key, (w * (w + 1) // 2,), mean, variance, dtype)
    if param_name in {"triangular_upper", "triangular_lower", "positive_definite"}:
        return _gaussian(key, (w * (w + 1) // 2,), mean, variance, dtype)
    if param_name == "definite_plus_skew":
        key1, key2 = jr.split(key, 2)
        lower = _gaussian(key1, (w * (w + 1) // 2,), mean, variance, dtype)
        upper = _gaussian(key2, (w * (w - 1) // 2,), mean, variance, dtype)
        return (lower, upper)
    if param_name == "block_diagonal":
        _require_blocks(cfg, require_kinds=False)
        keys = jr.split(key, len(cfg.blocks))
        return [_gaussian(k, (b, b), mean, variance, dtype) for k, b in zip(keys, cfg.blocks)]
    if param_name == "jordan_repeated":
        _require_blocks(cfg, require_kinds=True)
        keys = jr.split(key, len(cfg.blocks))
        return [
            _gaussian(k, (1 if kind == "real" else 2,), mean, variance, dtype)
            for k, kind in zip(keys, cfg.kinds or [])
        ]
    if param_name == "unconstrained_rotation":
        _require_blocks(cfg, require_kinds=True)
        keys = jr.split(key, len(cfg.blocks))
        params: List[jnp.ndarray] = []
        for k, size, kind in zip(keys, cfg.blocks, cfg.kinds):
            if kind == "real":
                params.append(_gaussian(k, (size,), mean, variance, dtype))
            elif kind == "complex":
                if size % 2 != 0:
                    raise ValueError(f"Complex Jordan block size must be even, got {size}")
                params.append(_gaussian(k, (size // 2, 2), mean, variance, dtype))
            else:
                raise ValueError(f"Unknown Jordan block kind '{kind}'")
        return params
    if param_name == "stable_block":
        if w % 2 != 0:
            raise ValueError("stable_block init requires even A width (2 rows per plane)")
        P = w // 2
        k_u, k_v, k_z, k_t = jr.split(key, 4)
        sr = min(float(getattr(cfg.init, "sigma_range", 0.0) or 0.0), 0.499)
        if sr > 0.0:
            aa = float(jnp.log((0.5 + sr) / (0.5 - sr)))
            u = jr.uniform(k_u, (P,), dtype=dtype, minval=-aa, maxval=aa)
        else:
            u = jnp.zeros((P,), dtype=dtype)
        v = cfg.damp_init_mean + cfg.damp_init_std * jr.normal(k_v, (P,), dtype=dtype)
        z = cfg.shear_init_mean + cfg.shear_init_std * jr.normal(k_z, (P,), dtype=dtype)
        t = jr.uniform(k_t, (P,), dtype=dtype, minval=0.0, maxval=float(2 * jnp.pi))
        return (v, u, z, t)
    if param_name in {
        "bounded_rotation",
        "unbounded_rotation",
        "damped_rotation",
        "relu_damped_rotation",
        "clamped_rotation",
    }:
        # The rotation families as init sources: `general` (whose init may be any family per
        # _allowed_a_inits) can start from e.g. a damped_rotation draw with the structural
        # constraint removed. Conventions mirror build_a_param's own branches (same key splits,
        # same distributions, sigma_range/damp_init_* read the same way), so a cross-init at
        # key K reproduces the class's own draw at key K. mean/variance are ignored here,
        # exactly as the classes ignore them.
        if w % 2 != 0:
            raise ValueError(f"{param_name} init requires even A width (2 rows per plane)")
        P = w // 2
        sr = min(float(getattr(cfg.init, "sigma_range", 0.0) or 0.0), 0.499)
        if param_name == "bounded_rotation":
            if sr > 0.0:
                a = float(jnp.log((0.5 + sr) / (0.5 - sr)))
                return jr.uniform(key, (P,), dtype=dtype, minval=-a, maxval=a)
            return jnp.zeros((P,), dtype=dtype)
        if param_name == "unbounded_rotation":
            if sr > 0.0:
                a = float(jnp.log((0.5 + sr) / (0.5 - sr)))
                v = jr.uniform(key, (P,), dtype=dtype, minval=-a, maxval=a)
            else:
                v = jnp.zeros((P,), dtype=dtype)
            return jnp.log(jnp.expm1(jax.nn.sigmoid(v)))
        k_u, k_v = jr.split(key, 2)
        if param_name == "clamped_rotation":
            u = jr.uniform(k_u, (P,), dtype=dtype, minval=0.0, maxval=1.0)
        elif sr > 0.0:
            aa = float(jnp.log((0.5 + sr) / (0.5 - sr)))
            u = jr.uniform(k_u, (P,), dtype=dtype, minval=-aa, maxval=aa)
        else:
            u = jnp.zeros((P,), dtype=dtype)
        v = cfg.damp_init_mean + cfg.damp_init_std * jr.normal(k_v, (P,), dtype=dtype)
        return (v, u)
    raise ValueError(f"Unknown parametrization '{param_name}'")


def _materialize_from_params(
    param_name: str, params, w: int, cfg: AConfig, dtype: jnp.dtype
) -> jnp.ndarray:
    if param_name == "sig":
        return jnp.zeros((w, w), dtype=dtype)
    if param_name == "general":
        return params
    if param_name == "efm":
        return jnp.diag(_positive_diagonal(params, cfg.positivity))
    if param_name == "skew_symmetric":
        upper = _upper_triangular_from_vec(params, w, include_diag=False)
        return upper - upper.T
    if param_name == "symmetric":
        upper = _upper_triangular_from_vec(params, w, include_diag=True)
        return upper + upper.T - jnp.diag(jnp.diag(upper))
    if param_name == "triangular_upper":
        return _upper_triangular_from_vec(params, w, include_diag=True)
    if param_name == "triangular_lower":
        return _lower_triangular_from_vec(params, w, include_diag=True)
    if param_name == "positive_definite":
        L = _lower_triangular_from_vec(params, w, include_diag=True)
        A = L @ L.T
        if cfg.epsilon > 0.0:
            A = A + cfg.epsilon * jnp.eye(w, dtype=dtype)
        return A
    if param_name == "definite_plus_skew":
        lower, upper_vec = params
        L = _lower_triangular_from_vec(lower, w, include_diag=True)
        pd = L @ L.T
        if cfg.epsilon > 0.0:
            pd = pd + cfg.epsilon * jnp.eye(w, dtype=dtype)
        upper = _upper_triangular_from_vec(upper_vec, w, include_diag=False)
        skew = upper - upper.T
        return pd + skew
    if param_name == "block_diagonal":
        return _build_block_diag(params, dtype=dtype)
    if param_name == "jordan_repeated":
        jblocks: List[jnp.ndarray] = []
        for size, kind, p in zip(cfg.blocks or [], cfg.kinds or [], params):
            jblocks.append(
                _build_real_jordan_repeated(p[0], size)
                if kind == "real"
                else _build_complex_jordan_repeated(p[0], p[1], size)
            )
        return _build_block_diag(jblocks, dtype=dtype)
    if param_name == "unconstrained_rotation":
        blocks: List[jnp.ndarray] = []
        for size, kind, p in zip(cfg.blocks or [], cfg.kinds or [], params):
            blocks.append(_build_rotation_planes(p))
        return _build_block_diag(blocks, dtype=dtype)
    if param_name == "stable_block":
        v, u, z, t = params
        P = w // 2
        a = jax.nn.softplus(v)
        b = cfg.omega_scale * jax.nn.sigmoid(u)
        sh = a * jax.nn.sigmoid(z)
        c, sn = jnp.cos(t), jnp.sin(t)
        A = jnp.zeros((w, w), dtype=dtype)
        idx = jnp.arange(P)
        A = A.at[2 * idx, 2 * idx].set(a + sh * c)
        A = A.at[2 * idx + 1, 2 * idx + 1].set(a - sh * c)
        A = A.at[2 * idx, 2 * idx + 1].set(-b + sh * sn)
        A = A.at[2 * idx + 1, 2 * idx].set(b + sh * sn)
        return A
    if param_name in {
        "bounded_rotation",
        "unbounded_rotation",
        "damped_rotation",
        "relu_damped_rotation",
        "clamped_rotation",
    }:
        # uniform-init: realise the dense matrix these classes' __call__ would produce from
        # the raw params sampled above: [[a, -b], [b, a]] per plane, block-diagonal.
        P = w // 2
        if param_name == "bounded_rotation":
            a_diag = jnp.zeros((P,), dtype=dtype)
            b_off = cfg.omega_scale * jax.nn.sigmoid(params)
        elif param_name == "unbounded_rotation":
            a_diag = jnp.zeros((P,), dtype=dtype)
            b_off = jax.nn.softplus(params)
        else:
            v, u = params
            a_diag = jax.nn.relu(v) if param_name == "relu_damped_rotation" else jax.nn.softplus(v)
            b_off = (
                cfg.omega_scale * jnp.clip(u, 0.0, 1.0)
                if param_name == "clamped_rotation"
                else cfg.omega_scale * jax.nn.sigmoid(u)
            )
        A = jnp.zeros((w, w), dtype=dtype)
        idx = jnp.arange(P)
        A = A.at[2 * idx, 2 * idx].set(a_diag)
        A = A.at[2 * idx + 1, 2 * idx + 1].set(a_diag)
        A = A.at[2 * idx, 2 * idx + 1].set(-b_off)
        A = A.at[2 * idx + 1, 2 * idx].set(b_off)
        return A
    raise ValueError(f"Unknown parametrization '{param_name}'")


def _make_param_module(param_name: str, params, w: int, cfg: AConfig, dtype: jnp.dtype):
    if param_name == "sig":
        return AParamSig(w, dtype)
    if param_name == "general":
        return AParamGeneral(params)
    if param_name == "efm":
        return AParamEFM(params, cfg.positivity)
    if param_name == "skew_symmetric":
        return AParamSkewSym(params, w)
    if param_name == "symmetric":
        return AParamSym(params, w)
    if param_name == "triangular_upper":
        return AParamTriUpper(params, w)
    if param_name == "triangular_lower":
        return AParamTriLower(params, w)
    if param_name == "positive_definite":
        return AParamPositiveDefinite(params, w, cfg.epsilon)
    if param_name == "definite_plus_skew":
        lower, upper = params
        return AParamDefinitePlusSkew(lower, upper, w, cfg.epsilon)
    if param_name == "block_diagonal":
        return AParamBlockDiag(params, cfg.blocks or [])
    if param_name == "jordan_repeated":
        return AParamJordanRepeated(cfg.blocks or [], cfg.kinds or [], params)
    if param_name == "unconstrained_rotation":
        return AParamUnconstrainedRotation(cfg.blocks or [], cfg.kinds or [], params)
    if param_name == "bounded_rotation":
        return AParamBoundedRotation(params, w, cfg.omega_scale)
    if param_name == "stable_block":
        v, u, z, t = params
        return AParamStableBlock(v, u, z, t, w, cfg.omega_scale)
    raise ValueError(f"Unknown parametrization '{param_name}'")


def build_a_param(cfg: AConfig, w: int, key: jax.random.PRNGKey, dtype: jnp.dtype):
    if cfg.parametrization not in A_PARAM_NAMES:
        raise ValueError(f"Unknown A parametrization '{cfg.parametrization}'")
    allowed = _allowed_a_inits(cfg.parametrization)
    init_name = cfg.init.type
    if init_name not in allowed:
        raise ValueError(
            f"Init '{init_name}' not allowed for A parametrization '{cfg.parametrization}'"
        )
    if cfg.parametrization == "sig":
        return AParamSig(w, dtype)

    if cfg.parametrization == "general":
        if init_name == "sig":
            matrix = jnp.zeros((w, w), dtype=dtype)
        else:
            params = _sample_params_for_param(init_name, w, cfg, key, dtype, init_name)
            matrix = _materialize_from_params(init_name, params, w, cfg, dtype)
        return AParamGeneral(matrix)

    if cfg.parametrization == "bounded_rotation":
        # u=0 -> sigmoid=1/2 -> max-dtheta token rotates by pi. With init.sigma_range=sr>0, spread
        # the per-plane frequencies: u_j ~ Uniform(+/- logit(0.5+sr)) so sigmoid(u_j) in
        # [0.5-sr, 0.5+sr] (median 0.5 -> pi preserved, the (0, 2pi) bound preserved, and the
        # plane-to-plane diversity restored: sr=0 collapses every plane to the same angle).
        P = w // 2
        sr = min(float(getattr(cfg.init, "sigma_range", 0.0) or 0.0), 0.499)
        if sr > 0.0:
            a = float(jnp.log((0.5 + sr) / (0.5 - sr)))
            u = jr.uniform(key, (P,), dtype=dtype, minval=-a, maxval=a)
        else:
            u = jnp.zeros((P,), dtype=dtype)
        return AParamBoundedRotation(u, w, cfg.omega_scale)

    if cfg.parametrization == "stable_block":
        # Key split mirrors damped_rotation (k_u, k_v first) so a stable_block and a damped_rotation built from the
        # same key share their frequency and damping draws exactly; the shear is the only thing added.
        P = w // 2
        sr = min(float(getattr(cfg.init, "sigma_range", 0.0) or 0.0), 0.499)
        k_u, k_v, k_z, k_t = jr.split(key, 4)
        if sr > 0.0:
            aa = float(jnp.log((0.5 + sr) / (0.5 - sr)))
            u = jr.uniform(k_u, (P,), dtype=dtype, minval=-aa, maxval=aa)
        else:
            u = jnp.zeros((P,), dtype=dtype)
        v = cfg.damp_init_mean + cfg.damp_init_std * jr.normal(k_v, (P,), dtype=dtype)
        z = cfg.shear_init_mean + cfg.shear_init_std * jr.normal(k_z, (P,), dtype=dtype)
        t = jr.uniform(k_t, (P,), dtype=dtype, minval=0.0, maxval=float(2 * jnp.pi))
        return AParamStableBlock(v, u, z, t, w, cfg.omega_scale)

    if cfg.parametrization == "damped_rotation":
        P = w // 2
        sr = min(float(getattr(cfg.init, "sigma_range", 0.0) or 0.0), 0.499)
        k_u, k_v = jr.split(key, 2)
        if sr > 0.0:
            aa = float(jnp.log((0.5 + sr) / (0.5 - sr)))
            u = jr.uniform(k_u, (P,), dtype=dtype, minval=-aa, maxval=aa)
        else:
            u = jnp.zeros((P,), dtype=dtype)
        v = cfg.damp_init_mean + cfg.damp_init_std * jr.normal(k_v, (P,), dtype=dtype)
        return AParamDampedRotation(v, u, w, cfg.omega_scale)

    if cfg.parametrization == "relu_damped_rotation":
        # Same init path as damped_rotation (same key split, same distributions) so the two
        # parametrisations differ only in the v -> a map.
        P = w // 2
        sr = min(float(getattr(cfg.init, "sigma_range", 0.0) or 0.0), 0.499)
        k_u, k_v = jr.split(key, 2)
        if sr > 0.0:
            aa = float(jnp.log((0.5 + sr) / (0.5 - sr)))
            u = jr.uniform(k_u, (P,), dtype=dtype, minval=-aa, maxval=aa)
        else:
            u = jnp.zeros((P,), dtype=dtype)
        v = cfg.damp_init_mean + cfg.damp_init_std * jr.normal(k_v, (P,), dtype=dtype)
        return AParamReluDampedRotation(v, u, w, cfg.omega_scale)

    if cfg.parametrization == "clamped_rotation":
        # u ~ U(0,1): uniform over one period (sigma_range not used). v: damped_rotation's
        # convention. Key split mirrors damped_rotation (k_u first) so the u draw at a given
        # key matches the earlier undamped variant's stream ordering.
        P = w // 2
        k_u, k_v = jr.split(key, 2)
        u = jr.uniform(k_u, (P,), dtype=dtype, minval=0.0, maxval=1.0)
        v = cfg.damp_init_mean + cfg.damp_init_std * jr.normal(k_v, (P,), dtype=dtype)
        return AParamClampedRotation(v, u, w, cfg.omega_scale)

    if cfg.parametrization == "unbounded_rotation":
        P = w // 2
        sr = min(float(getattr(cfg.init, "sigma_range", 0.0) or 0.0), 0.499)
        if sr > 0.0:
            a = float(jnp.log((0.5 + sr) / (0.5 - sr)))
            v = jr.uniform(key, (P,), dtype=dtype, minval=-a, maxval=a)
        else:
            v = jnp.zeros((P,), dtype=dtype)
        om0 = jax.nn.sigmoid(v)  # bounded_rotation's omega at init
        u = jnp.log(jnp.expm1(om0))  # softplus^-1 -> identical omega at step 0
        return AParamUnboundedRotation(u, w)

    params = _sample_params_for_param(cfg.parametrization, w, cfg, key, dtype, init_name)
    return _make_param_module(cfg.parametrization, params, w, cfg, dtype)


def build_b_matrix(
    cfg: BConfig, d_in: int, w: int, key: jax.random.PRNGKey, dtype: jnp.dtype
) -> jnp.ndarray:
    if cfg.parametrization not in B_PARAM_NAMES:
        raise ValueError(f"Unknown B parametrization '{cfg.parametrization}'")
    if cfg.init.type not in B_INIT_NAMES:
        raise ValueError(f"Unknown B init '{cfg.init.type}'")

    if cfg.preserve_first_channel:
        if d_in < 2 or w < 2:
            raise ValueError("_preserve_first_channel=True requires d_in >= 2 and out_dim >= 2")
        in_dim = d_in - 1
        out_dim = w - 1
    else:
        in_dim = d_in
        out_dim = w

    if cfg.init.type == "sig":
        return jnp.zeros((out_dim, in_dim), dtype=dtype)
    if cfg.init.type == "identity_block":
        return _identity_block(out_dim, in_dim, dtype)
    return _gaussian(key, (out_dim, in_dim), cfg.init.mean, cfg.init.variance, dtype)


# -- the lift ---------------------------------------------------------------------------------
class B(eqx.Module):
    """
    Learnable linear path transformation.

    If preserve_first_channel=True, the first channel is preserved and B acts
    on the remaining channels. Otherwise B mixes all channels.
    """

    matrix: jnp.ndarray
    preserve_first_channel: bool = eqx.field(static=True)

    def __init__(self, matrix: jnp.ndarray, preserve_first_channel: bool = True):
        self.matrix = matrix
        self.preserve_first_channel = bool(preserve_first_channel)

    def __call__(self, path: jnp.ndarray) -> jnp.ndarray:
        if self.preserve_first_channel:
            if path.shape[-1] < 2:
                raise ValueError(
                    "preserve_first_channel=True requires path with at least 2 channels"
                )
            time = path[:, :1]
            data = path[:, 1:]
            transformed = data @ self.matrix.T
            return jnp.concatenate([time, transformed], axis=-1)
        return path @ self.matrix.T


# -- building both from configuration --------------------------------------------------------
def build_ab_from_config(
    model_cfg: dict,
    d_in: int,
    key: jr.PRNGKey,
    dtype: jnp.dtype,
) -> Tuple[object, Optional[B], int, AConfig, BConfig]:
    a_cfg = parse_a_config(model_cfg)
    b_cfg = parse_b_config(model_cfg)

    key_a, key_b = jr.split(key, 2)

    if b_cfg.enabled:
        if b_cfg.out_dim is None:
            raise ValueError("model.B.out_dim required when model.B.enabled=true")
        w = int(b_cfg.out_dim)
        b_matrix = build_b_matrix(b_cfg, d_in=d_in, w=w, key=key_b, dtype=dtype)
        b_layer = B(matrix=b_matrix, preserve_first_channel=b_cfg.preserve_first_channel)
    else:
        w = int(d_in)
        b_layer = None

    a_param = build_a_param(a_cfg, w=w, key=key_a, dtype=dtype)
    return a_param, b_layer, w, a_cfg, b_cfg
