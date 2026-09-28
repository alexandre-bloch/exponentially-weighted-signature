"""Clocks: the monotone reparametrisation theta_t that drives the EWS reweighting.

A clock maps a time base and a driver (the signal whose variation it measures) to a
non-decreasing array theta of the same length. Six modes exist:

    time                     theta = t (scaled)
    blended_variation        theta = alpha * t + (1 - alpha) * V,   V = cumulative L2 increment norm
    blended_quad_variation   theta = alpha * t + (1 - alpha) * Q,   Q = cumulative squared increments
    blended_learn            blended_variation with a trainable alpha (one scalar per block)
    token_gated              theta = cumsum(relu(w . dx - b)) / c                 (learnable w, b)
    clamped_gate             theta = cumsum(sum_k clip(w_k . dx - b_k, 0, 1)) / c (learnable w_k, b_k)

The first three have no parameters; when their driver is raw data they are constants and
can be precomputed once (`build_clock_np`, the numpy path used by the UEA pipeline). The
last three are modules with parameters and are evaluated in the forward pass. All six share
one interface, `Clock`, so a model chooses its clock from configuration alone; the block-only
`source` field says which signal a block hands to its clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

CLOCK_MODES = (
    "time",
    "blended_variation",
    "blended_quad_variation",
    "token_gated",
    "clamped_gate",
    "blended_learn",
)
FIXED_MODES = ("time", "blended_variation", "blended_quad_variation")
GATED_MODES = ("token_gated", "clamped_gate")
LEARNED_MODES = ("blended_learn",)  # blended_variation with a trainable alpha
_CLOCK_MODES = set(FIXED_MODES)
_CLOCK_SCALINGS = {
    "none",
    "[0,1]",
    "[-1,1]",
    "steps",
}  # "steps": theta_i = i, one unit per sample (mode "time" only)
_CLOCK_NORMALIZE = {"per_sample", "constant"}
_ALPHA_PARAMETRISATIONS = {"clip", "sigmoid"}
CLOCK_SOURCES = ("block_input", "encoder_output", "raw_input")


@dataclass(frozen=True)
class ClockSpec:
    """Configuration of one clock. Field names are the config keys of the `clock` section."""

    mode: str = "time"
    alpha: float = 1.0
    scaling: str = "[0,1]"
    normalize: str = "per_sample"
    norm_constant: Optional[float] = None
    time_denominator: Optional[float] = None
    variation_eps: float = 1e-8
    stop_gradient: bool = False
    gate_init_scale: Optional[float] = None
    gate_init_bias: Union[float, Tuple[float, ...]] = -2.0
    gate_K: int = 4
    gate_in_scale: float = 1.0  # fixed factor on the driver before a gated clock; see GatedClock
    learn: bool = True  # a clock with parameters is trained; False leaves them at their init
    alpha_parametrisation: str = (
        "clip"  # blended_learn: how alpha is held in [0, 1]; see LearnableBlendedClock
    )
    source: str = "block_input"

    @classmethod
    def from_config(cls, cfg: Optional[dict], **overrides) -> "ClockSpec":
        d = dict(cfg or {})
        d.update(overrides)
        if isinstance(d.get("gate_init_bias"), list):
            d["gate_init_bias"] = tuple(float(x) for x in d["gate_init_bias"])
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown clock keys: {sorted(unknown)}")
        return cls(**d)

    def validate(self) -> "ClockSpec":
        if self.mode not in CLOCK_MODES:
            raise ValueError(f"clock mode must be one of {CLOCK_MODES}, got {self.mode!r}")
        if self.scaling not in _CLOCK_SCALINGS:
            raise ValueError("clock scaling must be one of {'none', '[0,1]', '[-1,1]'}.")
        if self.scaling == "steps" and self.mode != "time":
            raise ValueError("clock scaling 'steps' (theta_i = i) is defined for mode 'time' only.")
        if self.normalize not in _CLOCK_NORMALIZE:
            raise ValueError("clock normalize must be one of {'per_sample', 'constant'}.")
        if self.source not in CLOCK_SOURCES:
            raise ValueError(f"clock source must be one of {CLOCK_SOURCES}, got {self.source!r}")
        if (
            self.mode != "time" and not 0.0 <= float(self.alpha) <= 1.0
        ):  # for blended_learn this is alpha's init
            raise ValueError("alpha must be in [0, 1] for blended clock modes.")
        if self.alpha_parametrisation not in _ALPHA_PARAMETRISATIONS:
            raise ValueError(
                f"clock alpha_parametrisation must be one of {sorted(_ALPHA_PARAMETRISATIONS)}, "
                f"got {self.alpha_parametrisation!r}"
            )
        if self.mode in GATED_MODES:
            if not float(self.gate_in_scale) > 0.0:
                raise ValueError(
                    f"clock gate_in_scale must be positive, got {self.gate_in_scale!r}"
                )
            if self.normalize != "constant" or self.norm_constant is None:
                raise ValueError(
                    f"clock mode {self.mode!r} requires normalize='constant' with an explicit norm_constant."
                )
            if self.scaling != "none":
                raise ValueError(f"clock mode {self.mode!r} requires scaling='none'.")
        return self

    @property
    def has_parameters(self) -> bool:
        return self.mode in GATED_MODES or self.mode in LEARNED_MODES


# --------------------------------------------------------------------------------------------
# Parameter-free clocks, jnp (per sample) and numpy (batch precompute)
# --------------------------------------------------------------------------------------------
def _scale_clock_component_jnp(c: jnp.ndarray, scaling: str) -> jnp.ndarray:
    """Per-sample affine map of a monotone component onto the requested range.

    jnp twin of _scale_clock_component: 'none' leaves raw
    values; '[0,1]' and '[-1,1]' map [c_0, c_last] affinely (denominator
    guarded by +1e-12). Note: the denominator uses the component's final value,
    so this map is per-sample and non-causal at interior positions.
    """
    if scaling == "none":
        return c
    if scaling == "steps":  # the sample grid itself: one unit of clock per sample
        return jnp.arange(c.shape[0], dtype=c.dtype)
    denom = (c[-1] - c[0]) + 1e-12
    z = (c - c[0]) / denom
    if scaling == "[0,1]":
        return z
    if scaling == "[-1,1]":
        return 2.0 * z - 1.0
    raise ValueError("scaling must be one of {'none', '[0,1]', '[-1,1]', 'steps'}.")


def _range_map_jnp(z: jnp.ndarray, scaling: str) -> jnp.ndarray:
    """Map a ratio-in-[0,1] component onto the requested range (constant mode).

    'none' and '[0,1]' leave the ratio as is; '[-1,1]' maps it affinely.
    """
    if scaling in ("none", "[0,1]"):
        return z
    if scaling == "[-1,1]":
        return 2.0 * z - 1.0
    raise ValueError("scaling must be one of {'none', '[0,1]', '[-1,1]'}.")


def build_clock_jnp(
    times: jnp.ndarray,
    driver: Optional[jnp.ndarray],
    mode: str,
    alpha: float,
    scaling: str,
    *,
    mask: Optional[jnp.ndarray] = None,
    normalize: str = "per_sample",
    norm_constant: Optional[float] = None,
    time_denominator: Optional[float] = None,
    variation_eps: float = 1e-8,
    stop_gradient: bool = False,
) -> jnp.ndarray:
    """The clock in jnp, per sample: the twin of build_clock_np, with masking, the constant normalisation, an epsilon
    inside the variation's square root and an optional stop-gradient.

    theta = alpha * t_s + (1 - alpha) * V_s for the blended modes and theta = t_s for "time", where t_s is the scaled time
    and V_s the scaled cumulative variation of `driver`.

    Args:
        times: (seq_len,) monotone time base (the sample grid in a block).
        driver: (seq_len, channels), the signal whose variation is measured; None for "time".
        mode: "time", "blended_variation" (L2 increment norms) or "blended_quad_variation" (squared norms).
        alpha: the weight of the time component, in [0, 1]; 0 is a pure variation clock.
        scaling: "none", "[0,1]" or "[-1,1]".
        mask: (seq_len,) prefix validity mask. An increment counts only if both of its ends are real, and theta is held
            constant over the padding.
        normalize: "per_sample" divides V by the sample's own total (not causal: the total is known only at the end);
            "constant" divides it by `norm_constant` (causal).
        time_denominator: None scales the time component per sample over `times`; a number gives
            t_s = (t - t_0) / time_denominator.
        variation_eps: the epsilon in sqrt(sum(dx^2) + eps), a gradient guard at dx = 0.
        stop_gradient: detach the clock from the gradient.

    Returns theta, (seq_len,).
    """
    if mode not in _CLOCK_MODES:
        raise ValueError(
            "mode must be one of {'time', 'blended_variation', 'blended_quad_variation'}."
        )
    if scaling not in _CLOCK_SCALINGS:
        raise ValueError("scaling must be one of {'none', '[0,1]', '[-1,1]'}.")
    if normalize not in _CLOCK_NORMALIZE:
        raise ValueError("normalize must be one of {'per_sample', 'constant'}.")
    t = jnp.asarray(times)
    if t.ndim != 1:
        raise ValueError("times must be 1D (seq_len,).")
    if mask is not None:
        mask = jnp.asarray(mask)
        if mask.shape != t.shape:
            raise ValueError("mask must have the same shape as times.")

    if scaling == "steps" and (mode != "time" or time_denominator is not None):
        raise ValueError("scaling 'steps' is defined for mode 'time' without a time_denominator.")
    if time_denominator is None:
        t_s = _scale_clock_component_jnp(t, scaling)
    else:
        t_s = _range_map_jnp((t - t[0]) / float(time_denominator), scaling)

    if mode == "time":
        theta = t_s
    else:
        # alpha may be a traced scalar (the learnable-alpha clock, mode "blended_learn"), in which case its
        # value is not available at trace time and the range is enforced by the clock module's own clip
        # instead. Concrete floats keep the original check, so every existing caller is unaffected.
        if isinstance(alpha, jnp.ndarray) and alpha.ndim == 0:
            a = alpha
        else:
            a = float(alpha)
            if not 0.0 <= a <= 1.0:
                raise ValueError(
                    "alpha must be in [0, 1] for blended clock modes. NOTE: alpha=0 "
                    "(pure variation) loses strict monotonicity on zero-increment "
                    "steps (Definition 2.1 caveat, accepted for FL increment runs)."
                )
        if driver is None:
            raise ValueError("driver is required for blended clock modes.")
        p = jnp.asarray(driver)
        if p.ndim != 2 or p.shape[0] != t.shape[0]:
            raise ValueError("driver must be 2D (seq_len, channels) aligned with times.")
        dx = jnp.diff(p, axis=0)  # (seq_len - 1, channels)
        sq = (dx * dx).sum(axis=1)
        if mode == "blended_variation":
            incr = jnp.sqrt(sq + variation_eps)
        else:  # blended_quad_variation
            incr = sq
        if mask is not None:
            pair = (mask[1:] & mask[:-1]).astype(incr.dtype)
            incr = incr * pair
        V = jnp.concatenate([jnp.zeros((1,), dtype=incr.dtype), jnp.cumsum(incr)])
        if normalize == "per_sample":
            V_s = _scale_clock_component_jnp(V, scaling)
        else:
            if norm_constant is None:
                raise ValueError("norm_constant is required when normalize='constant'.")
            V_s = _range_map_jnp(V / float(norm_constant), scaling)
        theta = a * t_s + (1.0 - a) * V_s

    if mask is not None:
        n = jnp.sum(mask.astype(jnp.int32)) - 1
        theta = jnp.where(jnp.arange(theta.shape[0]) <= n, theta, jnp.take(theta, n))

    if stop_gradient:
        theta = jax.lax.stop_gradient(theta)
    return theta


# numpy twin used when the clock is a constant of the data (UEA preprocessing). Scaling is
# always per sample here; the variation is taken on the raw data channels.
def _ensure_2d(path: np.ndarray) -> np.ndarray:
    if path.ndim == 1:
        return path.reshape(-1, 1)
    return path


def _ensure_1d_clock(clock: np.ndarray) -> np.ndarray:
    c = np.asarray(clock)
    if c.ndim != 1:
        raise ValueError("clock arrays must be 1D")
    return c


def _scale_clock_component(c: np.ndarray, scaling: str) -> np.ndarray:
    """Scale a single monotone clock component to the requested range.

    'none' leaves raw values; '[0,1]' and '[-1,1]' map [c0, c_last] affinely.
    """
    c = np.asarray(c, dtype=np.float64)
    if scaling == "none":
        return c
    if scaling == "steps":  # the sample grid itself: one unit of clock per sample
        return np.arange(c.shape[0], dtype=np.float64)
    denom = (c[-1] - c[0]) + 1e-12
    z = (c - c[0]) / denom
    if scaling == "[0,1]":
        return z
    if scaling == "[-1,1]":
        return 2.0 * z - 1.0
    raise ValueError(f"clock scaling must be 'none', '[0,1]', or '[-1,1]', got {scaling}")


def build_clock_np(
    time_clock: np.ndarray,
    path_data: np.ndarray,
    mode: str,
    alpha: float,
    scaling: str,
) -> np.ndarray:
    """Build a monotone clock from a time base and the raw path.

    - ``time`` returns the (scaled) supplied time clock.
    - ``blended_variation`` blends scaled time with scaled total variation
      ``V_t = sum ||ΔX||_2`` as ``alpha*t + (1-alpha)*V``.
    - ``blended_quad_variation`` uses quadratic variation ``Q_t = sum ||ΔX||_2^2``.

    Each component is scaled once by ``scaling`` and then convex-blended, so for
    '[0,1]'/'[-1,1]' the result stays in range and is monotone (strictly so when
    alpha>0). Variation is computed on the raw ``path_data`` (data channels only).
    """
    t = np.asarray(time_clock, dtype=np.float64)
    if mode == "time":
        return _scale_clock_component(t, scaling)

    p = _ensure_2d(np.asarray(path_data)).astype(np.float64)
    dx = np.diff(p, axis=0)  # (T-1, d)
    if mode == "blended_variation":
        incr = np.sqrt((dx**2).sum(axis=1))
    elif mode == "blended_quad_variation":
        incr = (dx**2).sum(axis=1)
    else:
        raise ValueError(
            "clock mode must be 'time', 'blended_variation', or "
            f"'blended_quad_variation', got {mode}"
        )
    V = np.concatenate([np.zeros(1, dtype=np.float64), np.cumsum(incr)])  # (T,)

    t_s = _scale_clock_component(t, scaling)
    V_s = _scale_clock_component(V, scaling)
    a = float(alpha)
    return a * t_s + (1.0 - a) * V_s


# --------------------------------------------------------------------------------------------
# Clock modules
# --------------------------------------------------------------------------------------------
class Clock(eqx.Module):
    """Common interface: theta = clock(t, driver, mask); per-token increments for token caches."""

    spec: ClockSpec = eqx.field(static=True)

    @property
    def is_static(self) -> bool:
        """True when the clock has no parameters (its output is a constant of its inputs)."""
        return not self.spec.has_parameters

    def __call__(
        self, t: jnp.ndarray, driver: Optional[jnp.ndarray], mask: Optional[jnp.ndarray]
    ) -> jnp.ndarray:
        raise NotImplementedError

    def per_token_increment(self, embeds: jnp.ndarray) -> jnp.ndarray:
        """Delta theta contributed by each vocabulary token (V, d) -> (V,), for token caches.

        Defined only when the increment depends on the current token alone, i.e. under
        constant normalisation and, for the blended modes, alpha = 0."""
        raise NotImplementedError


class FixedClock(Clock):
    """time / blended_variation / blended_quad_variation: no parameters."""

    def __init__(self, spec: ClockSpec):
        self.spec = spec.validate()

    def __call__(self, t, driver, mask):
        s = self.spec
        return build_clock_jnp(
            t,
            driver,
            s.mode,
            s.alpha,
            s.scaling,
            mask=mask,
            normalize=s.normalize,
            norm_constant=s.norm_constant,
            time_denominator=s.time_denominator,
            variation_eps=s.variation_eps,
            stop_gradient=s.stop_gradient,
        )

    def per_token_increment(self, embeds):
        s = self.spec
        if s.normalize != "constant" or s.norm_constant is None:
            raise ValueError("per-token clock increments need normalize='constant'.")
        if s.mode == "time":
            if s.time_denominator is None:
                raise ValueError("time clock needs time_denominator for per-token increments.")
            return jnp.full((embeds.shape[0],), 1.0 / float(s.time_denominator), dtype=embeds.dtype)
        if float(s.alpha) != 0.0:
            raise ValueError("per-token increments of a blended clock need alpha = 0.")
        sq = jnp.sum(embeds**2, axis=1)
        if s.mode == "blended_variation":
            return jnp.sqrt(sq + s.variation_eps) / s.norm_constant
        return sq / s.norm_constant


class GatedClock(Clock):
    """token_gated / clamped_gate: learnable thresholds on the driver increments.

    `spec.gate_in_scale` multiplies the driver before the gate, so the gate's weights can stay O(1) on a
    dataset whose increments are not. EigenWorms' increments are about 7e-4: an unscaled gate needs weights
    of order 1e3 to be open at all, and Adam (whose step is about the learning rate whatever the gradient)
    then barely moves them in relative terms. Scaling by 1/rms||dx|| instead leaves the clock trainable.
    It is a fixed number, read from the configuration and never trained; scaling the driver by s is the
    same model as scaling w by s, so it changes only the parametrisation the optimiser sees. The driver is
    scaled before the difference, not after: the two agree in exact arithmetic but not in float32, and the
    selected EigenWorms model was produced with this order.
    """

    w: jnp.ndarray
    b: jnp.ndarray
    in_scale: float = eqx.field(static=True, default=1.0)

    def __init__(self, spec: ClockSpec, dim: int, key: jr.PRNGKey, dtype: jnp.dtype = jnp.float32):
        self.spec = spec.validate()
        self.in_scale = float(spec.gate_in_scale)
        gs = float(dim) ** -0.5 if spec.gate_init_scale is None else float(spec.gate_init_scale)
        if spec.mode == "token_gated":
            self.w = gs * jr.normal(key, (int(dim),), dtype=dtype)
            self.b = jnp.asarray(float(spec.gate_init_bias), dtype=dtype)
        else:
            if isinstance(spec.gate_init_bias, (list, tuple)):
                b_init = jnp.asarray([float(x) for x in spec.gate_init_bias], dtype=dtype)
                K = int(b_init.shape[0])
            else:
                K = int(spec.gate_K)
                b_init = jnp.full((K,), float(spec.gate_init_bias), dtype=dtype)
            self.w = gs * jr.normal(key, (K, int(dim)), dtype=dtype)
            self.b = b_init

    def __call__(self, t, driver, mask):
        if driver is None:
            raise ValueError(f"{self.spec.mode} clock requires a driver.")
        dx = jnp.diff(driver * self.in_scale, axis=0)
        if self.spec.mode == "clamped_gate":
            incr = jnp.clip(dx @ self.w.T - self.b, 0.0, 1.0).sum(axis=-1)
        else:
            incr = jax.nn.relu(dx @ self.w - self.b)
        if mask is not None:
            pair = (mask[1:] & mask[:-1]).astype(incr.dtype)
            incr = incr * pair
        V = jnp.concatenate([jnp.zeros((1,), dtype=incr.dtype), jnp.cumsum(incr)])
        c = self.spec.norm_constant
        theta = V / float(c if c is not None else 1.0)
        if mask is not None:
            n = jnp.sum(mask.astype(jnp.int32)) - 1
            theta = jnp.where(jnp.arange(theta.shape[0]) <= n, theta, jnp.take(theta, n))
        if self.spec.stop_gradient:
            theta = jax.lax.stop_gradient(theta)
        return theta

    def per_token_increment(self, embeds):
        c = self.spec.norm_constant
        embeds = embeds * self.in_scale
        if self.spec.mode == "token_gated":
            return jax.nn.relu(embeds @ self.w - self.b) / c
        return jnp.clip(embeds @ self.w.T - self.b, 0.0, 1.0).sum(axis=-1) / c


class LearnableBlendedClock(Clock):
    """blended_learn: theta = alpha * t + (1 - alpha) * V with a trainable alpha, one scalar per block.

    alpha = 1 is the time clock and alpha = 0 the variation clock; `spec.alpha` is the initialisation. At any fixed alpha
    this is exactly the `blended_variation` clock, since it calls `build_clock_jnp`.

    `spec.alpha_parametrisation` sets how alpha stays in [0, 1]:
      * `clip` (the default): alpha is stored raw and clipped. Outside [0, 1] the clip passes no gradient, so alpha can
        freeze at a pure clock (on EigenWorms it left the interval on four seeds of five).
      * `sigmoid`: alpha = sigmoid(raw), inside (0, 1) with a gradient everywhere; it never reaches a pure clock.
    The two are different models, so a configuration must say which it uses.
    """

    alpha_raw: jnp.ndarray

    _EPS = 1e-4  # keeps logit(alpha) finite when alpha is initialised at exactly 0 or 1

    def __init__(self, spec: ClockSpec, dtype: jnp.dtype = jnp.float32):
        self.spec = spec.validate()
        a0 = float(spec.alpha)
        if self.spec.alpha_parametrisation == "sigmoid":
            a0 = min(max(a0, self._EPS), 1.0 - self._EPS)
            a0 = math.log(a0 / (1.0 - a0))
        self.alpha_raw = jnp.asarray(a0, dtype)

    @property
    def alpha(self) -> jnp.ndarray:
        if self.spec.alpha_parametrisation == "sigmoid":
            return jax.nn.sigmoid(self.alpha_raw)
        return jnp.clip(self.alpha_raw, 0.0, 1.0)

    def __call__(self, t, driver, mask):
        s = self.spec
        return build_clock_jnp(
            t,
            driver,
            "blended_variation",
            self.alpha,
            s.scaling,
            mask=mask,
            normalize=s.normalize,
            norm_constant=s.norm_constant,
            time_denominator=s.time_denominator,
            variation_eps=s.variation_eps,
            stop_gradient=s.stop_gradient,
        )

    def per_token_increment(self, embeds):
        raise NotImplementedError(
            "blended_learn has a trainable alpha, so a token's clock increment is not a constant of the "
            "token: no token cache."
        )


def make_clock(
    spec: ClockSpec, dim: int, key: Optional[jr.PRNGKey] = None, dtype: jnp.dtype = jnp.float32
) -> Clock:
    """Build the clock module for `spec`; `key` is needed only for the gated modes."""
    spec = spec.validate()
    if spec.mode in LEARNED_MODES:
        return LearnableBlendedClock(spec, dtype)  # one scalar; no key needed
    if spec.has_parameters:
        if key is None:
            raise ValueError(f"clock mode {spec.mode!r} has parameters and needs a key.")
        return GatedClock(spec, dim, key, dtype)
    return FixedClock(spec)
