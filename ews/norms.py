"""Normalisation modules: RMSNorm, the per-level GradedRMSNorm used inside blocks, and the level-wise
FeatureNorm applied to a flattened signature feature in the layer-only models."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp


class RMSNorm(eqx.Module):
    """RMSNorm as in the SLiCE package: the root mean square over the last axis, no mean subtraction, no bias, one
    learnable gain initialised to ones. Input and output (..., dim).
    """

    weight: jnp.ndarray
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-6, dtype: jnp.dtype = jnp.float32):
        self.eps = float(eps)
        self.weight = jnp.ones((dim,), dtype=dtype)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        rms = jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x * rms) * self.weight


def graded_level_slices(d: int, n: int) -> Tuple[Tuple[int, int], ...]:
    """(start, end) of each level block in a level-0-free flat feature vector.

    Layout: [lvl1 (d) | lvl2 (d^2) | ... | lvlN (d^N)], total D = sum d^k.
    Deliberately not borrowed from FeatureNorm._level_slices, whose layout
    carries a level-0 constant at index 0: here level 1 starts
    at index 0.
    """
    slices = []
    start = 0
    for k in range(1, n + 1):
        size = d**k
        slices.append((start, start + size))
        start += size
    return tuple(slices)


class GradedRMSNorm(eqx.Module):
    """Per-level normalisation of a flat graded EWS feature [lvl1 | lvl2 | ... | lvlN], per sample and position.

        live_rms     g_k * block / (RMS_k + eps_k), the RMS over the level's d^k coordinates
        static_gain  g_k * block
        dilation     lam**k * block

    Levels are orders of magnitude apart and the top level holds most coordinates, so one flat statistic would be level
    N's. The statistics use the last axis only, so the norm is causal, and an all-zero level gives an exactly zero output
    and gradient.

    Args:
        d: the path width (m, or m + 1 with an appended clock). N: the depth.
        variant: "live_rms", "static_gain" or "dilation".
        eps: per-level floors for live_rms (a scalar is used for every level). dtype: the parameter dtype.

    Input and output (..., D) with D = sum_k d^k.
    """

    d: int = eqx.field(static=True)
    N: int = eqx.field(static=True)
    variant: str = eqx.field(static=True)
    level_slices: Tuple[Tuple[int, int], ...] = eqx.field(static=True)
    eps: Tuple[float, ...] = eqx.field(static=True)
    gains: Optional[jnp.ndarray]
    lam: Optional[jnp.ndarray]

    def __init__(
        self,
        d: int,
        N: int,
        variant: str = "live_rms",
        eps: Union[float, Sequence[float]] = 1e-3,
        dtype: jnp.dtype = jnp.float32,
    ):
        if variant not in {"live_rms", "static_gain", "dilation"}:
            raise ValueError("variant must be one of {'live_rms', 'static_gain', 'dilation'}.")
        self.d = int(d)
        self.N = int(N)
        self.variant = variant
        self.level_slices = graded_level_slices(self.d, self.N)
        if isinstance(eps, (int, float)):
            eps_tuple = tuple(float(eps) for _ in range(self.N))
        else:
            eps_tuple = tuple(float(e) for e in eps)
            if len(eps_tuple) != self.N:
                raise ValueError("eps must be a scalar or a sequence of length N.")
        self.eps = eps_tuple
        if variant == "dilation":
            self.gains = None
            self.lam = jnp.ones((), dtype=dtype)
        else:
            self.gains = jnp.ones((self.N,), dtype=dtype)
            self.lam = None

    def per_level_rms(self, x: jnp.ndarray) -> jnp.ndarray:
        """Positionwise RMS of each level block -> (..., N).

        RMS is over the level's d^k coordinates only; leading axes are kept
        (unlike FeatureNorm.per_level_rms, which pools everything). The
        harness reduces over leading axes (median) for the eps calibration.
        """
        out = []
        for s, e in self.level_slices:
            block = x[..., s:e]
            out.append(jnp.sqrt(jnp.mean(block * block, axis=-1)))
        return jnp.stack(out, axis=-1)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.shape[-1] != self.level_slices[-1][1]:
            raise ValueError(
                f"last axis must be D = {self.level_slices[-1][1]} "
                f"(level-0-free layout), got {x.shape[-1]}"
            )
        blocks = []
        for k, (s, e) in enumerate(self.level_slices):
            block = x[..., s:e]
            if self.variant == "live_rms":
                ms = jnp.mean(block * block, axis=-1, keepdims=True)
                # zero-safe sqrt: value and gradient are exactly 0 at ms == 0
                safe = jnp.where(ms > 0, ms, 1.0)
                rms = jnp.where(ms > 0, jnp.sqrt(safe), 0.0)
                blocks.append(self.gains[k] * block / (rms + self.eps[k]))
            elif self.variant == "static_gain":
                blocks.append(block * self.gains[k])
            else:  # dilation
                blocks.append(block * self.lam ** (k + 1))
        return jnp.concatenate(blocks, axis=-1)


def _level_slices(d: int, n: int) -> Tuple[Tuple[int, int], ...]:
    """(start, end) of each level block in the flat vector, level 0 at index 0."""
    slices = []
    start = 1  # index 0 is level 0 (the constant 1) -> excluded
    for k in range(1, n + 1):
        size = d**k
        slices.append((start, start + size))
        start += size
    return tuple(slices)


class FeatureNorm(eqx.Module):
    mode: str = eqx.field(static=True)
    d: int = eqx.field(static=True)
    N: int = eqx.field(static=True)
    level_slices: Tuple[Tuple[int, int], ...] = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    gains: Optional[jnp.ndarray] = None  # (N,) for learned_gain
    gamma: Optional[jnp.ndarray] = None  # (N,) for batchnorm
    beta: Optional[jnp.ndarray] = None  # (N,) for batchnorm

    def __init__(
        self,
        d: int,
        N: int,
        mode: str = "none",
        dtype: jnp.dtype = jnp.float32,
        eps: float = 1e-5,
        level_slices: Optional[Tuple[Tuple[int, int], ...]] = None,
    ):
        """``level_slices`` overrides the default signature layout (``d^k`` blocks with a
        level-0 constant at index 0). Pass log-signature slices (Witt-sized, no level-0;
        see ``log_level_slices`` in ``log_ews_layer``) to normalize a log-EWS feature.
        When ``None`` (default) the original signature behavior is preserved exactly."""
        if mode not in {"none", "learned_gain", "batchnorm"}:
            raise ValueError(
                f"feature_norm mode must be 'none', 'learned_gain', or 'batchnorm', got {mode}"
            )
        self.mode = mode
        self.d = int(d)
        self.N = int(N)
        self.level_slices = (
            tuple(level_slices) if level_slices is not None else _level_slices(int(d), int(N))
        )
        self.eps = float(eps)

        if mode == "learned_gain":
            self.gains = jnp.ones((self.N,), dtype=dtype)
        elif mode == "batchnorm":
            self.gamma = jnp.ones((self.N,), dtype=dtype)
            self.beta = jnp.zeros((self.N,), dtype=dtype)

    def per_level_rms(self, features: jnp.ndarray) -> jnp.ndarray:
        """RMS of each level block over a feature batch -> (N,). Used for 1/RMS init."""
        rms = []
        for s, e in self.level_slices:
            block = features[..., s:e]
            rms.append(jnp.sqrt(jnp.mean(block**2)))
        return jnp.stack(rms)

    def level_stats(self, features: jnp.ndarray):
        """Per-level (mean, variance) over a feature batch: the statistics `batchnorm` normalises with.

        Computed on the training set and handed to `apply_with_stats`, they make every evaluated prediction
        independent of the other evaluated samples (model.feature_norm_eval = "train")."""
        return [
            (jnp.mean(features[:, s:e]), jnp.var(features[:, s:e])) for s, e in self.level_slices
        ]

    def apply_with_stats(self, x: jnp.ndarray, stats) -> jnp.ndarray:
        """`batchnorm` with given per-level (mean, variance) instead of the input batch's own."""
        if self.mode != "batchnorm":
            raise ValueError("apply_with_stats is defined for feature_norm='batchnorm' only")
        out = x
        for k, ((s, e), (mean, var)) in enumerate(zip(self.level_slices, stats)):
            out = out.at[:, s:e].set(
                (x[:, s:e] - mean) / jnp.sqrt(var + self.eps) * self.gamma[k] + self.beta[k]
            )
        return out

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.mode == "none":
            return x
        if self.mode == "batchnorm" and x.ndim != 2:
            raise ValueError("feature_norm='batchnorm' requires batched (B, D) input")

        out = x
        for k, (s, e) in enumerate(self.level_slices):
            block = x[..., s:e]
            if self.mode == "learned_gain":
                new_block = block * self.gains[k]
            else:  # batchnorm: group stats over batch x coords
                mean = jnp.mean(block)
                var = jnp.var(block)
                new_block = (block - mean) / jnp.sqrt(var + self.eps)
                new_block = new_block * self.gamma[k] + self.beta[k]
            out = out.at[..., s:e].set(new_block)
        return out
