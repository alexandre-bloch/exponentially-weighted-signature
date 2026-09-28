"""Paths: what enters the signature as X, and the transforms applied to it.

The choices, in the order they are applied:

    representation   value (X_i = x_i), increment (X = cumulative sum of the rows, the discrete
                     integral of a per-token increment stream) or derivative (the path is replaced
                     by its N-th Savitzky-Golay derivative, length-preserving)
    per-sample normalisation   one scalar per instance, before anything else that is fitted
    (the clock is built here, from the representation, before the channel normalisation)
    channel normalisation      fitted on the training split (or all data) to a range
    basepoint        a zero row prepended, so the first increment is the first value
    clock channel    the clock (the main one, or a separately configured one) as channel 0

`PathSpec` holds the choices; `prepare_paths_np` applies them to a batch of numpy paths together
with the clock, which is the precomputation used whenever both stages are constants of the data.
`to_increment_path_jnp` is the in-model counterpart of the increment representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import jax.numpy as jnp
import numpy as np

from ews.clocks import ClockSpec, build_clock_np

NormalizeMode = Literal["train_only", "all_data"]
NormalizeRange = Literal["[0,1]", "[-1,1]"]
REPRESENTATIONS = ("value", "increment", "derivative")
CLOCK_CHANNELS = (None, "main", "separate")


@dataclass(frozen=True)
class PathSpec:
    """Configuration of the path stage. Field names are the config keys of the `path` section."""

    representation: str = "value"
    derivative_order: int = 0
    derivative_window: int = 15
    derivative_polyorder: int = 3
    normalize: bool = True
    normalize_mode: str = "train_only"
    normalize_range: str = "[0,1]"
    per_sample_normalize: bool = False
    per_sample_norm_stat: str = "std"
    basepoint: bool = False
    clock_channel: Optional[str] = None

    @classmethod
    def from_config(cls, cfg: Optional[dict], **overrides) -> "PathSpec":
        d = dict(cfg or {})
        d.update(overrides)
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown path keys: {sorted(unknown)}")
        return cls(**d).validate()

    def validate(self) -> "PathSpec":
        if self.representation not in REPRESENTATIONS:
            raise ValueError(
                f"path representation must be one of {REPRESENTATIONS}, got {self.representation!r}"
            )
        if self.representation == "derivative" and int(self.derivative_order) < 1:
            raise ValueError("representation 'derivative' needs derivative_order >= 1")
        if self.clock_channel not in CLOCK_CHANNELS:
            raise ValueError(
                f"clock_channel must be one of {CLOCK_CHANNELS}, got {self.clock_channel!r}"
            )
        if self.normalize_mode not in ("train_only", "all_data"):
            raise ValueError("normalize_mode must be 'train_only' or 'all_data'")
        return self


def legacy_specs(pp: dict) -> Tuple[PathSpec, ClockSpec, Optional[ClockSpec]]:
    """Translate the flat preprocessing keys of the older configs into (path, clock, append clock)."""
    order = int(pp.get("path_derivative", 0))
    append_mode = pp.get("append_clock_mode", None)
    path = PathSpec(
        representation="derivative" if order > 0 else "value",
        derivative_order=order,
        derivative_window=int(pp.get("savgol_window", 15)),
        derivative_polyorder=int(pp.get("savgol_polyorder", 3)),
        normalize=bool(pp.get("normalize_paths", True)),
        normalize_mode=pp.get("normalize_mode", "train_only"),
        normalize_range=pp.get("normalize_range", "[0,1]"),
        per_sample_normalize=bool(pp.get("per_sample_normalize", False)),
        per_sample_norm_stat=pp.get("per_sample_norm_stat", "std"),
        basepoint=bool(pp.get("add_basepoint", False)),
        clock_channel="separate"
        if append_mode is not None
        else ("main" if pp.get("augment_with_clock", False) else None),
    ).validate()
    clock = ClockSpec(
        mode=pp.get("clock_mode", "time"),
        alpha=float(pp.get("clock_alpha", 1.0)),
        scaling=pp.get("clock_normalize", "[0,1]"),
    ).validate()
    append = (
        None
        if append_mode is None
        else ClockSpec(
            mode=append_mode,
            alpha=float(pp.get("append_clock_alpha", 1.0)),
            scaling=path.normalize_range,
        ).validate()
    )
    return path, clock, append


def specs_from_config(pp: dict) -> Tuple[PathSpec, ClockSpec, Optional[ClockSpec]]:
    """Read (path, clock, append clock) from a config: sectioned (`path`, `clock`, `append_clock`) or legacy flat keys."""
    if "path" in pp or "clock" in pp:
        path = PathSpec.from_config(pp.get("path"))
        clock = ClockSpec.from_config(pp.get("clock")).validate()
        append = (
            None
            if pp.get("append_clock") is None
            else ClockSpec.from_config(pp.get("append_clock")).validate()
        )
        return path, clock, append
    return legacy_specs(pp)


def to_increment_path_jnp(p: jnp.ndarray, mask: Optional[jnp.ndarray]) -> jnp.ndarray:
    """Increment representation in the model: rows are per-step increments, the path is their
    cumulative sum. Pads are zeroed first so the path freezes after the last real step."""
    if mask is not None:
        p = p * mask[:, None].astype(p.dtype)
    return jnp.cumsum(p, axis=0)


@dataclass
class NormalizationScaler:
    data_min: np.ndarray  # shape (d,)
    data_max: np.ndarray  # shape (d,)


def _ensure_2d(path: np.ndarray) -> np.ndarray:
    if path.ndim == 1:
        return path.reshape(-1, 1)
    return path


def _ensure_1d_clock(clock: np.ndarray) -> np.ndarray:
    c = np.asarray(clock)
    if c.ndim != 1:
        raise ValueError("clock arrays must be 1D")
    return c


def make_uniform_clocks(paths: List[np.ndarray]) -> List[np.ndarray]:
    out = []
    for path in paths:
        p = _ensure_2d(np.asarray(path))
        out.append(np.linspace(0.0, 1.0, p.shape[0]))
    return out


def add_time_channel(paths: List[np.ndarray], time_arrays: List[np.ndarray]) -> List[np.ndarray]:
    if len(paths) != len(time_arrays):
        raise ValueError("paths and time_arrays must have the same length")
    out = []
    for path, t in zip(paths, time_arrays):
        p = _ensure_2d(np.asarray(path))
        t = np.asarray(t)
        t_norm = (t - t[0]) / (t[-1] - t[0] + 1e-12)
        out.append(np.column_stack([t_norm.reshape(-1, 1), p]))
    return out


def add_clock_channel(
    paths: List[np.ndarray],
    clocks: List[np.ndarray],
) -> List[np.ndarray]:
    if len(paths) != len(clocks):
        raise ValueError("paths and clocks must have the same length")
    out = []
    for path, clock in zip(paths, clocks):
        p = _ensure_2d(np.asarray(path))
        c = _ensure_1d_clock(clock)
        if p.shape[0] != c.shape[0]:
            raise ValueError("clock length must match path length")
        out.append(np.column_stack([c.reshape(-1, 1), p]))
    return out


def fit_normalization(paths: List[np.ndarray]) -> NormalizationScaler:
    if len(paths) == 0:
        raise ValueError("Cannot fit scaler on empty paths list")
    paths_np = [_ensure_2d(np.asarray(p)).astype(np.float32) for p in paths]
    stacked = np.vstack(paths_np)
    data_min = stacked.min(axis=0)
    data_max = stacked.max(axis=0)
    return NormalizationScaler(data_min=data_min, data_max=data_max)


def apply_normalization(
    paths: List[np.ndarray],
    scaler: NormalizationScaler,
    normalize_range: NormalizeRange,
) -> List[np.ndarray]:
    out = []
    data_min = scaler.data_min
    data_max = scaler.data_max
    denom = (data_max - data_min) + 1e-12
    for path in paths:
        p = _ensure_2d(np.asarray(path)).astype(np.float32)
        if normalize_range == "[0,1]":
            p_norm = (p - data_min) / denom
        elif normalize_range == "[-1,1]":
            p_norm = 2.0 * (p - data_min) / denom - 1.0
        else:
            raise ValueError(f"normalize_range must be '[0,1]' or '[-1,1]', got {normalize_range}")
        out.append(p_norm)
    return out


def per_sample_normalize_paths(
    paths: List[np.ndarray],
    stat: str = "std",
) -> List[np.ndarray]:
    """Divide each path by one scalar computed over the whole instance, every step and channel together: amplitudes are
    equalised across instances and the channels keep their relative scale. There is no mean subtraction, and each instance
    uses only its own statistic.

    stat: "std" (X.std() over (T, d)) or "rms_dx" (the RMS of the increments, which is what the signature integrates).
    The clock and an appended clock channel are unaffected; level k of the signature scales by 1 / s**k.
    """
    out = []
    for path in paths:
        p = _ensure_2d(np.asarray(path)).astype(np.float64)
        if stat == "std":
            s = float(p.std())
        elif stat == "rms_dx":
            dx = np.diff(p, axis=0)
            s = float(np.sqrt(np.mean(dx**2))) if dx.size > 0 else 0.0
        else:
            raise ValueError(f"per_sample_norm_stat must be 'std' or 'rms_dx', got {stat}")
        out.append((p / (s + 1e-8)).astype(np.float32))
    return out


def derivative_paths(
    paths: List[np.ndarray],
    order: int,
    window: int = 15,
    polyorder: int = 3,
) -> List[np.ndarray]:
    """Replace each path with its Nth Savitzky-Golay derivative (per channel), length-preserving.

    Standard spectral preprocessing: the smooth high-amplitude bulk (a broad bell curve) has a
    near-zero derivative, while a small localized roughness signal (the discriminative shoulder)
    is amplified: shifting the representation from amplitude-dominated (nuisance) to
    roughness-dominated (signal). SavGol fits a local polynomial and differentiates it, so it is
    length-preserving (T -> T, keeps the clock aligned) and less noise-amplifying than raw diff.
    order=0 is a no-op. Replaces, does not append (appending the raw re-injects the bulk nuisance).
    """
    from scipy.signal import savgol_filter

    out = []
    for path in paths:
        p = _ensure_2d(np.asarray(path)).astype(np.float64)
        if order <= 0:
            out.append(p.astype(np.float32))
            continue
        T = p.shape[0]
        w = window if window % 2 == 1 else window + 1  # window must be odd
        w = min(w, T if T % 2 == 1 else T - 1)  # and <= length
        po = min(polyorder, w - 1)  # and polyorder < window
        d = savgol_filter(p, w, po, deriv=order, axis=0)
        out.append(d.astype(np.float32))
    return out


def add_basepoint(paths: List[np.ndarray]) -> List[np.ndarray]:
    out = []
    for path in paths:
        p = _ensure_2d(np.asarray(path))
        zero = np.zeros((1, p.shape[1]), dtype=p.dtype)
        out.append(np.vstack([zero, p]))
    return out


def add_basepoint_with_clock(
    paths: List[np.ndarray],
    clocks: List[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if len(paths) != len(clocks):
        raise ValueError("paths and clocks must have the same length")
    out_paths = []
    out_clocks = []
    for path, clock in zip(paths, clocks):
        p = _ensure_2d(np.asarray(path))
        c = _ensure_1d_clock(clock)
        if p.shape[0] != c.shape[0]:
            raise ValueError("clock length must match path length")
        zero = np.zeros((1, p.shape[1]), dtype=p.dtype)
        out_paths.append(np.vstack([zero, p]))
        out_clocks.append(np.concatenate([c[:1], c], axis=0))
    return out_paths, out_clocks


def preprocess_paths(
    paths: List[np.ndarray],
    time_arrays: Optional[List[np.ndarray]] = None,
    augment_with_time: bool = True,
    normalize: bool = True,
    normalize_mode: NormalizeMode = "train_only",
    normalize_range: NormalizeRange = "[0,1]",
    basepoint_augmentation: bool = True,
    scaler: Optional[NormalizationScaler] = None,
    fit_scaler: bool = True,
    dtype: jnp.dtype = jnp.float32,
) -> Tuple[List[jnp.ndarray], Optional[NormalizationScaler], int]:
    if len(paths) == 0:
        return [], scaler, 0

    paths_np = [_ensure_2d(np.asarray(p)) for p in paths]

    if augment_with_time:
        if time_arrays is None:
            raise ValueError("time_arrays must be provided when augment_with_time=True")
        paths_np = add_time_channel(paths_np, time_arrays)

    if normalize:
        if fit_scaler:
            scaler = fit_normalization(paths_np)
        if scaler is None:
            raise ValueError("scaler must be provided when fit_scaler=False and normalize=True")
        paths_np = apply_normalization(paths_np, scaler, normalize_range)
    else:
        scaler = None

    if basepoint_augmentation:
        paths_np = add_basepoint(paths_np)

    d_path = paths_np[0].shape[1]
    paths_jax = [jnp.array(p, dtype=dtype) for p in paths_np]
    return paths_jax, scaler, d_path


def prepare_paths_np(
    paths: List[np.ndarray],
    clocks: List[np.ndarray],
    path: PathSpec,
    clock: ClockSpec,
    append_clock: Optional[ClockSpec] = None,
    scaler: Optional[NormalizationScaler] = None,
    fit_scaler: bool = True,
    dtype: jnp.dtype = jnp.float32,
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], Optional[NormalizationScaler], int]:
    """Apply the path stage and build the clock for a batch of numpy paths (the static precompute).

    Returns (paths, thetas, scaler, d_path). The clock is built from the representation before the
    channel normalisation; the appended clock channel, when configured, is scaled like a path channel.
    """
    if len(paths) == 0:
        return [], [], scaler, 0
    if len(paths) != len(clocks):
        raise ValueError("paths and clocks must have the same length")
    paths_np = [_ensure_2d(np.asarray(p)) for p in paths]
    clocks_np = [_ensure_1d_clock(c) for c in clocks]
    for p, c in zip(paths_np, clocks_np):
        if p.shape[0] != c.shape[0]:
            raise ValueError("clock length must match path length")
    if path.representation == "derivative":
        paths_np = derivative_paths(
            paths_np,
            int(path.derivative_order),
            int(path.derivative_window),
            int(path.derivative_polyorder),
        )
    elif path.representation == "increment":
        paths_np = [np.cumsum(p, axis=0) for p in paths_np]
    if path.per_sample_normalize:
        paths_np = per_sample_normalize_paths(paths_np, path.per_sample_norm_stat)
    if clock.has_parameters:
        # a clock with parameters is evaluated by the model; the loader supplies the scaled time base
        theta_clocks = [
            build_clock_np(c, p, "time", 1.0, clock.scaling) for p, c in zip(paths_np, clocks_np)
        ]
    else:
        theta_clocks = [
            build_clock_np(c, p, clock.mode, clock.alpha, clock.scaling)
            for p, c in zip(paths_np, clocks_np)
        ]
    if path.clock_channel == "separate":
        if append_clock is None:
            raise ValueError("clock_channel='separate' needs an append_clock spec")
        append_clocks = [
            build_clock_np(c, p, append_clock.mode, append_clock.alpha, path.normalize_range)
            for p, c in zip(paths_np, clocks_np)
        ]
    else:
        append_clocks = None
    if path.normalize:
        if fit_scaler:
            scaler = fit_normalization(paths_np)
        if scaler is None:
            raise ValueError("scaler must be provided when fit_scaler=False and normalize=True")
        paths_np = apply_normalization(paths_np, scaler, path.normalize_range)
    else:
        scaler = None
    if path.basepoint:
        paths_np, theta_clocks = add_basepoint_with_clock(paths_np, theta_clocks)
        if clock.mode == "time" and clock.scaling == "steps":
            # one unit of clock per sample including the basepoint step: theta_i = i on the basepointed path
            theta_clocks = [np.arange(len(p), dtype=np.float64) for p in paths_np]
        if append_clocks is not None:
            append_clocks = [np.concatenate([a[:1], a], axis=0) for a in append_clocks]
    if append_clocks is not None:
        paths_np = [np.column_stack([a.reshape(-1, 1), p]) for p, a in zip(paths_np, append_clocks)]
    elif path.clock_channel == "main":
        paths_np = add_clock_channel(paths_np, theta_clocks)
    d_path = paths_np[0].shape[1]
    paths_jax = [jnp.array(p, dtype=dtype) for p in paths_np]
    clocks_jax = [jnp.array(c, dtype=dtype) for c in theta_clocks]
    return paths_jax, clocks_jax, scaler, d_path


def deduplicate_paths(
    paths: List[np.ndarray],
    labels: List[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    seen = set()
    out_paths = []
    out_labels = []
    for path, label in zip(paths, labels):
        p = np.asarray(path)
        key = (p.shape, p.dtype.str, p.tobytes())
        if key in seen:
            continue
        seen.add(key)
        out_paths.append(path)
        out_labels.append(label)
    return out_paths, out_labels


def deduplicate_splits(
    train_paths: List[np.ndarray],
    train_labels: List[np.ndarray],
    val_paths: List[np.ndarray],
    val_labels: List[np.ndarray],
    test_paths: List[np.ndarray],
    test_labels: List[np.ndarray],
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
]:
    seen = set()

    def _filter(paths, labels):
        out_p, out_l = [], []
        for p, label in zip(paths, labels):
            arr = np.asarray(p)
            key = (arr.shape, arr.dtype.str, arr.tobytes())
            if key in seen:
                continue
            seen.add(key)
            out_p.append(p)
            out_l.append(label)
        return out_p, out_l

    train_paths, train_labels = _filter(train_paths, train_labels)
    val_paths, val_labels = _filter(val_paths, val_labels)
    test_paths, test_labels = _filter(test_paths, test_labels)

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels
