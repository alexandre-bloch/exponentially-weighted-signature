"""Signals, exact targets and the oracle check of the oscillatory-kernel experiment.

Everything here is numpy in float64 and independent of the EWS code: the targets are the causal
exponentially-windowed Fourier coefficients of a piecewise-constant signal, computed by the exact
zero-order-hold complex recurrence, and the oracle recomputes them by quadrature.

Grid: T samples on (0, 1], t_i = i / T, dt = 1 / T. The signal takes the value x_i on
(t_{i-1}, t_i]. The path fed to the models is X_i = (t_i, sum_{k<=i} x_k dt) with X_0 = (0, 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

T_DEFAULT = 256
BANK_HZ = (2.0, 4.0, 22.0, 27.0)
LOW_PLANES = (0, 1)
HIGH_PLANES = (2, 3)
DAMPING = 5.0
NOISE_SIGMA = 0.05
P_NO_HIGH = 0.15
OVERSAMPLE = 64

TARGET_DIMS = {"T1": 2 * len(BANK_HZ), "T2": 1, "T3": 1}
TARGET_KINDS = {"T1": "regression", "T2": "regression", "T3": "classification"}
# T1 output layout: (c_1, s_1, c_2, s_2, ...): one (Re, Im) pair per bank frequency, in bank order.
T1_NAMES = [f"{kind}_{f:g}Hz" for f in BANK_HZ for kind in ("c", "s")]


@dataclass
class Signals:
    t: np.ndarray  # (T,)
    x: np.ndarray  # (n, T)
    active: np.ndarray  # (n, T) bool: a high-band burst is on at t_i
    schedule: List[Dict] = field(default_factory=list)


def generate_signals(n: int, rng: np.random.Generator, T: int = T_DEFAULT) -> Signals:
    """Draw n signals: one or two low cosines (1-5 Hz), zero to two high bursts (20-30 Hz, each on for 0.15-0.4 of
    the window) and Gaussian noise. Returns the signals, the burst-active flags and the drawn components."""
    t = np.arange(1, T + 1, dtype=np.float64) / T
    x = np.zeros((n, T), dtype=np.float64)
    active = np.zeros((n, T), dtype=bool)
    schedule = []
    for i in range(n):
        comps = []
        k_low = int(rng.integers(1, 3))
        for _ in range(k_low):
            f = float(rng.uniform(1.0, 5.0))
            a = float(rng.uniform(0.5, 1.5))
            phi = float(rng.uniform(0.0, 2.0 * np.pi))
            x[i] += a * np.cos(2.0 * np.pi * f * t + phi)
            comps.append({"band": "low", "f": f, "a": a, "phi": phi})
        k_high = 0 if rng.uniform() < P_NO_HIGH else int(rng.integers(1, 3))
        for _ in range(k_high):
            f = float(rng.uniform(20.0, 30.0))
            a = float(rng.uniform(0.5, 1.5))
            phi = float(rng.uniform(0.0, 2.0 * np.pi))
            t_on = float(rng.uniform(0.0, 0.6))
            t_off = min(t_on + float(rng.uniform(0.15, 0.4)), 1.0)
            on = (t >= t_on) & (t <= t_off)
            x[i] += a * on * np.cos(2.0 * np.pi * f * t + phi)
            active[i] |= on
            comps.append({"band": "high", "f": f, "a": a, "phi": phi, "t_on": t_on, "t_off": t_off})
        x[i] += NOISE_SIGMA * rng.standard_normal(T)
        schedule.append({"components": comps})
    return Signals(t=t, x=x, active=active, schedule=schedule)


def _phi1(w: np.ndarray) -> np.ndarray:
    """phi_1(w) = (e^w - 1) / w, with the removable singularity at 0."""
    w = np.asarray(w, dtype=np.complex128)
    out = np.expm1(w) / np.where(w == 0, 1.0, w)
    return np.where(w == 0, 1.0, out)


def windowed_fourier(x: np.ndarray, bank_hz=BANK_HZ, damping: float = DAMPING) -> np.ndarray:
    """Exact ZOH recurrence: z_{j,i} = e^{-lam_j dt} z_{j,i-1} + dt phi_1(-lam_j dt) x_i,
    lam_j = a - i omega_j. Returns z of shape (n, T, J), complex128."""
    x = np.asarray(x, dtype=np.float64)
    n, T = x.shape
    dt = 1.0 / T
    lam = damping - 1j * 2.0 * np.pi * np.asarray(bank_hz, dtype=np.float64)
    decay = np.exp(-lam * dt)
    gain = dt * _phi1(-lam * dt)
    z = np.zeros((n, T, len(bank_hz)), dtype=np.complex128)
    z_prev = np.zeros((n, len(bank_hz)), dtype=np.complex128)
    for i in range(T):
        z_prev = decay[None, :] * z_prev + gain[None, :] * x[:, i][:, None]
        z[:, i] = z_prev
    return z


def make_targets(sig: Signals, bank_hz=BANK_HZ, damping: float = DAMPING) -> Dict[str, np.ndarray]:
    """T1 (n, T, 2J): (c_j, s_j) pairs; T2 (n, T, 1): high-band energy; T3 (n, T, 1): burst flag."""
    z = windowed_fourier(sig.x, bank_hz, damping)
    n, T, J = z.shape
    t1 = np.empty((n, T, 2 * J), dtype=np.float64)
    t1[:, :, 0::2] = z.real
    t1[:, :, 1::2] = z.imag
    high = list(HIGH_PLANES)
    t2 = (np.abs(z[:, :, high]) ** 2).sum(axis=2)[:, :, None]
    t3 = sig.active.astype(np.float64)[:, :, None]
    return {"T1": t1, "T2": t2, "T3": t3}


def make_paths(sig: Signals) -> np.ndarray:
    """(n, T + 1, 2): rows (t_i, sum_{k<=i} x_k dt) for i = 0..T, X_0 = (0, 0)."""
    n, T = sig.x.shape
    dt = 1.0 / T
    paths = np.zeros((n, T + 1, 2), dtype=np.float64)
    paths[:, 1:, 0] = sig.t[None, :]
    paths[:, 1:, 1] = np.cumsum(sig.x * dt, axis=1)
    return paths


def make_clocks(sig: Signals) -> np.ndarray:
    n, T = sig.x.shape
    clocks = np.zeros((n, T + 1), dtype=np.float64)
    clocks[:, 1:] = sig.t[None, :]
    return clocks


def oracle_quadrature(
    x: np.ndarray, bank_hz=BANK_HZ, damping: float = DAMPING, oversample: int = OVERSAMPLE
) -> np.ndarray:
    """Recompute z by quadrature on an `oversample`-times finer grid of the same piecewise-constant
    signal, without the phi_1 closed form and without the recurrence.

    Each interval integral I_k = int_{t_{k-1}}^{t_k} e^{-lam (t_k - u)} x_k du is a composite
    Simpson sum over `oversample` sub-intervals; z_i = sum_{k<=i} e^{-lam (t_i - t_k)} I_k is a
    direct sum over the intervals. Returns (n, T, J) complex128.
    """
    x = np.asarray(x, dtype=np.float64)
    n, T = x.shape
    dt = 1.0 / T
    lam = damping - 1j * 2.0 * np.pi * np.asarray(bank_hz, dtype=np.float64)  # (J,)
    delta = dt / oversample
    s = np.arange(oversample + 1) * delta  # offsets from the interval's left end
    w = np.full(oversample + 1, 2.0)
    w[1::2] = 4.0
    w[0] = w[-1] = 1.0
    w *= delta / 3.0
    kernel = np.exp(-lam[:, None] * (dt - s[None, :]))  # (J, S): e^{-lam (t_k - u)}
    unit_integral = (kernel * w[None, :]).sum(axis=1)  # (J,)
    I = x[:, :, None] * unit_integral[None, None, :]  # (n, T, J)
    t = np.arange(1, T + 1) * dt
    lag = t[:, None] - t[None, :]  # (T_i, T_k)
    mask = lag >= 0
    z = np.empty((n, T, len(bank_hz)), dtype=np.complex128)
    for j in range(len(bank_hz)):
        E = np.where(mask, np.exp(-lam[j] * np.where(mask, lag, 0.0)), 0.0)  # (T_i, T_k)
        z[:, :, j] = I[:, :, j] @ E.T
    return z


def oracle_check(x: np.ndarray, n_samples: int, rng: np.random.Generator) -> Dict:
    """Max relative error between the exact recurrence and the quadrature on random samples.
    Relative error per (sample, frequency) = max_i |z_quad - z_exact| / max_i |z_exact|."""
    idx = rng.choice(x.shape[0], size=min(n_samples, x.shape[0]), replace=False)
    z_exact = windowed_fourier(x[idx])
    z_quad = oracle_quadrature(x[idx])
    err = np.abs(z_quad - z_exact).max(axis=1)  # (n, J)
    scale = np.abs(z_exact).max(axis=1)
    rel = err / scale
    return {
        "n_samples": int(len(idx)),
        "sample_indices": [int(i) for i in idx],
        "oversample": OVERSAMPLE,
        "max_rel_err": float(rel.max()),
        "max_rel_err_per_freq": {f"{f:g}Hz": float(rel[:, j].max()) for j, f in enumerate(BANK_HZ)},
        "pass": bool(rel.max() <= 1e-6),
    }


def oscillation_count(bank_hz=BANK_HZ, damping: float = DAMPING) -> Dict:
    """High-band oscillations inside the effective window 3 / a, and the implied EFM bound."""
    window = 3.0 / damping
    counts = {f"{bank_hz[j]:g}Hz": bank_hz[j] * window for j in HIGH_PLANES}
    max_count = max(counts.values())
    return {
        "effective_window_s": window,
        "oscillations_in_window": counts,
        "efm_bound_nm_at_least": int(np.ceil(2.0 * max_count)),
    }


def feature_count(m: int, n: int) -> int:
    return int(sum(m**k for k in range(1, n + 1)))
