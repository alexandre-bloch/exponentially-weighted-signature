"""Brownian paths and EWS targets for the expressivity-gap experiment.

Each path is (W1, W2) on a uniform grid over [0, T]; the clock is the time grid. The target is one
component of the depth-2 EWS of the time-augmented path (t, W1, W2) under a fixed operator
A_true, with the first timestep dropped (n_steps - 1 values per path).

Flat depth-2 signature layout for d = 3 (13 values): index 0 is the level-0 constant, 1-3 the
level-1 terms (t, W1, W2), 4-12 the level-2 terms in row-major order, so index 11 = (2, 1) is the
W2 x W1 cross term used as the default target.

Target operators: "EWS" is the real part of a unitary similarity of diag(-0.5 +/- 5.2i, 0.8) (taking
the real part changes the spectrum: its eigenvalues are -0.44 +/- 3.22i and 0.68), "EFM" is
diag(0.5, 0.3, 0.8), "Sig" is the zero matrix (the classical signature).
"""

from __future__ import annotations

from typing import List, Tuple

import jax.numpy as jnp
import numpy as np

from ews.layer import EWS


def _construct_A_general() -> np.ndarray:
    """Real part of a random unitary similarity (numpy seed 42) of diag(-0.5 +/- 5.2j, 0.8); the real
    part has eigenvalues -0.44 +/- 3.22j and 0.68."""
    eigenvalues = np.array([-0.5 + 5.2j, -0.5 - 5.2j, 0.8], dtype=complex)
    np.random.seed(42)
    n = len(eigenvalues)
    Lambda = np.diag(eigenvalues)
    random_matrix = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    Q, _ = np.linalg.qr(random_matrix)
    A = Q @ Lambda @ Q.conj().T
    return np.real(A).astype(np.float64)


def _construct_A_general_pos() -> np.ndarray:
    """Same construction with eigenvalues [+0.5 +/- 5.2j, 0.8] (a growing oscillation)."""
    eigenvalues = np.array([0.5 + 5.2j, 0.5 - 5.2j, 0.8], dtype=complex)
    np.random.seed(42)
    n = len(eigenvalues)
    Lambda = np.diag(eigenvalues)
    random_matrix = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    Q, _ = np.linalg.qr(random_matrix)
    A = Q @ Lambda @ Q.conj().T
    return np.real(A).astype(np.float64)


def _construct_A_diagonal() -> np.ndarray:
    """diag([0.5, 0.3, 0.8])."""
    return np.diag([0.5, 0.3, 0.8]).astype(np.float64)


def get_A_true(name: str) -> np.ndarray:
    """The fixed operator that generates the target."""
    if name == "EWS":
        return _construct_A_general()
    if name == "EWS_pos":
        return _construct_A_general_pos()
    if name == "EFM":
        return _construct_A_diagonal()
    if name == "Sig":
        return np.zeros((3, 3), dtype=np.float64)
    raise ValueError(f"A_true must be 'EWS', 'EWS_pos', 'EFM', or 'Sig', got {name}")


def simulate_bm_paths(
    n_sequences: int,
    T: float,
    n_steps: int,
    seed: int,
    path_batch_size: int = 3,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Brownian paths (W1, W2) drawn in batches of `path_batch_size`, each batch from its own
    generator seeded by seed + batch_start; the clock is the time grid."""
    dt = T / (n_steps - 1)
    t_grid = np.linspace(0, T, n_steps)
    paths: List[np.ndarray] = []
    clocks: List[np.ndarray] = []
    i = 0
    while i < n_sequences:
        bsz = min(path_batch_size, n_sequences - i)
        rng = np.random.default_rng(seed + i)
        dW = rng.standard_normal((bsz, n_steps - 1, 2)) * np.sqrt(dt)
        W = np.concatenate([np.zeros((bsz, 1, 2)), np.cumsum(dW, axis=1)], axis=1)
        for j in range(bsz):
            paths.append(np.column_stack([W[j, :, 0], W[j, :, 1]]).astype(np.float32))
            clocks.append(t_grid.copy())
        i += bsz
    return paths, clocks


def compute_ews_targets(
    paths: List[np.ndarray],
    clocks: List[np.ndarray],
    A_np: np.ndarray,
    target_component: int,
    N: int = 2,
    M: int = 10,
    batch_size: int = 50,
    van_loan_expm: str = "exact",
    chen_expm: str = "exact",
) -> List[np.ndarray]:
    """One component of the depth-N EWS prefix signature of each path, in float64, with the first
    timestep dropped (n_steps - 1 values per path)."""
    ews_layer = EWS(
        A=jnp.array(A_np, dtype=jnp.float64),
        N=N,
        M=M,
        dtype=jnp.float64,
        reweighting_mode="displacement_uniform",
        van_loan_expm=van_loan_expm,
        chen_expm=chen_expm,
    )
    targets: List[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        batch = jnp.stack(
            [jnp.array(p, dtype=jnp.float64) for p in paths[start : start + batch_size]], axis=0
        )
        batch_clocks = jnp.stack(
            [jnp.array(c, dtype=jnp.float64) for c in clocks[start : start + batch_size]], axis=0
        )
        features = ews_layer(batch, batch_clocks, return_terminal=False, output_format="flat")
        features_np = np.array(features, dtype=np.float32)
        for j in range(features_np.shape[0]):
            targets.append(features_np[j][:, target_component])
    return targets
