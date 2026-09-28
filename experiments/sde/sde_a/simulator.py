"""Coupled oscillatory SDE for same-time regression.

    dX1 = (alpha sin(omega X2) - beta X1) dt + sigma dW1
    dX2 = (alpha cos(omega X1) - beta X2) dt + sigma dW2

with alpha = 3, omega = 1, beta = 0.5, sigma = 0.4 and X0 = (0.5, 0.5), simulated by the
Euler-Maruyama scheme. The input path is the driving Brownian motion (W1, W2), the clock is the
time grid, and the target is X1 at every step.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def simulate_sde(
    n_sequences: int,
    T: float,
    n_steps: int,
    seed: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Returns (paths, clocks, targets): (n_steps, 2) Brownian paths, (n_steps,) time grids and
    (n_steps,) X1 trajectories; sequence i uses numpy generator seed + i."""
    alpha1, alpha2 = 3.0, 3.0
    omega = 1.0
    beta1, beta2 = 0.5, 0.5
    sigma1, sigma2 = 0.4, 0.4
    X0 = np.array([0.5, 0.5])

    t_grid = np.linspace(0, T, n_steps)
    dt = T / (n_steps - 1)

    paths: List[np.ndarray] = []
    clocks: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for i in range(n_sequences):
        rng = np.random.default_rng(seed=seed + i)
        dW1 = rng.normal(0, np.sqrt(dt), n_steps - 1)
        dW2 = rng.normal(0, np.sqrt(dt), n_steps - 1)

        W1 = np.concatenate([[0], np.cumsum(dW1)])
        W2 = np.concatenate([[0], np.cumsum(dW2)])

        X = np.zeros((n_steps, 2))
        X[0] = X0
        for j in range(n_steps - 1):
            X1_curr, X2_curr = X[j]
            drift1 = alpha1 * np.sin(omega * X2_curr) - beta1 * X1_curr
            drift2 = alpha2 * np.cos(omega * X1_curr) - beta2 * X2_curr
            X[j + 1, 0] = X1_curr + drift1 * dt + sigma1 * dW1[j]
            X[j + 1, 1] = X2_curr + drift2 * dt + sigma2 * dW2[j]

        paths.append(np.column_stack([W1, W2]))
        clocks.append(t_grid.copy())
        targets.append(X[:, 0])

    return paths, clocks, targets
