"""Mamba-1 and Mamba-2 reference recurrences, written from the published equations; the tests use them, and verify.py
also compares against mamba_ssm itself.

Mamba's Delta_t, B_t and C_t depend on the input but A is fixed, so Abar_t = exp(Delta_t A) varies only through the
step size, which is what an EWS clock is: one EWS per channel (Mamba-1) or per head (Mamba-2), with a data-dependent
clock.

    Mamba-1 (S6), per channel, state N:  Abar_t = exp(Delta_t A), Bbar_t = Delta_t B_t,
                                         x_t = Abar_t x_{t-1} + Bbar_t u_t, y_t = C_t . x_t (+ D u_t)
    Mamba-2, per head:                   the same with A = a I, one decay rate per head
"""

import numpy as np


def mamba1_channel(A, Delta, B, C, u, D=None):
    """One Mamba-1 channel. A (N,) fixed; Delta (L,), B (L,N), C (L,N) input-dependent; u (L,)."""
    A = np.asarray(A, np.float64)
    Delta = np.asarray(Delta, np.float64)
    B = np.asarray(B, np.float64)
    C = np.asarray(C, np.float64)
    u = np.asarray(u, np.float64)
    x = np.zeros(A.shape[0], np.float64)
    ys = []
    for t in range(u.shape[0]):
        x = np.exp(Delta[t] * A) * x + Delta[t] * B[t] * u[t]
        ys.append(C[t] @ x)
    y = np.array(ys)
    return y if D is None else y + np.asarray(D, np.float64) * u


def mamba2_head(a, Delta, B, C, u, D=None):
    """One Mamba-2 head. `a` is a scalar decay shared by the head (the SSD restriction A = a I)."""
    a = float(a)
    Delta = np.asarray(Delta, np.float64)
    B = np.asarray(B, np.float64)
    C = np.asarray(C, np.float64)
    u = np.asarray(u, np.float64)
    N = B.shape[1]
    x = np.zeros(N, np.float64)
    ys = []
    for t in range(u.shape[0]):
        x = np.exp(Delta[t] * a) * x + Delta[t] * B[t] * u[t]
        ys.append(C[t] @ x)
    y = np.array(ys)
    return y if D is None else y + np.asarray(D, np.float64) * u
