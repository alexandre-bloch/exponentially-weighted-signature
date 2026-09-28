"""S4D reference recurrence, written from the published equations and evaluated step by step in complex128, so the
tests need no third-party code (Gu, Gupta, Goel and Re, NeurIPS 2022; their implementation, state-spaces/s4, applies
the same system as an FFT convolution).

Per channel h and mode n: Abar_n = exp(A_n dt_h), Bbar_n = A_n^-1 (exp(A_n dt_h) - 1) B_n,
x_n[l] = Abar_n x_n[l-1] + Bbar_n u[l] and y[l] = 2 Re(sum_n C_n x_n[l]) + D u[l], the factor 2 coming from their
conjugate symmetry (one mode of each pair is stored).
"""

import numpy as np


def s4d_discrete(A, dt, B=None, disc="zoh"):
    """The discrete transition of the diagonal system: A (H, N2) complex, dt (H,) real.

        zoh       Abar = exp(A dt)                     Bbar = (exp(A dt) - 1) / A * B
        bilinear  Abar = (1 + dt A/2) / (1 - dt A/2)   Bbar = (1 - dt A/2)^-1 dt B

    Only ZOH is a matrix exponential, so only under ZOH is Method A exact. Returns Abar and Bbar, (H, N2) complex128.
    """
    A = np.asarray(A, np.complex128)
    dt = np.asarray(dt, np.float64)[:, None]
    B = np.ones_like(A) if B is None else np.asarray(B, np.complex128)
    dtA = A * dt
    if disc == "zoh":
        Abar = np.exp(dtA)
        # (exp(dtA) - 1)/A * B. A is never 0 in this parametrisation (Re A = -exp(log_A_real) < 0), but keep the
        # expm1 form: it is the accurate one when |dtA| is small, which is exactly the dt_min end of their init.
        Bbar = np.expm1(dtA) / A * B
    elif disc == "bilinear":
        Abar = (1.0 + dtA / 2.0) / (1.0 - dtA / 2.0)
        Bbar = B * dt / (1.0 - dtA / 2.0)
    else:
        raise ValueError(f"disc must be 'zoh' or 'bilinear', got {disc!r}")
    return Abar, Bbar


def s4d_forward(A, dt, C, D, u, B=None, disc="zoh"):
    """Sequential reference forward. u (batch,H,L) real -> y (batch,H,L) real, float64 throughout.

    A (H,N2) complex, dt (H,) real, C (H,N2) complex, D (H,) real or None.
    """
    A = np.asarray(A, np.complex128)
    C = np.asarray(C, np.complex128)
    u = np.asarray(u, np.float64)
    H, N2 = A.shape
    bsz, Hu, L = u.shape
    assert Hu == H, f"u has {Hu} channels, A has {H}"

    Abar, Bbar = s4d_discrete(A, dt, B, disc)
    x = np.zeros((bsz, H, N2), np.complex128)
    y = np.zeros((bsz, H, L), np.float64)
    for l in range(L):
        # the input at step l enters the state at step l: their convolution includes the j=0 tap
        x = Abar[None] * x + Bbar[None] * u[:, :, l][:, :, None]
        y[:, :, l] = 2.0 * np.real(np.sum(C[None] * x, axis=-1))
    if D is not None:
        y = y + np.asarray(D, np.float64)[None, :, None] * u
    return y
