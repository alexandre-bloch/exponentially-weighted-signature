"""Method A: read the continuous generator off a family's parameters.

The EWS steps by exp(-A dtheta), so the read-off is exact when the family's own transition is that exponential, and
otherwise differs from it by the family's discretisation error:

    ZOH (S5, S4D zoh)                    exact
    bilinear (S4, S4D bilinear)          O(dt^3)
    LinOSS "IM" (LinOSS-IM, D-LinOSS)    O(dt^2): implicit, but not the symmetric midpoint rule
    LRU                                  does not exist (natively discrete)

Method B (method_b.py) converts every family exactly, LinOSS-IMEX included. `discretisation_residual()` reports how
the read-off's gap scales with dt.
"""

import numpy as np
import scipy.linalg


def linoss_continuous_generator(Acal, Delta):
    """LinOSS's own continuous system, per mode: z' = -Acal y + Bu, y' = z.

    Realified into the same layout Method B uses (`_convert_core.py`):

        index = 4k + 2*(re=0 / im=1) + (z=0 / y=1)

    so each mode contributes four real states, not two: z and y are complex, and Acal and Delta are real, so the
    generator acts identically on the real and imaginary parts. Returned in the EWS convention, where the layer
    applies exp(-A dtheta) with dtheta = 1, hence A_ews = -G * Delta.
    """
    Acal = np.asarray(Acal, np.float64)
    Delta = np.asarray(Delta, np.float64)
    P = Acal.shape[0]
    A = np.zeros((4 * P, 4 * P), np.float64)
    for k in range(P):
        for part in (0, 1):  # 0 = real, 1 = imaginary
            iz = 4 * k + 2 * part + 0
            iy = 4 * k + 2 * part + 1
            A[iz, iy] = Acal[k] * Delta[k]  # -G scaled by Delta, in EWS's exp(-A) convention
            A[iy, iz] = -Delta[k]
    return A


def linoss_im_transition(Acal, Delta):
    """The vendored LinOSS-IM discrete transition per mode: S * [[1, -Delta*Acal], [Delta, 1]] on (z, y)."""
    Acal = np.asarray(Acal, np.float64)
    Delta = np.asarray(Delta, np.float64)
    S = 1.0 / (1.0 + Delta**2 * Acal)
    P = Acal.shape[0]
    M = np.zeros((2 * P, 2 * P), np.float64)
    for k in range(P):
        i = 2 * k
        M[i, i] = S[k]
        M[i, i + 1] = -Delta[k] * Acal[k] * S[k]
        M[i + 1, i] = Delta[k] * S[k]
        M[i + 1, i + 1] = S[k]
    return M


def discretisation_residual(Acal, Delta_scales=(1.0, 0.5, 0.25, 0.125)):
    """How far the continuous read-off is from the family's own transition, as dt shrinks.

    Returns [(scale, max |exp(-A_ews) - Abar|)]. An implicit-midpoint family should show the gap falling like
    scale^3; a ZOH family should sit at float64 noise for every scale.
    """
    Acal = np.asarray(Acal, np.float64)
    out = []
    for s in Delta_scales:
        Delta = np.full_like(Acal, 0.5) * s
        A = linoss_continuous_generator(Acal, Delta)
        M_exp = scipy.linalg.expm(-A)
        M_fam = linoss_im_transition(Acal, Delta)
        out.append((s, float(np.max(np.abs(M_exp - M_fam)))))
    return out
