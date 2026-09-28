"""LinOSS-IMEX -> EWS, Method B.

Per mode on (z, y) (vendored/linoss.py::apply_linoss_imex):

    z_t = z_{t-1} - Delta Acal y_{t-1} + Delta (Bu)_t,   y_t = Delta z_t + y_{t-1},   output Re(C @ y)
    so M = [[1, -Delta Acal], [Delta, 1 - Delta^2 Acal]] and Bbar = [Delta, Delta^2]^T (Bu)_t

Every mode lies on the unit circle, which does not prevent a principal logarithm. Method A is not implemented: IMEX is
not the matrix exponential of the oscillator generator.
"""

import numpy as np

from ews.conversion.base import ConversionResult
from ews.conversion.method_b import from_discrete


def _realify_mode(A2, Bz, By, Cz, Cy, k, A, B, C):
    """Write one complex mode into the 4-per-mode real layout: index = 4k + 2*(re/im) + (z/y)."""
    for part in (0, 1):  # 0 = real component, 1 = imaginary
        iz, iy = 4 * k + 2 * part + 0, 4 * k + 2 * part + 1
        A[iz, iz] = A2[0, 0]
        A[iz, iy] = A2[0, 1]
        A[iy, iz] = A2[1, 0]
        A[iy, iy] = A2[1, 1]
    B[4 * k + 0, :] = Bz.real
    B[4 * k + 1, :] = By.real
    B[4 * k + 2, :] = Bz.imag
    B[4 * k + 3, :] = By.imag
    # output = Re(C y) = Re(C) Re(y) - Im(C) Im(y); z columns stay zero
    C[:, 4 * k + 1] = Cy.real
    C[:, 4 * k + 3] = -Cy.imag


def to_ews(layer, method="B", *, check_tol=1e-10, dtheta=1.0):
    import jax

    if method != "B":
        raise NotImplementedError(
            "LinOSS-IMEX Method A is not implemented: IMEX is not exp of the oscillator generator, so the "
            "continuous read-off would carry the family's discretisation error (O(dt^2) for LinOSS-IM)."
        )
    if getattr(layer, "discretization", None) != "IMEX":
        raise ValueError(
            f"expected an IMEX layer, got discretization={getattr(layer, 'discretization', None)!r}"
        )

    Acal = np.asarray(jax.nn.relu(layer.A_diag), np.float64)
    Delta = np.asarray(jax.nn.sigmoid(layer.steps), np.float64)
    Bc = np.asarray(layer.B[..., 0], np.float64) + 1j * np.asarray(
        layer.B[..., 1], np.float64
    )  # (P, H)
    Cc = np.asarray(layer.C[..., 0], np.float64) + 1j * np.asarray(
        layer.C[..., 1], np.float64
    )  # (H, P)
    D = np.asarray(layer.D, np.float64)
    P, H = Bc.shape
    m = 4 * P
    A = np.zeros((m, m))
    B = np.zeros((m, H))
    C = np.zeros((H, m))
    diags = []
    for k in range(P):
        d, a = Delta[k], Acal[k]
        M = np.array([[1.0, -d * a], [d, 1.0 - d * d * a]])
        Bbar = np.array([[d], [d * d]])  # per unit (Bu); the complex B factors out
        A2, B2, dg = from_discrete(M, Bbar, dtheta=dtheta, check_tol=check_tol)
        diags.append(dg)
        _realify_mode(A2, B2[0, 0] * Bc[k], B2[1, 0] * Bc[k], None, Cc[:, k], k, A, B, C)
    return ConversionResult(
        A=A,
        B=B,
        C=C,
        D=D,
        m=m,
        family="linoss_imex",
        method="B",
        dtheta=float(dtheta),
        diag=dict(
            P=P,
            H=H,
            cond_G_max=max(d["cond_G"] for d in diags),
            roundtrip_M_max=max(d["roundtrip_M"] for d in diags),
            max_abs_im_dtA=max(d["max_abs_im_dtA"] for d in diags),
        ),
    )
