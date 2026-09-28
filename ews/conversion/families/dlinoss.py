"""D-LinOSS -> EWS. Methods A and B.

D-LinOSS is LinOSS with damping: the same oscillator system with a decay term, so the transition and the
continuous generator both pick up a damping coefficient. Method B inverts the family's own transition and is
exact. Method A reads the continuous system off the parameters and carries the family's discretisation error,
exactly as for LinOSS-IM: measured there at O(dt^2).

Equations from the D-LinOSS paper; the reference implementation (damped-linoss, MIT) is read from a read-only
clone and never vendored here.
"""

import numpy as np

from ews.conversion.base import ConversionResult


def dlinoss_transition(Acal, G, Delta):
    """The damped LinOSS-IMEX transition, per mode on (z, y), transcribed from their own
    `make_damped_linoss_imex_recurrence` (damped-linoss a396997, linoss/models/LinOSS.py:134):

        S    = 1 + Delta*G
        M_11 = 1/S                      M_12 = -Delta*Acal/S
        M_21 = Delta/S                  M_22 = 1 - Delta^2*Acal/S

    Damping is only implemented for IMEX in their code: the IM + damping branch raises NotImplementedError, and
    `G_diag` is None there.
    """
    Acal = np.asarray(Acal, np.float64)
    G = np.asarray(G, np.float64)
    Delta = np.asarray(Delta, np.float64)
    P = Acal.shape[0]
    M = np.zeros((2 * P, 2 * P), np.float64)
    for k in range(P):
        d, a, g = Delta[k], Acal[k], G[k]
        S = 1.0 + d * g
        M[2 * k : 2 * k + 2, 2 * k : 2 * k + 2] = np.array(
            [[1.0 / S, -d * a / S], [d / S, 1.0 - d * d * a / S]]
        )
    return M


def dlinoss_continuous_generator(Acal, G, Delta):
    """The damped oscillator: z' = -Acal y - G z + Bu, y' = z, realified into the 4-per-mode layout
    (index = 4k + 2*(re/im) + (z/y)) and returned in EWS's exp(-A dtheta) convention with dtheta = 1."""
    Acal = np.asarray(Acal, np.float64)
    G = np.asarray(G, np.float64)
    Delta = np.asarray(Delta, np.float64)
    P = Acal.shape[0]
    A = np.zeros((4 * P, 4 * P), np.float64)
    for k in range(P):
        for part in (0, 1):
            iz, iy = 4 * k + 2 * part, 4 * k + 2 * part + 1
            A[iz, iz] = G[k] * Delta[k]  # damping on z
            A[iz, iy] = Acal[k] * Delta[k]
            A[iy, iz] = -Delta[k]
    return A


def to_ews(layer, method="B", *, check_tol=1e-10, dtheta=1.0):
    """D-LinOSS = damped LinOSS-IMEX. Damping is IMEX-only in their code: the IM + damping branch raises
    NotImplementedError and G_diag is None there, so there is no damped-IM variant to convert."""
    import jax

    from ews.conversion.families.linoss_imex import _realify_mode
    from ews.conversion.method_b import from_discrete

    if method != "B":
        raise NotImplementedError(
            "D-LinOSS Method A is not implemented: its transition is not exp of the damped oscillator generator, "
            "so the continuous read-off carries the family's discretisation error -- measured O(dt^2), ratio 3.96."
        )
    if not getattr(layer, "damping", False):
        raise ValueError(
            "expected a damped layer (damping=True); an undamped IMEX layer is families/linoss_imex"
        )

    Acal = np.asarray(layer.A_diag, np.float64)
    G = np.asarray(jax.nn.relu(layer.G_diag), np.float64)
    Delta = np.asarray(jax.nn.sigmoid(layer.steps), np.float64)
    Bc = np.asarray(layer.B[..., 0], np.float64) + 1j * np.asarray(layer.B[..., 1], np.float64)
    Cc = np.asarray(layer.C[..., 0], np.float64) + 1j * np.asarray(layer.C[..., 1], np.float64)
    D = np.asarray(getattr(layer, "D", np.zeros(Bc.shape[1])), np.float64)
    P, H = Bc.shape
    m = 4 * P
    A = np.zeros((m, m))
    B = np.zeros((m, H))
    C = np.zeros((H, m))
    diags = []
    for k in range(P):
        d, a, g = Delta[k], Acal[k], G[k]
        S = 1.0 + d * g
        M = np.array([[1.0 / S, -d * a / S], [d / S, 1.0 - d * d * a / S]])
        Bbar = np.array([[d / S], [d * d / S]])  # their F1, F2 per unit (Bu)
        A2, B2, dg = from_discrete(M, Bbar, dtheta=dtheta, check_tol=check_tol)
        diags.append(dg)
        _realify_mode(A2, B2[0, 0] * Bc[k], B2[1, 0] * Bc[k], None, Cc[:, k], k, A, B, C)
    return ConversionResult(
        A=A,
        B=B,
        C=C,
        D=D,
        m=m,
        family="dlinoss",
        method="B",
        dtheta=float(dtheta),
        diag=dict(
            P=P,
            H=H,
            cond_G_max=max(x["cond_G"] for x in diags),
            roundtrip_M_max=max(x["roundtrip_M"] for x in diags),
            max_abs_im_dtA=max(x["max_abs_im_dtA"] for x in diags),
        ),
    )
