"""S4 (NPLR/DPLR) -> depth-1 EWS, Methods A and B.

Equations from Gu, Goel and Re, "Efficiently Modeling Long Sequences with Structured State Spaces" (ICLR 2022); the
reference implementation state-spaces/s4 (Apache 2.0) is read from a clone, not vendored. S4's DPLR kernel always uses
the bilinear transform (it never reads `disc`), so there is no ZOH variant.

The conversion starts from their explicit discrete system, `_setup_step(mode='dense')`: per channel
x[l] = dA x[l-1] + dB u[l] and y[l] = Re(dC . x[l]), with x in C^n the conjugate-extended state (hence Re, not 2 Re).
Method B takes a = -logm(dA) / dtheta and b = G_VL(a)^-1 dB, exact by construction. Method A inverts the bilinear
transform, Ac = (2/dt)(dA - I)(dA + I)^-1 and Bc = (I - dt/2 Ac) dB / dt; since the EWS steps by exp(-A dtheta) and S4
by the Pade form, its residual is S4's own O(dt^3) discretisation gap, not a conversion error.

The complex state of each channel is realified in blocks, [Re x | Im x], so M becomes [[Re M, -Im M], [Im M, Re M]].
"""

import numpy as np
import scipy.linalg

from ews.conversion.base import ConversionResult


def _G_VL(A, dtheta=1.0):
    """G_VL(A) = int_0^dtheta exp(-sA) ds, via the Van Loan augmented exponential.

    The (1,2) block of expm([[-A*dtheta, I*dtheta], [0, 0]]). Always defined, never A^-1 (I - e^-A), which is
    singular exactly at the undamped modes.
    """
    n = A.shape[0]
    M = np.zeros((2 * n, 2 * n), A.dtype)
    M[:n, :n] = -A * dtheta
    M[:n, n:] = np.eye(n) * dtheta
    return scipy.linalg.expm(M)[:n, n:]


def _realify(M):
    """Complex (n,n) -> real (2n,2n) in the block layout [Re; Im]."""
    return np.block([[M.real, -M.imag], [M.imag, M.real]])


def to_ews_arrays(dA, dB, dC, dt, *, method="A", check_tol=1e-9, dtheta=1.0):
    """Convert S4's explicit discrete system to depth-1 EWS weights.

    dA (H, n, n) complex, dB (H, n) complex, dC (H, n) complex, dt (H,) real.
    Returns a ConversionResult with m = H * 2n.
    """
    if method not in ("A", "B"):
        raise NotImplementedError(
            f"S4 implements Method A (continuous read-off, via the exact Cayley inverse of its bilinear "
            f"discretisation) and Method B (matrix logarithm); got {method!r}."
        )
    dA = np.asarray(dA, np.complex128)
    dB = np.asarray(dB, np.complex128)
    dC = np.asarray(dC, np.complex128)
    dt = np.asarray(dt, np.float64).reshape(-1)
    H, n, n2 = dA.shape
    assert n == n2, f"dA must be square per channel, got {dA.shape}"
    assert dB.shape == (H, n) and dC.shape == (H, n), (
        f"dB {dB.shape} dC {dC.shape} vs dA {dA.shape}"
    )
    assert dt.shape == (H,), f"dt {dt.shape} != ({H},)"

    m = H * 2 * n
    A_ews = np.zeros((m, m), np.float64)
    B_ews = np.zeros((m, H), np.float64)
    C_ews = np.zeros((H, m), np.float64)
    I = np.eye(n)
    diag = {
        "H": H,
        "n": n,
        "disc": "bilinear",
        "transition_exact": method == "B",
        "per_channel": [],
    }

    for h in range(H):
        if method == "B":
            a = -scipy.linalg.logm(dA[h]) / dtheta
            gap = float(np.max(np.abs(scipy.linalg.expm(-a * dtheta) - dA[h])))
            if gap > check_tol:
                raise AssertionError(f"channel {h}: expm(-a dtheta) != dA (max err {gap:.3e})")
            b = np.linalg.solve(_G_VL(a, dtheta), dB[h])
        else:
            # exact inverse of their discretisation: see the module docstring on why (A,P,Q) cannot be used
            Ac = (2.0 / dt[h]) * np.linalg.solve((dA[h] + I).T, (dA[h] - I).T).T
            back = np.linalg.solve(I - dt[h] / 2.0 * Ac, I + dt[h] / 2.0 * Ac)
            rt = float(np.max(np.abs(back - dA[h])))
            if rt > check_tol:
                raise AssertionError(
                    f"channel {h}: Cayley round-trip failed ({rt:.3e}); their transition is not the bilinear "
                    f"transform of any generator, so Method A has no continuous system to read off."
                )
            a = -Ac * dt[h] / dtheta
            b = (I - dt[h] / 2.0 * Ac) @ dB[h] / dt[h]
            gap = float(np.max(np.abs(scipy.linalg.expm(-a * dtheta) - dA[h])))
        diag["per_channel"].append({"transition_gap": gap, "dt": float(dt[h])})

        s = slice(h * 2 * n, (h + 1) * 2 * n)
        A_ews[s, s] = _realify(a)
        B_ews[h * 2 * n : h * 2 * n + n, h] = b.real
        B_ews[h * 2 * n + n : (h + 1) * 2 * n, h] = b.imag
        # y = Re(dC . x) = Re(dC).Re(x) - Im(dC).Im(x): single Re, the conj pair is inside the state
        C_ews[h, h * 2 * n : h * 2 * n + n] = dC[h].real
        C_ews[h, h * 2 * n + n : (h + 1) * 2 * n] = -dC[h].imag

    diag["max_transition_gap"] = float(max(c["transition_gap"] for c in diag["per_channel"]))
    diag["A_op_norm"] = float(np.linalg.norm(A_ews, 2))
    return ConversionResult(
        A=A_ews,
        B=B_ews,
        C=C_ews,
        D=None,
        m=m,
        family="s4",
        method=method,
        dtheta=float(dtheta),
        lti=True,
        diag=diag,
    )


def from_torch_kernel(kernel, *, method="A", **kw):
    """Pull the explicit discrete system out of a live state-spaces/s4 SSMKernelDPLR, then convert.

    Uses their `_setup_step(mode='dense')`, so `(dA, dB, dC)` is their construction, not our re-derivation.
    """
    import torch

    def npy(t):
        return t.detach().resolve_conj().cpu().numpy()

    kernel._setup_step(mode="dense")
    with torch.no_grad():
        dA, dB, dC = npy(kernel.dA), npy(kernel.dB), npy(kernel.dC)
        dt = npy(kernel._get_params()[0]).reshape(-1)
    if dC.ndim == 3:  # (C, H, n): single output channel in the standard configuration
        if dC.shape[0] != 1:
            raise NotImplementedError(
                f"multi-channel dC (C={dC.shape[0]}) is not handled; got {dC.shape}"
            )
        dC = dC[0]
    if dt.size == 1:
        dt = np.repeat(dt, dA.shape[0])
    return to_ews_arrays(dA, dB, dC, dt, method=method, **kw)


def reference_forward(dA, dB, dC, u):
    """Their discrete system, run sequentially. u (H, L) real -> y (H, L) real.

    This is the recurrence their own `step()` implements; running it by hand reproduces their FFT convolution to
    8.882e-16, which is what licenses using it as ground truth.

    Note on independence: unlike `reference/s4d_ref.py`, this is not a from-the-published-equations reference:
    it starts from the (dA, dB, dC) their code builds. Their NPLR parameter convention could not be reconstructed
    (see the module docstring), so a fully independent S4 reference does not exist here and the S4 rows are checked
    against their implementation only. That is a weaker check than S4D's and the README should say so.
    """
    dA = np.asarray(dA, np.complex128)
    dB = np.asarray(dB, np.complex128)
    dC = np.asarray(dC, np.complex128)
    u = np.asarray(u, np.float64)
    H, n, _ = dA.shape
    L = u.shape[1]
    x = np.zeros((H, n), np.complex128)
    y = np.zeros((H, L), np.float64)
    for l in range(L):
        x = np.einsum("hmn,hn->hm", dA, x) + dB * u[:, l][:, None]
        y[:, l] = np.real(np.einsum("hn,hn->h", dC, x))
    return y
