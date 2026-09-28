"""Method B, the matrix logarithm: one converter for any family given its discrete pair x_t = M x_{t-1} + Bbar u_t.

The EWS applies exp(-A dtheta) and lifts through G(A) = int_0^dtheta exp(-sA) ds, so A = -logm(M) / dtheta and
B_ews = G(A)^-1 Bbar, with C and D unchanged. LinOSS-IM, LinOSS-IMEX and D-LinOSS all use it. `from_discrete` checks
the two conditions (G(A) is singular iff A has an eigenvalue 2 pi i k / dtheta with k != 0; a real logarithm needs
each Jordan block of a negative real eigenvalue to occur an even number of times) and raises with the offending
eigenvalue.
"""

import numpy as np
import scipy.linalg


def van_loan_G(A, dtheta=1.0):
    """G(A) = int_0^dtheta exp(-sA) ds, by the Van Loan block-exponential trick (no quadrature, no eigendecomposition)."""
    A = np.asarray(A, np.float64)
    n = A.shape[0]
    Z = np.zeros((2 * n, 2 * n), np.float64)
    Z[:n, :n] = -A * dtheta
    Z[:n, n:] = np.eye(n) * dtheta
    return scipy.linalg.expm(Z)[:n, n:]


def logm_real_2x2(M):
    """Exact real logarithm of a 2x2 real matrix with complex-conjugate eigenvalues.

    scipy.linalg.logm is a general iterative routine and loses precision on these blocks: on D-LinOSS's damped
    IMEX modes it warned at 3.8e-12 and the expm round-trip came back at 4.3e-09, which tripped the round-trip
    assertion. The closed form below is exact and uses only the trace and determinant: no eigendecomposition
    anywhere.

        lambda = rho e^{+-i theta},  rho = sqrt(det M),  cos theta = tr M / (2 rho)
        log M  = log(rho) I + (theta / (rho sin theta)) (M - (tr M / 2) I)
    """
    M = np.asarray(M, np.float64)
    tr, det = M[0, 0] + M[1, 1], M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if det <= 0:
        raise ValueError(f"2x2 block has det = {det:.3e} <= 0; no real logarithm by this route")
    rho = np.sqrt(det)
    c = np.clip(tr / (2.0 * rho), -1.0, 1.0)
    th = np.arccos(c)
    # the 1/rho matters: M - (tr/2)I equals rho*sin(theta)*[[0,-1],[1,0]], so the coefficient must divide it out.
    # Dropping it gave a round-trip of 1.94e+01 instead of ~1e-16. theta/sin(theta) -> 1 as theta -> 0.
    k = (1.0 / rho) if abs(th) < 1e-12 else th / (rho * np.sin(th))
    return np.log(rho) * np.eye(2) + k * (M - (tr / 2.0) * np.eye(2))


def check_convertible(M, dtheta=1.0, tol=1e-9):
    """Raise if Method B cannot produce a real generator for this transition. Returns diagnostics."""
    M = np.asarray(M, np.float64)
    lam = np.linalg.eigvals(M)
    neg_real = lam[(np.abs(lam.imag) < tol) & (lam.real < 0)]
    if neg_real.size:
        raise ValueError(
            f"no real principal logarithm: {neg_real.size} eigenvalue(s) on the negative real axis, first "
            f"{neg_real[0]}. A real log needs each such Jordan block to occur an even number of times; check "
            f"pairing before forcing this."
        )
    if np.any(np.abs(lam) < tol):
        raise ValueError(
            f"transition is singular (min |lambda| = {np.abs(lam).min():.3e}); no logarithm exists"
        )
    A = (-logm_real_2x2(M) / dtheta) if M.shape == (2, 2) else (-scipy.linalg.logm(M).real / dtheta)
    ev = np.linalg.eigvals(A)
    k = np.abs(ev.imag) * dtheta / (2 * np.pi)
    at_2pik = (np.abs(ev.real) < tol) & (np.abs(k - np.round(k)) < tol) & (k > tol)
    if np.any(at_2pik):
        raise ValueError(
            f"the lift G(A) is singular: generator eigenvalue at 2*pi*i*k/dtheta ({ev[at_2pik][0]}). "
            f"This is the real obstruction, not |lambda| = 1."
        )
    return A, dict(
        max_abs_lam=float(np.abs(lam).max()),
        min_abs_lam=float(np.abs(lam).min()),
        n_neg_real=0,
        max_abs_im_dtA=float(np.abs(ev.imag).max() * dtheta),
        cond_G=float(np.linalg.cond(van_loan_G(A, dtheta))),
    )


def from_discrete(M, Bbar, dtheta=1.0, check_tol=1e-10):
    """(M, Bbar) -> (A, B_ews, diagnostics), with a round-trip assertion so a layout error dies here."""
    A, diag = check_convertible(M, dtheta)
    G = van_loan_G(A, dtheta)
    B_ews = np.linalg.solve(G, np.asarray(Bbar, np.float64))
    err_M = float(np.max(np.abs(scipy.linalg.expm(-A * dtheta) - np.asarray(M, np.float64))))
    err_B = float(np.max(np.abs(G @ B_ews - np.asarray(Bbar, np.float64))))
    # This guard exists to catch layout errors, which are O(1): it must not fail a correct conversion whose
    # accuracy is capped by the family's own spectrum. Inverting the lift G(A) amplifies float64 noise by
    # cond(G), so a mode near the 2*pi*i*k/dtheta singularity cannot reach a flat 1e-10: D-LinOSS's mode 3
    # sits at |Im(A dt)| = 3.1145, essentially pi, giving cond G = 1.8e5 and an achievable floor near 1e-9.
    # Same policy as pass_threshold() in tests/test_conversion.py. Still orders of magnitude tighter than any
    # layout error, which is what this assertion is for.
    tol = check_tol * max(1.0, float(diag["cond_G"]) / 1e3)
    if err_M > tol or err_B > tol:
        raise AssertionError(
            f"round-trip failed: |expm(-A dtheta) - M| = {err_M:.3e}, |G B_ews - Bbar| = {err_B:.3e}, "
            f"against a conditioning-aware bar of {tol:.3e} (cond G = {float(diag['cond_G']):.3e})"
        )
    diag.update(roundtrip_M=err_M, roundtrip_B=err_B)
    return A, B_ews, diag
