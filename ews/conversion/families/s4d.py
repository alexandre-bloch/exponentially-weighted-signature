"""S4D -> depth-1 EWS, Methods A and B.

Equations from Gu, Gupta, Goel and Re, "On the Parameterization and Initialization of Diagonal State Space Models"
(NeurIPS 2022); the reference implementation state-spaces/s4 (Apache 2.0) is read from a clone, not vendored.

Per channel h and mode n, S4D is Abar_n = exp(A_n dt_h), Bbar_n = (exp(A_n dt_h) - 1) / A_n and
y = 2 Re(sum_n C_n x_n) + D u; the EWS steps by exp(-A dtheta) with the lift G_VL(A) B. With dtheta = 1 and dt_h folded
into the generator, as for S5:

  Method A: a = -A_n dt_h, and then b = Bbar / G_VL(a) = dt_h: the lift cancels the ZOH factor.
  Method B: a = -Log(Abar) / dtheta on the principal branch, then b = Bbar / G_VL(a).

Both reproduce the recurrence exactly, but Log inverts exp only on Im in (-pi, pi], so a mode with
|Im(A_n dt_h)| > pi comes back wrapped by 2 pi i k. That is invisible at depth 1 and matters at depth >= 2; the default
initialisation makes it common (A_imag = pi n, dt up to 0.1). `diag["n_wrapped"]` counts such modes.
"""

import numpy as np
import scipy.linalg

from ews.conversion.base import ConversionResult


def _G_VL_scalar(a):
    """G_VL(a) = int_0^1 exp(-s a) ds for a complex scalar, with the a -> 0 guard.

    Closed form -expm1(-a)/a. Never a bare division: at |a| below the guard the series 1 - a/2 + a^2/6 is
    already exact to float64, and a = 0 (a mode with no decay and no rotation) is a legitimate limit.
    """
    a = np.asarray(a, np.complex128)
    small = np.abs(a) < 1e-8
    safe = np.where(small, 1.0, a)
    return np.where(small, 1.0 - a / 2.0 + a * a / 6.0, -np.expm1(-safe) / safe)


def _realify(w):
    """Complex scalar w as the 2x2 real matrix acting on (re, im). Realification is a ring homomorphism,
    so it commutes with exp and with G_VL: that is what makes the block assembly below exact."""
    return np.array([[w.real, -w.imag], [w.imag, w.real]], np.float64)


def to_ews(layer, method="A", *, check_tol=1e-12, dtheta=1.0, B=None, allow_missing_D=False):
    """Convert a live state-spaces/s4 layer to depth-1 EWS weights. Same signature shape as the other families.

    `layer` is an S4D module (has .kernel and .D) or a bare S4DKernel. Their materialisation, mirrored exactly:
        dt = exp(log_dt);  C = view_as_complex(C);  A = -exp(log_A_real) + 1j*A_imag

    A bare S4DKernel is refused by default. Their `y = y + u * self.D` skip lives on the S4D module, not on the
    kernel, so a kernel alone cannot know D and converting one silently drops the feedthrough: a conversion
    that still passes a reference check whose reference omits D too, and is simply wrong against their layer.
    Pass allow_missing_D=True to convert the SSM alone on purpose.

    For parameters already in numpy (the torch-free path the committed tests use), call `to_ews_arrays`.
    """
    if method not in ("A", "B"):
        raise NotImplementedError(
            f"S4D implements Method A (continuous read-off) and Method B (matrix logarithm); got {method!r}."
        )
    import torch

    kernel = getattr(layer, "kernel", layer)
    if not hasattr(kernel, "log_A_real"):
        raise TypeError(f"expected an S4D module or S4DKernel, got {type(layer).__name__}")
    D = getattr(layer, "D", None)
    if D is None and not allow_missing_D:
        raise ValueError(
            "converting a bare S4DKernel would silently drop the D feedthrough, which lives on the S4D module. "
            "Pass the S4D module, or allow_missing_D=True to convert the SSM alone deliberately."
        )
    with torch.no_grad():
        dt_np = torch.exp(kernel.log_dt.double()).cpu().numpy()
        C_np = torch.view_as_complex(kernel.C.double().contiguous()).cpu().numpy()
        A_np = (-torch.exp(kernel.log_A_real.double()) + 1j * kernel.A_imag.double()).cpu().numpy()
        D_np = None if D is None else D.double().cpu().numpy()
    return to_ews_arrays(
        A_np, dt_np, C_np, D_np, method=method, B=B, check_tol=check_tol, dtheta=dtheta
    )


def to_ews_arrays(A, dt, C, D=None, *, method="A", B=None, check_tol=1e-12, dtheta=1.0, disc="zoh"):
    """Convert S4D parameters to depth-1 EWS weights.

    A  (H, N2) complex diagonal modes        dt (H,) real step sizes
    C  (H, N2) complex readout               D  (H,) real feedthrough, or None
    B  (H, N2) complex input map, or None for their B = 1

    Returns a ConversionResult with m = 2*H*N2. State layout: mode j = h*N2 + n occupies
    [2j] = Re x_j, [2j+1] = Im x_j.
    """
    if method not in ("A", "B"):
        raise NotImplementedError(
            f"S4D implements Method A (continuous read-off) and Method B (matrix logarithm); got {method!r}."
        )
    A = np.asarray(A, np.complex128)
    C = np.asarray(C, np.complex128)
    dt = np.asarray(dt, np.float64)
    H, N2 = A.shape
    assert C.shape == (H, N2), f"C {C.shape} != A {A.shape}"
    assert dt.shape == (H,), f"dt {dt.shape} != ({H},)"
    Bc = np.ones_like(A) if B is None else np.asarray(B, np.complex128)

    dtA = A * dt[:, None]
    if disc == "zoh":
        Abar = np.exp(dtA)
        Bbar = np.expm1(dtA) / A * Bc
    elif disc == "bilinear":
        # The Pade(1,1) transition. Not a matrix exponential: see the Method A note below.
        Abar = (1.0 + dtA / 2.0) / (1.0 - dtA / 2.0)
        Bbar = Bc * dt[:, None] / (1.0 - dtA / 2.0)
    else:
        raise ValueError(f"disc must be 'zoh' or 'bilinear', got {disc!r}")

    # --- the generator, per method -------------------------------------------------
    if method == "A":
        a = -dtA / dtheta
        n_wrapped = int(np.sum(np.abs(dtA.imag) > np.pi))
    else:
        # principal log exists unless Abar sits on the closed negative real axis
        on_cut = (Abar.imag == 0.0) & (Abar.real <= 0.0)
        if np.any(on_cut):
            bad = np.argwhere(on_cut)
            raise AssertionError(
                f"Method B impossible: {len(bad)} mode(s) have Abar on the negative real axis, "
                f"no principal logarithm. First at (h,n)={tuple(bad[0])}, Abar={Abar[tuple(bad[0])]}. "
                f"Record this as an obstruction rather than perturbing past it."
            )
        a = -np.log(Abar) / dtheta
        # base.py defines n_wrapped as "how many differ by a non-zero multiple of 2pi". Counting any difference
        # instead is wrong under a non-exponential discretisation: bilinear's log-generator legitimately differs
        # from the continuous A on essentially every mode (that is the discretisation gap), which reported
        # 248/256 "branch-wrapped" and would have overstated the wrapping finding in the README.
        k = ((-a * dtheta).imag - dtA.imag) / (2.0 * np.pi)
        n_wrapped = int(
            np.sum(np.abs(k - np.round(k)) < 1e-6) * 0 + np.sum(np.abs(np.round(k)) >= 1)
        )

    # G_VL(a) = int_0^dtheta exp(-s a) ds = dtheta * int_0^1 exp(-s a dtheta) ds. The dtheta factor is not
    # optional: the matrix Van Loan form in ews_forward integrates over [0, dtheta], and the two must agree
    # or every conversion with dtheta != 1 is silently off by that factor.
    g = dtheta * _G_VL_scalar(a * dtheta)
    b = Bbar / g

    # --- exactness asserts, before anything is assembled ---------------------------
    # these are the whole point: a sign or layout error dies here, not in a residual table
    err_A = np.max(np.abs(np.exp(-a * dtheta) - Abar))
    exact = (method == "B") or (disc == "zoh")
    if err_A > check_tol and exact:
        raise AssertionError(f"exp(-a*dtheta) != Abar (max err {err_A:.3e})")
    # Under a non-exponential discretisation, Method A cannot reproduce the transition and is not supposed to:
    # the EWS layer steps by exp(-A dtheta) while the family steps by the Pade form. The gap is the family's own
    # O(dt^3) discretisation error, so it is recorded rather than raised: measuring it is the point. Method B
    # takes the log of whatever the transition actually is and stays exact either way.
    err_B = np.max(np.abs(g * b - Bbar))
    if err_B > check_tol:
        raise AssertionError(f"G_VL(a) * b != Bbar (max err {err_B:.3e})")
    if method == "A" and B is None and disc == "zoh":
        # the closed form claimed in the docstring, checked rather than trusted
        err_cf = np.max(np.abs(b - dt[:, None] / dtheta))
        if err_cf > 1e-10:
            raise AssertionError(
                f"Method A closed form b == dt/dtheta failed (max err {err_cf:.3e})"
            )

    # --- assembly ------------------------------------------------------------------
    m = 2 * H * N2
    A_ews = np.zeros((m, m), np.float64)
    B_ews = np.zeros((m, H), np.float64)
    C_ews = np.zeros((H, m), np.float64)
    for h in range(H):
        for n in range(N2):
            j = h * N2 + n
            s = slice(2 * j, 2 * j + 2)
            A_ews[s, s] = _realify(a[h, n])
            B_ews[2 * j + 0, h] = b[h, n].real
            B_ews[2 * j + 1, h] = b[h, n].imag
            # y = 2 Re(C x) = 2(C_re x_re - C_im x_im)
            C_ews[h, 2 * j + 0] = 2.0 * C[h, n].real
            C_ews[h, 2 * j + 1] = -2.0 * C[h, n].imag
    D_ews = None if D is None else np.diag(np.asarray(D, np.float64))

    return ConversionResult(
        A=A_ews,
        B=B_ews,
        C=C_ews,
        D=D_ews,
        m=m,
        family="s4d",
        method=method,
        dtheta=float(dtheta),
        lti=True,
        diag={
            "H": H,
            "N2": N2,
            "n_wrapped": n_wrapped,
            "max_abs_im_dtA": float(np.max(np.abs(dtA.imag))),
            "assert_expm_err": float(err_A),
            "assert_drive_err": float(err_B),
            "A_op_norm": float(np.linalg.norm(A_ews, 2)),
            "dt_min": float(dt.min()),
            "dt_max": float(dt.max()),
            "disc": disc,
            "transition_exact": bool(exact),
            "method_a_disc_residual": float(
                err_A
            ),  # 0 when exact; the O(dt^3) gap under bilinear + Method A
        },
    )


def ews_forward(res: ConversionResult, u):
    """Run the depth-1 EWS recurrence from a ConversionResult. u (batch, d_in, L) -> y (batch, d_out, L).

    Deliberately uses the dense (m,m) generator, not the 2x2 blocks: that exercises the assembled layout, so a
    mis-indexed state shows up here instead of cancelling out.
    """
    u = np.asarray(u, np.float64)
    Abar = scipy.linalg.expm(-res.A * res.dtheta)
    # G_VL over the whole matrix via the Van Loan augmented exponential: always defined, never A^{-1}(...)
    M = np.zeros((2 * res.m, 2 * res.m), np.float64)
    M[: res.m, : res.m] = -res.A * res.dtheta
    M[: res.m, res.m :] = np.eye(res.m) * res.dtheta
    G = scipy.linalg.expm(M)[: res.m, res.m :]
    drive = G @ res.B
    bsz, d_in, L = u.shape
    x = np.zeros((bsz, res.m), np.float64)
    y = np.zeros((bsz, res.C.shape[0], L), np.float64)
    for l in range(L):
        x = x @ Abar.T + u[:, :, l] @ drive.T
        y[:, :, l] = x @ res.C.T
    if res.D is not None:
        y = y + np.einsum("oi,bil->bol", res.D, u)
    return y
