"""SSM -> EWS weight conversion (Method B in closed form), float64 throughout.

The eigenvalue conditions are asserted on every channel before any logarithm, and after the conversion
expm(-A_k) == Abar_k and G_VL(A_k) @ B_k == Bbar_k are checked to `check_tol`, so a layout or formula error fails here.
The weights describe the depth-1 EWS model itself; the leaf is chosen by the wrapper (`apply.py`).

State layout: index = 4k + 2*(re=0 / im=1) + (z=0 / y=1), i.e. [4k] = Re z_k, [4k+1] = Re y_k, [4k+2] = Im z_k and
[4k+3] = Im y_k for source channel k.
"""

import numpy as np
import scipy.linalg


def _f_arctan_over(omega: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """f(w) = arctan(w)/w with the w -> 0 Taylor guard.

    Never a bare division: below `eps` use 1 - w^2/3 (next term w^4/5 is
    ~2e-17 at w=1e-4, below float64 resolution).
    """
    small = 1.0 - omega**2 / 3.0
    safe = np.where(omega < eps, 1.0, omega)
    return np.where(omega < eps, small, np.arctan(safe) / safe)


def _van_loan_G(A_blocks: np.ndarray) -> np.ndarray:
    """G_VL(A) = int_0^1 exp(-sA) ds per 2x2 block, via the Van Loan augmented
    exponential: the (1,2) block of expm([[-A, I], [0, 0]]).

    Always well defined, never the closed form A^{-1}(I - e^{-A}), which is
    singular exactly at the relu-clipped channels.
    """
    P = A_blocks.shape[0]
    G = np.zeros_like(A_blocks)
    for k in range(P):
        M = np.zeros((4, 4))
        M[:2, :2] = -A_blocks[k]
        M[:2, 2:] = np.eye(2)
        G[k] = scipy.linalg.expm(M)[:2, 2:]
    return G


def linoss_im_to_ews(layer, *, check_tol: float = 1e-12) -> dict:
    """Convert a vendored LinOSSLayer (discretization='IM') to depth-1 EWS weights.

    Source recurrence (vendored linoss.py::apply_linoss_im, state (z, y)):
        Abar_k = [[S, -Delta*Acal*S], [Delta*S, S]],  Bbar_k = (Delta*S, Delta^2*S),
        S = 1/(1 + Delta^2*Acal),  Acal = relu(A_diag),  Delta = sigmoid(steps),
        drive (B_C u)_k complex, output Re(C_C y) + D*u.

    Returns dict:
        A (m, m) float64  block-diagonal 2x2, m = 4P
        B (m, H) float64  lift
        C (H, m) float64  readout (y-components only)
        D (H,)   float64  feedthrough
        m, P, H, diag (per-channel diagnostics)
    """
    if layer.discretization != "IM":
        raise ValueError(
            f"only LinOSS-IM converts (IMEX has |lambda|=1, no principal log); "
            f"got discretization={layer.discretization!r}"
        )
    A_diag = np.asarray(layer.A_diag, dtype=np.float64)
    steps = np.asarray(layer.steps, dtype=np.float64)
    B_re = np.asarray(layer.B[..., 0], dtype=np.float64)  # (P, H)
    B_im = np.asarray(layer.B[..., 1], dtype=np.float64)
    C_re = np.asarray(layer.C[..., 0], dtype=np.float64)  # (H, P)
    C_im = np.asarray(layer.C[..., 1], dtype=np.float64)
    D = np.asarray(layer.D, dtype=np.float64)
    P, H = B_re.shape

    Acal = np.maximum(A_diag, 0.0)  # relu, as in the layer
    Delta = 1.0 / (1.0 + np.exp(-steps))  # sigmoid, as in the layer
    S = 1.0 / (1.0 + Delta**2 * Acal)

    # --- source transition/drive, state (z, y) ------------------------------
    Abar = np.zeros((P, 2, 2))
    Abar[:, 0, 0] = S
    Abar[:, 0, 1] = -Delta * Acal * S
    Abar[:, 1, 0] = Delta * S
    Abar[:, 1, 1] = S
    Bbar = np.stack([Delta * S, Delta**2 * S], axis=-1)  # (P, 2)

    # --- eigenvalue assertions, every channel, before any log --------------
    lam_formula = S + 1j * Delta * np.sqrt(Acal) * S  # (P,), + conjugate
    for k in range(P):
        eig = np.linalg.eigvals(Abar[k])
        want = np.array([lam_formula[k], np.conj(lam_formula[k])])
        # match by sorted imaginary part
        eig = eig[np.argsort(eig.imag)]
        want = want[np.argsort(want.imag)]
        if not np.allclose(eig, want, atol=1e-12, rtol=0.0):
            raise AssertionError(f"channel {k}: eig(Abar)={eig} != formula {want}")
        if not np.allclose(np.abs(eig), np.sqrt(S[k]), atol=1e-12, rtol=0.0):
            raise AssertionError(f"channel {k}: |lambda| != sqrt(S)")
        if np.any((eig.real <= 0.0) & (np.abs(eig.imag) < 1e-300)):
            raise AssertionError(f"channel {k}: eigenvalue on R_<=0, no principal log")

    # --- A block (closed form) ----------------------------------------------
    omega = Delta * np.sqrt(Acal)
    f = _f_arctan_over(omega)
    decay = 0.5 * np.log1p(Delta**2 * Acal)  # = -1/2 log S >= 0
    A_blk = np.zeros((P, 2, 2))
    A_blk[:, 0, 0] = decay
    A_blk[:, 0, 1] = f * Delta * Acal
    A_blk[:, 1, 0] = -f * Delta
    A_blk[:, 1, 1] = decay

    # --- B block ------------------------------------------------------------
    G = _van_loan_G(A_blk)  # (P, 2, 2)
    B_blk = np.linalg.solve(G, Bbar[..., None])[..., 0]  # (P, 2)

    # --- internal exactness asserts -----------------------------------------
    for k in range(P):
        eA = scipy.linalg.expm(-A_blk[k])
        if np.max(np.abs(eA - Abar[k])) > check_tol:
            raise AssertionError(
                f"channel {k}: expm(-A_k) != Abar_k (max err {np.max(np.abs(eA - Abar[k])):.2e})"
            )
        if np.max(np.abs(G[k] @ B_blk[k] - Bbar[k])) > check_tol:
            raise AssertionError(f"channel {k}: G_VL @ B_k != Bbar_k")

    # --- assembly (layout 4k + 2*(re/im) + (z/y)) ---------------------------
    m = 4 * P
    A_ews = np.zeros((m, m))
    B_ews = np.zeros((m, H))
    C_ews = np.zeros((H, m))
    for k in range(P):
        re = slice(4 * k, 4 * k + 2)  # (z_re, y_re)
        im = slice(4 * k + 2, 4 * k + 4)  # (z_im, y_im)
        A_ews[re, re] = A_blk[k]
        A_ews[im, im] = A_blk[k]  # same block twice: real & imag systems
        B_ews[4 * k + 0] = B_blk[k, 0] * B_re[k]
        B_ews[4 * k + 1] = B_blk[k, 1] * B_re[k]
        B_ews[4 * k + 2] = B_blk[k, 0] * B_im[k]
        B_ews[4 * k + 3] = B_blk[k, 1] * B_im[k]
        C_ews[:, 4 * k + 1] = C_re[:, k]  # Re(C) y_re
        C_ews[:, 4 * k + 3] = -C_im[:, k]  # -Im(C) y_im
        # z-columns stay zero: readout is y-components only

    return {
        "A": A_ews,
        "B": B_ews,
        "C": C_ews,
        "D": D,
        "m": m,
        "P": P,
        "H": H,
        "diag": {
            "Acal": Acal,
            "Delta": Delta,
            "S": S,
            "omega": omega,
            "n_clipped": int(np.sum(A_diag <= 0.0)),
            "A_op_norm": float(np.linalg.norm(A_ews, 2)),
        },
    }


def s5_to_ews(layer, *, check_tol: float = 1e-12, wrap_branch: bool = True) -> dict:
    """A vendored S5Layer (discretisation='zoh') as depth-1 EWS weights.

    S5 has complex diagonal modes with Lambda_bar = exp(Lambda Delta) and B_bar = Lambda^-1 (Lambda_bar - 1) B~, and
    y = c Re(C~ x) + D u with c = 2 under conjugate symmetry. ZOH gives the generator directly, a_j = -Lambda_j Delta_j,
    so exp(-a_j) = Lambda_bar_j with no logarithm. Every a_j + 2 pi i k has the same exponential; `wrap_branch=True` (the
    default) takes Im(a) in (-pi, pi], which keeps the Van Loan leaf accurate on fast modes and gives level 2 the minimal
    generator (the depth-1 output is the same either way). B_EWS = G_VL(a)^-1 B_bar, which for unwrapped modes equals
    Delta B~ (asserted). Mode j is realified to the pair (2j, 2j + 1), m = 2P.
    """
    if layer.discretisation != "zoh":
        raise ValueError(
            f"only ZOH S5 converts via the direct generator; got "
            f"discretisation={layer.discretisation!r} (bilinear = Cayley would "
            f"need a principal log and is not what the walker harness runs)"
        )
    Lam_re = np.asarray(layer.Lambda_re, dtype=np.float64)
    if layer.clip_eigs:
        Lam_re = np.minimum(Lam_re, -1e-4)
    lam = Lam_re + 1j * np.asarray(layer.Lambda_im, dtype=np.float64)  # (P,)
    Delta = float(layer.step_rescale) * np.exp(
        np.asarray(layer.log_step[:, 0], dtype=np.float64)
    )  # (P,)
    B_t = np.asarray(layer.B[..., 0], np.float64) + 1j * np.asarray(
        layer.B[..., 1], np.float64
    )  # (P, H)
    C_t = np.asarray(layer.C[..., 0], np.float64) + 1j * np.asarray(
        layer.C[..., 1], np.float64
    )  # (H, P)
    D = np.asarray(layer.D, dtype=np.float64)
    P, H = B_t.shape

    if not np.all(lam.real < 0.0):
        raise AssertionError("S5 modes must be stable (Re Lambda < 0; clip_eigs)")
    if not np.all(Delta > 0.0):
        raise AssertionError("Delta must be positive")

    lam_bar = np.exp(lam * Delta)
    B_bar = ((lam_bar - 1.0) / lam)[:, None] * B_t

    a_raw = -lam * Delta  # true ZOH generator, no log taken
    # derivation check (unwrapped): B = G_VL(a_raw)^{-1} B_bar == Delta * B~
    G_raw = (1.0 - np.exp(-a_raw)) / a_raw
    if np.max(np.abs(G_raw[:, None] * (Delta[:, None] * B_t) - B_bar)) > check_tol:
        raise AssertionError("closed-form B_EWS = Delta*B~ failed vs G_VL * B_bar")

    if wrap_branch:
        a = a_raw.real + 1j * (np.mod(a_raw.imag + np.pi, 2.0 * np.pi) - np.pi)
    else:
        a = a_raw
    # exp(-a) == lam_bar for every branch; B from the same generator actually used
    G = (1.0 - np.exp(-a)) / a  # scalar G_VL per mode; a != 0 (Re a > 0)
    B_c = B_bar / G[:, None]

    m = 2 * P
    A_ews = np.zeros((m, m))
    B_ews = np.zeros((m, H))
    C_ews = np.zeros((H, m))
    c = 2.0 if layer.conj_sym else 1.0
    for j in range(P):
        blk = np.array([[a[j].real, -a[j].imag], [a[j].imag, a[j].real]])
        A_ews[2 * j : 2 * j + 2, 2 * j : 2 * j + 2] = blk
        eA = scipy.linalg.expm(-blk)
        want = np.array([[lam_bar[j].real, -lam_bar[j].imag], [lam_bar[j].imag, lam_bar[j].real]])
        if np.max(np.abs(eA - want)) > check_tol:
            raise AssertionError(f"mode {j}: expm(-A_real) != realify(Lambda_bar)")
        B_ews[2 * j + 0] = B_c[j].real
        B_ews[2 * j + 1] = B_c[j].imag
        C_ews[:, 2 * j + 0] = c * C_t[:, j].real
        C_ews[:, 2 * j + 1] = -c * C_t[:, j].imag

    return {
        "A": A_ews,
        "B": B_ews,
        "C": C_ews,
        "D": D,
        "m": m,
        "P": P,
        "H": H,
        "diag": {
            "lam": lam,
            "Delta": Delta,
            "max_im_a": float(np.max(np.abs(a.imag))),
            "n_wrapped": int(np.sum(np.abs(a_raw.imag) > np.pi)),
            "A_op_norm": float(np.max(np.abs(a))),
        },
    }


def lru_to_ews(layer, *, check_tol: float = 1e-12, wrap_branch: bool = True) -> dict:
    """A vendored LRULayer as depth-1 EWS weights.

    The LRU has lambda = exp(-exp(nu_log) + i exp(theta_log)), B_norm = (B_re + i B_im) exp(gamma_log) and
    y = Re(C x) + D u. Its parameters give the generator directly, a_j = exp(nu_log_j) - i exp(theta_log_j), with
    exp(-a_j) = lambda_j and Re(a) > 0. The phases run up to 2 pi, so `wrap_branch=True` (the default) reduces Im(a) to
    (-pi, pi], as for S5. B_EWS = G_VL(a)^-1 B_norm, in the layout of S5 (m = 2N).
    """
    nu = np.exp(np.asarray(layer.nu_log, dtype=np.float64))  # (N,) > 0
    theta = np.exp(np.asarray(layer.theta_log, dtype=np.float64))  # (N,) > 0
    lam = np.exp(-nu + 1j * theta)
    B_norm = (
        np.asarray(layer.B_re, np.float64) + 1j * np.asarray(layer.B_im, np.float64)
    ) * np.exp(np.asarray(layer.gamma_log, np.float64))[:, None]  # (N, H)
    C_c = np.asarray(layer.C_re, np.float64) + 1j * np.asarray(layer.C_im, np.float64)
    D = np.asarray(layer.D, dtype=np.float64)
    N, H = B_norm.shape

    if not np.all(nu > 0.0):
        raise AssertionError("exp(nu_log) must be positive (|lambda| < 1)")
    if not np.all(np.abs(lam) < 1.0):
        raise AssertionError("LRU modes must be strictly stable")

    a_raw = nu - 1j * theta  # exp(-a_raw) == lambda by construction
    if wrap_branch:
        a = a_raw.real + 1j * (np.mod(a_raw.imag + np.pi, 2.0 * np.pi) - np.pi)
    else:
        a = a_raw
    if np.max(np.abs(np.exp(-a) - lam)) > check_tol:
        raise AssertionError("exp(-a) != lambda after branch handling")
    G = (1.0 - np.exp(-a)) / a  # scalar G_VL per mode; Re(a) > 0
    B_c = B_norm / G[:, None]
    if np.max(np.abs(G[:, None] * B_c - B_norm)) > check_tol:
        raise AssertionError("G_VL * B_EWS != B_norm")

    m = 2 * N
    A_ews = np.zeros((m, m))
    B_ews = np.zeros((m, H))
    C_ews = np.zeros((H, m))
    for j in range(N):
        blk = np.array([[a[j].real, -a[j].imag], [a[j].imag, a[j].real]])
        A_ews[2 * j : 2 * j + 2, 2 * j : 2 * j + 2] = blk
        eA = scipy.linalg.expm(-blk)
        want = np.array([[lam[j].real, -lam[j].imag], [lam[j].imag, lam[j].real]])
        if np.max(np.abs(eA - want)) > check_tol:
            raise AssertionError(f"mode {j}: expm(-A_real) != realify(lambda)")
        B_ews[2 * j + 0] = B_c[j].real
        B_ews[2 * j + 1] = B_c[j].imag
        C_ews[:, 2 * j + 0] = C_c[:, j].real
        C_ews[:, 2 * j + 1] = -C_c[:, j].imag

    return {
        "A": A_ews,
        "B": B_ews,
        "C": C_ews,
        "D": D,
        "m": m,
        "P": N,
        "H": H,
        "diag": {
            "lam": lam,
            "max_im_a": float(np.max(np.abs(a.imag))),
            "n_wrapped": int(np.sum(np.abs(a_raw.imag) > np.pi)),
            "A_op_norm": float(np.max(np.abs(a))),
        },
    }


def jordanize(weights: dict, *, copies: int = 2, check_tol: float = 1e-10) -> dict:
    """Rotate converted weights into the Jordan-canonical basis.

    The converted 2x2 blocks are [[d, a], [-b, d]] with a != b (LinOSS's
    (velocity, position) units). The exact similarity T = diag(s, 1/s) per
    block, s = (b/a)^(1/4), makes each block canonical [[d, w], [-w, d]],
    w = sqrt(ab); T is absorbed into B (rows) and C (cols), so the model
    function is unchanged (level-1 pathway exact; C2 is zero at conversion so
    level 2 has nothing to transform). Requires all channels oscillatory
    (a, b > 0, true for every trained Heartbeat checkpoint) and the
    tied copies identical.

    Returns a new weights dict (A canonical, B/C transformed) plus per-channel
    "jordan_d"/"jordan_w" arrays (P,) for JordanTiedA.
    """
    A = np.asarray(weights["A"], dtype=np.float64)
    B = np.array(weights["B"], dtype=np.float64)
    C = np.array(weights["C"], dtype=np.float64)
    m = A.shape[0]
    nblk = m // 2
    d = A[0::2, 0::2].diagonal().copy()
    a = np.array([A[2 * i, 2 * i + 1] for i in range(nblk)])
    b = np.array([-A[2 * i + 1, 2 * i] for i in range(nblk)])
    if not (np.all(a > 0) and np.all(b > 0)):
        raise AssertionError(
            "jordanize requires oscillatory blocks (a, b > 0); a clipped/defective "
            "channel needs a real-kind block instead"
        )
    for k in range(nblk // copies):  # tied copies must be identical
        base = copies * k
        for c in range(1, copies):
            if (
                np.max(
                    np.abs(
                        A[2 * base : 2 * base + 2, 2 * base : 2 * base + 2]
                        - A[
                            2 * (base + c) : 2 * (base + c) + 2, 2 * (base + c) : 2 * (base + c) + 2
                        ]
                    )
                )
                > check_tol
            ):
                raise AssertionError(f"copies of channel {k} differ; cannot tie")
    s = (b / a) ** 0.25
    w_can = np.sqrt(a * b)
    # T = diag(s, 1/s) per block: z-dims scaled by s, y-dims by 1/s
    scale = np.empty(m)
    scale[0::2] = s
    scale[1::2] = 1.0 / s
    A_new = np.zeros_like(A)
    for i in range(nblk):
        A_new[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = [[d[i], w_can[i]], [-w_can[i], d[i]]]
    if np.max(np.abs((scale[:, None] * A * (1.0 / scale)[None, :]) - A_new)) > check_tol:
        raise AssertionError("similarity did not canonicalise A")
    out = dict(weights)
    out["A"] = A_new
    out["B"] = scale[:, None] * B  # B' = T B
    out["C"] = C * (1.0 / scale)[None, :]  # C' = C T^-1
    out["jordan_d"] = d[::copies].copy()
    out["jordan_w"] = w_can[::copies].copy()
    return out
