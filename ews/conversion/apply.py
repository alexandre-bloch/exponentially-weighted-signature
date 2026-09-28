"""EWS-as-SSM wrapper: run converted SSM weights through the real EWS layer.

The wrapper routes through ews.layer.EWS (Van Loan / series leaf, Chen scan),
never a bespoke recurrence.

Path/clock convention: the driving path is the integral path of the input with
a prepend-origin basepoint whose interval lasts one full clock step:

    knots  X = [0, u_1, u_1+u_2, ..., sum u]      (L+1 knots)
    clock  theta = (0, 1, 2, ..., L)              (dtheta = 1 everywhere,
                                                   including the basepoint hop)

so the prefix states align to h_1..h_L with h_n = e^{-A} h_{n-1} + G_VL B u_n.
Note: the repo block's `basepoint=True` duplicates theta_0 (dtheta=0 jump):
that is the parity convention, wrong here; this wrapper builds both arrays
itself and never routes through it.

The non-"integral" path conventions exist only as negative controls for the
tests: each encodes one of three silent convention bugs and must fail.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from ews.layer import EWS

PATH_CONVENTIONS = (
    "integral",  # correct: [0; cumsum(u)] with theta = arange(L+1)
    "value",  # control (i):  knots = u, no cumsum
    "no_basepoint",  # control (ii): cumsum(u) without the zero knot -> u_1 dropped
    "jump_basepoint",  # control (iii): zero knot but theta_0 duplicated -> dtheta=0 hop
)


def ews_ssm_apply(
    weights: dict,
    u: jnp.ndarray,
    *,
    N: int = 1,
    leaf: str = "van_loan",
    van_loan_expm: str = "exact",
    chen_expm: str = "exact",
    M: int = 4,
    series_q=None,
    path_convention: str = "integral",
    reweighting_mode: str | None = None,
    clock: jnp.ndarray | None = None,
    dtype=jnp.float64,
):
    """Apply the depth-N EWS with converted SSM weights to one input sequence.

    Args:
        weights: dict from a family's conversion (A (m,m), B (m,H), C (H,m), D (H,);
            optional "C2" (H, m^2) level-2 readout columns: absent/None means
            the zero-extension: level 2 contributes nothing).
        u: (L, H) input sequence (the SSM layer input), any float dtype.
        N: truncation depth (1 = containment; 2 = grafting test).
        leaf / van_loan_expm / chen_expm / M / series_q: the EWS engine options
            (M defaults to 4; at N=1 with exact exponentials it has no effect).
        dtype: compute dtype (float64 for the exactness checks; float32 to
            measure the practical deployment gap).
        path_convention: see PATH_CONVENTIONS; anything but "integral" is a
            deliberate negative control.
        reweighting_mode: override; defaults to "displacement_uniform" (one
            cached expm: valid because dtheta is uniform) except for the
            jump control whose clock is non-uniform ("displacement").
        clock: optional explicit clock overriding the built one;
            must match the knot count of the chosen convention.

    Returns:
        (y, h): y (L, H) layer output = h @ C.T (+ level-2 term if C2) + D*u;
                h (L, m) level-1 prefix states aligned to SSM states h_1..h_L.
    """
    if path_convention not in PATH_CONVENTIONS:
        raise ValueError(f"path_convention must be one of {PATH_CONVENTIONS}")
    A = jnp.asarray(weights["A"], dtype)
    B = jnp.asarray(weights["B"], dtype)
    C = jnp.asarray(weights["C"], dtype)
    D = jnp.asarray(weights["D"], dtype)
    m = A.shape[0]
    u = jnp.asarray(u, dtype)
    L = u.shape[0]

    v = u @ B.T  # lift-then-cumsum; commutes with cumsum-then-lift (linear)
    if path_convention == "integral":
        path = jnp.concatenate([jnp.zeros((1, m)), jnp.cumsum(v, axis=0)])
        theta = jnp.arange(L + 1, dtype=dtype)
    elif path_convention == "value":
        path = v
        theta = jnp.arange(L, dtype=dtype)
    elif path_convention == "no_basepoint":
        path = jnp.cumsum(v, axis=0)
        theta = jnp.arange(L, dtype=dtype)
    else:  # jump_basepoint
        path = jnp.concatenate([jnp.zeros((1, m)), jnp.cumsum(v, axis=0)])
        t = jnp.arange(L, dtype=dtype)
        theta = jnp.concatenate([t[:1], t])  # (0, 0, 1, ..., L-1): dtheta_0 = 0
    if clock is not None:
        clock = jnp.asarray(clock, dtype)
        if clock.shape[0] != path.shape[0]:
            raise ValueError(f"explicit clock has {clock.shape[0]} knots, path has {path.shape[0]}")
        theta = clock

    if reweighting_mode is None:
        reweighting_mode = (
            "displacement" if path_convention == "jump_basepoint" else "displacement_uniform"
        )

    ewms = EWS(
        A,
        N=N,
        M=M,
        dtype=dtype,
        reweighting_mode=reweighting_mode,
        terminal_reduce="tree",
        van_loan_expm=van_loan_expm,
        chen_expm=chen_expm,
        leaf=leaf,
        series_q=series_q,
        # combine is deliberately left at the EWS default "assoc.build":
        # seq.direct.remat does not work correctly and must not be used.
    )
    flat = ewms(path, theta, readout="prefix", output_format="flat")

    h = flat[:, 1 : 1 + m]  # level-1 slice; index 0 is the level-0 constant
    if h.shape[0] == L - 1:
        # L-knot conventions yield L-1 prefix rows; re-align like the repo
        # block (zero row = trivial prefix). Negative controls only.
        h = jnp.concatenate([jnp.zeros((1, m), dtype=h.dtype), h])

    y = h @ C.T + u * D
    C2 = weights.get("C2")
    if N >= 2 and C2 is not None:
        lvl2 = flat[:, 1 + m : 1 + m + m * m]
        if lvl2.shape[0] == L - 1:
            lvl2 = jnp.concatenate([jnp.zeros((1, m * m), dtype=lvl2.dtype), lvl2])
        y = y + lvl2 @ jnp.asarray(C2, dtype).T
    return y, h
