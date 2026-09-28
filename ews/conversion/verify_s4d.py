"""S4D checked against their implementation (state-spaces/s4). Requires torch and a local clone of their
repository, so it is skipped (not failed) wherever either is absent. The published-equation reference
in reference/s4d_ref.py needs neither and is what the committed tests use.
"""

import importlib.util
import os

S4_CLONE = os.environ.get("S4_REPO", "")  # a clone of github.com/state-spaces/s4
SKIP_REASON = (
    None
    if (importlib.util.find_spec("torch") and os.path.isdir(S4_CLONE))
    else (
        "torch not installed"
        if not importlib.util.find_spec("torch")
        else f"no S4 clone at {S4_CLONE} (set S4_REPO)"
    )
)

"""Verify the S4D conversion: EWS-from-converted-weights vs (a) our reference recurrence, (b) their code.

Run:
    python -m ews.conversion.verify_s4d            # reference only, no torch needed
    S4_REPO=<s4 clone> python -m ews.conversion.verify_s4d --theirs     # needs torch

Their code is read from the local clone of state-spaces/s4 (Apache 2.0) named by S4_REPO (or S4_WORKTREE)
and is never vendored into this tree.
"""
import argparse
import os
import sys

import numpy as np

from ews.conversion.base import Residual, residual
from ews.conversion.families import s4d as s4d_conv
from ews.conversion.reference import s4d_ref

S4_WORKTREE = os.environ.get("S4_WORKTREE", S4_CLONE)


def make_params(H=3, N=16, L=24, bsz=2, seed=0, dt_min=1e-3, dt_max=1e-1):
    """Their initialisation, with C and log_dt randomised. N is the full state size; N2 = N//2 modes."""
    rng = np.random.default_rng(seed)
    N2 = N // 2
    log_dt = rng.uniform(np.log(dt_min), np.log(dt_max), size=H)
    dt = np.exp(log_dt)
    # S4D-Lin init, exactly as their S4DKernel.__init__: A = -exp(log(0.5)) + i*pi*n = -0.5 + i*pi*n
    A = -0.5 * np.ones((H, N2)) + 1j * (np.pi * np.arange(N2)[None, :]).repeat(H, 0)
    C = rng.standard_normal((H, N2)) + 1j * rng.standard_normal((H, N2))
    D = rng.standard_normal(H)
    u = rng.standard_normal((bsz, H, L))
    return dict(A=A, dt=dt, C=C, D=D, u=u, log_dt=log_dt, H=H, N=N, N2=N2, L=L)


def theirs_forward(p):
    """Ground truth: their S4DKernel builds the kernel, their S4D.forward applies it (FFT conv + D skip).

    The activation / dropout / output_linear that follow the SSM in their block are replaced by identity, so
    this compares the state-space layer itself and not the pointwise head wrapped around it.
    """
    import torch

    sys.path.insert(0, S4_WORKTREE)
    from models.s4.s4d import S4D

    torch.manual_seed(0)
    model = S4D(d_model=p["H"], d_state=p["N"], dropout=0.0, transposed=True).double()
    with torch.no_grad():
        model.kernel.log_dt.copy_(torch.tensor(p["log_dt"], dtype=torch.float64))
        model.kernel.log_A_real.copy_(
            torch.log(torch.full((p["H"], p["N2"]), 0.5, dtype=torch.float64))
        )
        model.kernel.A_imag.copy_(
            torch.tensor(np.pi * np.arange(p["N2"])[None].repeat(p["H"], 0), dtype=torch.float64)
        )
        model.kernel.C.copy_(torch.view_as_real(torch.tensor(p["C"], dtype=torch.complex128)))
        model.D.copy_(torch.tensor(p["D"], dtype=torch.float64))
    model.activation = torch.nn.Identity()
    model.dropout = torch.nn.Identity()
    model.output_linear = torch.nn.Identity()
    with torch.no_grad():
        y, _ = model(torch.tensor(p["u"], dtype=torch.float64))
    return y.cpu().numpy(), model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--theirs", action="store_true", help="also compare against state-spaces/s4 (needs torch)"
    )
    ap.add_argument("--H", type=int, default=3)
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--L", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    p = make_params(H=args.H, N=args.N, L=args.L, seed=args.seed)
    ref = s4d_ref.s4d_forward(p["A"], p["dt"], p["C"], p["D"], p["u"])

    rows, convs = [], {}
    for method in ("A", "B"):
        res = s4d_conv.to_ews_arrays(p["A"], p["dt"], p["C"], p["D"], method=method)
        convs[method] = res
        y = s4d_conv.ews_forward(res, p["u"])
        ma, mr = residual(y, ref)
        rows.append(
            Residual(
                family="s4d",
                method=method,
                setting="random",
                max_abs=ma,
                max_rel=mr,
                n_params=int(p["H"] * p["N2"] * 3 + p["H"]),
                notes=f"vs our reference recurrence; {res.diag['n_wrapped']}/{p['H'] * p['N2']} modes branch-wrapped",
            )
        )

    if args.theirs:
        y_theirs, model = theirs_forward(p)
        ma, mr = residual(ref, y_theirs)
        rows.append(
            Residual(
                family="s4d",
                method="-",
                setting="reference vs state-spaces/s4",
                max_abs=ma,
                max_rel=mr,
                notes="our published-equation recurrence vs their FFT kernel",
            )
        )
        for method in ("A", "B"):
            res = s4d_conv.to_ews(model, method)
            y = s4d_conv.ews_forward(res, p["u"])
            ma, mr = residual(y, y_theirs)
            rows.append(
                Residual(
                    family="s4d",
                    method=method,
                    setting="random",
                    max_abs=ma,
                    max_rel=mr,
                    notes="vs state-spaces/s4 (ground truth)",
                )
            )

    print(
        f"\nS4D conversion residuals  (H={args.H}, N={args.N}, L={args.L}, seed={args.seed}, float64)\n"
    )
    print(Residual.header())
    for r in rows:
        print(r.row())

    a_gen, b_gen = convs["A"].A, convs["B"].A
    print(f"\ngenerator A vs B: max |A_A - A_B| = {np.max(np.abs(a_gen - b_gen)):.3e}")
    print(
        f"branch-wrapped modes: {convs['A'].diag['n_wrapped']} of {args.H * (args.N // 2)}"
        f"   max |Im(dt*A)| = {convs['A'].diag['max_abs_im_dtA']:.3f}  (pi = {np.pi:.3f})"
    )


if __name__ == "__main__":
    main()
