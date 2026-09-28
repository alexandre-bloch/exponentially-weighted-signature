"""The conversion contract that every family module and reference recurrence follows.

A conversion takes a layer of a linear state-space family and returns the depth-1 EWS weights that reproduce it: a
generator A, an input map B, a readout C and a feedthrough D, in the real state layout. The EWS layer driven by these
weights must reproduce the family's own recurrence to float64 tolerance.

  Method A: continuous read-off. The family discretises an ODE, so its generator is read off its parameters (S4, S4D,
            S5, LinOSS-IM, LinOSS-IMEX, D-LinOSS).
  Method B: the matrix logarithm, A = -log(Abar) / dtheta. It needs G(A) = int_0^dtheta exp(-sA) ds to be invertible
            (no eigenvalue of A at 2 pi i k / dtheta with k != 0) and, for a real logarithm, each Jordan block of a
            negative real eigenvalue of Abar to occur an even number of times; |lambda| = 1 is no obstruction. The LRU
            needs it.

Mamba-1 and Mamba-2 convert exactly too: A is fixed and only the step Delta_t depends on the input, so the EWS clock
takes dtheta_t = Delta_t, A_ews = -A, and the path carries Delta_t B_t u_t (one EWS per channel or per head).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ConversionResult:
    """What every `<family>.to_ews(...)` returns."""

    A: np.ndarray  # (m, m) real generator, our layout
    B: np.ndarray  # (m, d_in) real input map
    C: np.ndarray  # (d_out, m) real readout
    D: Optional[np.ndarray]  # feedthrough: (d_out,) diagonal, or (d_out, d_in) dense, or None.
    # The EWS wrapper (ews_ssm_apply) computes y = h @ C.T + u * D and so
    # supports the diagonal case only. verify.py reduces a diagonal (d_out,
    # d_in) D to its diagonal and raises on any non-zero off-diagonal, so a
    # family with a genuinely dense feedthrough fails loudly instead of
    # silently losing terms. Both shapes are legal here; the restriction
    # belongs to the wrapper, not to the conversion.
    m: int  # real state width
    family: str  # "s4" | "s4d" | "s5" | "lru" | "linoss_im" | "linoss_imex" | "dlinoss" | "mamba1" | "mamba2"
    method: str  # "A" or "B"
    dtheta: float = 1.0  # clock step the conversion assumes
    lti: bool = True  # False for Mamba (frozen-input conversion)
    frozen_input: Optional[np.ndarray] = None  # the input Mamba's conversion was frozen at
    diag: Dict[str, Any] = field(default_factory=dict)  # per-family diagnostics, free-form

    def __post_init__(self):
        assert self.A.shape == (self.m, self.m), f"A must be (m,m), got {self.A.shape}"
        assert self.B.shape[0] == self.m, f"B must have m rows, got {self.B.shape}"
        assert self.C.shape[1] == self.m, f"C must have m columns, got {self.C.shape}"
        assert self.method in ("A", "B"), self.method
        if not self.lti:
            assert self.frozen_input is not None, (
                "a non-LTI conversion must record its frozen input"
            )


@dataclass
class Residual:
    """One row of the residual table."""

    family: str
    method: str
    setting: str  # "random" or a dataset name for a trained checkpoint
    max_abs: float  # max |ews_out - reference_out| over the whole output, float64
    max_rel: float  # max |diff| / (|reference_out| + tiny), same
    n_params: Optional[int] = None
    metric_before: Optional[float] = None  # trained rows only
    metric_after: Optional[float] = None
    # Methods A and B can agree on output to float64 and still differ in the generator: Log inverts exp only on
    # Im in (-pi, pi], so any mode with |Im(A dt)| > pi comes back branch-wrapped by a multiple of 2pi. Whole extra
    # rotations per step are invisible at the sample times, which is why the output residual cannot see them, but
    # they are not invisible at depth >= 2.
    n_modes: Optional[int] = None  # modes compared between methods
    n_wrapped: Optional[int] = None  # how many differ by a non-zero multiple of 2pi
    max_abs_im_dtA: Optional[float] = None  # max |Im(A dt)|; above pi, wrapping is possible
    notes: str = ""

    def row(self):
        mb = "-" if self.metric_before is None else f"{self.metric_before:.4f}"
        ma = "-" if self.metric_after is None else f"{self.metric_after:.4f}"
        wr = "-" if self.n_wrapped is None else f"{self.n_wrapped}/{self.n_modes}"
        return (
            f"| {self.family} | {self.method} | {self.setting} | {self.max_abs:.3e} | "
            f"| {self.max_rel:.3e} | {wr} | {mb} | {ma} | {self.notes} |".replace("| |", "|")
        )

    @staticmethod
    def header():
        return (
            "| family | method | setting | max abs err | max rel err | branch-wrapped | metric before | "
            "metric after | notes |\n|---|---|---|---|---|---|---|---|---|"
        )


def residual(ews_out: np.ndarray, ref_out: np.ndarray):
    """max absolute and max relative output error, float64, computed the same way for every family."""
    a = np.asarray(ews_out, np.float64)
    b = np.asarray(ref_out, np.float64)
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    d = np.abs(a - b)
    return float(d.max()), float((d / (np.abs(b) + 1e-300)).max())
