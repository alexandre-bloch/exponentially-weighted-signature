"""Mamba-1 (per channel) and Mamba-2 (per head) -> EWS, with a data-dependent clock.

Not a frozen-input approximation: see reference/mamba_ref.py. `A` is fixed, so the whole of Mamba's selectivity
lives in the clock increments `dtheta_t = Delta_t` and in the driving path increments `Delta_t * B_t * u_t`, both
of which are data. The generator is constant, which is what an EWS needs.

`ConversionResult` is not used here: it describes a fixed (A, B, C) triple, and these conversions produce a
constant A with a per-step clock and per-step path. The clock is the conversion, so it is returned explicitly.
"""

import numpy as np


class ClockedConversion:
    """A constant generator plus the per-step clock and path that carry the input-dependence."""

    def __init__(self, A, dtheta, dX, C, D=None, family="mamba1", n_units=1):
        self.A, self.dtheta, self.dX, self.C, self.D = A, dtheta, dX, C, D
        self.family, self.n_units, self.lti = family, n_units, True
        assert dtheta.shape[0] == dX.shape[0], "one clock increment per path increment"

    def forward(self):
        """Direct recurrence h_t = exp(-A dtheta_t) h_{t-1} + dX_t. Kept only as a cheap algebra check; the row
        that counts is `forward_ews`, which drives the real ews.layer.EWS."""
        h = np.zeros(self.A.shape[0], np.float64)
        out = []
        for t in range(self.dtheta.shape[0]):
            h = np.exp(-self.A * self.dtheta[t]) * h + self.dX[t]
            out.append(self.C[t] @ h)
        return np.array(out)

    def forward_ews(self, phi1_correction=True):
        """Drive `ews.layer.EWS` through the wrapper (`apply.py`) with the variable clock theta = cumsum(dtheta).

        The EWS integrates each increment over its segment, S_t = e^{-dtheta_t A} S_{t-1} + phi1(-dtheta_t A) dX_t with
        phi1(z) = (e^z - 1)/z, whereas Mamba adds dtheta_t B_t u_t. The layer is therefore given
        dX_t = phi1(-dtheta_t A)^-1 dtheta_t B_t u_t, per channel [dtheta_t a / (1 - e^{-dtheta_t a})] dtheta_t B_t u_t: the
        inversion of Method B, applied at every step.
        """
        from ews.conversion.apply import ews_ssm_apply

        m = self.A.shape[0]
        dX = self.dX
        if phi1_correction:
            z = self.dtheta[:, None] * self.A[None, :]  # dtheta_t * a  (L, m)
            dX = dX * np.where(np.abs(z) < 1e-12, 1.0, z / (1.0 - np.exp(-z)))
        theta = np.concatenate(
            [[0.0], np.cumsum(self.dtheta)]
        )  # L+1 knots, matching the integral path
        # A is stored as a diagonal vector (one decay per channel); the layer needs the full (m, m) generator.
        w = dict(
            A=np.diag(self.A), B=np.eye(m), C=np.eye(m), D=np.zeros(m)
        )  # B = I: dX is already lifted
        # a variable clock requires the non-uniform reweighting; "displacement_uniform" assumes one dt for all
        # segments and takes a path in ews/core/van_loan.py that cannot see per-step dtheta at all.
        _y, h = ews_ssm_apply(w, dX, N=1, clock=theta, reweighting_mode="displacement")
        h = np.asarray(h, np.float64)
        return np.einsum("tm,tm->t", self.C, h)


def mamba1_to_ews(A, Delta, B, C, u, D=None):
    """One Mamba-1 channel -> EWS. A_ews = -A (EWS applies exp(-A dtheta)); clock = Delta; path carries Delta*B*u."""
    A = np.asarray(A, np.float64)
    Delta = np.asarray(Delta, np.float64)
    B = np.asarray(B, np.float64)
    u = np.asarray(u, np.float64)
    return ClockedConversion(
        A=-A,
        dtheta=Delta,
        dX=(Delta * u)[:, None] * B,
        C=np.asarray(C, np.float64),
        D=D,
        family="mamba1",
        n_units=1,
    )


def mamba2_to_ews(a, Delta, B, C, u, D=None):
    """One Mamba-2 head -> EWS. `a` scalar, broadcast across the head's N state coordinates."""
    Delta = np.asarray(Delta, np.float64)
    B = np.asarray(B, np.float64)
    u = np.asarray(u, np.float64)
    N = B.shape[1]
    return ClockedConversion(
        A=-np.full(N, float(a)),
        dtheta=Delta,
        dX=(Delta * u)[:, None] * B,
        C=np.asarray(C, np.float64),
        D=D,
        family="mamba2",
        n_units=1,
    )
