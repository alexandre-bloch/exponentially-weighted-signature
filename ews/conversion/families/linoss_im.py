"""LinOSS-IM -> EWS. Method B (matrix logarithm).

Method A reads the oscillator ODE's generator off the parameters; see `to_ews`.
"""

import numpy as np

from ews.conversion.base import ConversionResult


def _archive():
    """The shared conversion maths (`_convert_core`)."""
    from ews.conversion import _convert_core

    return _convert_core


def to_ews(layer, method="B", *, check_tol=1e-12, dtheta=1.0):
    """Method B inverts the family's own transition and is exact (1e-14).

    Method A reads LinOSS's continuous system off the parameters (z' = -Acal y + Bu, y' = z) and is not exact,
    because the family's transition is not the matrix exponential of that generator: the gap falls 4.00x per
    halving of dt, i.e. O(dt^2), and at their own dt (sigmoid(steps) ~ 0.5) it is 7.9e-1. Their
    "IM" is implicit but is not the symmetric midpoint (Cayley) rule, which would be O(dt^3) and 9x closer to exp.
    Method A is offered because the continuous generator is what a depth >= 2 model would need; its residual is the
    family's discretisation error, not a conversion error, and the row must say so.
    """
    if method not in ("A", "B"):
        raise NotImplementedError(f"LinOSS-IM has Methods A and B; got {method!r}")
    d = _archive().linoss_im_to_ews(layer, check_tol=check_tol)
    if method == "A":
        import jax
        import numpy as _np

        from ews.conversion.method_a import linoss_continuous_generator

        Acal = _np.asarray(jax.nn.relu(layer.A_diag), _np.float64)
        Delta = _np.asarray(jax.nn.sigmoid(layer.steps), _np.float64)
        d = dict(d, A=linoss_continuous_generator(Acal, Delta))
    return ConversionResult(
        A=np.asarray(d["A"], np.float64),
        B=np.asarray(d["B"], np.float64),
        C=np.asarray(d["C"], np.float64),
        D=None if d.get("D") is None else np.asarray(d["D"], np.float64),
        m=int(d["m"]),
        family="linoss_im",
        method=method,
        dtheta=float(dtheta),
        diag=dict(d.get("diag", {}), P=d.get("P"), H=d.get("H")),
    )
