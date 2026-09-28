"""S5 (ZOH) -> EWS. Method A (continuous read-off).

S5 is a ZOH discretisation of a diagonal continuous system, so its continuous generator is read straight off Lambda
-- this is Method A, not a matrix logarithm.
"""

import numpy as np

from ews.conversion.base import ConversionResult
from ews.conversion.families.linoss_im import _archive


def to_ews(layer, method="A", *, check_tol=1e-12, wrap_branch=True, dtheta=1.0):
    """Method A is the direct ZOH generator. Method B is the same conversion with `wrap_branch=False`, i.e. the
    generator obtained by inverting exp: for a ZOH family those are the same map up to branch choice, so the
    pair is exactly the cross-check we want: identical output, possibly different generator."""
    if method not in ("A", "B"):
        raise NotImplementedError(f"S5 has Methods A and B; got {method!r}")
    d = _archive().s5_to_ews(layer, check_tol=check_tol, wrap_branch=(method == "B"))
    return ConversionResult(
        A=np.asarray(d["A"], np.float64),
        B=np.asarray(d["B"], np.float64),
        C=np.asarray(d["C"], np.float64),
        D=None if d.get("D") is None else np.asarray(d["D"], np.float64),
        m=int(d["m"]),
        family="s5",
        method=method,
        dtheta=float(dtheta),
        diag=dict(d.get("diag", {})),
    )
