"""LRU -> EWS. Method B (matrix logarithm) only.

The LRU is parameterised directly in discrete time: there is no underlying ODE, so Method A does not exist for this
family. That is a property of the model, not a gap in the implementation.
"""

import numpy as np

from ews.conversion.base import ConversionResult
from ews.conversion.families.linoss_im import _archive


def to_ews(layer, method="B", *, check_tol=1e-12, wrap_branch=True, dtheta=1.0):
    if method != "B":
        raise NotImplementedError(
            "the LRU has no continuous-time parametrisation, so Method A does not exist for it; use B."
        )
    d = _archive().lru_to_ews(layer, check_tol=check_tol, wrap_branch=wrap_branch)
    return ConversionResult(
        A=np.asarray(d["A"], np.float64),
        B=np.asarray(d["B"], np.float64),
        C=np.asarray(d["C"], np.float64),
        D=None if d.get("D") is None else np.asarray(d["D"], np.float64),
        m=int(d["m"]),
        family="lru",
        method="B",
        dtheta=float(dtheta),
        diag=dict(d.get("diag", {})),
    )
