"""EWS: the Exponentially Weighted Signature in JAX.

    from ews import EWS, LogEWS, B, FeatureNorm            # the layer-only pieces
    from ews import EWSBlock, StackedEWS                    # the block and the stack
    from ews import ClockSpec, PathSpec, build_ab_from_config

The package is organised by the stages of the method: `paths` (what the path is), `clocks` (the
clock), `params` (the A and B parametrisations), `core` (the signature computation), `layer`,
`norms`, `block`; `config` holds the sectioned config schema and `determinism` the XLA switches.
"""

from ews.block import EWSBlock, StackedEWS, TokenCachedStackedEWS
from ews.clocks import ClockSpec
from ews.layer import EWS, B, BToSignature, LogEWS
from ews.norms import FeatureNorm, GradedRMSNorm, RMSNorm
from ews.params import build_ab_from_config
from ews.paths import PathSpec

__all__ = [
    "EWS",
    "LogEWS",
    "B",
    "BToSignature",
    "FeatureNorm",
    "GradedRMSNorm",
    "RMSNorm",
    "EWSBlock",
    "StackedEWS",
    "TokenCachedStackedEWS",
    "ClockSpec",
    "PathSpec",
    "build_ab_from_config",
]
__version__ = "0.1.0"
