"""The documentation and the code agree on the option vocabulary.

Forward: every option value or key the code defines appears in docs/. Reverse: every key named in
the first column of an option table in docs/ is one the code defines."""

import glob
import os
import re

from ews.clocks import CLOCK_MODES, CLOCK_SOURCES, ClockSpec
from ews.config import (
    BLOCK_SCAFFOLD_KEYS,
    BLOCK_SECTIONS,
    COMPUTE_KEYS,
    LAYER_MODEL_KEYS,
    LAYER_TRAINING_KEYS,
    PREPROCESSING_KEYS,
    STACK_MODEL_KEYS,
    STACK_TRAINING_KEYS,
)
from ews.core.kernels.combine import COMBINE_KINDS
from ews.core.leaf import LEAF_KINDS
from ews.params import A_PARAM_NAMES, B_INIT_NAMES
from ews.paths import CLOCK_CHANNELS, REPRESENTATIONS, PathSpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = "\n".join(open(p).read() for p in sorted(glob.glob(f"{ROOT}/docs/*.md")))

CODE_VALUES = {
    "clock modes": CLOCK_MODES,
    "clock sources": CLOCK_SOURCES,
    "representations": REPRESENTATIONS,
    "clock channels": [c for c in CLOCK_CHANNELS if c],
    "A parametrisations": A_PARAM_NAMES,
    "B inits": B_INIT_NAMES,
    "expm methods": ("exact", "cayley", "pade", "euler"),
    "leaves": LEAF_KINDS,
    "combines": COMBINE_KINDS,
    "reweighting": ("displacement", "displacement_uniform", "displacement_uniform_basepoint"),
    "terminal reduce": ("tree", "assoc"),
    "graded norm": ("live_rms", "static_gain", "dilation"),
    "norm placement": ("pre", "slice_post", "none"),
    "channel stage": ("mlp", "tanh", "glu", "gelu_glu"),
    "heads": ("seq2seq", "mean_pool"),
    "readouts": ("terminal", "mean", "max"),
    "feature norm": ("none", "learned_gain", "batchnorm"),
    "features": ("signature", "logsignature"),
}
CODE_KEYS = {
    "ClockSpec": ClockSpec.__dataclass_fields__,
    "PathSpec": PathSpec.__dataclass_fields__,
    "compute": COMPUTE_KEYS,
    "block": BLOCK_SCAFFOLD_KEYS,
    "block sections": BLOCK_SECTIONS,
    "stack model": STACK_MODEL_KEYS,
    "layer model": LAYER_MODEL_KEYS,
    "stack training": STACK_TRAINING_KEYS,
    "layer training": LAYER_TRAINING_KEYS,
    "preprocessing": PREPROCESSING_KEYS,
}
EXTRA_KEYS = {  # documented keys that live inside A / B / dataset sections rather than in a schema list
    "parametrization",
    "init",
    "learn",
    "blocks",
    "kinds",
    "epsilon",
    "omega_scale",
    "damp_init_mean",
    "positivity",
    "damp_init_std",
    "enabled",
    "out_dim",
    "_preserve_first_channel",
    "file",
    "task",
    "length",
    "derivative_order",
    "derivative_window",
    "derivative_polyorder",
    "normalize",
    "normalize_mode",
    "normalize_range",
    "per_sample_normalize",
    "per_sample_norm_stat",
    "basepoint",
    "clock_channel",
    "path",
    "clock",
    "append_clock",
    "compute",
    "A",
    "B",
    "block",
    "blocks",
    "training",
    "model",
    "preprocessing",
    "dataset",
    "num_blocks",
    "hidden_dim",
    "autotune",
    "deterministic-ops",
    "type",
    "mean",
    "variance",
    "sigma_range",  # init sub-keys
    "a_scale",
    "output_format",  # layer call arguments
}


def test_every_code_option_is_documented():
    missing = [
        (group, v)
        for group, vals in CODE_VALUES.items()
        for v in vals
        if f"`{v}`" not in DOCS and f"`{v}" not in DOCS
    ]
    missing += [
        (group, k)
        for group, keys in CODE_KEYS.items()
        for k in keys
        if f"`{k}`" not in DOCS and f"`{k}" not in DOCS and f"{k}`" not in DOCS
    ]
    assert not missing, f"undocumented: {missing}"


def test_every_documented_key_exists():
    known = (
        set(EXTRA_KEYS)
        | {v for vals in CODE_VALUES.values() for v in vals}
        | {k for keys in CODE_KEYS.values() for k in keys}
    )
    unknown = set()
    for line in DOCS.splitlines():
        if not line.startswith("| `"):
            continue
        first = line.split("|")[1]
        for tok in re.findall(r"`([A-Za-z_][A-Za-z0-9_.\-]*)`", first):
            base = tok.split(".")[-1] if tok.startswith(("init.", "model.", "path.")) else tok
            if base not in known and tok not in known:
                unknown.add(tok)
    assert not unknown, f"documented but not defined in code: {sorted(unknown)}"
