"""The initial weights of a signature classifier (general A, lift B, MLP head, learned-gain feature
norm, fixed seed) are pinned in tests/reference_model_init.npz, so a change in the order in which the
random keys are consumed at construction shows up as a mismatch.

Runs on CPU. To regenerate the reference after an intended change:
EWS_WRITE_REFERENCES=1 pytest tests/test_model_init_pinned.py
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.argv = [sys.argv[0], "--gpu", ""]

import equinox as eqx
import jax
import jax.random as jr
import numpy as np

from experiments.uea.train import Config, build_model

REFERENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_model_init.npz")
WRITE = os.environ.get("EWS_WRITE_REFERENCES") == "1"


def _weights():
    model_cfg = {
        "A": {"parametrization": "general", "init": {"type": "general", "variance": 1.0}},
        "B": {
            "enabled": True,
            "out_dim": 5,
            "parametrization": "general",
            "init": {"type": "general", "variance": 1.0},
        },
        "ews_depth": 2,
        "M": 4,
        "head_type": "mlp",
        "head_hidden_dims": [16],
        "feature_norm": "learned_gain",
        "readout": "terminal",
    }
    cfg = Config(data={}, preprocessing={"dtype": "float32"}, model=model_cfg, training={})
    model = build_model(cfg, d_in=4, n_classes=3, key=jr.PRNGKey(0))
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))]


if WRITE:
    np.savez(REFERENCE, *_weights())


def test_initial_weights_match_reference():
    ref = np.load(REFERENCE)
    got = _weights()
    assert len(ref.files) == len(got), (
        f"number of weight arrays changed: {len(ref.files)} -> {len(got)}"
    )
    for key, g in zip(ref.files, got):
        assert ref[key].shape == g.shape
        np.testing.assert_array_equal(g, ref[key])
