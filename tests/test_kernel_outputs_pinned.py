"""The classifier's logits are pinned for every combination of reweighting mode, readout and
terminal reduce (and the log-signature feature), so any change to the reweighting kernels or the
scan shows up as a mismatch against tests/reference_kernel_outputs.npz.

Runs on CPU with uniform clocks (so the uniform Van Loan modes are exercised). To regenerate the
reference after an intended change: EWS_WRITE_REFERENCES=1 pytest tests/test_kernel_outputs_pinned.py
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.argv = [sys.argv[0], "--gpu", ""]

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from experiments.uea.train import Config, build_model

REFERENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_kernel_outputs.npz")
WRITE = os.environ.get("EWS_WRITE_REFERENCES") == "1"

B, L, D, NCLASS = 4, 9, 3, 3
_rng = np.random.RandomState(0)
PATHS = jnp.asarray(np.cumsum(0.1 * _rng.randn(B, L, D).astype("float32"), axis=1))
CLOCKS = jnp.asarray(np.broadcast_to(np.linspace(0, 1, L, dtype="float32"), (B, L)))


def _model_cfg(reweighting_mode, readout, terminal_reduce, feature="signature"):
    return {
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
        "readout": readout,
        "reweighting_mode": reweighting_mode,
        "terminal_reduce": terminal_reduce,
        "van_loan_expm": "exact",
        "chen_expm": "exact",
        "feature": feature,
    }


def _combinations():
    out = []
    for mode in ["displacement", "displacement_uniform", "displacement_uniform_basepoint"]:
        for reduce in ["tree", "assoc"]:  # terminal_reduce matters only for the terminal readout
            out.append((f"{mode}|terminal|{reduce}", _model_cfg(mode, "terminal", reduce)))
        for readout in ["mean", "max"]:
            out.append((f"{mode}|{readout}|-", _model_cfg(mode, readout, "tree")))
    out.append(
        (
            "displacement|terminal|tree|log",
            _model_cfg("displacement", "terminal", "tree", "logsignature"),
        )
    )
    return out


def _logits(model_cfg):
    cfg = Config(data={}, preprocessing={"dtype": "float32"}, model=model_cfg, training={})
    model = build_model(cfg, d_in=D, n_classes=NCLASS, key=jr.PRNGKey(0))
    return np.asarray(model(PATHS, CLOCKS))


if WRITE:
    np.savez(REFERENCE, **{name: _logits(cfg) for name, cfg in _combinations()})


@pytest.mark.parametrize("name,model_cfg", _combinations(), ids=[n for n, _ in _combinations()])
def test_logits_match_reference(name, model_cfg):
    ref = np.load(REFERENCE)
    assert name in ref.files, f"combination missing from the reference file: {name}"
    got = _logits(model_cfg)
    assert ref[name].shape == got.shape
    np.testing.assert_array_equal(got, ref[name])
