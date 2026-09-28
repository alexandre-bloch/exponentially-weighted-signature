"""The log-EWS feature (LogEWS in ews/layer.py): with A = 0 it is the compressed log-signature,
exp(log(EWS)) recovers the EWS for a general A, the flat length is the Witt dimension, every
readout has the right shape, gradients through it are finite and match finite differences, and a
classifier built on it trains end to end.
"""

from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.argv = [sys.argv[0], "--gpu", ""]

import jax

jax.config.update("jax_enable_x64", True)

from math import factorial

import jax.numpy as jnp
import jax.random as jr
import signax
from signax.tensor_ops import log as tlog
from signax.tensor_ops import otimes

from ews.core.output_format import flatten_signature_list
from ews.layer import EWS, LogEWS, log_signature_dim

DT = jnp.float64


# ----------------------------------------------------------------------------- helpers
def _mult_nonscalar(X, Y, N, d):
    """Truncated tensor product of two elements with zero scalar part.
    X[i], Y[i] are the degree-(i+1) components; result Z[i] likewise, degrees 1..N."""
    Z = [jnp.zeros((d,) * (i + 1), dtype=DT) for i in range(N)]
    for i in range(N):
        for j in range(N):
            n = (i + 1) + (j + 1)
            if n <= N:
                Z[n - 1] = Z[n - 1] + otimes(X[i], Y[j])
    return Z


def tensor_exp(ell, N, d):
    """exp of an expanded log (scalar part 0), truncated at depth N.
    ell: list [L1..LN], Lk shape (d,)*k. Returns levels 1..N (scalar 1 implicit)."""
    out = [jnp.asarray(Lk, dtype=DT) for Lk in ell]  # k=1 term (/1!)
    P = [jnp.asarray(Lk, dtype=DT) for Lk in ell]  # P = ell^{otimes 1}
    for k in range(2, N + 1):
        P = _mult_nonscalar(P, ell, N, d)  # P = ell^{otimes k}
        out = [out[m] + P[m] / factorial(k) for m in range(N)]
    return out


def _maxabs(a, b):
    return float(jnp.max(jnp.abs(jnp.asarray(a) - jnp.asarray(b))))


def _rand_path(key, L, d):
    return jnp.cumsum(0.3 * jr.normal(key, (L, d), dtype=DT), axis=0)


def _witt(d, N):
    """sum_k W_d(k) via signax's Lyndon-word enumeration."""
    return int(sum(int(w.shape[0]) for w in signax.utils.lyndon_words(N, d)))


# ------------------------------------------------------------------------------- tests
def test_tensor_exp_helper_inverts_log():
    """exp(log(sig)) == sig for a plain signature: proves the helper is trustworthy."""
    d, N, L = 3, 3, 7
    path = _rand_path(jr.PRNGKey(0), L, d)
    sig = signax.signature(path, depth=N, stream=False, flatten=False)
    rt = tensor_exp(tlog(sig), N, d)
    err = max(_maxabs(a, b) for a, b in zip(sig, rt))
    assert err < 1e-10, f"tensor_exp helper broken: err={err:.2e}"


def test_zero_drift_gives_the_log_signature():
    """A=0: reweighting is the identity, so LogEWS terminal == compressed log-signature."""
    d, N, L = 3, 3, 8
    path = _rand_path(jr.PRNGKey(1), L, d)
    clock = jnp.linspace(0.0, 1.0, L, dtype=DT)
    layer = LogEWS(jnp.zeros((d, d), dtype=DT), N=N, M=8, dtype=DT)
    got = layer(path, clock, readout="terminal")

    ref_list = signax.signature_to_logsignature(
        signax.signature(path, depth=N, stream=False, flatten=False)
    )
    ref = flatten_signature_list(ref_list, include_level0=False)
    err = _maxabs(got, ref)
    assert got.shape == ref.shape == (_witt(d, N),)
    assert err < 1e-8, f"A=0 log-EWS != log-signature: err={err:.2e}"


def test_exp_of_log_recovers_the_ews():
    """exp(log(EWS)) == EWS on the expanded log, for a nonzero (general) A."""
    d, N, L = 3, 3, 8
    path = _rand_path(jr.PRNGKey(2), L, d)
    clock = jnp.linspace(0.0, 1.0, L, dtype=DT)
    A = 0.5 * jr.normal(jr.PRNGKey(3), (d, d), dtype=DT)
    ews = EWS(A, N=N, M=8, dtype=DT, van_loan_expm="exact", chen_expm="exact")
    ews_list = ews(path, clock, readout="terminal", output_format="list")
    rt = tensor_exp(tlog(ews_list), N, d)
    err = max(_maxabs(a, b) for a, b in zip(ews_list, rt))
    assert err < 1e-9, f"exp(log(EWS)) != EWS: err={err:.2e}"


def test_flat_length_is_the_witt_dimension():
    """Flat length == sum_k W_d(k); no level-0 entry."""
    for d, N, expect in [(3, 3, 14), (4, 2, 4 + 6), (2, 4, 2 + 1 + 2 + 3)]:
        assert log_signature_dim(d, N) == _witt(d, N) == expect, (d, N)
    d, N, L = 4, 3, 9
    path = _rand_path(jr.PRNGKey(4), L, d)
    clock = jnp.linspace(0.0, 1.0, L, dtype=DT)
    layer = LogEWS(jnp.zeros((d, d), dtype=DT), N=N, M=4, dtype=DT)
    got = layer(path, clock, readout="terminal")
    assert got.shape == (log_signature_dim(d, N),)


def test_readout_shapes():
    """terminal/mean/max -> (D_log,); prefix -> (L, D_log); batched adds a leading axis."""
    d, N, L, B = 3, 2, 8, 5
    D = log_signature_dim(d, N)
    key = jr.PRNGKey(6)
    path = _rand_path(key, L, d)
    clock = jnp.linspace(0.0, 1.0, L, dtype=DT)
    layer = LogEWS(jnp.zeros((d, d), dtype=DT), N=N, M=4, dtype=DT)

    for mode in ("terminal", "mean", "max"):
        out = layer(path, clock, readout=mode)
        assert out.shape == (D,), (mode, out.shape)
        assert bool(jnp.all(jnp.isfinite(out)))
    pref = layer(path, clock, readout="prefix")
    assert pref.ndim == 2 and pref.shape[1] == D, pref.shape

    paths = jnp.stack([_rand_path(k, L, d) for k in jr.split(key, B)])
    clocks = jnp.broadcast_to(clock, (B, L))
    for mode in ("terminal", "mean", "max"):
        out = layer(paths, clocks, readout=mode)
        assert out.shape == (B, D), (mode, out.shape)
    pref_b = layer(paths, clocks, readout="prefix")
    assert pref_b.ndim == 3 and pref_b.shape[0] == B and pref_b.shape[2] == D


def test_gradients_are_finite_and_match_finite_differences():
    """jax.grad of a scalar loss through LogEWS w.r.t. A is finite, non-zero, and matches
    finite differences, i.e. the log/compress readout is safely trainable."""
    d, N, L = 3, 3, 6
    path = _rand_path(jr.PRNGKey(7), L, d)
    clock = jnp.linspace(0.0, 1.0, L, dtype=DT)
    A0 = 0.2 * jr.normal(jr.PRNGKey(8), (d, d), dtype=DT)

    def loss(A, readout):
        feat = LogEWS(A, N=N, M=6, dtype=DT)(path, clock, readout=readout)
        return jnp.sum(feat**2)

    # terminal + a pooled (per-knot log) readout both differentiate cleanly
    for readout in ("terminal", "mean"):
        g = jax.grad(lambda A: loss(A, readout))(A0)
        assert bool(jnp.all(jnp.isfinite(g))), f"non-finite grad ({readout})"
        assert float(jnp.max(jnp.abs(g))) > 0, f"zero grad ({readout})"

    # finite-difference check a few entries (terminal)
    g = jax.grad(lambda A: loss(A, "terminal"))(A0)
    eps, maxrel = 1e-5, 0.0
    for i, j in [(0, 0), (1, 2), (2, 1)]:
        fd = (loss(A0.at[i, j].add(eps), "terminal") - loss(A0.at[i, j].add(-eps), "terminal")) / (
            2 * eps
        )
        maxrel = max(maxrel, abs(float(fd) - float(g[i, j])) / (abs(float(fd)) + 1e-9))
    assert maxrel < 1e-4, f"grad != finite-diff: relerr={maxrel:.2e}"


def _classifier(feature, d_in=4, n_classes=3, seed=0):
    from experiments.uea.train import Config, build_model

    model_cfg = {
        "A": {"parametrization": "general", "init": {"type": "general", "variance": 1.0}},
        "ews_depth": 3,
        "M": 4,
        "head_type": "mlp",
        "head_hidden_dims": [16],
        "feature_norm": "learned_gain",
        "readout": "terminal",
        "feature": feature,
    }
    cfg = Config(data={}, preprocessing={"dtype": "float32"}, model=model_cfg, training={})
    return build_model(cfg, d_in=d_in, n_classes=n_classes, key=jr.PRNGKey(seed))


def test_log_signature_classifier_trains_end_to_end():
    """A classifier on the log-signature feature has log-sized features (no level-0 constant), the
    same logit shape as the signature classifier, and finite, non-zero gradients through the whole
    model."""
    import equinox as eqx
    import numpy as np

    B, L, d, n_classes = 5, 7, 4, 3
    rng = np.random.RandomState(0)
    paths = jnp.asarray(np.cumsum(0.1 * rng.randn(B, L, d).astype("float32"), axis=1))
    clocks = jnp.asarray(np.broadcast_to(np.linspace(0, 1, L, dtype="float32"), (B, L)))
    log_model, sig_model = _classifier("logsignature"), _classifier("signature")
    assert isinstance(log_model.ewms, LogEWS) and isinstance(sig_model.ewms, EWS)
    log_out, sig_out = log_model(paths, clocks), sig_model(paths, clocks)
    assert log_out.shape == sig_out.shape == (B, n_classes)
    assert bool(jnp.all(jnp.isfinite(log_out)))
    d_eff, N = log_model.ewms.d, log_model.ewms.N
    log_feat = log_model(paths, clocks, return_features=True)
    assert log_feat.shape == (B, log_signature_dim(d_eff, N))
    assert log_model.head.layers[0].weight.shape[1] == log_signature_dim(d_eff, N)
    assert log_model.feature_norm.level_slices[-1][1] == log_signature_dim(d_eff, N)

    labels = jnp.array([0, 1, 2, 0, 1])

    def cross_entropy(model):
        logits = model(paths, clocks)
        return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(B), labels])

    grads = eqx.filter_grad(cross_entropy)(log_model)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
    assert any(float(jnp.max(jnp.abs(g))) > 0 for g in leaves)
