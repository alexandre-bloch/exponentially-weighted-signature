"""The shared regression trainer pieces: batching, the efm positivity option, the shipped configs."""

import glob
import json
import os
import sys

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.argv = [sys.argv[0], "--gpu", ""]
os.environ["JAX_PLATFORMS"] = "cpu"

from ews.params import AConfig, InitConfig, build_a_param  # noqa: E402
from experiments.regression_common import (  # noqa: E402
    build_model,
    create_batch_iterator,
    eigenvalues,
    group_by_length,
    load_config,
    make_optimizer,
    split_rmse,
)

SDE_CONFIGS = sorted(glob.glob(f"{ROOT}/sde_results/*/*/config.json"))


def _naive_batches(paths, clocks, targets, batch_size, key, n_batches):
    """The reference batching: one permutation per epoch, each batch stacked from the lists."""
    out = []
    n = len(paths)
    while len(out) < n_batches:
        key, subkey = jr.split(key)
        idx = jr.permutation(subkey, jnp.arange(n))
        for i in range(0, n, batch_size):
            batch = idx[i : i + batch_size]
            out.append(
                tuple(jnp.stack([arr[int(j)] for j in batch]) for arr in (paths, clocks, targets))
            )
            if len(out) == n_batches:
                break
    return out


def test_batch_iterator_matches_reference_order():
    rng = np.random.default_rng(0)
    paths = [jnp.asarray(rng.standard_normal((7, 3)), dtype=jnp.float32) for _ in range(11)]
    clocks = [jnp.asarray(np.linspace(0, 1, 7), dtype=jnp.float32) for _ in range(11)]
    targets = [jnp.asarray(rng.standard_normal(6), dtype=jnp.float32) for _ in range(11)]
    key = jr.PRNGKey(3)
    it = create_batch_iterator(paths, clocks, targets, 4, key)
    got = [next(it) for _ in range(7)]  # more than one epoch
    ref = _naive_batches(paths, clocks, targets, 4, key, 7)
    for g, r in zip(got, ref):
        for a, b in zip(g, r):
            assert a.shape == b.shape
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_batch_iterator_mixed_lengths():
    rng = np.random.default_rng(1)
    paths = [jnp.asarray(rng.standard_normal((5, 2))) for _ in range(3)] + [
        jnp.asarray(rng.standard_normal((6, 2)))
    ]
    clocks = [jnp.zeros(len(p)) for p in paths]
    targets = [jnp.zeros(len(p) - 1) for p in paths]
    it = create_batch_iterator(paths[:3], clocks[:3], targets[:3], 3, jr.PRNGKey(0))
    bp, bc, bt = next(it)
    assert bp.shape == (3, 5, 2)
    groups = group_by_length(paths, clocks, targets)
    assert [g[0].shape[0] for g in groups] == [3, 1]


def test_efm_positivity():
    theta = jnp.asarray([-1.0, 0.5, 2.0])
    soft = build_a_param(
        AConfig("efm", InitConfig("efm"), positivity="softplus"),
        w=3,
        key=jr.PRNGKey(0),
        dtype=jnp.float32,
    )
    square = build_a_param(
        AConfig("efm", InitConfig("efm"), positivity="square"),
        w=3,
        key=jr.PRNGKey(0),
        dtype=jnp.float32,
    )
    soft = jax.tree_util.tree_map(lambda x: theta if x.shape == theta.shape else x, soft)
    square = jax.tree_util.tree_map(lambda x: theta if x.shape == theta.shape else x, square)
    np.testing.assert_allclose(
        np.diag(np.asarray(soft())), np.log1p(np.exp(np.asarray(theta))), rtol=1e-6
    )
    np.testing.assert_allclose(np.diag(np.asarray(square())), np.asarray(theta) ** 2, rtol=1e-6)
    with pytest.raises(ValueError):
        build_a_param(
            AConfig("efm", InitConfig("efm"), positivity="abs"),
            w=3,
            key=jr.PRNGKey(0),
            dtype=jnp.float32,
        )


def test_eigenvalues_pairs():
    A = np.array([[0.0, -2.0], [2.0, 0.0]])
    ev = eigenvalues(A)
    np.testing.assert_allclose(sorted(v[1] for v in ev), [-2.0, 2.0], atol=1e-12)
    assert all(abs(v[0]) < 1e-12 for v in ev)


@pytest.mark.parametrize("path", SDE_CONFIGS, ids=lambda p: os.path.relpath(p, ROOT))
def test_sde_config_builds_and_trains_one_step(path):
    cfg = load_config(path)
    model = build_model(cfg, d_in=3, key=jr.PRNGKey(0))
    assert model.ews.N == 2 and model.ews.M == 10
    assert model.ews.van_loan_expm == "exact" and model.ews.chen_expm == "exact"
    a_cfg = cfg.model["A"]
    if a_cfg["parametrization"] == "efm":
        assert a_cfg["positivity"] == "square"
        assert model.ews.A_param.positivity == "square"
    opt, opt_state = make_optimizer(model, cfg, 2000)
    rng = np.random.default_rng(0)
    paths = [jnp.asarray(rng.standard_normal((9, 3)), dtype=jnp.float32) for _ in range(4)]
    clocks = [jnp.asarray(np.linspace(0, 1, 9), dtype=jnp.float32) for _ in range(4)]
    targets = [jnp.asarray(rng.standard_normal(8), dtype=jnp.float32) for _ in range(4)]
    rmse = split_rmse(model, group_by_length(paths, clocks, targets), lambda p: p)
    assert np.isfinite(rmse)


def test_expressivity_targets_and_learners_cover_the_grid():
    def entry(p):
        return os.path.basename(os.path.dirname(p))

    names = {entry(p) for p in SDE_CONFIGS if "/expressivity/" in p}
    assert names == {f"{t}_{m}" for t in ("ews", "efm", "sig") for m in ("ews", "efm", "sig")}
    for p in SDE_CONFIGS:
        cfg = json.load(open(p))
        learner = entry(p).split("_")[-1]
        expected = {"ews": "general", "efm": "efm", "sig": "sig"}[learner]
        assert cfg["model"]["A"]["parametrization"] == expected
        assert cfg["model"]["A"].get("learn", True) == (learner != "sig")
