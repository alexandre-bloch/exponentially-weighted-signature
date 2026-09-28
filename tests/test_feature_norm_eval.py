"""feature_norm='batchnorm' at evaluation: statistics of the evaluated batch (default) or given ones (training set)."""

import jax.numpy as jnp
import numpy as np
import pytest

from ews.norms import FeatureNorm


def _norm(d=3, N=2):
    return FeatureNorm(d=d, N=N, mode="batchnorm")


def test_own_statistics_reproduce_the_call():
    fn = _norm()
    x = jnp.asarray(np.random.default_rng(0).normal(size=(16, fn.level_slices[-1][1])))
    np.testing.assert_allclose(
        np.asarray(fn.apply_with_stats(x, fn.level_stats(x))), np.asarray(fn(x)), rtol=0, atol=1e-6
    )


def test_given_statistics_make_predictions_independent():
    fn = _norm()
    rng = np.random.default_rng(1)
    D = fn.level_slices[-1][1]
    train = jnp.asarray(rng.normal(size=(64, D)))
    a, b = jnp.asarray(rng.normal(size=(8, D))), jnp.asarray(3.0 + rng.normal(size=(8, D)))
    stats = fn.level_stats(train)
    alone = fn.apply_with_stats(a[:1], stats)
    with_others = fn.apply_with_stats(jnp.concatenate([a[:1], b]), stats)[:1]
    assert np.array_equal(
        np.asarray(alone), np.asarray(with_others)
    )  # a sample's output ignores its batch
    batch_a = fn(jnp.concatenate([a, a]))[:1]
    batch_b = fn(jnp.concatenate([a, b]))[:1]
    assert not np.allclose(
        np.asarray(batch_a), np.asarray(batch_b)
    )  # the default (batch statistics) does not


def test_only_defined_for_batchnorm():
    fn = FeatureNorm(d=3, N=2, mode="learned_gain")
    with pytest.raises(ValueError):
        fn.apply_with_stats(jnp.zeros((2, fn.level_slices[-1][1])), [(0.0, 1.0)] * 2)
