"""Clock modules: the five modes share one interface; the module wrappers reproduce the builders exactly."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from ews.clocks import (
    ClockSpec,
    FixedClock,
    GatedClock,
    build_clock_jnp,
    build_clock_np,
    make_clock,
)

jax.config.update("jax_enable_x64", True)


def _driver(L=9, d=3, seed=0):
    return jnp.asarray(np.random.default_rng(seed).normal(size=(L, d)))


@pytest.mark.parametrize("mode", ["time", "blended_variation", "blended_quad_variation"])
@pytest.mark.parametrize("scaling", ["none", "[0,1]", "[-1,1]"])
def test_fixed_clock_matches_builder(mode, scaling):
    spec = ClockSpec(mode=mode, alpha=0.4, scaling=scaling)
    clock = make_clock(spec, dim=3)
    assert isinstance(clock, FixedClock) and clock.is_static
    t = jnp.arange(9, dtype=jnp.float64)
    d = _driver()
    expected = build_clock_jnp(t, d, mode, 0.4, scaling)
    np.testing.assert_array_equal(np.asarray(clock(t, d, None)), np.asarray(expected))


@pytest.mark.parametrize("mode", ["time", "blended_variation", "blended_quad_variation"])
def test_numpy_and_jnp_builders_agree(mode):
    t = np.arange(9, dtype=np.float64)
    d = np.asarray(_driver())
    a = build_clock_np(t, d, mode, 0.3, "[0,1]")
    b = build_clock_jnp(jnp.asarray(t), jnp.asarray(d), mode, 0.3, "[0,1]", variation_eps=0.0)
    np.testing.assert_allclose(np.asarray(b), a, rtol=1e-12, atol=1e-12)


def test_gated_clock_formula_and_mask():
    spec = ClockSpec(
        mode="token_gated",
        scaling="none",
        normalize="constant",
        norm_constant=4.0,
        gate_init_bias=-0.5,
    )
    clock = make_clock(spec, dim=3, key=jr.PRNGKey(1), dtype=jnp.float64)
    assert isinstance(clock, GatedClock) and not clock.is_static
    d = _driver()
    t = jnp.arange(9, dtype=jnp.float64)
    dx = jnp.diff(d, axis=0)
    incr = jax.nn.relu(dx @ clock.w - clock.b)
    expected = jnp.concatenate([jnp.zeros(1), jnp.cumsum(incr)]) / 4.0
    np.testing.assert_allclose(np.asarray(clock(t, d, None)), np.asarray(expected), rtol=1e-12)
    mask = jnp.arange(9) < 6
    theta = clock(t, d, mask)
    assert np.all(np.asarray(theta[6:]) == np.asarray(theta[5]))  # frozen pads
    assert np.all(np.diff(np.asarray(theta)) >= 0)  # monotone


def test_clamped_gate_staggered_bias():
    spec = ClockSpec(
        mode="clamped_gate",
        scaling="none",
        normalize="constant",
        norm_constant=2.0,
        gate_init_bias=(-1.0, 0.0, 1.0),
    )
    clock = make_clock(spec, dim=3, key=jr.PRNGKey(2), dtype=jnp.float64)
    assert clock.w.shape == (3, 3) and tuple(np.asarray(clock.b)) == (-1.0, 0.0, 1.0)
    inc = clock.per_token_increment(_driver(L=5))
    assert inc.shape == (5,) and np.all(np.asarray(inc) >= 0) and np.all(np.asarray(inc) <= 3 / 2.0)


def test_per_token_increment_matches_call_for_token_streams():
    """For an increment stream of one-hot-like tokens the per-token increment equals diff(theta)."""
    spec = ClockSpec(
        mode="blended_variation", alpha=0.0, scaling="none", normalize="constant", norm_constant=3.0
    )
    clock = make_clock(spec, dim=4)
    embeds = jnp.asarray(np.random.default_rng(3).normal(size=(5, 4)))
    ids = jnp.asarray([2, 0, 3, 3, 1])
    rows = embeds[ids]
    path = jnp.cumsum(rows, axis=0)  # increment representation
    theta = clock(jnp.arange(5, dtype=jnp.float64), path, None)
    per_tok = clock.per_token_increment(embeds)
    np.testing.assert_allclose(np.asarray(jnp.diff(theta)), np.asarray(per_tok[ids[1:]]), rtol=1e-9)


def test_guards():
    with pytest.raises(ValueError):
        ClockSpec(mode="token_gated").validate()  # needs constant normalisation
    with pytest.raises(ValueError):
        ClockSpec(mode="nope").validate()
    with pytest.raises(ValueError):
        make_clock(
            ClockSpec(mode="token_gated", scaling="none", normalize="constant", norm_constant=1.0),
            dim=2,
        )  # no key
    with pytest.raises(ValueError):
        make_clock(
            ClockSpec(
                mode="blended_variation",
                alpha=0.5,
                scaling="none",
                normalize="constant",
                norm_constant=1.0,
            ),
            2,
        ).per_token_increment(jnp.ones((2, 2)))
    with pytest.raises(ValueError):
        ClockSpec.from_config({"mode": "time", "bogus": 1})


def test_steps_scaling_is_the_sample_counter():
    """scaling='steps': a time clock that advances one unit per sample (theta_i = i), not normalised by length; a
    saturated clamped-gate clock (every gate at its ceiling) is exactly this clock."""
    t = np.linspace(0.0, 1.0, 7)
    assert np.array_equal(build_clock_np(t, None, "time", 1.0, "steps"), np.arange(7.0))
    th = build_clock_jnp(jnp.asarray(t), None, "time", 1.0, "steps")
    assert np.array_equal(np.asarray(th), np.arange(7.0))
    clock = make_clock(ClockSpec(mode="time", scaling="steps"), dim=3)
    assert isinstance(clock, FixedClock) and np.array_equal(
        np.asarray(clock(jnp.asarray(t), None, None)), np.arange(7.0)
    )
    # the saturated gated clock: w = 0, b = -2 -> each of the K gates clips to 1, increment K / norm_constant = 1
    spec = ClockSpec(
        mode="clamped_gate",
        scaling="none",
        normalize="constant",
        norm_constant=4.0,
        gate_init_bias=-2.0,
    )
    gated = make_clock(spec, dim=3, key=jr.PRNGKey(0))
    import equinox as eqx

    gated = eqx.tree_at(lambda c: c.w, gated, jnp.zeros_like(gated.w))
    assert np.array_equal(np.asarray(gated(jnp.arange(7.0), _driver(L=7), None)), np.arange(7.0))
    with pytest.raises(ValueError):
        ClockSpec(mode="blended_variation", alpha=0.5, scaling="steps").validate()


def test_steps_scaling_counts_the_basepoint_step():
    from ews.paths import PathSpec, prepare_paths_np

    path_spec = PathSpec(normalize=False, basepoint=True).validate()
    clock_spec = ClockSpec(mode="time", scaling="steps").validate()
    paths = [np.random.default_rng(0).normal(size=(5, 2))]
    clocks = [np.linspace(0.0, 1.0, 5)]
    _, thetas, _, _ = prepare_paths_np(paths, clocks, path_spec, clock_spec)
    assert np.array_equal(
        np.asarray(thetas[0]), np.arange(6.0)
    )  # 5 samples + the basepoint: 0, 1, ..., 5
