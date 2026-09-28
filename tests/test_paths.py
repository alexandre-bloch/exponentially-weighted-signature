"""Path stage: spec parsing, the numpy precompute, and the increment representation."""

import jax.numpy as jnp
import numpy as np
import pytest

from ews.clocks import ClockSpec
from ews.paths import (
    PathSpec,
    legacy_specs,
    prepare_paths_np,
    specs_from_config,
    to_increment_path_jnp,
)


def _paths(n=3, L=8, d=2, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(L, d)) for _ in range(n)], [np.linspace(0, 1, L) for _ in range(n)]


def test_legacy_keys_translate():
    pp = {
        "path_derivative": 2,
        "savgol_window": 11,
        "add_basepoint": True,
        "augment_with_clock": True,
        "clock_mode": "blended_variation",
        "clock_alpha": 0.7,
        "clock_normalize": "[-1,1]",
        "normalize_range": "[-1,1]",
    }
    path, clock, append = legacy_specs(pp)
    assert (
        path.representation == "derivative"
        and path.derivative_order == 2
        and path.derivative_window == 11
    )
    assert path.basepoint and path.clock_channel == "main" and append is None
    assert clock.mode == "blended_variation" and clock.alpha == 0.7 and clock.scaling == "[-1,1]"
    sectioned = {"path": {"representation": "value"}, "clock": {"mode": "time"}}
    p2, c2, a2 = specs_from_config(sectioned)
    assert p2.representation == "value" and c2.mode == "time" and a2 is None


def test_prepare_shapes_and_clock_channel():
    paths, clocks = _paths()
    spec = PathSpec(representation="value", normalize=True, basepoint=True, clock_channel="main")
    out_paths, thetas, scaler, d_path = prepare_paths_np(
        paths, clocks, spec, ClockSpec(mode="blended_variation", alpha=0.5)
    )
    assert d_path == 3 and out_paths[0].shape == (9, 3) and thetas[0].shape == (9,)
    np.testing.assert_array_equal(
        np.asarray(out_paths[0][:, 0]), np.asarray(thetas[0])
    )  # the clock is channel 0
    assert float(thetas[0][0]) == float(thetas[0][1])  # basepoint duplicates theta_0


def test_separate_clock_channel_needs_spec():
    paths, clocks = _paths()
    with pytest.raises(ValueError):
        prepare_paths_np(paths, clocks, PathSpec(clock_channel="separate"), ClockSpec())
    out, _, _, d = prepare_paths_np(
        paths,
        clocks,
        PathSpec(clock_channel="separate", normalize=False),
        ClockSpec(),
        append_clock=ClockSpec(mode="blended_quad_variation", alpha=0.2),
    )
    assert d == 3 and out[0].shape == (8, 3)


def test_increment_representation():
    paths, clocks = _paths(L=5)
    out, _, _, _ = prepare_paths_np(
        paths, clocks, PathSpec(representation="increment", normalize=False), ClockSpec()
    )
    np.testing.assert_allclose(
        np.asarray(out[0]), np.cumsum(paths[0], axis=0).astype(np.float32), rtol=1e-6
    )
    p = jnp.asarray(paths[0])
    mask = jnp.asarray([True, True, True, False, False])
    inc = to_increment_path_jnp(p, mask)
    assert np.all(np.asarray(inc[3]) == np.asarray(inc[2])) and np.all(
        np.asarray(inc[4]) == np.asarray(inc[2])
    )


def test_derivative_replaces_path():
    paths, clocks = _paths(L=31)
    out, _, _, d = prepare_paths_np(
        paths,
        clocks,
        PathSpec(representation="derivative", derivative_order=1, normalize=False),
        ClockSpec(),
    )
    assert d == 2 and out[0].shape == (31, 2)
    assert not np.allclose(np.asarray(out[0]), paths[0])


def test_unknown_keys_rejected():
    with pytest.raises(ValueError):
        PathSpec.from_config({"representation": "value", "bogus": 1})
    with pytest.raises(ValueError):
        PathSpec(representation="derivative").validate()
