"""The block and the stack (ews/block.py) and their norms and clocks.

RMSNorm against the SLiCE package's torch implementation; the jnp clock builder against the numpy
one (modes, scalings, masking of padded positions, the constant normalisation); the graded norm;
block shapes, dropout, padding invariance, basepoint alignment and the Van Loan recursion at depth
one; stack shapes, heads, causality and a short training run. CPU only, float64 where a tolerance
is asserted.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from ews.block import EWSBlock, GradedRMSNorm, RMSNorm, StackedEWS, graded_level_slices
from ews.clocks import build_clock_jnp
from ews.clocks import build_clock_np as build_clock
from ews.norms import FeatureNorm
from tests.block_kwargs import specs

# The reference RMSNorm is the SLiCE package's (torch): install it or set SLICES_SRC to its src/ directory,
# otherwise that test is skipped.
SLICES_SRC = os.environ.get("SLICES_SRC")


def _rand(shape, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return scale * rng.standard_normal(shape)


# -----------------------
# RMSNorm
# -----------------------


def test_rmsnorm_formula_and_unit_rms():
    x = jnp.asarray(_rand((5, 7), seed=1))
    norm = RMSNorm(7, eps=1e-12, dtype=jnp.float64)
    y = norm(x)
    expected = x / np.sqrt((np.asarray(x) ** 2).mean(axis=-1, keepdims=True) + 1e-12)
    np.testing.assert_allclose(np.asarray(y), expected, rtol=1e-10)
    rms = np.sqrt((np.asarray(y) ** 2).mean(axis=-1))
    np.testing.assert_allclose(rms, np.ones(5), rtol=1e-6)


def test_rmsnorm_shapes():
    norm = RMSNorm(4, dtype=jnp.float64)
    assert norm(jnp.zeros((9, 4))).shape == (9, 4)
    assert norm(jnp.asarray(_rand((3, 9, 4)))).shape == (3, 9, 4)


def test_rmsnorm_matches_slices_reference():
    torch = pytest.importorskip("torch")
    import sys

    if SLICES_SRC and SLICES_SRC not in sys.path:
        sys.path.insert(0, SLICES_SRC)
    TorchRMSNorm = pytest.importorskip(
        "slices.slices",
        reason="no SLiCE reference: install the SLiCE package or set SLICES_SRC to its src/",
    ).RMSNorm

    dim = 6
    x = _rand((4, 8, dim), seed=2).astype(np.float32)
    gains = _rand((dim,), seed=3).astype(np.float32)

    ref = TorchRMSNorm(dim)
    with torch.no_grad():
        ref.weight.copy_(torch.tensor(gains))
    y_ref = ref(torch.tensor(x)).detach().numpy()

    import equinox as eqx

    ours = eqx.tree_at(lambda m: m.weight, RMSNorm(dim, dtype=jnp.float32), jnp.asarray(gains))
    y = np.asarray(ours(jnp.asarray(x)))

    np.testing.assert_allclose(y, y_ref, rtol=1e-5, atol=1e-6)


# -----------------------
# build_clock_jnp: numpy parity (per_sample, no mask, eps=0)
# -----------------------


@pytest.mark.parametrize("scaling", ["none", "[0,1]", "[-1,1]"])
@pytest.mark.parametrize("mode", ["time", "blended_variation", "blended_quad_variation"])
@pytest.mark.parametrize("alpha", [1.0, 0.6])
def test_build_clock_jnp_matches_numpy(mode, alpha, scaling):
    L, c = 11, 3
    driver = _rand((L, c), seed=4)
    for times in (np.arange(L, dtype=np.float64), np.cumsum(_rand((L,), seed=5) ** 2)):
        expected = build_clock(times, driver, mode, alpha, scaling)
        got = build_clock_jnp(
            jnp.asarray(times),
            jnp.asarray(driver),
            mode,
            alpha,
            scaling,
            variation_eps=0.0,
        )
        np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-10, atol=1e-12)


def test_time_mode_allows_none_driver():
    L = 8
    got = build_clock_jnp(jnp.arange(L, dtype=jnp.float64), None, "time", 1.0, "[0,1]")
    np.testing.assert_allclose(np.asarray(got), np.arange(L) / (L - 1), rtol=0, atol=1e-12)


# -----------------------
# build_clock_jnp: guards
# -----------------------


# alpha=0.0 is allowed (pure variation clock); real FL configs use it.
@pytest.mark.parametrize("alpha", [-0.1, 1.5])
def test_alpha_guard_blended(alpha):
    L = 6
    driver = jnp.asarray(_rand((L, 2), seed=6))
    with pytest.raises(ValueError, match="alpha must be in"):
        build_clock_jnp(
            jnp.arange(L, dtype=jnp.float64), driver, "blended_variation", alpha, "[0,1]"
        )


def test_constant_requires_norm_constant():
    L = 6
    driver = jnp.asarray(_rand((L, 2), seed=7))
    with pytest.raises(ValueError, match="norm_constant is required"):
        build_clock_jnp(
            jnp.arange(L, dtype=jnp.float64),
            driver,
            "blended_variation",
            0.5,
            "[0,1]",
            normalize="constant",
        )


def test_variation_eps_gradient_guard():
    L = 5
    flat = jnp.zeros((L, 3), dtype=jnp.float64)  # repeated token -> zero increments
    times = jnp.arange(L, dtype=jnp.float64)

    def loss(d, eps):
        theta = build_clock_jnp(
            times,
            d,
            "blended_variation",
            0.5,
            "[0,1]",
            normalize="constant",
            norm_constant=1.0,
            variation_eps=eps,
        )
        return jnp.sum(theta)

    g_guarded = jax.grad(lambda d: loss(d, 1e-8))(flat)
    assert np.all(np.isfinite(np.asarray(g_guarded)))
    g_bare = jax.grad(lambda d: loss(d, 0.0))(flat)
    assert np.any(~np.isfinite(np.asarray(g_bare)))  # documents why the guard exists


def test_stop_gradient():
    L = 7
    driver = jnp.asarray(_rand((L, 2), seed=8))
    times = jnp.arange(L, dtype=jnp.float64)

    def loss(d, sg):
        theta = build_clock_jnp(times, d, "blended_variation", 0.5, "[0,1]", stop_gradient=sg)
        return jnp.sum(theta)

    g_on = np.asarray(jax.grad(lambda d: loss(d, False))(driver))
    g_off = np.asarray(jax.grad(lambda d: loss(d, True))(driver))
    assert np.any(g_on != 0.0)
    np.testing.assert_array_equal(g_off, np.zeros_like(g_off))


# -----------------------
# build_clock_jnp: mask gating + frozen pads
# -----------------------


def _padded_driver(L_real, L_pad, c=2, seed=9, pad_value=5.0):
    real = _rand((L_real, c), seed=seed)
    pad = np.full((L_pad - L_real, c), pad_value)  # big real->pad boundary jump
    driver = np.concatenate([real, pad], axis=0)
    mask = np.zeros(L_pad, dtype=bool)
    mask[:L_real] = True
    return real, driver, mask


def test_mask_gating_hand_computed():
    L_real, L_pad, C, T = 4, 9, 3.0, 10.0
    real, driver, mask = _padded_driver(L_real, L_pad)
    times = np.arange(L_pad, dtype=np.float64)

    got = np.asarray(
        build_clock_jnp(
            jnp.asarray(times),
            jnp.asarray(driver),
            "blended_quad_variation",
            0.5,
            "[0,1]",
            mask=jnp.asarray(mask),
            normalize="constant",
            norm_constant=C,
            time_denominator=T,
        )
    )
    dq = (np.diff(real, axis=0) ** 2).sum(axis=1)
    Q = np.concatenate([[0.0], np.cumsum(dq)])  # gated quad variation, real segment
    expected_real = 0.5 * (times[:L_real] / T) + 0.5 * (Q / C)
    np.testing.assert_allclose(got[:L_real], expected_real, rtol=1e-12)
    # boundary jump excluded and clock frozen at the true end:
    np.testing.assert_allclose(
        got[L_real:], np.full(L_pad - L_real, got[L_real - 1]), rtol=0, atol=0
    )


def test_per_sample_padded_matches_truncated_at_real_positions():
    L_real, L_pad = 6, 13
    real, driver, mask = _padded_driver(L_real, L_pad, seed=10)
    n = L_real - 1

    truncated = build_clock(
        np.arange(L_real, dtype=np.float64), real, "blended_variation", 0.5, "[0,1]"
    )
    padded = np.asarray(
        build_clock_jnp(
            jnp.arange(L_pad, dtype=jnp.float64),
            jnp.asarray(driver),
            "blended_variation",
            0.5,
            "[0,1]",
            mask=jnp.asarray(mask),
            variation_eps=0.0,
            time_denominator=float(n),
        )
    )
    np.testing.assert_allclose(padded[:L_real], truncated, rtol=1e-10)
    np.testing.assert_allclose(padded[L_real:], np.full(L_pad - L_real, truncated[-1]), rtol=1e-10)


def test_frozen_beyond_true_end_time_mode():
    L_real, L_pad = 5, 12
    _, driver, mask = _padded_driver(L_real, L_pad, seed=11)
    got = np.asarray(
        build_clock_jnp(
            jnp.arange(L_pad, dtype=jnp.float64),
            jnp.asarray(driver),
            "time",
            1.0,
            "[0,1]",
            mask=jnp.asarray(mask),
        )
    )
    assert np.all(got[L_real:] == got[L_real - 1])
    assert np.all(np.diff(got[:L_real]) > 0)


# -----------------------
# build_clock_jnp: constant-mode semantics
# -----------------------


def test_constant_dispatch_honored():
    L = 9
    driver = jnp.asarray(_rand((L, 2), seed=12))
    times = jnp.arange(L, dtype=jnp.float64)
    kw = dict(mask=None, normalize="constant", time_denominator=float(L - 1))
    th1 = np.asarray(
        build_clock_jnp(times, driver, "blended_variation", 0.5, "[0,1]", norm_constant=2.0, **kw)
    )
    th2 = np.asarray(
        build_clock_jnp(times, driver, "blended_variation", 0.5, "[0,1]", norm_constant=4.0, **kw)
    )
    t_s = np.arange(L) / (L - 1)
    np.testing.assert_allclose(th1 - 0.5 * t_s, 2.0 * (th2 - 0.5 * t_s), rtol=1e-10)


def _graded_D(d, n):
    return sum(d**k for k in range(1, n + 1))


# -----------------------
# GradedRMSNorm (step 2)
# -----------------------


def test_graded_level_slices_level0_free():
    assert graded_level_slices(3, 2) == ((0, 3), (3, 12))
    assert graded_level_slices(2, 3) == ((0, 2), (2, 6), (6, 14))


def test_graded_static_gain_matches_feature_norm_learned_gain():
    import equinox as eqx

    d, n, B = 3, 3, 5
    D = _graded_D(d, n)
    gains = jnp.asarray(_rand((n,), seed=20) ** 2 + 0.5)
    flat_with_l0 = jnp.asarray(np.concatenate([np.ones((B, 1)), _rand((B, D), seed=21)], axis=1))

    fn = FeatureNorm(d, n, mode="learned_gain", dtype=jnp.float64)
    fn = eqx.tree_at(lambda m: m.gains, fn, gains)
    gr = GradedRMSNorm(d, n, variant="static_gain", dtype=jnp.float64)
    gr = eqx.tree_at(lambda m: m.gains, gr, gains)

    np.testing.assert_array_equal(
        np.asarray(fn(flat_with_l0)[:, 1:]), np.asarray(gr(flat_with_l0[:, 1:]))
    )


def test_graded_live_rms_unit_rms_per_level():
    d, n, L = 3, 3, 7
    D = _graded_D(d, n)
    # levels on wildly different scales, as in a real signature
    x = np.asarray(_rand((L, D), seed=22))
    for k, (s, e) in enumerate(graded_level_slices(d, n)):
        x[:, s:e] *= 10.0 ** (-2 * k)
    gr = GradedRMSNorm(d, n, variant="live_rms", eps=0.0, dtype=jnp.float64)
    y = gr(jnp.asarray(x))
    rms = np.asarray(gr.per_level_rms(y))  # (L, N)
    np.testing.assert_allclose(rms, np.ones((L, n)), rtol=1e-10)


def test_graded_gains_gradflow_all_levels():
    d, n, L = 2, 3, 6
    D = _graded_D(d, n)
    x = jnp.asarray(_rand((L, D), seed=23))

    def loss_gains(gains):
        import equinox as eqx

        gr = GradedRMSNorm(d, n, variant="live_rms", dtype=jnp.float64)
        gr = eqx.tree_at(lambda m: m.gains, gr, gains)
        return jnp.sum(gr(x) ** 2)

    g = np.asarray(jax.grad(loss_gains)(jnp.ones((n,), dtype=jnp.float64)))
    assert np.all(g != 0.0) and np.all(np.isfinite(g))

    def loss_lam(lam):
        import equinox as eqx

        gr = GradedRMSNorm(d, n, variant="dilation", dtype=jnp.float64)
        gr = eqx.tree_at(lambda m: m.lam, gr, lam)
        return jnp.sum(gr(x) ** 2)

    gl = np.asarray(jax.grad(loss_lam)(jnp.ones((), dtype=jnp.float64)))
    assert gl != 0.0 and np.isfinite(gl)


def test_graded_dilation_exact():
    import equinox as eqx

    d, n = 2, 3
    D = _graded_D(d, n)
    x = jnp.asarray(_rand((4, D), seed=24))
    lam = jnp.asarray(1.7)
    gr = GradedRMSNorm(d, n, variant="dilation", dtype=jnp.float64)
    gr = eqx.tree_at(lambda m: m.lam, gr, lam)
    y = np.asarray(gr(x))
    for k, (s, e) in enumerate(graded_level_slices(d, n)):
        np.testing.assert_allclose(y[:, s:e], 1.7 ** (k + 1) * np.asarray(x)[:, s:e], rtol=1e-12)


def test_graded_live_rms_zero_input_safe():
    d, n, L = 3, 2, 4
    D = _graded_D(d, n)
    gr = GradedRMSNorm(d, n, variant="live_rms", eps=1e-3, dtype=jnp.float64)
    zeros = jnp.zeros((L, D), dtype=jnp.float64)
    np.testing.assert_array_equal(np.asarray(gr(zeros)), np.zeros((L, D)))
    g = np.asarray(jax.grad(lambda x: jnp.sum(gr(x) ** 2))(zeros))
    assert np.all(np.isfinite(g))


def test_graded_live_rms_eps_floor_vanishing_inputs():
    d, n, L = 2, 2, 5
    D = _graded_D(d, n)
    base = _rand((L, D), seed=25)
    gr = GradedRMSNorm(d, n, variant="live_rms", eps=1e-3, dtype=jnp.float64)
    norms = []
    for s in (1.0, 1e-2, 1e-4, 0.0):
        y = np.asarray(gr(jnp.asarray(s * base)))
        assert np.all(np.isfinite(y))
        norms.append(np.linalg.norm(y))
    assert norms[0] > norms[1] > norms[2] > norms[3] == 0.0


def test_graded_shapes_and_validation():
    d, n = 2, 2
    D = _graded_D(d, n)
    gr = GradedRMSNorm(d, n, dtype=jnp.float64)
    assert gr(jnp.zeros((D,))).shape == (D,)
    assert gr(jnp.zeros((5, D))).shape == (5, D)
    assert gr(jnp.zeros((3, 5, D))).shape == (3, 5, D)
    with pytest.raises(ValueError, match="last axis must be D"):
        gr(jnp.zeros((5, D + 1)))
    with pytest.raises(ValueError, match="variant must be one of"):
        GradedRMSNorm(d, n, variant="flat")
    with pytest.raises(ValueError, match="eps must be a scalar"):
        GradedRMSNorm(d, n, eps=[1e-3])  # wrong length (N=2)


def test_graded_per_level_rms_helper():
    d, n, L = 3, 2, 6
    D = _graded_D(d, n)
    x = _rand((L, D), seed=26)
    gr = GradedRMSNorm(d, n, dtype=jnp.float64)
    got = np.asarray(gr.per_level_rms(jnp.asarray(x)))
    for k, (s, e) in enumerate(graded_level_slices(d, n)):
        expected = np.sqrt((x[:, s:e] ** 2).mean(axis=1))
        np.testing.assert_allclose(got[:, k], expected, rtol=1e-12)


def test_one_hot_apriori_bound():
    # Alternating one-hots: every step is a token change of norm sqrt(2);
    # with norm_constant = sqrt(2) * (L - 1) the variation component ends at 1.
    L, V = 10, 4
    tokens = np.arange(L) % 2
    onehot = np.eye(V, dtype=np.float64)[tokens]
    const = np.sqrt(2.0) * (L - 1)
    got = np.asarray(
        build_clock_jnp(
            jnp.arange(L, dtype=jnp.float64),
            jnp.asarray(onehot),
            "blended_variation",
            0.5,
            "[0,1]",
            normalize="constant",
            norm_constant=float(const),
            time_denominator=float(L - 1),
            variation_eps=0.0,
        )
    )
    np.testing.assert_allclose(got[-1], 1.0, rtol=1e-12)
    assert np.all(np.diff(got) > 0)


# -----------------------
# EWSBlock (step 3)
# -----------------------

H_SMALL = 4


def _mk_block(key_seed=0, hidden_dim=H_SMALL, m=2, n=1, **kw):
    kw.setdefault("M", 3)
    kw.setdefault("dtype", jnp.float64)
    return EWSBlock(hidden_dim, m, n, key=jr.PRNGKey(key_seed), **specs(kw))


@pytest.mark.parametrize("channel_stage", ["mlp", "tanh", "glu", "gelu_glu"])
@pytest.mark.parametrize("norm_placement", ["pre", "slice_post", "none"])
def test_block_shapes_norm_placement_and_channel_stage(norm_placement, channel_stage):
    L = 6
    block = _mk_block(norm_placement=norm_placement, channel_stage=channel_stage)
    x = jnp.asarray(_rand((L, H_SMALL), seed=30))
    y = block(x, inference=True)
    assert y.shape == (L, H_SMALL)
    assert np.all(np.isfinite(np.asarray(y)))


@pytest.mark.parametrize(
    "kw",
    [
        dict(basepoint=True),
        dict(append_clock_mode="time"),  # preserve defaults on -> w_eff = m+1
        dict(
            append_clock_mode="time",
            b_config={
                "enabled": True,
                "out_dim": 2,
                "_preserve_first_channel": False,
                "init": {"type": "general", "mean": 0.0, "variance": 0.25},
            },
        ),
        dict(b_config={"enabled": False}),  # lift disabled -> w_eff = H
        dict(output_gate=True),
        dict(n=2, graded_norm="static_gain"),
        dict(clock_mode="blended_variation", clock_alpha=0.5),
        dict(
            clock_mode="blended_quad_variation",
            clock_alpha=0.5,
            clock_normalize="constant",
            clock_norm_constant=3.0,
        ),
    ],
)
def test_block_shapes_variants(kw):
    L = 6
    block = _mk_block(**kw)
    if kw.get("append_clock_mode") and kw.get("b_config", {}).get("_preserve_first_channel", True):
        assert block.w_eff == block.m + 1 or not kw.get("b_config")
    x = jnp.asarray(_rand((L, H_SMALL), seed=31))
    y = block(x, inference=True)
    assert y.shape == (L, H_SMALL)
    assert np.all(np.isfinite(np.asarray(y)))


def test_block_batched_matches_loop():
    B_, L = 3, 6
    block = _mk_block(n=2, graded_norm="static_gain")
    x = jnp.asarray(_rand((B_, L, H_SMALL), seed=32))
    y = np.asarray(block(x, inference=True))
    for i in range(B_):
        yi = np.asarray(block(x[i], inference=True))
        np.testing.assert_allclose(y[i], yi, rtol=1e-12)


def test_block_dropout_train_eval_parity():
    L = 6
    x = jnp.asarray(_rand((L, H_SMALL), seed=33))
    block = _mk_block(key_seed=5, dropout_rate=0.5)
    y1 = np.asarray(block(x, key=jr.PRNGKey(1), inference=True))
    y2 = np.asarray(block(x, key=jr.PRNGKey(2), inference=True))
    np.testing.assert_array_equal(y1, y2)  # inference ignores the key
    twin = _mk_block(key_seed=5, dropout_rate=0.0)
    y0 = np.asarray(twin(x, inference=False))  # p=0 needs no key
    np.testing.assert_array_equal(y1, y0)  # same init key -> same weights
    t1 = np.asarray(block(x, key=jr.PRNGKey(1), inference=False))
    t2 = np.asarray(block(x, key=jr.PRNGKey(2), inference=False))
    assert np.any(t1 != t2)


def test_block_dropout_key_required():
    block = _mk_block(dropout_rate=0.1)
    x = jnp.asarray(_rand((5, H_SMALL), seed=34))
    with pytest.raises(ValueError, match="key is required"):
        block(x, inference=False)


def test_block_raw_input_requires_driver():
    block = _mk_block(clock_mode="blended_variation", clock_alpha=0.5, clock_source="raw_input")
    x = jnp.asarray(_rand((5, H_SMALL), seed=35))
    with pytest.raises(ValueError, match="clock_driver is required"):
        block(x, inference=True)


def test_block_blended_alpha_guard_constructor():
    # alpha=0.0 is allowed (pure variation clock): real FL configs use it. Out-of-range
    # values must still raise.
    _mk_block(clock_mode="blended_variation", clock_alpha=0.0)
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="alpha must be in"):
            _mk_block(clock_mode="blended_variation", clock_alpha=bad)


@pytest.mark.parametrize(
    # exact pins the wiring; cayley/euler document each approximant's genuine
    # gap to the exact Van Loan step (a wiring bug would be O(1), not O(1e-2))
    "expm_method,tol",
    [("exact", 1e-9), ("cayley", 1e-4), ("euler", 5e-2)],
)
def test_depth_one_block_is_the_van_loan_affine_recursion(expm_method, tol):
    from scipy.linalg import expm as scipy_expm

    H, L = 3, 6
    block = _mk_block(
        hidden_dim=H,
        m=H,
        n=1,
        b_config={"enabled": False},
        a_config={
            "parametrization": "skew_symmetric",
            "init": {"type": "skew_symmetric", "mean": 0.0, "variance": 0.01},
        },
        norm_placement="none",
        van_loan_expm=expm_method,
        chen_expm="exact",
    )
    x = jnp.asarray(_rand((L, H), seed=36))
    levels = np.asarray(block.mixer_levels(x))  # (L, H) since D = w = H at n=1
    A = np.asarray(block.ewms.A_param())
    theta = np.arange(L) / (L - 1)
    S = np.zeros(H)
    expected = [S]
    xs = np.asarray(x)
    for j in range(1, L):
        aug = np.zeros((H + 1, H + 1))
        aug[:H, :H] = -(theta[j] - theta[j - 1]) * A
        aug[:H, H] = xs[j] - xs[j - 1]
        Mv = scipy_expm(aug)
        S = Mv[:H, :H] @ S + Mv[:H, H]
        expected.append(S)
    np.testing.assert_allclose(levels, np.stack(expected), rtol=tol, atol=tol)


def test_gelu_glu_channel_stage_wiring():
    L = 6
    block = _mk_block(
        key_seed=7,
        channel_stage="gelu_glu",
        graded_norm="static_gain",
        norm_placement="pre",
    )
    x = jnp.asarray(_rand((L, H_SMALL), seed=37))
    got = np.asarray(block(x, inference=True))

    levels = block.mixer_levels(x)
    r = jax.vmap(block.c_down)(block.graded(levels))
    x1 = x + r
    xhat = block.norm2(x1)
    h = jax.nn.gelu(xhat)
    mch = jax.vmap(block.ff1)(h) * jax.nn.sigmoid(jax.vmap(block.ff2)(h))
    manual = np.asarray(x1 + mch)
    np.testing.assert_allclose(got, manual, rtol=1e-12)


def test_slice_post_norm_placement():
    L = 6
    block = _mk_block(key_seed=8, norm_placement="slice_post", channel_stage="tanh")
    assert block.norm1 is None and isinstance(block.norm2, eqx.nn.LayerNorm)
    x = jnp.asarray(_rand((L, H_SMALL), seed=38))
    got = np.asarray(block(x, inference=True))

    levels = block.mixer_levels(x)  # mixer consumes raw x (no pre-norm)
    r = jax.vmap(block.c_down)(block.graded(levels))
    y1 = x + r
    y2 = y1 + jnp.tanh(jax.vmap(block.ff1)(y1))
    manual = np.asarray(jax.vmap(block.norm2)(y2))  # LayerNorm last, dropout inert
    np.testing.assert_allclose(got, manual, rtol=1e-12)


def test_norm_placement_none_removes_the_stream_norms():
    L = 6
    block = _mk_block(key_seed=9, norm_placement="none", channel_stage="tanh")
    assert block.norm1 is None and block.norm2 is None
    x = jnp.asarray(_rand((L, H_SMALL), seed=39))
    got = np.asarray(block(x, inference=True))
    levels = block.mixer_levels(x)
    r = jax.vmap(block.c_down)(block.graded(levels))
    x1 = x + r
    manual = np.asarray(x1 + jnp.tanh(jax.vmap(block.ff1)(x1)))
    np.testing.assert_allclose(got, manual, rtol=1e-12)


def test_eps_floor_keeps_the_block_finite():
    L = 6
    block = _mk_block(
        key_seed=10,
        n=2,
        graded_norm="live_rms",
        graded_eps=1e-3,
        clock_mode="blended_variation",
        clock_alpha=0.5,
    )
    base = _rand((L, H_SMALL), seed=40)
    for s in (1.0, 1e-2, 1e-4, 0.0):
        y = np.asarray(block(jnp.asarray(s * base), inference=True))
        assert np.all(np.isfinite(y)), f"non-finite output at scale {s}"
    g = jax.grad(lambda z: jnp.sum(block(z, inference=True) ** 2))(
        jnp.zeros((L, H_SMALL), dtype=jnp.float64)
    )
    assert np.all(np.isfinite(np.asarray(g)))


def test_block_padding_invariance():
    L_real = 5
    block = _mk_block(
        key_seed=11,
        n=2,
        graded_norm="live_rms",
        clock_mode="blended_variation",
        clock_alpha=0.5,
        clock_normalize="constant",
        clock_norm_constant=5.0,
        clock_time_denominator=8.0,
    )
    x_real = _rand((L_real, H_SMALL), seed=41)
    outs = []
    for L_pad in (9, 13):
        x = np.full((L_pad, H_SMALL), 3.3)
        x[:L_real] = x_real
        mask = np.zeros(L_pad, dtype=bool)
        mask[:L_real] = True
        y = np.asarray(block(jnp.asarray(x), mask=jnp.asarray(mask), inference=True))
        outs.append(y[:L_real])
        levels = np.asarray(block.mixer_levels(jnp.asarray(x), mask=jnp.asarray(mask)))
        for i in range(L_real, L_pad):  # frozen pad states == real terminal state
            np.testing.assert_allclose(levels[i], levels[L_real - 1], rtol=1e-12)
    np.testing.assert_allclose(outs[0], outs[1], rtol=1e-10)


def test_basepoint_row_alignment():
    H, L = 3, 5
    x = jnp.asarray(_rand((L, H), seed=42))
    common = dict(hidden_dim=H, m=H, n=1, b_config={"enabled": False}, norm_placement="none")
    no_bp = _mk_block(key_seed=12, **common)
    lv0 = np.asarray(no_bp.mixer_levels(x))
    assert lv0.shape == (L, H)
    np.testing.assert_array_equal(lv0[0], np.zeros(H))  # trivial prefix row
    with_bp = _mk_block(key_seed=12, basepoint=True, **common)
    lv1 = np.asarray(with_bp.mixer_levels(x))
    assert lv1.shape == (L, H)
    # basepoint jump 0 -> p_1 with zero time increment: level 1 == p_1 = x[0]
    np.testing.assert_allclose(lv1[0], np.asarray(x)[0], rtol=1e-9)


# -----------------------
# StackedEWS (step 4)
# -----------------------


def _mk_stack(key_seed=0, num_blocks=2, data_dim=7, hidden_dim=H_SMALL, label_dim=5, **kw):
    block_kwargs = kw.pop("block_kwargs", {})
    block_kwargs.setdefault("m", 2)
    block_kwargs.setdefault("M", 3)
    block_kwargs.setdefault("dropout_rate", 0.0)
    kw.setdefault("dtype", jnp.float64)
    if "per_block_kwargs" in kw:
        kw["per_block_kwargs"] = [specs(x) for x in kw["per_block_kwargs"]]
    return StackedEWS(
        num_blocks,
        data_dim,
        hidden_dim,
        label_dim,
        block_kwargs=specs(block_kwargs),
        key=jr.PRNGKey(key_seed),
        **kw,
    )


def test_stack_shapes_tokens_seq2seq():
    L, B_ = 6, 3
    model = _mk_stack()
    X = jnp.asarray(np.arange(L) % 7)
    y = model(X, inference=True)
    assert y.shape == (L, 5)
    Xb = jnp.asarray(_rand((B_, L), seed=50).astype(np.int64) % 7 + 3 % 7) % 7
    yb = model(Xb, inference=True)
    assert yb.shape == (B_, L, 5)
    assert np.all(np.isfinite(np.asarray(yb)))


def test_stack_shapes_timeseries():
    L, B_, d = 6, 3, 3
    model = _mk_stack(tokens=False, data_dim=d)
    X = jnp.asarray(_rand((L, d), seed=51))
    assert model(X, inference=True).shape == (L, 5)
    Xb = jnp.asarray(_rand((B_, L, d), seed=52))
    assert model(Xb, inference=True).shape == (B_, L, 5)


def test_stack_mean_pool_masked():
    L, L_real = 8, 5
    model = _mk_stack(head_mode="mean_pool")
    X = jnp.asarray(np.arange(L) % 7)
    mask = np.zeros(L, dtype=bool)
    mask[:L_real] = True
    y = model(X, mask=jnp.asarray(mask), inference=True)
    assert y.shape == (5,)
    h = model.hidden(X, mask=jnp.asarray(mask), inference=True)
    pooled = np.asarray(h)[:L_real].mean(axis=0)
    manual = np.asarray(model.head(jnp.asarray(pooled)))
    np.testing.assert_allclose(np.asarray(y), manual, rtol=1e-10)
    # batched
    Xb = jnp.asarray(np.stack([np.arange(L) % 7] * 2))
    maskb = jnp.asarray(np.stack([mask] * 2))
    assert model(Xb, mask=maskb, inference=True).shape == (2, 5)


def test_stack_tuple_heads():
    L = 6
    model = _mk_stack(label_dim=(5, 3))
    X = jnp.asarray(np.arange(L) % 7)
    out = model(X, inference=True)
    assert isinstance(out, tuple) and len(out) == 2
    assert out[0].shape == (L, 5) and out[1].shape == (L, 3)
    with pytest.raises(TypeError, match="label_dim must be int or tuple of int"):
        _mk_stack(label_dim=[5])


def test_stack_per_block_kwargs_and_final_norm():
    model = _mk_stack(
        per_block_kwargs=[{"n": 1}, {"n": 2, "graded_norm": "static_gain"}],
        final_norm=True,
    )
    assert model.blocks[0].n == 1 and model.blocks[1].n == 2
    assert model.blocks[1].graded.variant == "static_gain"
    assert isinstance(model.final_norm, RMSNorm)
    X = jnp.asarray(np.arange(6) % 7)
    assert model(X, inference=True).shape == (6, 5)


def test_stack_block1_block_input_equals_encoder_output():
    kw = dict(clock_mode="blended_variation", clock_alpha=0.5)
    m1 = _mk_stack(key_seed=3, num_blocks=1, block_kwargs={**kw, "clock_source": "block_input"})
    m2 = _mk_stack(key_seed=3, num_blocks=1, block_kwargs={**kw, "clock_source": "encoder_output"})
    X = jnp.asarray(np.arange(6) % 7)
    y1 = np.asarray(m1(X, inference=True))
    y2 = np.asarray(m2(X, inference=True))
    np.testing.assert_array_equal(y1, y2)


def test_stack_raw_input_driver():
    model = _mk_stack(
        num_blocks=1,
        block_kwargs=dict(
            clock_mode="blended_variation", clock_alpha=0.5, clock_source="raw_input"
        ),
    )
    X = jnp.asarray(np.arange(6) % 7)
    with pytest.raises(ValueError, match="clock_driver is required"):
        model(X, inference=True)
    onehot = jnp.asarray(np.eye(7)[np.asarray(X)])
    y = model(X, raw_driver=onehot, inference=True)
    assert y.shape == (6, 5)


@pytest.mark.parametrize(
    "clock_kwargs",
    [
        dict(clock_mode="time"),
        dict(
            clock_mode="blended_variation",
            clock_alpha=0.5,
            clock_normalize="constant",
            clock_norm_constant=4.0,
            clock_time_denominator=7.0,
        ),
    ],
)
def test_stack_causality_jacobian_causal_clocks(clock_kwargs):
    L, d = 5, 3
    model = _mk_stack(
        key_seed=4,
        tokens=False,
        data_dim=d,
        label_dim=2,
        block_kwargs={**clock_kwargs, "n": 2},
    )
    X = jnp.asarray(_rand((L, d), seed=53))
    J = np.asarray(jax.jacobian(lambda z: model(z, inference=True))(X))  # (L, 2, L, d)
    for i in range(L):
        for j in range(i + 1, L):
            np.testing.assert_allclose(J[i, :, j, :], 0.0, atol=1e-12)


def test_stack_per_sample_clock_leaks_future():
    # positive control: with per_sample scaling the denominator V_L makes
    # earlier positions depend on later inputs (documented non-causality)
    L, d = 5, 3
    model = _mk_stack(
        key_seed=4,
        tokens=False,
        data_dim=d,
        label_dim=2,
        block_kwargs=dict(clock_mode="blended_variation", clock_alpha=0.5, n=2),
    )
    X = jnp.asarray(_rand((L, d), seed=54))
    J = np.asarray(jax.jacobian(lambda z: model(z, inference=True))(X))
    beyond = max(np.abs(J[i, :, j, :]).max() for i in range(L) for j in range(i + 1, L))
    assert beyond > 0.0


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("expm_method", ["exact", "euler", "cayley"])
def test_short_training_run_stays_finite(expm_method, n):
    import optax

    vocab, L, B_, steps = 4, 8, 8, 120
    model = _mk_stack(
        key_seed=20 + n,
        data_dim=vocab,
        hidden_dim=6,
        label_dim=2,
        block_kwargs=dict(m=2, n=n, van_loan_expm=expm_method, M=3),
    )
    rng = np.random.default_rng(60 + n)
    X = rng.integers(0, 2, size=(B_, L))  # binary tokens
    y = np.cumsum(X, axis=1) % 2  # running parity per position
    Xj, yj = jnp.asarray(X), jnp.asarray(y)

    opt = optax.adam(3e-3)
    params, static = eqx.partition(model, eqx.is_inexact_array)
    opt_state = opt.init(params)

    @eqx.filter_jit
    def step(params, opt_state):
        def loss_fn(p):
            mdl = eqx.combine(p, static)
            logits = mdl(Xj, inference=True)
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, yj))

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    losses = []
    for _ in range(steps):
        params, opt_state, loss = step(params, opt_state)
        losses.append(float(loss))
    assert np.all(np.isfinite(losses))
    assert losses[-1] < losses[0]
    leaves = jax.tree_util.tree_leaves(params)
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)
