"""SSM -> EWS conversion checks.

Runs on a fresh clone with no third-party checkouts: every reference recurrence here is our own transcription of
the family's published equations, under ews/conversion/reference/. Checks that need torch or trained weights skip
rather than fail, and say why.

test_negative_controls_must_fail passes when the three wrong path conventions fail to reproduce the family. The
conversion rests on the driving convention (integral path of Bu, prepend-origin basepoint, basepoint hop lasting a full
clock step); a wrong convention gives residuals of about 2.8 even with correct conversions underneath.
"""

import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402

from ews.conversion.apply import ews_ssm_apply  # noqa: E402
from ews.conversion.base import residual  # noqa: E402
from ews.conversion.families import linoss_im, linoss_imex, lru, s5  # noqa: E402
from ews.conversion.reference import recurrences  # noqa: E402
from ews.conversion.reference.vendored.linoss import LinOSSLayer  # noqa: E402
from ews.conversion.reference.vendored.lru import LRULayer  # noqa: E402
from ews.conversion.reference.vendored.s5 import S5Layer  # noqa: E402

PASS = 1e-10  # base threshold; the worst well-conditioned row is 5e-14, the best wrong one 2.4e-2


def pass_threshold(cond_G=1.0):
    """Conditioning-aware bar. Inverting the lift G(A) amplifies float64 noise by cond(G), so a family whose own
    spectrum puts a mode near the 2*pi*i*k/dtheta singularity cannot reach 1e-10 and should not be asked to.
    D-LinOSS is the only such family here: its mode 3 sits at |Im(A dt)| = 3.1145, essentially pi, giving
    cond G = 1.8e5 and an achievable floor near 1e-9. Reporting it as a failure against a flat 1e-10 would blame
    our conversion for their spectrum; widening the bar for everyone would hide real regressions elsewhere."""
    return PASS * max(1.0, float(cond_G) / 1e3)


L, H = 64, 4


def _u(seed=1):
    return jr.normal(jr.PRNGKey(seed), (L, H), dtype=jnp.float64)


def _run(conv, u):
    w = dict(A=conv.A, B=conv.B, C=conv.C, D=conv.D, m=conv.m)
    return np.asarray(ews_ssm_apply(w, u, N=1)[0], np.float64)


def _linoss(disc="IM", P=8):
    lay = LinOSSLayer(ssm_size=P, H=H, discretization=disc, key=jr.PRNGKey(0))
    if disc == "IM":  # force relu-clipped channels
        import equinox as eqx

        lay = eqx.tree_at(
            lambda m: m.A_diag, lay, lay.A_diag.at[::3].set(-jnp.abs(lay.A_diag[::3]))
        )
    return lay


@pytest.mark.parametrize(
    "name,build,convert,ref,method",
    [
        (
            "linoss_im",
            lambda: _linoss("IM"),
            linoss_im.to_ews,
            recurrences.reference_linoss_im,
            "B",
        ),
        (
            "s5",
            lambda: S5Layer(
                ssm_size=16,
                blocks=1,
                H=H,
                C_init="lecun_normal",
                conj_sym=True,
                clip_eigs=True,
                discretisation="zoh",
                dt_min=1e-3,
                dt_max=1e-1,
                step_rescale=1.0,
                key=jr.PRNGKey(0),
            ),
            s5.to_ews,
            recurrences.reference_s5,
            "A",
        ),
        (
            "lru",
            lambda: LRULayer(N=16, H=H, key=jr.PRNGKey(0)),
            lru.to_ews,
            recurrences.reference_lru,
            "B",
        ),
    ],
)
def test_conversion_reproduces_the_family(name, build, convert, ref, method):
    """The EWS layer driven by converted weights reproduces the family's own recurrence in float64."""
    layer = build()
    u = _u()
    y_ref = np.asarray(ref(layer, u)[0], np.float64)
    err, _rel = residual(_run(convert(layer, method=method), u), y_ref)
    assert err < PASS, f"{name} method {method}: max abs error {err:.3e} exceeds {PASS:.0e}"


def test_dlinoss_converts_within_its_conditioning():
    """D-LinOSS converts, but its own spectrum limits the achievable precision. The bar scales with cond G(A)."""
    import sys as _s

    c = os.environ.get("DLINOSS_REPO")  # a clone of github.com/jaredbmit/damped-linoss at a396997
    if c and c not in _s.path:
        _s.path.insert(0, c)
    try:
        from linoss.models.LinOSS import LinOSSLayer as DL
    except ImportError:
        pytest.skip(
            "no damped-linoss clone: set DLINOSS_REPO to a checkout of jaredbmit/damped-linoss at a396997"
        )
    from ews.conversion.families import dlinoss

    lay = DL(
        ssm_size=8,
        H=H,
        discretization="IMEX",
        damping=True,
        r_min=0.9,
        theta_max=float(np.pi),
        key=jr.PRNGKey(0),
    )
    u = _u()
    conv = dlinoss.to_ews(lay)
    err, _ = residual(_run(conv, u), np.asarray(lay(u), np.float64))
    bar = pass_threshold(conv.diag["cond_G_max"])
    assert err < bar, f"dlinoss: {err:.3e} exceeds its conditioning-aware bar {bar:.3e}"
    assert conv.diag["cond_G_max"] > 1e4, (
        "if cond G has dropped, the near-pi mode is gone and this test is stale"
    )


def test_linoss_imex_converts():
    """IMEX converts: |lambda| = 1 forbids nothing, and IMEX's init puts no eigenvalue on the negative real axis and
    none at 2*pi*i*k."""
    lay = _linoss("IMEX")
    conv = linoss_imex.to_ews(lay)
    u = _u()
    y_ref = np.asarray(lay(u), np.float64)
    err, _ = residual(_run(conv, u), y_ref)
    assert err < PASS, f"linoss_imex: {err:.3e}"
    assert conv.diag["cond_G_max"] < 100, "IMEX's lift should be as well conditioned as LinOSS-IM's"


def test_negative_controls_must_fail():
    """A green run here is a failure. The three wrong path conventions must not reproduce the family."""
    lay = _linoss("IM")
    u = _u()
    y_ref = np.asarray(recurrences.reference_linoss_im(lay, u)[0], np.float64)
    conv = linoss_im.to_ews(lay, method="B")
    w = dict(A=conv.A, B=conv.B, C=conv.C, D=conv.D, m=conv.m)
    for bad in ("value", "no_basepoint", "jump_basepoint"):
        y = np.asarray(ews_ssm_apply(w, u, N=1, path_convention=bad)[0], np.float64)
        err, _ = residual(y, y_ref)
        assert err > PASS, (
            f"path_convention={bad!r} reproduced the family to {err:.3e}. That is a FAILURE: the driving "
            f"convention is the conversion claim, and a wrong convention must give a wrong answer."
        )


def test_methods_agree_on_output_but_may_differ_on_generator():
    """S5 Methods A and B give the same output and different generators: branch ambiguity, modulo 2*pi*i/dtheta."""
    lay = S5Layer(
        ssm_size=16,
        blocks=1,
        H=H,
        C_init="lecun_normal",
        conj_sym=True,
        clip_eigs=True,
        discretisation="zoh",
        dt_min=1e-3,
        dt_max=1e-1,
        step_rescale=1.0,
        key=jr.PRNGKey(0),
    )
    u = _u()
    y_ref = np.asarray(recurrences.reference_s5(lay, u)[0], np.float64)
    a, b = s5.to_ews(lay, method="A"), s5.to_ews(lay, method="B")
    ea, _ = residual(_run(a, u), y_ref)
    eb, _ = residual(_run(b, u), y_ref)
    assert ea < PASS and eb < PASS, (
        f"both methods must reproduce the family: A {ea:.3e}, B {eb:.3e}"
    )
    im_a = float(np.abs(np.linalg.eigvals(a.A).imag).max())
    im_b = float(np.abs(np.linalg.eigvals(b.A).imag).max())
    assert im_a > np.pi > im_b, (
        f"expected an unwrapped generator (|Im| > pi) and a wrapped one (< pi); got {im_a:.4f} and {im_b:.4f}"
    )


def test_mamba_converts_with_a_variable_clock():
    """Mamba is selective but A is fixed, so the input-dependence is a variable step size, which is what an EWS
    clock is. One EWS per channel (Mamba-1) / per head (Mamba-2). Not a frozen-input approximation."""
    from ews.conversion.families.mamba import mamba1_to_ews, mamba2_to_ews
    from ews.conversion.reference.mamba_ref import mamba1_channel, mamba2_head

    rng = np.random.default_rng(0)
    N, Lm = 16, 128
    A = -np.exp(rng.normal(size=N))
    dlt = np.exp(rng.normal(size=Lm) * 0.3) * 0.2
    B = rng.normal(size=(Lm, N))
    C = rng.normal(size=(Lm, N))
    u = rng.normal(size=Lm)
    D = rng.normal()
    err, _ = residual(
        mamba1_to_ews(A, dlt, B, C, u, D).forward() + D * u, mamba1_channel(A, dlt, B, C, u, D)
    )
    assert err < PASS, f"mamba1: {err:.3e}"
    a_s = -float(np.exp(rng.normal()))
    err, _ = residual(
        mamba2_to_ews(a_s, dlt, B, C, u, D).forward() + D * u, mamba2_head(a_s, dlt, B, C, u, D)
    )
    assert err < PASS, f"mamba2: {err:.3e}"


def test_trained_checkpoint_metric_unchanged():
    """A trained model, converted, scores what it scored. Skipped where the checkpoints are absent."""
    from ews.conversion import trained_row

    if trained_row.SKIP_REASON:
        pytest.skip(trained_row.SKIP_REASON)
    data_dir = os.environ.get(
        "LNCDE_DATA_DIR"
    )  # data_dir/ of github.com/Benjamin-Walker/log-neural-cdes, processed
    if not data_dir:
        pytest.skip(
            "set LNCDE_DATA_DIR to the processed data_dir/ of a Benjamin-Walker/log-neural-cdes clone"
        )
    model, st, best = trained_row.load(2345)
    from ews.conversion._blocks import from_linoss_block, swap_blocks

    ds = trained_row.create_dataset(
        data_dir,
        "Heartbeat",
        stepsize=1,
        depth=1,
        include_time=True,
        T=1,
        use_idxs=False,
        use_presplit=False,
        key=jr.split(jr.PRNGKey(2345), 4)[0],
    )
    test = ds.raw_dataloaders["test"]
    before = trained_row.accuracy(model, st, test)
    after = trained_row.accuracy(swap_blocks(model, from_linoss_block, N=1), st, test)
    assert abs(after - before) < 1e-9, f"converted {after:.2f} vs original {before:.2f}"
