"""`stable_block`: the free 2x2 block restricted to the region that survives a long path.

The unconstrained `block_diagonal` family cannot be used on real data. The clock is normalised to a mean increment of
1, so a 3,000-token dataset applies exp(-A dtheta) three thousand times; a Gaussian 2x2 block has an expanding
direction with probability ~1/2, and the state overflows float32 at every init variance we would ever use. These
tests pin the three properties that make `stable_block` usable instead: it cannot expand, it keeps all four degrees
of freedom of a real 2x2 block, and it contains `damped_rotation` exactly as its zero-shear face.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import ews.params as P

jax.config.update("jax_enable_x64", True)
W = 12
PLANES = W // 2


def cfg(name, **kw):
    kw.setdefault("omega_scale", 2.0)
    kw.setdefault("damp_init_mean", -2.0)
    kw.setdefault("damp_init_std", 2.0)
    return P.AConfig(parametrization=name, init=P.InitConfig(type=name, sigma_range=0.3), **kw)


def sym_min_eig_2x2(b):
    """Smallest eigenvalue of a symmetric 2x2, in closed form (the repo never calls an eigendecomposition)."""
    tr, det = b[0, 0] + b[1, 1], b[0, 0] * b[1, 1] - b[0, 1] * b[1, 0]
    return (tr - np.sqrt(max(tr * tr - 4 * det, 0.0))) / 2


@pytest.mark.parametrize("shear_mean", [-2.0, 0.0, 2.0, 6.0])
def test_symmetric_part_is_psd_so_nothing_can_expand(shear_mean):
    worst = np.inf
    for seed in range(50):
        ap = P.build_a_param(
            cfg("stable_block", shear_init_mean=shear_mean, shear_init_std=2.0),
            W,
            jr.PRNGKey(seed),
            jnp.float64,
        )
        A = np.array(ap())
        S = (A + A.T) / 2
        for j in range(PLANES):
            worst = min(worst, sym_min_eig_2x2(S[2 * j : 2 * j + 2, 2 * j : 2 * j + 2]))
    assert worst >= -1e-12, f"an expanding direction appeared: lambda_min = {worst}"


def test_does_not_blow_up_over_three_thousand_tokens():
    """The measurement that condemned `block_diagonal`: growth of exp(-A dtheta) raised to the path length."""
    from jax.scipy.linalg import expm

    def log10_growth(A, T=3000):
        acc, base, log10, logbase, rem = (
            np.eye(A.shape[0]),
            np.array(expm(-jnp.asarray(A))),
            0.0,
            0.0,
            T,
        )
        while rem:
            if rem & 1:
                acc = acc @ base
                n = np.linalg.norm(acc)
                acc /= n
                log10 += np.log10(n) + logbase
            base = base @ base
            n = np.linalg.norm(base)
            base /= n
            logbase = 2 * logbase + np.log10(n)
            rem >>= 1
        return log10

    for seed in range(10):
        A = np.array(
            P.build_a_param(
                cfg("stable_block", shear_init_mean=1.0), W, jr.PRNGKey(seed), jnp.float64
            )()
        )
        assert log10_growth(A) <= 1.0, "stable_block expanded over a 3,000-token path"
    # the control: the unconstrained family, at the smallest init variance anyone would use
    bd = P.AConfig(
        parametrization="block_diagonal",
        init=P.InitConfig(type="block_diagonal", variance=1e-3),
        blocks=[2] * PLANES,
    )
    A = np.array(P.build_a_param(bd, W, jr.PRNGKey(3), jnp.float64)())
    assert log10_growth(A) > 10.0, (
        "block_diagonal was expected to overflow; this test documents why stable_block exists"
    )


def test_zero_shear_is_exactly_damped_rotation():
    a = P.build_a_param(
        cfg("stable_block", shear_init_mean=-60.0, shear_init_std=0.0, damp_init_std=1.0),
        W,
        jr.PRNGKey(7),
        jnp.float64,
    )
    b = P.build_a_param(cfg("damped_rotation", damp_init_std=1.0), W, jr.PRNGKey(7), jnp.float64)
    assert np.abs(np.array(a()) - np.array(b())).max() == 0.0


def test_shear_is_a_real_extra_degree_of_freedom():
    """With shear the off-diagonal is no longer antisymmetric and the diagonal is no longer constant per plane:
    neither is reachable by any rotation family."""
    A = np.array(
        P.build_a_param(cfg("stable_block", shear_init_mean=2.0), W, jr.PRNGKey(1), jnp.float64)()
    )
    off = [abs(A[2 * j, 2 * j + 1] + A[2 * j + 1, 2 * j]) for j in range(PLANES)]
    diag = [abs(A[2 * j, 2 * j] - A[2 * j + 1, 2 * j + 1]) for j in range(PLANES)]
    assert max(off) > 1e-6 and max(diag) > 1e-6


def test_block_structure_is_preserved():
    A = np.array(P.build_a_param(cfg("stable_block"), W, jr.PRNGKey(0), jnp.float64)())
    mask = np.ones((W, W), dtype=bool)
    for j in range(PLANES):
        mask[2 * j : 2 * j + 2, 2 * j : 2 * j + 2] = False
    assert np.abs(A[mask]).max() == 0.0
