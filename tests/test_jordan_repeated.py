"""`jordan_repeated` (Jordan blocks with a repeated eigenvalue) and `unconstrained_rotation` (2x2 rotation planes
[[a, -b], [b, a]] with a free scale).
"""

import json
import os

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax.scipy.linalg import expm

import ews.params as P

jax.config.update("jax_enable_x64", True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cfg(name, blocks, kinds, variance=1.0):
    return P.AConfig(
        parametrization=name,
        init=P.InitConfig(type=name, mean=0.0, variance=variance),
        blocks=list(blocks),
        kinds=list(kinds),
    )


def build(name, blocks, kinds, key=0, variance=1.0):
    return P.build_a_param(
        cfg(name, blocks, kinds, variance), sum(blocks), jr.PRNGKey(key), jnp.float64
    )


# ----------------------------------------------------------------- the structure is a Jordan form
def test_real_block_repeats_one_eigenvalue():
    A = np.array(build("jordan_repeated", [3], ["real"])())
    assert np.allclose(np.diag(A), np.diag(A)[0]), (
        f"a Jordan block repeats ONE eigenvalue, got {np.diag(A)}"
    )
    assert np.allclose(np.diag(A, 1), 1.0)
    assert np.count_nonzero(A) == 5


def test_complex_block_repeats_one_rotation():
    A = np.array(build("jordan_repeated", [4], ["complex"])())
    assert np.allclose(A[0:2, 0:2], A[2:4, 2:4]), (
        "every diagonal slot must carry the SAME rotation-scaling"
    )
    assert np.allclose(A[0:2, 2:4], np.eye(2)), "the complex Jordan coupling is the 2x2 identity"
    assert np.allclose(A[2:4, 0:2], 0.0)


def test_different_eigenvalues_come_from_different_blocks():
    A = np.array(build("jordan_repeated", [2, 2], ["real", "real"])())
    assert A[0, 0] == A[1, 1] and A[2, 2] == A[3, 3]  # repeated within a block
    assert A[0, 0] != A[2, 2]  # free across blocks
    assert A[1, 2] == 0.0  # no coupling between blocks


# ----------------------------------------------------------------- it is defective (the point of the family)
@pytest.mark.parametrize(
    "blocks,kinds,chain", [([2], ["real"], 2), ([3], ["real"], 3), ([4], ["complex"], 2)]
)
def test_nilpotent_part_certifies_the_jordan_chain(blocks, kinds, chain):
    """A = S + N with S the repeated semisimple part. N must be non-zero and nilpotent of index = chain length,
    which is exactly what "defective" means. Checked by matrix multiplication: no decomposition."""
    A = np.array(build("jordan_repeated", blocks, kinds)())
    n = A.shape[0]
    if kinds[0] == "real":
        S = np.diag(np.diag(A))
    else:
        R = A[0:2, 0:2]
        S = np.kron(np.eye(n // 2), np.zeros((2, 2)))
        for i in range(0, n, 2):
            S[i : i + 2, i : i + 2] = R
    N = A - S
    assert np.count_nonzero(N) > 0, "a Jordan block has a non-zero nilpotent part"
    power = np.linalg.matrix_power(N, chain - 1)
    assert np.count_nonzero(power) > 0, (
        f"N^{chain - 1} must still be non-zero (chain length {chain})"
    )
    assert np.allclose(np.linalg.matrix_power(N, chain), 0.0), f"N^{chain} must vanish"


@pytest.mark.parametrize("blocks,kinds", [([2], ["real"]), ([4], ["complex"])])
def test_geometric_multiplicity_is_half_the_algebraic(blocks, kinds):
    """One eigenvector per Jordan chain: algebraic 2, geometric 1. (matrix_rank is SVD-based, test-only.)"""
    A = np.array(build("jordan_repeated", blocks, kinds)())
    n = A.shape[0]
    S = np.diag(np.diag(A)) if kinds[0] == "real" else np.kron(np.eye(n // 2), A[0:2, 0:2])
    geometric = n - np.linalg.matrix_rank(A - S, tol=1e-10)
    assert geometric == n // 2, f"expected {n // 2} eigenvectors for {n} algebraic, got {geometric}"


def test_the_ramp_is_exactly_linear_in_the_step():
    """exp(-hA) = exp(-h*lam) [[1, -h], [0, 1]] for a real J_2: the off-diagonal is proportional to h, and that is
    the accumulation a rotation cannot produce. A free diagonal (distinct eigenvalues) does not do this."""

    def ratio(M):
        e1, e2 = np.array(expm(-0.3 * jnp.asarray(M))), np.array(expm(-0.6 * jnp.asarray(M)))
        return (e2[0, 1] / e2[0, 0]) / (e1[0, 1] / e1[0, 0])

    A = np.array(build("jordan_repeated", [2], ["real"])())
    assert ratio(jnp.asarray(A)) == pytest.approx(2.0, abs=1e-9)
    assert ratio(jnp.array([[0.5, 1.0], [0.0, 0.8]])) != pytest.approx(
        2.0, abs=1e-3
    )  # distinct -> no ramp


def test_complex_ramp_carries_the_rotation():
    """For the complex block, exp(-hA)'s coupling block is h * exp(-ha) * R(-hb): a ramp times a rotation."""
    A = jnp.asarray(np.array(build("jordan_repeated", [4], ["complex"])()))
    for h in (0.2, 0.4):
        E = np.array(expm(-h * A))
        diag, coup = E[0:2, 0:2], E[0:2, 2:4]
        # coupling = -h * diag  (the identity coupling commutes with R), so the ratio is exactly -h
        assert np.allclose(coup, -h * diag, atol=1e-10), (
            f"coupling is not h * exp(-hA_block) at h={h}"
        )


# ----------------------------------------------------------------- the eigenvalue is bounded from the start
def test_eigenvalue_is_non_negative_by_construction():
    for k in range(50):
        for blocks, kinds in (([2, 2], ["real", "real"]), ([4, 4], ["complex", "complex"])):
            A = np.array(build("jordan_repeated", blocks, kinds, key=k, variance=4.0)())
            assert np.all(np.diag(A) >= 0.0), "softplus must keep the real part non-negative"


def test_parameter_count_is_per_block():
    ap = build("jordan_repeated", [4, 4], ["complex", "complex"])
    assert sum(int(p.size) for p in ap.block_params) == 4  # 2 per block whatever the size
    ap = build("jordan_repeated", [3, 3], ["real", "real"])
    assert sum(int(p.size) for p in ap.block_params) == 2  # 1 per block


@pytest.mark.parametrize("blocks,kinds", [([2], ["complex"]), ([1], ["real"]), ([3], ["complex"])])
def test_degenerate_block_sizes_are_rejected(blocks, kinds):
    """A complex pair already fills a 2x2 real block, so size 2 complex is one pair with nothing to chain to:
    that is `unconstrained_rotation`, not a Jordan block."""
    with pytest.raises(ValueError):
        build("jordan_repeated", blocks, kinds)


# ----------------------------------------------------------------- the rename
def test_heartbeat_is_rotation_planes_under_its_new_name():
    c = json.load(open(os.path.join(ROOT, "uea_results", "heartbeat", "config.json")))["model"]["A"]
    assert (
        c["parametrization"] == "unconstrained_rotation"
        and c["init"]["type"] == "unconstrained_rotation"
    )
    assert set(c["blocks"]) == {2} and set(c["kinds"]) == {"complex"}
    w = sum(c["blocks"])
    A = np.array(
        build(
            "unconstrained_rotation",
            c["blocks"],
            c["kinds"],
            key=11,
            variance=c["init"]["variance"],
        )()
    )
    for i in range(0, w, 2):
        assert A[i, i] == A[i + 1, i + 1] and A[i, i + 1] == -A[i + 1, i]
    off = A.copy()
    for i in range(0, w, 2):
        off[i : i + 2, i : i + 2] = 0
    assert np.count_nonzero(off) == 0, "the size-2 case has no super-diagonal at all"


@pytest.mark.parametrize(
    "blocks,kinds", [([4], ["complex"]), ([2], ["real"]), ([2, 4], ["complex", "complex"])]
)
def test_unconstrained_rotation_rejects_anything_but_2x2_complex(blocks, kinds):
    with pytest.raises(ValueError):
        build("unconstrained_rotation", blocks, kinds)


def test_bidiagonal_is_gone():
    assert "bidiagonal" not in P.A_PARAM_NAMES and not hasattr(P, "AParamBidiagonal")


def test_engine_sees_the_block_structure():
    import ews.core.kernels.combine as C

    ap = build("jordan_repeated", [4, 4], ["complex", "complex"])
    assert C.block_sizes_of(ap) == (4, 4)
    assert C.is_rotation_planes(ap) is False  # a Jordan block is not a canonical rotation plane
    hb = build("unconstrained_rotation", [2] * 5, ["complex"] * 5)
    assert C.block_sizes_of(hb) == (2,) * 5
    assert (
        C.is_rotation_planes(hb) is False
    )  # kept False on purpose: Heartbeat's engine path must not move
