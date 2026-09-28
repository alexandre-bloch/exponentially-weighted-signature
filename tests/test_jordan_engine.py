"""Gates for the closed-form Jordan leaf (``EWS(leaf="jordan_closed")``, ews/core/kernels/jordan_engine.py), to the standard
of tests/test_engines.py, whose helpers and criteria are reused: the leaf must reproduce the default path (series leaf +
chen_scan associative scan), forward for every readout, and the parameter and path gradients.

CPU (always, float64): real chains of 2, 3, 4 and complex chains of size 4, 6, 8, depths 1 and 2, both leaves
(``jordan_closed``: the fast form where it exists; ``jordan_closed_tables``: the float64 tables throughout), default and lean
combines, the uniform-clock modes, a rescaled generator (``a_scale``), fail-fast cases, the any-argument tables against
mpmath, and large regimes against a direct matrix-exponential quadrature.
GPU (skipped without one, float32): the block kernel route, measured against the float64 default path and required to be as
accurate as the float32 default engine (within 4x of its own error), exactly as the existing kernels are.
"""

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ews import params as P  # noqa: E402
from ews.core.kernels import jordan_engine as je  # noqa: E402
from tests import test_engines as T  # noqa: E402  (the shared layer, data and comparison helpers)

CASES = [("real", 2), ("real", 3), ("real", 4), ("complex", 4), ("complex", 6), ("complex", 8)]
LEAVES = ("jordan_closed", "jordan_closed_tables")


def jordan_A(kind, size, K, seed=0, dtype=jnp.float64):
    rng = np.random.default_rng(seed)
    params = []
    for _ in range(K):
        v = rng.normal(-1.5, 0.7)
        params.append(
            jnp.asarray(
                [v] if kind == "real" else [v, rng.uniform(0.3, 1.5) * rng.choice([-1.0, 1.0])],
                dtype,
            )
        )
    return P.AParamJordanRepeated([size] * K, [kind] * K, params)


class Jitted(eqx.Module):
    """A layer whose call is compiled. Every layer compared here is wrapped in it, the reference included: on the CPU
    backend of jaxlib 0.9.0 the series leaf run eagerly corrupts the heap when its gradient is taken repeatedly in one
    process (glibc aborts, or silently wrong numbers); under jit it is exact and repeatable, jit is how every trainer runs
    it, and jax 0.10.2 does not show it. The comparison criteria are unchanged: test_engines' ``_compare`` / ``_compare_f32``."""

    layer: eqx.Module
    a_scale: float = eqx.field(static=True, default=1.0)

    @eqx.filter_jit
    def __call__(self, path, clock, readout="prefix", output_format="flat"):
        return self.layer(
            path, clock, readout=readout, output_format=output_format, a_scale=self.a_scale
        )


def layer(*args, a_scale=1.0, **kw):
    return Jitted(T._layer(*args, **kw), a_scale)


@pytest.mark.parametrize("N", [1, 2])
@pytest.mark.parametrize("kind,size", CASES)
def test_jordan_closed_matches_default_path(N, kind, size):
    """cf. test_plane_closed_leaf_matches_series: prefix, terminal and mean readouts and the path / A gradients against
    series q 24 + assoc at 1e-7, then against series q 60 at 1e-10."""
    K, L = 2, 13
    A = jordan_A(kind, size, K)
    path, clock = T._data(L, size * K)
    assert je.jordan_chains_of(A) == (kind, size)
    for leaf in LEAVES:
        T._compare(
            layer(A, N),
            layer(A, N, leaf=leaf),
            path,
            clock,
            ["prefix", "terminal", "mean"],
            tol=1e-7,
        )
        T._compare(
            layer(A, N, series_q=60),
            layer(A, N, leaf=leaf),
            path,
            clock,
            ["prefix", "terminal"],
            tol=1e-10,
        )


@pytest.mark.parametrize("kind,size", CASES)
def test_jordan_closed_with_lean_combine_and_assoc_terminal(kind, size):
    K, L = 2, 9
    A = jordan_A(kind, size, K, seed=1)
    path, clock = T._data(L, size * K)
    T._compare(
        layer(A, 2, series_q=60),
        layer(A, 2, leaf="jordan_closed", combine="lean"),
        path,
        clock,
        ["prefix", "terminal"],
        tol=1e-10,
    )
    T._compare(
        layer(A, 2, series_q=60, terminal_reduce="assoc"),
        layer(A, 2, leaf="jordan_closed", terminal_reduce="assoc"),
        path,
        clock,
        ["terminal"],
        tol=1e-10,
    )


@pytest.mark.parametrize("mode", ["displacement_uniform", "displacement_uniform_basepoint"])
def test_jordan_closed_uniform_clock_modes(mode):
    K, L, kind, size = 2, 12, "real", 2
    A = jordan_A(kind, size, K, seed=2)
    path, clock = T._data(L, size * K, uniform=True)
    if mode.endswith("basepoint"):
        clock = jnp.asarray(np.concatenate([[0.0], np.arange(L) * 0.7]))
        path = path.at[0].set(path[1])
    T._compare(
        layer(A, 2, series_q=60, reweighting_mode=mode),
        layer(A, 2, leaf="jordan_closed", reweighting_mode=mode),
        path,
        clock,
        ["prefix", "terminal"],
        tol=1e-10,
    )


@pytest.mark.parametrize("kind,size", CASES)
def test_jordan_closed_with_rescaled_generator(kind, size):
    """``a_scale`` multiplies the whole generator, so the chain coupling is a_scale, not 1: the leaf reads it from A."""
    K, L = 2, 9
    A = jordan_A(kind, size, K, seed=8)
    path, clock = T._data(L, size * K)
    for leaf in LEAVES:
        T._compare(
            layer(A, 2, series_q=60, a_scale=0.6),
            layer(A, 2, leaf=leaf, a_scale=0.6),
            path,
            clock,
            ["prefix", "terminal"],
            tol=1e-10,
        )


def _terms(kind, size, A, dx, dth, tables):
    return je.local_terms(
        jnp.asarray(dx), jnp.asarray(A), jnp.asarray(dth), 2, kind, size, tables=tables
    )


def test_fast_and_table_forms_agree_at_every_regime():
    rng = np.random.default_rng(4)
    for kind, size in je.FAST_CHAINS:
        K, L = 3, 30
        A = jordan_A(kind, size, K, seed=3)()
        for scale in (0.02, 1.0, 8.0, 40.0):
            dx = rng.normal(size=(L, size * K))
            dth = np.concatenate([[0.0], rng.lognormal(0, 0.5, L - 1) * scale])
            for lv_f, lv_t in zip(
                _terms(kind, size, A, dx, dth, False), _terms(kind, size, A, dx, dth, True)
            ):
                assert T.relerr(lv_f, lv_t) < 1e-9, (
                    f"{kind} {size} scale {scale}: {T.relerr(lv_f, lv_t):.2e}"
                )


def test_fail_fast():
    A = jordan_A("real", 2, 2)
    path, clock = T._data(6, 4)
    with pytest.raises(NotImplementedError):
        T._layer(A, 3, leaf="jordan_closed")(path, clock, readout="prefix", output_format="flat")
    with pytest.raises(ValueError):
        T._layer(T._rotation_A(2), 2, leaf="jordan_closed")  # not a Jordan A
    with pytest.raises(ValueError):
        T._layer(
            P.AParamJordanRepeated([2, 4], ["real", "real"], [jnp.zeros(1), jnp.zeros(1)]),
            2,
            leaf="jordan_closed",
        )  # mixed sizes
    with pytest.raises(ValueError):
        T._layer(jordan_A("real", 5, 1), 2, leaf="jordan_closed_tables")  # unsupported length
    assert je.jordan_chains_of(T._rotation_A(2)) is None


def test_tables_against_mpmath():
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 30
    ys = [
        -1e-9,
        -0.01,
        -3.0,
        -7.9,
        -8.1,
        -40.0,
        -100.0,
        complex(-0.5, 2.0),
        complex(-5, 30),
        complex(-0.5, 100),
        complex(-7.0, 4.2),
    ]
    for y in ys:
        ref = [complex(mp.quad(lambda t: t**r * mp.exp(t * y), [0, 1])) for r in range(18)]
        got = np.asarray(je.phi_derivative_table(jnp.asarray([y]), 17))[0]
        assert max(abs(got[r] - ref[r]) for r in range(18)) < 1e-12, y
    pairs = [
        (-0.7, -0.7),
        (-0.76, -0.76),
        (-2.0, -0.24),
        (-2.0, -0.26),
        (-30.0, -45.0),
        (-0.05, -80.0),
        (complex(-1, 3), complex(-1, -3)),
        (complex(-0.2, 40), complex(-0.2, 40)),
        (complex(-0.01, 0.2), complex(-0.01, -0.2)),
    ]
    for xv, dv in pairs:
        got = je.pair_integral_derivatives(jnp.asarray([[xv]]), jnp.asarray([[dv]]), 3, 3)
        for p in range(4):
            for q in range(4):
                ref = complex(
                    mp.quad(
                        lambda t, u: t**q * u ** (p + q + 1) * mp.exp(u * (xv + t * dv)),
                        [0, 1],
                        [0, 1],
                    )
                )
                assert abs(complex(np.asarray(got[(p, q)])[0, 0, 0]) - ref) / abs(ref) < 1e-8, (
                    xv,
                    dv,
                    p,
                    q,
                )


def _quad_reference(A, dx, h, nodes=16):
    """Levels 1 and 2 of one interval by Gauss-Legendre quadrature of the matrix exponential, resolving the oscillation."""
    from scipy.linalg import expm

    A = np.asarray(A, np.float64)
    w = np.asarray(dx, np.float64) / h
    n_uni = int(np.ceil(h * max(np.abs(A).sum(1).max(), 1e-9) / 0.5)) + 4
    edges = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, h, n_uni + 1),
                h * (1.0 - np.geomspace(1.0, 1e-7, 12))[::-1],
                [0.0, h],
            ]
        )
    )
    xg, wg = np.polynomial.legendre.leggauss(nodes)
    x8, w8 = np.polynomial.legendre.leggauss(8)
    E = lambda u: expm(-(h - u) * A) @ w  # noqa: E731
    S1, S2, cum, last = np.zeros_like(w), np.zeros((w.size, w.size)), np.zeros_like(w), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        for xi, wi in zip(xg, wg):
            u2 = mid + half * xi
            m2, h2 = (last + u2) / 2, (u2 - last) / 2
            cum = cum + h2 * sum(wj * E(m2 + h2 * xj) for xj, wj in zip(x8, w8))
            last = u2
            S1 = S1 + half * wi * E(u2)
            S2 = S2 + half * wi * np.outer(cum, E(u2))
    return S1, S2


@pytest.mark.parametrize("kind,size", [("real", 2), ("real", 4), ("complex", 4), ("complex", 6)])
def test_large_regimes_against_matrix_exponential_quadrature(kind, size):
    """Where no series order can serve as the reference (clock step x |eigenvalue| up to about 60)."""
    rng = np.random.default_rng(7)
    A = jordan_A(kind, size, 1, seed=5)()
    for h in (0.6, 12.0, 40.0):
        dx = rng.normal(size=(size,))
        S1r, S2r = _quad_reference(A, dx, h)
        for tables in (True, False):
            new = _terms(kind, size, A, dx[None], [h], tables)
            assert T.relerr(new[0][0], S1r) < 1e-8 and T.relerr(new[1][0], S2r) < 1e-7, (
                kind,
                size,
                h,
                tables,
            )


# ---------------------------------------------------------------------------- GPU: the block kernel route, float32


@T.gpu_only
@pytest.mark.parametrize("N", [1, 2])
@pytest.mark.parametrize(
    "kind,size,K",
    [
        ("real", 2, 3),
        ("real", 2, 4),
        ("real", 3, 2),
        ("real", 4, 2),
        ("complex", 4, 2),
        ("complex", 4, 3),
        ("complex", 6, 2),
        ("complex", 8, 2),
    ],
)
def test_block_kernel_with_jordan_closed(N, kind, size, K):
    """cf. test_block_kernel_rotation_with_closed_forms: closed-form leaf + block kernel in float32 against the float64
    default path; below 2e-3 and within 4x of the float32 default engine's own error, forward and gradients."""
    L = 48
    A = jordan_A(kind, size, K, seed=6)
    path, clock = T._data(L, size * K)
    ref64 = layer(A, N)
    ref32 = layer(T._to32(A), N, dtype=jnp.float32)
    for leaf in LEAVES if (kind, size) in je.FAST_CHAINS else LEAVES[:1]:
        new32 = layer(
            T._to32(A), N, dtype=jnp.float32, leaf=leaf, combine="block_kernel", combine_chunk=16
        )
        T._compare_f32(ref64, ref32, new32, path, clock, ["prefix", "terminal"])
