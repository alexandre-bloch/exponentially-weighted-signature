"""Gates for the optional EWS engines (``EWS(leaf=..., combine=...)``, ews/core/kernels).

Every non-default leaf / combine must reproduce the default path (Van Loan or series leaf +
chen_scan associative scan): forward for every readout and reweighting mode, and the parameter
and path gradients. The defaults themselves are pinned by tests/test_kernel_outputs_pinned.py.

CPU (always): the lean combine (any A, depths 1-3), the closed-form plane leaf and the
block-diagonal series leaf (f64, tight tolerances).
GPU (skipped without one): the fused block kernel (rotation planes, 2x2 / 4x4 / mixed blocks,
prefix and terminal, forward + gradients) and the dense kernel: f32 engines are measured against the
f64 default path and must be as accurate as the f32 default engine (within 4x of its own error).
"""

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ews import params as P  # noqa: E402
from ews.core.kernels.combine import block_sizes_of  # noqa: E402
from ews.layer import EWS  # noqa: E402

HAS_GPU = any(d.platform == "gpu" for d in jax.devices())
gpu_only = pytest.mark.skipif(not HAS_GPU, reason="Pallas/Triton kernels need a GPU")


def relerr(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def _general_A(m, seed=0, dtype=jnp.float64):
    rng = np.random.default_rng(seed)
    A0 = rng.normal(size=(m, m)) * 0.2
    A0 = (A0 - A0.T) / 2 + np.diag(rng.uniform(0.05, 0.3, m))
    return P.AParamGeneral(jnp.asarray(A0, dtype))


def _blockdiag_A(sizes, seed=0, dtype=jnp.float64):
    rng = np.random.default_rng(seed)
    blocks = [jnp.asarray(rng.normal(size=(s, s)) * 0.3, dtype) for s in sizes]
    return P.AParamBlockDiag(blocks, list(sizes))


def _planes_A(K, seed=0, dtype=jnp.float64):
    """Canonical 2x2 planes [[d, w], [-w, d]] as an AParamBlockDiag (the closed-form leaf's structure)."""
    rng = np.random.default_rng(seed)
    blocks = []
    for _ in range(K):
        d, w = rng.uniform(0.05, 0.4), rng.uniform(-1.0, 1.0)
        blocks.append(jnp.asarray([[d, w], [-w, d]], dtype))
    return P.AParamBlockDiag(blocks, [2] * K)


def _rotation_A(K, seed=0):
    rng = np.random.default_rng(seed)
    return P.AParamDampedRotation(
        jnp.asarray(rng.normal(size=K)), jnp.asarray(rng.normal(size=K)), 2 * K, 1.5
    )


def _data(L, m, seed=1, dtype=jnp.float64, uniform=False, basepoint=False):
    rng = np.random.default_rng(seed)
    path = jnp.asarray(np.cumsum(rng.normal(size=(L + 1, m)) * 0.3, 0), dtype)
    if uniform:
        clock = jnp.asarray(np.arange(L + 1) * 0.7, dtype)
        if basepoint:
            clock = jnp.concatenate([clock[:1], clock[:-1] + 0.0]) * 0 + jnp.asarray(
                np.concatenate([[0.0, 0.0], np.arange(1, L) * 0.7]), dtype
            )
    else:
        clock = jnp.asarray(np.cumsum(rng.lognormal(0, 0.3, L + 1)), dtype)
    return path, clock


def _layer(A, N, dtype=jnp.float64, **kw):
    base = dict(
        M=4,
        dtype=dtype,
        reweighting_mode="displacement",
        van_loan_expm="exact",
        chen_expm="exact",
        leaf="series",
        series_q=24,
    )
    base.update(kw)
    return EWS(A, N=N, **base)


def _value_and_grads(layer, path, clock, readout, weights):
    def loss(mod, pth):
        out = mod(pth, clock, readout=readout, output_format="flat")
        return jnp.sum(out * weights)

    spec = jax.tree_util.tree_map(eqx.is_inexact_array, layer)
    diff, static = eqx.partition(layer, spec)
    v, (ga, gp) = jax.value_and_grad(
        lambda d, pth: loss(eqx.combine(d, static), pth), argnums=(0, 1)
    )(diff, path)
    return v, jax.tree_util.tree_leaves(eqx.filter(ga, eqx.is_array)), gp


def _compare(ref, new, path, clock, readouts, tol, grads=True):
    for readout in readouts:
        out_r = ref(path, clock, readout=readout, output_format="flat")
        out_n = new(path, clock, readout=readout, output_format="flat")
        assert out_r.shape == out_n.shape
        e = relerr(out_n, out_r)
        assert e < tol, f"{readout}: forward relerr {e:.2e}"
        if grads:
            w = jnp.asarray(np.random.default_rng(3).normal(size=out_r.shape), out_r.dtype)
            v_r, ga_r, gp_r = _value_and_grads(ref, path, clock, readout, w)
            v_n, ga_n, gp_n = _value_and_grads(new, path, clock, readout, w)
            assert relerr(v_n, v_r) < tol
            assert relerr(gp_n, gp_r) < tol, f"{readout}: path grad relerr {relerr(gp_n, gp_r):.2e}"
            for a, b in zip(ga_n, ga_r):
                if np.max(np.abs(np.asarray(b))) > 0:
                    assert relerr(a, b) < tol, f"{readout}: A grad relerr {relerr(a, b):.2e}"


def _to32(A):
    return jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float32) if eqx.is_inexact_array(x) else x, A
    )


def _compare_f32(ref64, ref32, new32, path64, clock64, readouts, tol=2e-3, factor=4.0):
    """f32 engines against the f64 default path: the new engine's forward and gradient errors must be below
    `tol` and within `factor` x the f32 default engine's own error (the kernels are as accurate as what they
    replace; f32 associative scans carry ~1e-4-1e-3 noise in this regime, which is not the kernel's fault)."""
    path32, clock32 = path64.astype(jnp.float32), clock64.astype(jnp.float32)
    for readout in readouts:
        truth = ref64(path64, clock64, readout=readout, output_format="flat")
        out_r = ref32(path32, clock32, readout=readout, output_format="flat")
        out_n = new32(path32, clock32, readout=readout, output_format="flat")
        assert out_n.shape == truth.shape
        e_n, e_r = relerr(out_n, truth), relerr(out_r, truth)
        assert e_n < tol and e_n < factor * e_r + 1e-5, (
            f"{readout}: forward relerr kernel {e_n:.2e} vs default f32 {e_r:.2e}"
        )
        w = np.random.default_rng(3).normal(size=truth.shape)
        v_t, ga_t, gp_t = _value_and_grads(
            ref64, path64, clock64, readout, jnp.asarray(w, jnp.float64)
        )
        v_r, ga_r, gp_r = _value_and_grads(
            ref32, path32, clock32, readout, jnp.asarray(w, jnp.float32)
        )
        v_n, ga_n, gp_n = _value_and_grads(
            new32, path32, clock32, readout, jnp.asarray(w, jnp.float32)
        )
        e_n, e_r = relerr(gp_n, gp_t), relerr(gp_r, gp_t)
        assert e_n < tol and e_n < factor * e_r + 1e-5, (
            f"{readout}: path grad relerr kernel {e_n:.2e} vs default f32 {e_r:.2e}"
        )
        for a_n, a_r, a_t in zip(ga_n, ga_r, ga_t):
            if np.max(np.abs(np.asarray(a_t))) > 0:
                e_n, e_r = relerr(a_n, a_t), relerr(a_r, a_t)
                assert e_n < tol and e_n < factor * e_r + 1e-5, (
                    f"{readout}: A grad relerr kernel {e_n:.2e} vs default f32 {e_r:.2e}"
                )


# ---------------------------------------------------------------------------- defaults & structure


def test_defaults_unchanged():
    A = _general_A(4)
    layer = EWS(A, N=2)
    assert layer.leaf == "van_loan" and layer.combine == "assoc" and layer.terminal_reduce == "tree"
    assert layer.block_sizes is None
    assert block_sizes_of(_blockdiag_A((2, 3, 1))) == (2, 3, 1)
    assert block_sizes_of(_rotation_A(3)) == (2, 2, 2)
    assert block_sizes_of(_general_A(6)) is None


def test_option_validation():
    with pytest.raises(ValueError):
        EWS(_general_A(4), N=1, combine="block_kernel")
    with pytest.raises(ValueError):
        EWS(_general_A(4), N=1, leaf="plane_closed")
    with pytest.raises(ValueError):
        EWS(
            _blockdiag_A((2, 2)), N=1, leaf="plane_closed"
        )  # free 2x2 blocks are not (known to be) planes
    with pytest.raises(ValueError):
        EWS(_general_A(4), N=1, combine="nonsense")
    with pytest.raises(ValueError):
        EWS(_blockdiag_A((2, 2)), N=2, leaf="block_series")  # depth 2 needs series_q


# ---------------------------------------------------------------------------- CPU: lean combine, new leaves


@pytest.mark.parametrize("N", [1, 2, 3])
@pytest.mark.parametrize(
    "mode", ["displacement", "displacement_uniform", "displacement_uniform_basepoint"]
)
def test_lean_combine_matches_assoc(N, mode):
    m, L = 4, 13
    A = _general_A(m)
    uniform = mode != "displacement"
    path, clock = _data(L, m, uniform=uniform)
    if mode == "displacement_uniform_basepoint":
        clock = jnp.asarray(
            np.concatenate([[0.0], np.arange(L) * 0.7])
        )  # first interval has dtheta 0
        path = path.at[0].set(path[1])  # and a zero increment (the basepoint knot)
    ref = _layer(A, N, reweighting_mode=mode, leaf="van_loan")
    new = _layer(A, N, reweighting_mode=mode, leaf="van_loan", combine="lean")
    _compare(ref, new, path, clock, ["prefix", "terminal", "mean", "max"], tol=1e-10, grads=(N < 3))


def test_lean_combine_terminal_matches_tree_and_assoc():
    m, L = 3, 10
    A = _general_A(m)
    path, clock = _data(L, m)
    for reduce in ("tree", "assoc"):
        ref = _layer(A, 2, terminal_reduce=reduce)
        new = _layer(A, 2, terminal_reduce=reduce, combine="lean")
        _compare(ref, new, path, clock, ["terminal"], tol=1e-10)


@pytest.mark.parametrize("N", [1, 2])
def test_plane_closed_leaf_matches_series(N):
    K, L = 3, 17
    A = _rotation_A(K)
    path, clock = _data(L, 2 * K)
    ref = _layer(A, N)
    new = _layer(A, N, leaf="plane_closed")
    assert new.block_sizes == (2,) * K
    _compare(
        ref, new, path, clock, ["prefix", "terminal", "mean"], tol=1e-7
    )  # series q 24 truncation ~1e-8


@pytest.mark.parametrize("N", [1, 2])
@pytest.mark.parametrize("sizes", [(2, 2, 2), (2, 3, 1), (4, 4), (3, 4)])
def test_block_series_leaf_matches_series(N, sizes):
    L = 11
    A = _blockdiag_A(sizes)
    path, clock = _data(L, sum(sizes))
    ref = _layer(A, N)
    new = _layer(A, N, leaf="block_series")
    _compare(ref, new, path, clock, ["prefix", "terminal"], tol=1e-7)


def test_new_leaf_with_lean_combine():
    """Leaf and combine are independent choices: closed-form leaf + lean scan == series + assoc."""
    K, L = 2, 9
    A = _rotation_A(K)
    path, clock = _data(L, 2 * K)
    _compare(
        _layer(A, 2),
        _layer(A, 2, leaf="plane_closed", combine="lean"),
        path,
        clock,
        ["prefix", "terminal"],
        tol=1e-8,
    )


def test_depth3_leaves_fail_fast():
    A = _rotation_A(2)
    path, clock = _data(6, 4)
    with pytest.raises(NotImplementedError):
        _layer(A, 3, leaf="plane_closed")(path, clock, readout="prefix", output_format="flat")
    with pytest.raises(NotImplementedError):
        _layer(A, 3, leaf="block_series")(path, clock, readout="prefix", output_format="flat")


# ---------------------------------------------------------------------------- GPU: fused kernels (f32)


@gpu_only
@pytest.mark.parametrize("N", [1, 2])
@pytest.mark.parametrize("sizes", [(2, 2, 2), (2, 2, 2, 2), (4, 4, 4, 4), (4, 2, 4), (3, 4)])
def test_block_kernel_matches_assoc(N, sizes):
    L = 32
    A = _blockdiag_A(sizes)
    path, clock = _data(L, sum(sizes))
    ref64 = _layer(A, N)
    ref32 = _layer(_to32(A), N, dtype=jnp.float32)
    new32 = _layer(_to32(A), N, dtype=jnp.float32, combine="block_kernel", combine_chunk=8)
    _compare_f32(ref64, ref32, new32, path, clock, ["prefix", "terminal", "mean"])


@gpu_only
@pytest.mark.parametrize("N", [1, 2])
def test_block_kernel_rotation_with_closed_forms(N):
    """The production route for rotation planes: closed-form leaf + block kernel, non-power-of-2 plane count."""
    K, L = 3, 48
    A = _rotation_A(K)
    path, clock = _data(L, 2 * K)
    ref64 = _layer(A, N)
    ref32 = _layer(_to32(A), N, dtype=jnp.float32)
    new32 = _layer(
        _to32(A),
        N,
        dtype=jnp.float32,
        leaf="plane_closed",
        combine="block_kernel",
        combine_chunk=16,
    )
    _compare_f32(ref64, ref32, new32, path, clock, ["prefix", "terminal"])


@gpu_only
@pytest.mark.parametrize("sizes", [(2, 2, 2), (2, 2, 2, 2), (2,) * 16])
def test_block_kernel_depth3_series_leaf(sizes):
    """Depth 3 on the block kernel: the general series leaf supplies levels 1-3 and the kernel combines them; the
    register path when the level-3 tiles fit (m 6, 8), the buffered path when they do not (m 32)."""
    L = 32
    A = _blockdiag_A(sizes)
    path, clock = _data(L, sum(sizes))
    ref64 = _layer(A, 3)
    ref32 = _layer(_to32(A), 3, dtype=jnp.float32)
    new32 = _layer(_to32(A), 3, dtype=jnp.float32, combine="block_kernel", combine_chunk=8)
    # depth 3 in f32: the default engine itself sits at ~4e-3 on the level-3 gradients, so the absolute bound is
    # widened to 5e-3; the kernel must still be within 4x of the default f32 error (it is more accurate in practice)
    _compare_f32(ref64, ref32, new32, path, clock, ["prefix", "terminal", "mean"], tol=5e-3)


@gpu_only
def test_block_kernel_depth3_rotation_planes():
    """The production route at depth 3: damped-rotation planes, series leaf, block kernel, non-power-of-2 plane count."""
    K, L = 5, 48
    A = _rotation_A(K)
    path, clock = _data(L, 2 * K)
    ref64 = _layer(A, 3)
    ref32 = _layer(_to32(A), 3, dtype=jnp.float32)
    new32 = _layer(_to32(A), 3, dtype=jnp.float32, combine="block_kernel", combine_chunk=16)
    _compare_f32(ref64, ref32, new32, path, clock, ["prefix", "terminal"], tol=5e-3)


@gpu_only
def test_block_kernel_uniform_modes():
    K, L = 2, 16
    A = _planes_A(K)
    for mode in ("displacement_uniform", "displacement_uniform_basepoint"):
        path, clock = _data(L, 2 * K, uniform=True)
        if mode.endswith("basepoint"):
            clock = jnp.asarray(np.concatenate([[0.0], np.arange(L) * 0.7]))
            path = path.at[0].set(path[1])
        ref64 = _layer(A, 2, reweighting_mode=mode, leaf="van_loan")
        ref32 = _layer(_to32(A), 2, dtype=jnp.float32, reweighting_mode=mode, leaf="van_loan")
        new32 = _layer(
            _to32(A),
            2,
            dtype=jnp.float32,
            reweighting_mode=mode,
            leaf="van_loan",
            combine="block_kernel",
        )
        _compare_f32(ref64, ref32, new32, path, clock, ["prefix", "terminal"])


@gpu_only
@pytest.mark.parametrize("N", [1, 2])
def test_dense_kernel_matches_assoc(N):
    m, L = 16, 32
    A = _general_A(m)
    path, clock = _data(L, m)
    ref64 = _layer(A, N)
    ref32 = _layer(_to32(A), N, dtype=jnp.float32)
    new32 = _layer(_to32(A), N, dtype=jnp.float32, combine="dense_kernel", combine_chunk=8)
    _compare_f32(ref64, ref32, new32, path, clock, ["prefix"])
