"""The leaf: the per-interval stage of the EWS (leaf -> combine), routed to one of its implementations.

    kind="van_loan"      van_loan.py + signature.py      reweighted knots, then their signature (signax)
    kind="series"        series_leaf.py                  the level tensors directly from the series letters
    kind="plane_closed"  kernels/plane_engine.py         closed forms for 2x2 rotation planes, depths 1-2
    kind="block_series"  kernels/blockdiag_engine.py     the series block by block for block-diagonal A, depths 1-2
    kind="jordan_closed" kernels/jordan_engine.py        closed forms for uniform Jordan chains, depths 1-2
                                                         ("jordan_closed_tables": the float64-table form throughout)

Every kind returns level k with shape (n_intervals,) + (d,)*k, so the combine is the same for all; the default is
"van_loan". The series order `series_q` sizes an array, so it is fixed for a run (changing it would change the pytree
and invalidate the optimiser state): size it with `suggest_series_q` and check it with `check_series_q`.
"""

import jax

# Match the convention in the sibling ews/core modules (van_loan.py:20, chen_scan.py:9,
# signature.py:10): x64 is enabled at import, and several modules depend on that side effect.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from ews.core.series_leaf import (  # noqa: E402
    series_error_bound,
    series_local_signatures,
    series_regime,
)
from ews.core.signature import compute_reweighted_signatures  # noqa: E402
from ews.core.van_loan import reweighted_segments_van_loan  # noqa: E402

__all__ = [
    "LEAF_KINDS",
    "check_series_q",
    "local_signatures",
    "series_error_bound",
    "series_regime",
    "suggest_series_q",
]

LEAF_KINDS = (
    "van_loan",
    "series",
    "plane_closed",
    "block_series",
    "jordan_closed",
    "jordan_closed_tables",
)


def local_signatures(
    path,
    clock,
    A,
    trunc,
    *,
    kind="van_loan",
    dtype=jnp.float32,
    M=4,
    van_loan_expm="cayley",
    reweighting_mode="displacement",
    series_q=None,
    block_sizes=None,
    jordan_chain=None,
):
    """Per-interval local signatures, level k of shape (n_intervals,) + (d,)*k.

    Args:
        path: (L, d) knots.  clock: (L,) monotone clock.  A: (d, d) effective generator.
        trunc: truncation depth n.
        kind: "van_loan" (default), "series" or one of the block leaves below.
        M, van_loan_expm, reweighting_mode: Van Loan sub-options, ignored when kind="series".
        series_q: series order, required when kind="series" / "block_series", ignored otherwise.
        block_sizes: diagonal-block sizes of A, required when kind="block_series"
            (see ews.core.kernels.combine.block_sizes_of); ignored otherwise.
        jordan_chain: (kind, size) of A's uniform Jordan chains, required when kind="jordan_closed" /
            "jordan_closed_tables" (see ews.core.kernels.jordan_engine.jordan_chains_of); ignored otherwise.

    "plane_closed" needs A block-diagonal with canonical 2x2 planes [[d, w], [-w, d]] (the
    rotation parametrisations) and trunc <= 2; "block_series" needs a block-diagonal A of any
    block sizes and trunc <= 2. Both ignore the Van Loan options (they are exact in dtheta).
    "jordan_closed" needs uniform Jordan chains (AParamJordanRepeated: real chains of 2-4 or complex chains of
    size 4, 6, 8) and trunc <= 2; it is exact at any x = ||A|| dtheta and has no series order. It is
    the faster leaf for real chains of 2 and complex chains of 4, and for any chain at large x; for longer chains at
    small x, "series" is as fast or faster.
    """
    if kind == "van_loan":
        # Preserve ews_layer.py's original validation exactly: an unknown mode must raise, not
        # silently fall through to the non-uniform path.
        if reweighting_mode not in (
            "displacement",
            "displacement_uniform",
            "displacement_uniform_basepoint",
        ):
            raise ValueError(
                "reweighting_mode must be 'displacement', 'displacement_uniform', or "
                f"'displacement_uniform_basepoint', got {reweighting_mode}"
            )
        interval_times = clock[1:] - clock[:-1]
        reweighted = reweighted_segments_van_loan(
            path,
            interval_times,
            A,
            M,
            dtype=dtype,
            expm_method=van_loan_expm,
            uniform=reweighting_mode in ("displacement_uniform", "displacement_uniform_basepoint"),
            basepoint=reweighting_mode == "displacement_uniform_basepoint",
        )
        return compute_reweighted_signatures(reweighted, trunc, dtype=dtype)

    if kind == "series":
        if series_q is None:
            raise ValueError("kind='series' requires series_q (see check_series_q).")
        return series_local_signatures(path, clock, A, trunc, int(series_q), dtype=dtype)

    if kind == "plane_closed":
        if trunc > 2:
            raise NotImplementedError(
                "leaf='plane_closed': closed-form local terms exist for depths 1 and 2"
            )
        from ews.core.kernels import plane_engine as pe

        path = jnp.asarray(path, dtype)
        A = jnp.asarray(A, dtype)
        v = path[1:] - path[:-1]
        dtheta = (clock[1:] - clock[:-1]).astype(dtype)
        d = A.shape[0]
        if d % 2:
            raise ValueError("leaf='plane_closed' needs an even dimension (2x2 planes)")
        ii = jnp.arange(0, d, 2)
        dw = jnp.stack([A[ii, ii], A[ii, ii + 1]], -1)  # (K, 2) [d, w] of each plane
        _rho, s1, s2 = pe.local_terms(v, dw, dtheta, int(trunc))
        levels = [pe.from_complex_planes(s1)]  # (L, d)
        if trunc == 2:
            levels.append(pe.blocks_to_flat(s2).reshape(v.shape[0], d, d))
        return levels

    if kind == "block_series":
        if trunc > 2:
            raise NotImplementedError(
                "leaf='block_series': blockwise series local terms exist for depths 1 and 2"
            )
        if block_sizes is None:
            raise ValueError("leaf='block_series' needs block_sizes (a block-diagonal A)")
        if trunc == 2 and series_q is None:
            raise ValueError(
                "leaf='block_series' at depth 2 requires series_q (see check_series_q)."
            )
        from ews.core.kernels import blockdiag_engine as bd
        from ews.core.kernels.combine import pad_block_problem

        path = jnp.asarray(path, dtype)
        A = jnp.asarray(A, dtype)
        v = path[1:] - path[:-1]
        dtheta = (clock[1:] - clock[:-1]).astype(dtype)
        L, d = v.shape
        # uniform block size b = max(sizes), zero-padded (exact); the power-of-two block count is harmless here
        blocks, (vp,), keep, mp = pad_block_problem(A, [v], block_sizes)
        _E, s1, s2 = bd.local_terms(
            vp,
            blocks,
            dtheta,
            int(trunc),
            expm_fn=bd.expm_pade,
            series_q=None if series_q is None else int(series_q),
        )
        lvl1 = s1.reshape(L, mp)
        levels = [lvl1[:, keep]]
        if trunc == 2:
            levels.append(bd.blocks_to_flat(s2).reshape(L, mp, mp)[:, keep][:, :, keep])
        return levels

    if kind in ("jordan_closed", "jordan_closed_tables"):
        if trunc > 2:
            raise NotImplementedError(
                f"leaf={kind!r}: closed-form local terms exist for depths 1 and 2"
            )
        if jordan_chain is None:
            raise ValueError(
                f"leaf={kind!r} needs jordan_chain (uniform Jordan chains; see jordan_engine.jordan_chains_of)"
            )
        from ews.core.kernels import jordan_engine as je

        path = jnp.asarray(path, dtype)
        A = jnp.asarray(A, dtype)
        v = path[1:] - path[:-1]
        dtheta = (clock[1:] - clock[:-1]).astype(dtype)
        return je.local_terms(
            v,
            A,
            dtheta,
            int(trunc),
            jordan_chain[0],
            jordan_chain[1],
            tables=kind == "jordan_closed_tables",
        )

    raise ValueError(f"leaf kind must be one of {LEAF_KINDS}, got {kind!r}")


def suggest_series_q(x, trunc, tol=1e-6, q_max=200):
    """Smallest q whose per-interval error bound at level `trunc` is below `tol`.

    Bound: (k*x)^(q+1)/(q+1)! * e^(k*x), evaluated in log space so it does not overflow for
    x >~ 20 (the direct form builds an integer factorial).
    """
    import math

    if x <= 0.0:
        return 1
    lx, ltol = math.log(trunc * x), math.log(tol)
    for q in range(1, q_max):
        if (q + 1) * lx - math.lgamma(q + 2) + trunc * x < ltol:
            return q
    return q_max


def check_series_q(A, dtheta_max, trunc, series_q, tol=1e-6, where=""):
    """Warn if `series_q` is too small for the observed x = ||A||_op * max dtheta.

    Returns (x, bound, sufficient_q). Warns rather than raises: q is fixed for a run, so a
    warning at build and at each eval surfaces drift without being able to act on it.
    """
    import warnings

    x = float(jnp.linalg.norm(jnp.asarray(A), 2) * float(dtheta_max))
    bound = series_error_bound(x, int(trunc), int(series_q))
    need = suggest_series_q(x, int(trunc), tol=tol)
    if bound > tol:
        warnings.warn(
            f"[series leaf{' ' + where if where else ''}] series_q={series_q} gives a "
            f"per-interval error bound of {bound:.2e} at x={x:.3f} (||A||*max dtheta), "
            f"above tol={tol:.0e}. series_q={need} would suffice. q is fixed for the run; "
            f"raise it in config and restart if this matters.",
            RuntimeWarning,
            stacklevel=2,
        )
    return x, bound, need
