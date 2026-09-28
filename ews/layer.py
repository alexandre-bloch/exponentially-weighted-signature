"""The EWS layer: (path, clock) -> signature features, and the log-signature variant."""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import signax

from ews.core.chen_scan import (
    chen_associative_scan_final_packed_with_M,
    chen_associative_scan_prefix_packed_with_M,
    chen_tree_reduce_final_packed_with_M,
    unpack_signature_batch,
    unpack_signature_vector,
)
from ews.core.kernels.combine import (
    COMBINE_KINDS,
    block_sizes_of,
    combine_packed,
    is_rotation_planes,
)
from ews.core.kernels.jordan_engine import JORDAN_LEAVES, jordan_chains_of
from ews.core.leaf import LEAF_KINDS, local_signatures
from ews.core.output_format import (
    flatten_signature_list,
    flatten_signature_prefix,
)
from ews.params import AParamConstant, B


def _normalize_expm_method(method: str) -> str:
    method = method.lower()
    if method in {"pade_lite", "pade-lite"}:
        return "pade"
    if method not in {"exact", "euler", "cayley", "pade"}:
        raise ValueError("expm method must be one of 'exact', 'euler', 'cayley', 'pade'")
    return method


class EWS(eqx.Module):
    """
    EWS layer using Van Loan + Signax + Chen associative scan.

    Attributes:
        A_param: Learnable drift parametrization (d, d)
        N: Truncation level (static)
        M: Van Loan discretization points (static)
        dtype: Data type (static)
    """

    A_param: eqx.Module
    d: int = eqx.field(static=True)
    N: int = eqx.field(static=True)
    M: int = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)
    reweighting_mode: str = eqx.field(static=True)
    terminal_reduce: str = eqx.field(static=True)
    van_loan_expm: str = eqx.field(static=True)
    chen_expm: str = eqx.field(static=True)
    # Leaf implementation. "van_loan" (default) is the Van Loan + signax leaf;
    # "series" uses the closed-form series leaf. series_q is its order: fixed for a run
    # (it sizes a coefficient array, so changing it changes the pytree structure and would
    # invalidate optimiser state). Use ews.core.leaf.check_series_q to warn if it is too low.
    leaf: str = eqx.field(static=True)
    series_q: Optional[int] = eqx.field(static=True)
    # Combine implementation. "assoc" (default) is the original chen_scan path; "lean",
    # "block_kernel" and "dense_kernel" are the engines in ews.core.kernels (see combine.py).
    # block_sizes is the diagonal-block structure of A_param (from its class; None = dense),
    # used by the block kernel and the block-diagonal leaves. combine_chunk / combine_tile are
    # kernel tuning knobs (no effect on the result).
    combine: str = eqx.field(static=True)
    combine_chunk: int = eqx.field(static=True)
    combine_tile: int = eqx.field(static=True)
    block_sizes: Optional[tuple] = eqx.field(static=True)
    # jordan_flows: the block kernel builds the flows of a uniform Jordan chain A in closed form
    # (combine.jordan_block_operators) instead of by matrix exponentials.
    jordan_flows: bool = eqx.field(static=True)

    def __init__(
        self,
        A,
        N: int,
        M: int = 20,
        dtype: jnp.dtype = jnp.float32,
        reweighting_mode: str = "displacement",
        terminal_reduce: str = "tree",
        van_loan_expm: str = "cayley",
        chen_expm: str = "exact",
        leaf: str = "van_loan",
        series_q: Optional[int] = None,
        combine: str = "assoc",
        combine_chunk: int = 1024,
        combine_tile: int = 4,
        jordan_flows: bool = False,
    ):
        if isinstance(A, jnp.ndarray):
            A_param = AParamConstant(A.astype(dtype))
        else:
            A_param = A
        self.A_param = A_param
        if hasattr(A_param, "dim"):
            self.d = int(A_param.dim)
        else:
            self.d = int(A_param().shape[0])
        self.N = N
        self.M = M
        self.dtype = dtype
        self.reweighting_mode = reweighting_mode
        if terminal_reduce not in {"tree", "assoc"}:
            raise ValueError("terminal_reduce must be 'tree' or 'assoc'")
        self.terminal_reduce = terminal_reduce
        self.van_loan_expm = _normalize_expm_method(van_loan_expm)
        self.chen_expm = _normalize_expm_method(chen_expm)
        if leaf not in LEAF_KINDS:
            raise ValueError(f"leaf must be one of {LEAF_KINDS}, got {leaf!r}")
        if leaf == "series" and series_q is None:
            raise ValueError("leaf='series' requires series_q (see ews.core.leaf.check_series_q)")
        self.leaf = leaf
        self.series_q = None if series_q is None else int(series_q)
        if combine not in COMBINE_KINDS:
            raise ValueError(f"combine must be one of {COMBINE_KINDS}, got {combine!r}")
        self.combine = combine
        self.combine_chunk = int(combine_chunk)
        self.combine_tile = int(combine_tile)
        self.block_sizes = block_sizes_of(A_param)
        if combine == "block_kernel" and self.block_sizes is None:
            raise ValueError(
                "combine='block_kernel' needs a block-diagonal A parametrisation "
                "(rotation planes, AParamUnconstrainedRotation, AParamJordanRepeated or AParamBlockDiag)"
            )
        self.jordan_flows = bool(jordan_flows)
        if self.jordan_flows and (combine != "block_kernel" or jordan_chains_of(A_param) is None):
            raise ValueError(
                "jordan_flows needs combine='block_kernel' and a jordan_repeated A with uniform chains"
            )
        if leaf == "block_series" and self.block_sizes is None:
            raise ValueError("leaf='block_series' needs a block-diagonal A parametrisation")
        if leaf == "plane_closed" and not is_rotation_planes(A_param):
            raise ValueError(
                "leaf='plane_closed' needs a rotation-plane A parametrisation "
                "(bounded/damped/relu_damped/clamped/unbounded rotation)"
            )
        if leaf == "block_series" and N == 2 and series_q is None:
            raise ValueError("leaf='block_series' at depth 2 requires series_q")
        if leaf in JORDAN_LEAVES and jordan_chains_of(A_param) is None:
            raise ValueError(
                f"leaf={leaf!r} needs a jordan_repeated A with uniform chains (every block the same kind and size: "
                "real 2-4 or complex 4, 6, 8); use leaf='series' or 'block_series' otherwise"
            )

    def _single_path(
        self,
        path: jnp.ndarray,
        clock: jnp.ndarray,
        readout: str,
        output_format: str,
        a_scale=1.0,
    ) -> jnp.ndarray:
        path = path.astype(self.dtype)
        clock = clock.astype(self.dtype)
        if clock.ndim != 1 or clock.shape[0] != path.shape[0]:
            raise ValueError("clock must be 1D and aligned with path length")
        # a_scale rescales the drift generator; bounded_rotation supplies 2*pi/dtheta_max
        # from the enclosing block. Default 1.0 leaves all other parametrizations untouched.
        A = self.A_param().astype(self.dtype) * jnp.asarray(a_scale, dtype=self.dtype)
        interval_times = clock[1:] - clock[:-1]
        # Leaf stage. Routed through ews.core.leaf so the implementation (Van Loan + signax,
        # or the series) is selectable via the `leaf` field, default "van_loan".
        local_sigs = local_signatures(
            path,
            clock,
            A,
            self.N,
            kind=self.leaf,
            dtype=self.dtype,
            M=self.M,
            van_loan_expm=self.van_loan_expm,
            reweighting_mode=self.reweighting_mode,
            series_q=self.series_q,
            block_sizes=self.block_sizes,
            jordan_chain=jordan_chains_of(self.A_param) if self.leaf in JORDAN_LEAVES else None,
        )

        cached = self.reweighting_mode in ("displacement_uniform", "displacement_uniform_basepoint")
        basepoint = self.reweighting_mode == "displacement_uniform_basepoint"
        if readout == "terminal":
            if self.combine != "assoc":
                packed_final = combine_packed(
                    self.combine,
                    local_sigs,
                    A,
                    interval_times,
                    self.N,
                    self.chen_expm,
                    cached,
                    basepoint,
                    terminal=True,
                    block_sizes=self.block_sizes,
                    chunk_target=self.combine_chunk,
                    tile=self.combine_tile,
                    jordan_chain=jordan_chains_of(self.A_param) if self.jordan_flows else None,
                )
            elif self.terminal_reduce == "assoc":
                packed_final, _ = chen_associative_scan_final_packed_with_M(
                    local_sigs,
                    interval_times,
                    A,
                    path.shape[1],
                    self.N,
                    self.chen_expm,
                    cached,
                    basepoint,
                )
            else:
                packed_final, _ = chen_tree_reduce_final_packed_with_M(
                    local_sigs, interval_times, A, path.shape[1], self.N, self.chen_expm
                )
            sig_list = unpack_signature_vector(packed_final, path.shape[1], self.N)
            if output_format == "list":
                return sig_list
            return flatten_signature_list(sig_list, include_level0=True)

        if self.combine != "assoc":
            packed_prefix = combine_packed(
                self.combine,
                local_sigs,
                A,
                interval_times,
                self.N,
                self.chen_expm,
                cached,
                basepoint,
                terminal=False,
                block_sizes=self.block_sizes,
                chunk_target=self.combine_chunk,
                tile=self.combine_tile,
                jordan_chain=jordan_chains_of(self.A_param) if self.jordan_flows else None,
            )
        else:
            packed_prefix, _ = chen_associative_scan_prefix_packed_with_M(
                local_sigs,
                interval_times,
                A,
                path.shape[1],
                self.N,
                self.chen_expm,
                cached,
                basepoint,
            )
        prefix_list = unpack_signature_batch(packed_prefix, path.shape[1], self.N)
        if output_format == "list":
            if readout in ("mean", "max"):
                raise ValueError(f"readout='{readout}' requires output_format='flat'")
            return prefix_list
        flat_prefix = flatten_signature_prefix(
            prefix_list, path.shape[1], self.N, include_level0=True
        )
        if readout == "mean":
            return jnp.mean(flat_prefix, axis=0)
        if readout == "max":
            return jnp.max(flat_prefix, axis=0)
        return flat_prefix

    def __call__(
        self,
        path: jnp.ndarray,
        clock: jnp.ndarray,
        return_terminal: bool = True,
        output_format: str = "flat",
        readout: Optional[str] = None,
        a_scale=1.0,
    ):
        """
        Args:
            path: (n_steps, d) or (batch, n_steps, d)
            clock: (n_steps,) or (batch, n_steps)
            return_terminal: If True, return terminal EWS; else return prefixes.
                Used only when ``readout`` is None (back-compat).
            output_format: "flat" (default) or "list"
            readout: "terminal", "prefix", "mean", or "max". If None, derived from
                ``return_terminal`` ("terminal" if True else "prefix"). "mean"/"max"
                compute the prefix EWS at every knot and pool over time coordinate-wise
                (average / maximum), returning a single vector like "terminal"; both
                require output_format="flat". "max" is a translation-invariant pool that
                surfaces the strongest local signature anywhere along the path.
        """
        if readout is None:
            readout = "terminal" if return_terminal else "prefix"
        if readout not in {"terminal", "prefix", "mean", "max"}:
            raise ValueError("readout must be 'terminal', 'prefix', 'mean', or 'max'")
        if path.ndim == 2:
            return self._single_path(path, clock, readout, output_format, a_scale)
        if path.ndim == 3:
            if clock.ndim != 2:
                raise ValueError("clock must be 2D for batched paths")
            return jax.vmap(lambda p, c: self._single_path(p, c, readout, output_format, a_scale))(
                path, clock
            )
        raise ValueError(f"Path must be 2D or 3D, got shape {path.shape}")

    @property
    def signature_dim(self) -> int:
        d = self.d
        return sum([d**k for k in range(1, self.N + 1)])


class BToSignature(eqx.Module):
    """
    Optional B matrix + EWS layer.
    """

    b: Optional[B]
    ewms: EWS

    def __init__(self, b: Optional[B], ewms: EWS):
        self.b = b
        self.ewms = ewms

    def __call__(
        self,
        path: jnp.ndarray,
        clock: jnp.ndarray,
        return_terminal: bool = True,
        output_format: str = "flat",
        readout: Optional[str] = None,
    ):
        if self.b is not None:
            if path.ndim == 2:
                path = self.b(path)
            elif path.ndim == 3:
                path = jax.vmap(self.b)(path)
        return self.ewms(
            path,
            clock,
            return_terminal=return_terminal,
            output_format=output_format,
            readout=readout,
        )


def ews_forward_single(
    path: jnp.ndarray,
    clock: jnp.ndarray,
    A: jnp.ndarray,
    N: int,
    M: int = 20,
    dtype: jnp.dtype = jnp.float32,
    reweighting_mode: str = "displacement",
    terminal_reduce: str = "tree",
    van_loan_expm: str = "cayley",
    chen_expm: str = "exact",
    return_terminal: bool = True,
    output_format: str = "flat",
    readout: Optional[str] = None,
) -> jnp.ndarray:
    """Non-batched wrapper for EWS with selectable reweighting mode."""
    layer = EWS(
        A,
        N=N,
        M=M,
        dtype=dtype,
        reweighting_mode=reweighting_mode,
        terminal_reduce=terminal_reduce,
        van_loan_expm=van_loan_expm,
        chen_expm=chen_expm,
    )
    return layer(
        path,
        clock,
        return_terminal=return_terminal,
        output_format=output_format,
        readout=readout,
    )


def ews_forward_batched(
    paths: jnp.ndarray,
    clocks: jnp.ndarray,
    A: jnp.ndarray,
    N: int,
    M: int = 20,
    dtype: jnp.dtype = jnp.float32,
    reweighting_mode: str = "displacement",
    terminal_reduce: str = "tree",
    van_loan_expm: str = "cayley",
    chen_expm: str = "exact",
    return_terminal: bool = True,
    output_format: str = "flat",
    readout: Optional[str] = None,
) -> jnp.ndarray:
    """Batched wrapper for EWS with selectable reweighting mode."""
    layer = EWS(
        A,
        N=N,
        M=M,
        dtype=dtype,
        reweighting_mode=reweighting_mode,
        terminal_reduce=terminal_reduce,
        van_loan_expm=van_loan_expm,
        chen_expm=chen_expm,
    )
    return jax.vmap(
        lambda p, c: layer(
            p,
            c,
            return_terminal=return_terminal,
            output_format=output_format,
            readout=readout,
        )
    )(paths, clocks)


# -- log-signature variant --------------------------------------------------------------------
def log_level_sizes(d: int, N: int):
    """Per-level coordinate counts of the compressed log-signature: ``[W_d(1), .., W_d(N)]``
    (Witt numbers, = number of Lyndon words per length), straight from signax."""
    return [int(w.shape[0]) for w in signax.utils.lyndon_words(N, d)]


def log_signature_dim(d: int, N: int) -> int:
    """Compressed (Lyndon) log-signature length: ``sum_{k=1..N} W_d(k)`` (Witt numbers)."""
    return int(sum(log_level_sizes(d, N)))


def log_level_slices(d: int, N: int):
    """(start, end) slice of each level in the flat compressed log-signature feature.
    Witt-sized blocks, contiguous from index 0 (a log-signature has no level-0 constant).
    Feed to ``FeatureNorm(level_slices=...)`` to level-normalize a log-EWS feature."""
    slices = []
    start = 0
    for size in log_level_sizes(d, N):
        slices.append((start, start + size))
        start += size
    return tuple(slices)


class LogEWS(eqx.Module):
    """
    Log-EWS feature: ``log(EWS)`` in the compressed Lyndon basis.

    Wraps an inner ``EWS`` with identical config; the reweighting + Chen scan run unchanged.
    The learnable drift ``A`` lives inside ``self.ewms``.
    """

    ewms: EWS
    d: int = eqx.field(static=True)
    N: int = eqx.field(static=True)

    def __init__(
        self,
        A,
        N: int,
        M: int = 20,
        dtype: jnp.dtype = jnp.float32,
        reweighting_mode: str = "displacement",
        terminal_reduce: str = "tree",
        van_loan_expm: str = "cayley",
        chen_expm: str = "exact",
        leaf: str = "van_loan",
        series_q=None,
        combine: str = "assoc",
        combine_chunk: int = 1024,
        combine_tile: int = 4,
        jordan_flows: bool = False,
    ):
        self.ewms = EWS(
            A,
            N=N,
            M=M,
            dtype=dtype,
            reweighting_mode=reweighting_mode,
            terminal_reduce=terminal_reduce,
            van_loan_expm=van_loan_expm,
            chen_expm=chen_expm,
            leaf=leaf,
            series_q=series_q,
            combine=combine,
            combine_chunk=combine_chunk,
            combine_tile=combine_tile,
            jordan_flows=jordan_flows,
        )
        self.d = self.ewms.d
        self.N = self.ewms.N

    def _single_path(
        self, path: jnp.ndarray, clock: jnp.ndarray, readout: str, output_format: str
    ) -> jnp.ndarray:
        if readout == "terminal":
            # Terminal EWS as list-of-tensors -> compressed log (no level-0).
            sig_list = self.ewms(path, clock, readout="terminal", output_format="list")
            log_list = signax.signature_to_logsignature(sig_list)
            if output_format == "list":
                return log_list
            return flatten_signature_list(log_list, include_level0=False)

        # prefix / mean / max: log each prefix, then pool over time (log-then-pool).
        prefix_list = self.ewms(path, clock, readout="prefix", output_format="list")
        # prefix_list: [(L,d), (L,d,d), ...] -> vmap the log over the L (knot) axis.
        logsig_trace = jax.vmap(signax.signature_to_logsignature)(prefix_list)
        # logsig_trace: [(L,W1), (L,W2), ...]  (per-level compressed log, leading L axis)
        if output_format == "list":
            if readout in ("mean", "max"):
                raise ValueError(f"readout='{readout}' requires output_format='flat'")
            return logsig_trace
        # concat to (L, D_log). No level-0 in a log-signature.
        L = logsig_trace[0].shape[0]
        feat_trace = jnp.concatenate([lvl.reshape(L, -1) for lvl in logsig_trace], axis=1)
        if readout == "prefix":
            return feat_trace
        if readout == "mean":
            return jnp.mean(feat_trace, axis=0)
        if readout == "max":
            return jnp.max(feat_trace, axis=0)
        raise ValueError(f"readout must be 'terminal', 'prefix', 'mean', or 'max', got {readout}")

    def __call__(
        self,
        path: jnp.ndarray,
        clock: jnp.ndarray,
        return_terminal: bool = True,
        output_format: str = "flat",
        readout: Optional[str] = None,
    ):
        """
        Drop-in for ``EWS.__call__``: same signature, produces the compressed log-EWS.

        Args:
            path:  (n_steps, d) or (batch, n_steps, d)
            clock: (n_steps,)  or (batch, n_steps)
            return_terminal: used only when ``readout`` is None (terminal if True else prefix).
            output_format: "flat" (default) or "list" (per-level compressed log tensors).
            readout: "terminal" | "prefix" | "mean" | "max".

        Returns (flat): "terminal"/"mean"/"max" -> (D_log,); "prefix" -> (L, D_log).
        Batched paths add a leading batch axis.
        """
        if readout is None:
            readout = "terminal" if return_terminal else "prefix"
        if readout not in {"terminal", "prefix", "mean", "max"}:
            raise ValueError("readout must be 'terminal', 'prefix', 'mean', or 'max'")
        if path.ndim == 2:
            return self._single_path(path, clock, readout, output_format)
        if path.ndim == 3:
            if clock.ndim != 2:
                raise ValueError("clock must be 2D for batched paths")
            return jax.vmap(lambda p, c: self._single_path(p, c, readout, output_format))(
                path, clock
            )
        raise ValueError(f"Path must be 2D or 3D, got shape {path.shape}")

    @property
    def log_signature_dim(self) -> int:
        return log_signature_dim(self.d, self.N)
