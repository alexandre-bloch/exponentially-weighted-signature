"""The EWS block, the stack, and the token-cached variants."""

import dataclasses
from typing import Optional, Sequence, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from ews.clocks import Clock, ClockSpec, build_clock_jnp, make_clock  # noqa: F401
from ews.core.token_scan import build_token_cache, token_cached_prefix_scan
from ews.layer import EWS
from ews.norms import GradedRMSNorm, RMSNorm, graded_level_slices  # noqa: F401
from ews.params import (
    AParamBoundedRotation,
    AParamClampedRotation,
    AParamDampedRotation,
    AParamReluDampedRotation,
    AParamUnboundedRotation,
    B,
    build_ab_from_config,
)
from ews.paths import to_increment_path_jnp

_BOUNDED_PAD_ID = 0


def _forward_fill(arr: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Forward-fill rows beyond the true last real index (prefix-structured mask)."""
    n_idx = jnp.sum(mask.astype(jnp.int32)) - 1
    idx = jnp.minimum(jnp.arange(arr.shape[0]), n_idx)
    return jnp.take(arr, idx, axis=0)


class EWSBlock(eqx.Module):
    """A residual layer around an EWS mixer, on the pre-norm scaffold of the SLiCE block.

    With norm_placement="pre" (the default) the forward pass is

      theta = clock(t, driver)                    the clock, from the pre-norm driver, masked
      x_t   = RMSNorm1(x)
      u     = [appended clock | x_t], or x_t
      p     = B(u)                                the lift, forward-filled past the end of the mask
      S     = EWS prefix of (theta, p)            level 0 dropped, one row per position
      x     = x + Dropout(C(GradedRMSNorm(S)))    C optionally gated by a sigmoid of the branch input
      x     = x + Dropout(channel_stage(RMSNorm2(x)))

    norm_placement="slice_post" is the older [mix, res, ff, res, LayerNorm, drop] assembly, and "none" drops the stream
    norms but keeps the residuals and the branch dropout.

    Args:
        hidden_dim: the stream width H. m: the lift width (m + 1 with an appended, preserved clock channel). n: the depth.
        a_config, b_config: the "model.A" and "model.B" configs of the drift and of the lift.
        M, reweighting_mode, terminal_reduce, van_loan_expm, chen_expm, leaf, series_q, combine, combine_chunk,
            combine_tile, jordan_flows: passed to the EWS layer.
        basepoint: prepend a zero knot with a zero time increment.
        clock_*: the clock (see ews.clocks.build_clock_jnp); a "constant" normalisation needs clock_norm_constant.
        append_clock_*: an independent clock added as channel 0 of the lift input.
        norm_placement, second_norm, graded_norm, graded_eps, channel_stage, ff_mult, output_gate, dropout_rate,
            norm_eps: the scaffold (docs/block.md).
        dtype: the compute dtype. key: the PRNG key (keyword-only).

    Input (seq_len, hidden_dim) or (batch, seq_len, hidden_dim); the output has the same shape.
    """

    # --- submodules ---
    norm1: Optional[RMSNorm]
    b_layer: Optional[B]
    ewms: EWS
    graded: GradedRMSNorm
    c_down: eqx.nn.Linear
    gate: Optional[eqx.nn.Linear]
    norm2: Optional[Union[RMSNorm, eqx.nn.LayerNorm]]
    ff1: eqx.nn.Linear
    ff2: Optional[eqx.nn.Linear]
    drop: eqx.nn.Dropout
    # Clock modules (fixed or learnable). A token-gated clock has increments relu(dx.w - b)/c,
    # so the exact-zero increment is an open set (any token with dx.w <= b) instead of the
    # measure-zero point ||e|| = 0.
    clock: Clock
    append_clock: Optional[Clock]
    # --- static config ---
    hidden_dim: int = eqx.field(static=True)
    m: int = eqx.field(static=True)
    n: int = eqx.field(static=True)
    w_eff: int = eqx.field(static=True)
    D: int = eqx.field(static=True)
    basepoint: bool = eqx.field(static=True)
    norm_placement: str = eqx.field(static=True)
    graded_prescale: Optional[str] = eqx.field(static=True)
    path_mode: str = eqx.field(static=True)
    second_norm: bool = eqx.field(static=True)
    channel_stage: str = eqx.field(static=True)
    output_gate: bool = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        hidden_dim: int,
        m: int = 8,
        n: int = 1,
        *,
        a_config: Optional[dict] = None,
        b_config: Optional[dict] = None,
        M: int = 4,
        reweighting_mode: str = "displacement",
        terminal_reduce: str = "tree",
        van_loan_expm: str = "exact",
        chen_expm: str = "exact",
        leaf: str = "van_loan",
        series_q: Optional[int] = None,
        combine: str = "assoc",
        combine_chunk: int = 1024,
        combine_tile: int = 4,
        jordan_flows: bool = False,
        basepoint: bool = False,
        clock: Union[ClockSpec, dict, None] = None,
        append_clock: Union[ClockSpec, dict, None] = None,
        norm_placement: str = "pre",
        second_norm: bool = True,
        graded_norm: str = "live_rms",
        graded_eps: Union[float, Sequence[float]] = 1e-3,
        channel_stage: str = "mlp",
        graded_prescale: Optional[str] = None,
        path_mode: str = "value",
        ff_mult: int = 4,
        output_gate: bool = False,
        dropout_rate: float = 0.01,
        norm_eps: float = 1e-6,
        dtype: jnp.dtype = jnp.float32,
        key: jr.PRNGKey,
    ):
        if norm_placement not in {"pre", "slice_post", "none"}:
            raise ValueError("norm_placement must be one of {'pre', 'slice_post', 'none'}.")
        if channel_stage not in {"mlp", "tanh", "glu", "gelu_glu"}:
            raise ValueError("channel_stage must be one of {'mlp', 'tanh', 'glu', 'gelu_glu'}.")
        if graded_prescale not in {None, "binomial"}:
            raise ValueError("graded_prescale must be None or 'binomial'.")
        self.graded_prescale = graded_prescale
        if path_mode not in {"value", "increment"}:
            raise ValueError("path_mode must be 'value' or 'increment'.")
        self.path_mode = path_mode
        clock_spec = (
            clock if isinstance(clock, ClockSpec) else ClockSpec.from_config(clock)
        ).validate()
        append_spec = (
            None
            if append_clock is None
            else (
                append_clock
                if isinstance(append_clock, ClockSpec)
                else ClockSpec.from_config(append_clock)
            ).validate()
        )
        self.hidden_dim = int(hidden_dim)
        self.m = int(m)
        self.n = int(n)
        self.basepoint = bool(basepoint)
        self.norm_placement = norm_placement
        self.second_norm = bool(second_norm)
        self.channel_stage = channel_stage
        self.output_gate = bool(output_gate)
        self.dropout_rate = float(dropout_rate)
        self.dtype = dtype

        key_ab, key_c, key_gate, key_ff1, key_ff2 = jr.split(key, 5)

        if a_config is None:
            a_config = {
                "parametrization": "definite_plus_skew",
                "init": {"type": "definite_plus_skew", "mean": 0.0, "variance": 0.005},
                "learn": True,
            }
        append_on = append_spec is not None
        if b_config is None:
            b_config = {
                "enabled": True,
                "out_dim": int(m) + (1 if append_on else 0),
                "_preserve_first_channel": append_on,
                "parametrization": "general",
                "init": {"type": "general", "mean": 0.0, "variance": 1.0 / float(hidden_dim)},
                "learn": True,
            }
        elif "_preserve_first_channel" not in b_config and "_time_handling" not in b_config:
            # parse_b_config's own default is True; the block's default is
            # "preserve iff a clock channel is appended".
            b_config = dict(b_config)
            b_config["_preserve_first_channel"] = append_on

        d_in = self.hidden_dim + (1 if append_on else 0)
        a_param, b_layer, w, _, _ = build_ab_from_config(
            {"A": a_config, "B": b_config}, d_in=d_in, key=key_ab, dtype=dtype
        )
        self.b_layer = b_layer
        self.w_eff = int(w)
        self.D = sum(self.w_eff**k for k in range(1, self.n + 1))
        self.ewms = EWS(
            a_param,
            N=self.n,
            M=int(M),
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
        self.graded = GradedRMSNorm(
            self.w_eff, self.n, variant=graded_norm, eps=graded_eps, dtype=dtype
        )
        self.c_down = eqx.nn.Linear(self.D, self.hidden_dim, key=key_c)
        self.gate = (
            eqx.nn.Linear(self.hidden_dim, self.hidden_dim, key=key_gate) if output_gate else None
        )

        if norm_placement == "pre":
            self.norm1 = RMSNorm(self.hidden_dim, eps=norm_eps, dtype=dtype)
            self.norm2 = (
                RMSNorm(self.hidden_dim, eps=norm_eps, dtype=dtype) if second_norm else None
            )
        elif norm_placement == "slice_post":
            self.norm1 = None
            self.norm2 = eqx.nn.LayerNorm(self.hidden_dim, eps=norm_eps)
        else:
            self.norm1 = None
            self.norm2 = None

        H = self.hidden_dim
        if channel_stage == "mlp":
            if ff_mult < 1:
                raise ValueError("ff_mult must be at least 1.")
            self.ff1 = eqx.nn.Linear(H, ff_mult * H, key=key_ff1)
            self.ff2 = eqx.nn.Linear(ff_mult * H, H, key=key_ff2)
        elif channel_stage == "tanh":
            self.ff1 = eqx.nn.Linear(H, H, key=key_ff1)
            self.ff2 = None
        elif channel_stage == "glu":
            self.ff1 = eqx.nn.Linear(H, 2 * H, key=key_ff1)
            self.ff2 = None
        else:  # gelu_glu
            self.ff1 = eqx.nn.Linear(H, H, key=key_ff1)
            self.ff2 = eqx.nn.Linear(H, H, key=key_ff2)

        self.drop = eqx.nn.Dropout(p=self.dropout_rate)

        # Clock modules. The gated modes draw their parameters from fold_in(key, 7) (token_gated)
        # or fold_in(key, 11) (clamped_gate), independent of the 5-way split above.
        gate_key = jr.fold_in(key, 7 if clock_spec.mode == "token_gated" else 11)
        self.clock = make_clock(clock_spec, self.hidden_dim, key=gate_key, dtype=dtype)
        self.append_clock = (
            None
            if append_spec is None
            else make_clock(append_spec, self.hidden_dim, key=jr.fold_in(key, 13), dtype=dtype)
        )

    # -- clock ---------------------------------------------------------------

    def _to_increment_path(self, p: jnp.ndarray, mask: Optional[jnp.ndarray]) -> jnp.ndarray:
        """Increment representation: rows are per-step increments; see ews.paths.to_increment_path_jnp."""
        return to_increment_path_jnp(p, mask)

    def _resolve_driver(
        self, x: jnp.ndarray, clock_driver: Optional[jnp.ndarray]
    ) -> Optional[jnp.ndarray]:
        if self.clock.spec.source == "block_input":
            return x
        if clock_driver is None and self.clock.spec.mode != "time":
            raise ValueError(
                f"clock_driver is required for clock source '{self.clock.spec.source}'."
            )
        return clock_driver

    # -- mixer ---------------------------------------------------------------

    def _mixer_levels(
        self,
        branch_in: jnp.ndarray,
        theta: jnp.ndarray,
        theta_app: Optional[jnp.ndarray],
        mask: Optional[jnp.ndarray],
        a_scale=1.0,
    ) -> jnp.ndarray:
        u = branch_in
        if theta_app is not None:
            u = jnp.concatenate([theta_app[:, None].astype(u.dtype), u], axis=1)
        p = self.b_layer(u) if self.b_layer is not None else u
        if mask is not None:
            p = _forward_fill(p, mask)  # theta is already frozen by build_clock_jnp
        if self.basepoint:
            p_in = jnp.concatenate([jnp.zeros((1, p.shape[1]), dtype=p.dtype), p])
            theta_in = jnp.concatenate([theta[:1], theta])
        else:
            p_in = p
            theta_in = theta
        flat = self.ewms(p_in, theta_in, readout="prefix", output_format="flat", a_scale=a_scale)
        levels = flat[:, 1:]  # drop the level-0 ones column
        if not self.basepoint:
            levels = jnp.concatenate([jnp.zeros((1, levels.shape[1]), dtype=levels.dtype), levels])
        return levels

    def mixer_levels(
        self,
        x: jnp.ndarray,
        *,
        t: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
        clock_driver: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Pre-GradedRMSNorm mixer features of one sample -> (seq_len, D).

        Public hook for tests and for the harness's init calibrations
        (graded-eps floors and constant-mode clock denominators).
        """
        if x.ndim != 2:
            raise ValueError("mixer_levels expects a single sample (seq_len, hidden_dim).")
        L = x.shape[0]
        if t is None:
            t = jnp.arange(L, dtype=x.dtype)
        driver = self._resolve_driver(x, clock_driver)
        if self.path_mode == "increment" and driver is not None:
            driver = self._to_increment_path(driver, mask)
        theta = self.clock(t, driver, mask)
        branch_in = self.norm1(x) if self.norm_placement == "pre" else x
        if self.path_mode == "increment":
            branch_in = self._to_increment_path(branch_in, mask)
        theta_app = None if self.append_clock is None else self.append_clock(t, driver, mask)
        return self._prescale(self._mixer_levels(branch_in, theta, theta_app, mask))

    def _prescale(self, levels: jnp.ndarray) -> jnp.ndarray:
        """Optional fixed positionwise per-level normalizer (H2 test).

        "binomial": level-k features at row t are scaled by 1/C(t_len, k)
        where t_len = t+1 is the prefix length: converting the ~C(t,k)
        subsequence-count growth of one-hot level-k coefficients into
        densities, so a shared head sees a position-stationary code instead
        of asking the learned gains to absorb the scale."""
        if self.graded_prescale is None:
            return levels
        L = levels.shape[0]
        tlen = jnp.arange(1, L + 1, dtype=levels.dtype)
        cols = []
        for k, (start, end) in enumerate(self.graded.level_slices, start=1):
            tk = jnp.maximum(tlen, float(k))
            logc = (
                jax.scipy.special.gammaln(tk + 1.0)
                - jax.scipy.special.gammaln(tk - k + 1.0)
                - jax.scipy.special.gammaln(k + 1.0)
            )
            cols.append(jnp.tile(jnp.exp(-logc)[:, None], (1, end - start)))
        return levels * jnp.concatenate(cols, axis=1)

    # -- channel stage -------------------------------------------------------

    def _channel(self, xhat: jnp.ndarray) -> jnp.ndarray:
        if self.channel_stage == "mlp":
            h = jax.nn.gelu(jax.vmap(self.ff1)(xhat))
            return jax.vmap(self.ff2)(h)
        if self.channel_stage == "tanh":
            return jnp.tanh(jax.vmap(self.ff1)(xhat))
        if self.channel_stage == "glu":
            h = jax.vmap(self.ff1)(xhat)
            a, b = jnp.split(h, 2, axis=-1)
            return a * jax.nn.sigmoid(b)
        h = jax.nn.gelu(xhat)  # gelu_glu (S5/LinOSS stage)
        return jax.vmap(self.ff1)(h) * jax.nn.sigmoid(jax.vmap(self.ff2)(h))

    # -- forward -------------------------------------------------------------

    def _call_single(
        self,
        x: jnp.ndarray,
        t: Optional[jnp.ndarray],
        mask: Optional[jnp.ndarray],
        clock_driver: Optional[jnp.ndarray],
        key: Optional[jr.PRNGKey],
        inference: bool,
        a_scale=1.0,
    ) -> jnp.ndarray:
        L = x.shape[0]
        if t is None:
            t = jnp.arange(L, dtype=x.dtype)
        no_drop = inference or self.dropout_rate == 0.0
        if not no_drop and key is None:
            raise ValueError(
                "key is required when dropout is active (inference=False and dropout_rate > 0)."
            )
        k1, k2 = jr.split(key) if key is not None else (None, None)

        driver = self._resolve_driver(x, clock_driver)
        if self.path_mode == "increment" and driver is not None:
            driver = self._to_increment_path(driver, mask)
        theta = self.clock(t, driver, mask)
        theta_app = None if self.append_clock is None else self.append_clock(t, driver, mask)

        branch_in = self.norm1(x) if self.norm_placement == "pre" else x
        if self.path_mode == "increment":
            branch_in = self._to_increment_path(branch_in, mask)
        levels = self._prescale(self._mixer_levels(branch_in, theta, theta_app, mask, a_scale))
        r = jax.vmap(self.c_down)(self.graded(levels))
        if self.gate is not None:
            r = r * jax.nn.sigmoid(jax.vmap(self.gate)(branch_in))

        if self.norm_placement == "slice_post":
            y = x + r
            y = y + self._channel(y)
            y = jax.vmap(self.norm2)(y)
            return self.drop(y, key=k1, inference=no_drop)

        r = self.drop(r, key=k1, inference=no_drop)
        x = x + r
        if self.norm_placement == "pre" and self.norm2 is not None:
            xhat = self.norm2(x)
        else:
            xhat = x
        mch = self._channel(xhat)
        mch = self.drop(mch, key=k2, inference=no_drop)
        return x + mch

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        t: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
        clock_driver: Optional[jnp.ndarray] = None,
        key: Optional[jr.PRNGKey] = None,
        inference: bool = False,
        a_scale=1.0,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: (seq_len, hidden_dim) single sample or (batch, seq_len,
               hidden_dim) batch (vmapped internally, per-sample keys).
            t: Optional (seq_len,) time base shared across the batch;
               defaults to the sample grid arange(seq_len).
            mask: Optional boolean validity mask, (seq_len,) or (batch,
               seq_len), prefix-structured. Gates the clock and freezes the
               lifted path at the true end (frozen pads).
            clock_driver: Required when clock_source is 'encoder_output' or
               'raw_input' and the clock is blended: (seq_len, c) / batched.
            key: PRNG key for dropout; may be None iff inference or
               dropout_rate == 0.
            inference: Disables dropout.
        """
        if x.ndim == 2:
            return self._call_single(x, t, mask, clock_driver, key, inference, a_scale)
        if x.ndim == 3:
            n_batch = x.shape[0]
            keys = jr.split(key, n_batch) if key is not None else None
            in_axes = (
                0,
                None,
                None if mask is None else 0,
                None if clock_driver is None else 0,
                None if keys is None else 0,
                None,
                None,
            )
            return jax.vmap(self._call_single, in_axes=in_axes)(
                x, t, mask, clock_driver, keys, inference, a_scale
            )
        raise ValueError(f"x must be 2D or 3D, got shape {x.shape}")


class StackedEWS(eqx.Module):
    """An encoder, `num_blocks` EWSBlocks and a head.

    The encoder is an Embedding for tokens (`tokens=True`) or a Linear map for time series; the head acts per position
    (`head_mode="seq2seq"`) or on a masked mean over the positions (`"mean_pool"`), and `label_dim` may be a tuple for
    several heads. The encoder output is passed to every block whose clock source is "encoder_output"; "raw_input" blocks
    read the `raw_driver` argument and "block_input" blocks (the default) their own input.

    Args:
        num_blocks, data_dim (vocabulary size or input channels), hidden_dim, label_dim, tokens, head_mode,
        final_norm (an RMSNorm after the last block), block_kwargs (EWSBlock arguments for every block),
        per_block_kwargs (per-block overrides merged over block_kwargs), dtype, key (keyword-only).

    Input: tokens (seq_len,) or (batch, seq_len) as ints, or series (seq_len, data_dim) or (batch, seq_len, data_dim).
    Output: (..., seq_len, label_dim) for seq2seq and (..., label_dim) for mean_pool.
    """

    embedding: eqx.Module
    blocks: Tuple[EWSBlock, ...]
    head: Union[eqx.nn.Linear, Tuple[eqx.nn.Linear, ...]]
    final_norm: Optional[RMSNorm]
    tokens: bool = eqx.field(static=True)
    head_mode: str = eqx.field(static=True)
    # Cosine readout (anti-margin-inflation): when set, logits = (W_hat @ h_hat) / tau with
    # both the head rows and the hidden vector unit-normalised (bias dropped). Removes the
    # ||C||- and ||h||-growth routes to CE margin so the only way up is code precision.
    cosine_head_tau: Optional[float] = eqx.field(static=True)
    # Vocab ids that actually occur as inputs. Tasks like
    # cycle_nav share one id space between input tokens and label ids; label ids never occur
    # in sequences, but _bounded_a_scale's max-over-vocab would otherwise anchor the 2pi
    # normalisation on such a phantom row. None = all non-pad ids.
    input_token_ids: Optional[Tuple[int, ...]] = eqx.field(static=True)
    num_blocks: int = eqx.field(static=True)
    data_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    label_dim: Union[int, Tuple[int, ...]] = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        num_blocks: int,
        data_dim: int,
        hidden_dim: int,
        label_dim: Union[int, Tuple[int, ...]],
        *,
        tokens: bool = True,
        head_mode: str = "seq2seq",
        final_norm: bool = False,
        cosine_head_tau: Optional[float] = None,
        input_token_ids: Optional[Sequence[int]] = None,
        block_kwargs: Optional[dict] = None,
        per_block_kwargs: Optional[Sequence[dict]] = None,
        dtype: jnp.dtype = jnp.float32,
        key: jr.PRNGKey,
    ):
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1.")
        if head_mode not in {"seq2seq", "mean_pool"}:
            raise ValueError("head_mode must be one of {'seq2seq', 'mean_pool'}.")
        if per_block_kwargs is not None and len(per_block_kwargs) != num_blocks:
            raise ValueError("per_block_kwargs must have length num_blocks.")

        self.tokens = bool(tokens)
        self.head_mode = head_mode
        self.cosine_head_tau = None if cosine_head_tau is None else float(cosine_head_tau)
        self.input_token_ids = (
            None if input_token_ids is None else tuple(int(i) for i in input_token_ids)
        )
        self.num_blocks = int(num_blocks)
        self.data_dim = int(data_dim)
        self.hidden_dim = int(hidden_dim)
        self.dtype = dtype

        key_emb, key_blocks, key_head = jr.split(key, 3)

        if self.tokens:
            self.embedding = eqx.nn.Embedding(data_dim, hidden_dim, key=key_emb)
        else:
            self.embedding = eqx.nn.Linear(data_dim, hidden_dim, key=key_emb)

        base_kwargs = dict(block_kwargs or {})
        blocks = []
        for i in range(self.num_blocks):
            kw = dict(base_kwargs)
            if per_block_kwargs is not None:
                kw.update(per_block_kwargs[i])
            blocks.append(EWSBlock(hidden_dim, dtype=dtype, key=jr.fold_in(key_blocks, i), **kw))
        self.blocks = tuple(blocks)

        if isinstance(label_dim, int):
            self.label_dim = int(label_dim)
            self.head = eqx.nn.Linear(hidden_dim, label_dim, key=key_head)
        elif isinstance(label_dim, tuple):
            if not all(isinstance(d, int) for d in label_dim):
                raise TypeError("label_dim must be int or tuple of int")
            self.label_dim = tuple(int(d) for d in label_dim)
            head_keys = jr.split(key_head, len(label_dim))
            self.head = tuple(
                eqx.nn.Linear(hidden_dim, d, key=k) for d, k in zip(label_dim, head_keys)
            )
        else:
            raise TypeError("label_dim must be int or tuple of int")

        self.final_norm = RMSNorm(hidden_dim, dtype=dtype) if final_norm else None

    def _is_single(self, X: jnp.ndarray) -> bool:
        return X.ndim == (1 if self.tokens else 2)

    def hidden(
        self,
        X: jnp.ndarray,
        *,
        t: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
        raw_driver: Optional[jnp.ndarray] = None,
        key: Optional[jr.PRNGKey] = None,
        inference: bool = False,
    ) -> jnp.ndarray:
        """Encoder + block loop for one sample, no head -> (seq_len, hidden_dim).

        Args:
            X: (seq_len,) int tokens or (seq_len, data_dim) floats.
            raw_driver: (seq_len, c) driver for blocks with
                clock_source='raw_input' (e.g. one-hot floats for tokens).
        """
        if not self._is_single(X):
            raise ValueError("hidden expects a single sample; use __call__ for batches.")
        if self.tokens:
            h = jax.vmap(self.embedding)(X.astype(jnp.int32))
        else:
            h = jax.vmap(self.embedding)(X)
        enc = h  # encoder output, captured once
        for i, block in enumerate(self.blocks):
            driver = None
            if block.clock.spec.source == "encoder_output":
                driver = enc
            elif block.clock.spec.source == "raw_input":
                driver = raw_driver
            k_i = jr.fold_in(key, i) if key is not None else None
            a_scale = self._bounded_a_scale(block)
            h = block(
                h,
                t=t,
                mask=mask,
                clock_driver=driver,
                key=k_i,
                inference=inference,
                a_scale=a_scale,
            )
        if self.final_norm is not None:
            h = self.final_norm(h)
        return h

    def _bounded_a_scale(self, block):
        """a_scale for a bounded_rotation block: 2*pi / dtheta_max, where dtheta_max is the
        max over non-pad vocab tokens of ||embed(g)|| / clock_norm_constant: the exact
        per-token clock increment under path_mode='increment' (diff(cumsum(x)) = x, so the
        increment at token g is ||x_g|| = ||embed(g)||; V2). Read from the embedding table
        (deterministic across batches and at eval, identical when a batch is missing tokens),
        stop-gradient, pad token excluded. Returns 1.0 for any other A parametrization.
        """
        # Note: this gate keys on the class, so every bounded-rotation variant must be listed:
        # a missing entry silently returns a_scale = 1.0 instead of 2*pi/dtheta_max, which is not
        # an error, just a wrong model. a_scale itself is computed from the embedding table (and
        # the gate, for the block whose clock reads it); it does not depend on omega or on
        # omega_scale, so the normaliser cannot drift with the frequency range. dtheta_max is
        # stop-gradiented below.
        if not (
            self.tokens
            and isinstance(
                block.ewms.A_param,
                (
                    AParamBoundedRotation,
                    AParamClampedRotation,
                    AParamDampedRotation,
                    AParamReluDampedRotation,
                    AParamUnboundedRotation,
                ),
            )
        ):
            return 1.0
        c = block.clock.spec.norm_constant or 1.0
        # The gate branches evaluate the block's gate on the embedding table, which is only the
        # block's real clock input when the block's clock driver is the embedding stream: the
        # first block under clock_source='block_input', or any block reading 'encoder_output'.
        # A gated later block (e.g. a clamped gate on b2, whose clock reads b1's output) would
        # otherwise be normalised by a quantity its clock never sees: measured at init that
        # silently moves a_scale from 16.2 to 114, and if the gate closes on every input row the
        # 1e-6 guard fires and a_scale jumps to ~6.3e6 mid-training. No config in the repo gates
        # a non-first block, so this is a no-op for parity, even_pairs and cycle_nav.
        gate_reads_embeddings = block.clock.spec.source == "encoder_output" or (
            block.clock.spec.source == "block_input" and block is self.blocks[0]
        )
        if gate_reads_embeddings and block.clock.spec.mode == "token_gated":
            dth = jax.nn.relu(self.embedding.weight @ block.clock.w - block.clock.b)
        elif gate_reads_embeddings and block.clock.spec.mode == "clamped_gate":
            dth = jnp.clip(self.embedding.weight @ block.clock.w.T - block.clock.b, 0.0, 1.0).sum(
                axis=-1
            )
        else:
            dth = jnp.linalg.norm(self.embedding.weight, axis=1)  # ||embed(g)||
        dth = self._mask_phantom(dth)
        # guard: if every token is gated off, the clock is degenerate; avoid inf scale
        dtheta_max = jax.lax.stop_gradient(jnp.maximum(jnp.max(dth), 1e-6) / c)
        return 2.0 * jnp.pi / dtheta_max

    def _mask_phantom(self, vals: jnp.ndarray) -> jnp.ndarray:
        """-inf at pad and at any vocab id not in input_token_ids (label rows that never occur
        as inputs must not anchor the 2pi normalisation)."""
        vals = vals.at[_BOUNDED_PAD_ID].set(-jnp.inf)
        if self.input_token_ids is not None:
            keep = jnp.zeros((vals.shape[0],), dtype=bool)
            keep = keep.at[jnp.asarray(self.input_token_ids)].set(True)
            vals = jnp.where(keep, vals, -jnp.inf)
        return vals

    def _cosine_logits(self, head: eqx.nn.Linear, h: jnp.ndarray) -> jnp.ndarray:
        W = head.weight
        W_hat = W / (jnp.linalg.norm(W, axis=1, keepdims=True) + 1e-8)
        h_hat = h / (jnp.linalg.norm(h, axis=-1, keepdims=True) + 1e-8)
        return (h_hat @ W_hat.T) / self.cosine_head_tau

    def _apply_head(self, head: eqx.nn.Linear, h: jnp.ndarray, mask: Optional[jnp.ndarray]):
        if self.head_mode == "seq2seq":
            if self.cosine_head_tau is not None:
                return self._cosine_logits(head, h)
            return jax.vmap(head)(h)
        if mask is not None:
            w = mask.astype(h.dtype)[:, None]
            pooled = (h * w).sum(axis=0) / jnp.maximum(w.sum(), 1.0)
        else:
            pooled = h.mean(axis=0)
        if self.cosine_head_tau is not None:
            return self._cosine_logits(head, pooled)
        return head(pooled)

    def _call_single(
        self,
        X: jnp.ndarray,
        t: Optional[jnp.ndarray],
        mask: Optional[jnp.ndarray],
        raw_driver: Optional[jnp.ndarray],
        key: Optional[jr.PRNGKey],
        inference: bool,
    ):
        h = self.hidden(X, t=t, mask=mask, raw_driver=raw_driver, key=key, inference=inference)
        if isinstance(self.head, tuple):
            return tuple(self._apply_head(hd, h, mask) for hd in self.head)
        return self._apply_head(self.head, h, mask)

    def __call__(
        self,
        X: jnp.ndarray,
        *,
        t: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
        raw_driver: Optional[jnp.ndarray] = None,
        key: Optional[jr.PRNGKey] = None,
        inference: bool = False,
    ):
        """Forward pass; single samples and batches (vmapped internally).

        Args:
            X: tokens (seq_len,) / (batch, seq_len); series (seq_len,
               data_dim) / (batch, seq_len, data_dim).
            t: Optional (seq_len,) time base shared across the batch.
            mask: Optional (seq_len,) or (batch, seq_len) validity mask.
            raw_driver: Optional (seq_len, c) or (batch, seq_len, c).
            key: PRNG key for dropout (split per sample, folded per block).
            inference: Disables dropout.
        """
        if self._is_single(X):
            return self._call_single(X, t, mask, raw_driver, key, inference)
        n_batch = X.shape[0]
        keys = jr.split(key, n_batch) if key is not None else None
        in_axes = (
            0,
            None,
            None if mask is None else 0,
            None if raw_driver is None else 0,
            None if keys is None else 0,
            None,
        )
        return jax.vmap(self._call_single, in_axes=in_axes)(X, t, mask, raw_driver, keys, inference)


class TokenCachedEWSBlock(EWSBlock):
    """EWSBlock whose mixer levels come from the token cache. No new fields (retyped from an
    EWSBlock instance via `_as_token_cached`); only method overrides."""

    # -- the cached replacement for `self.ewms(p_in, theta_in, ...)` in _mixer_levels ----------
    def _token_cached_levels(self, token_ids, vocab_embeds, mask, a_scale):
        """Reproduce `_mixer_levels(branch_in, theta, None, mask, a_scale)` via the vocab cache.

        Per-token quantities (V2, exact for linear b_layer):
          Δθ_g = sqrt(||embed(g)||^2 + variation_eps) / clock_norm_constant   (raw embed; clock path)
          ΔX_g = b_layer(norm1(embed(g)))                                      (normed embed; b-lift)
        A_eff = A_param() * a_scale. Basepoint: interval 0 is (ΔX_{g0}, Δθ=0) i.e. M=I, a different
        segment than the cache's (ΔX_{g0}, Δθ_{g0}): handled by a Δθ=0 basepoint cache used only at
        position 0.
        """
        # per-vocab-token ΔX_g (normed -> b-lift) and Δθ_g (raw embed clock increment)
        branch_vocab = self.norm1(vocab_embeds) if self.norm_placement == "pre" else vocab_embeds
        dx = (
            jax.vmap(self.b_layer)(branch_vocab) if self.b_layer is not None else branch_vocab
        )  # (V, m)
        dtheta = self.clock.per_token_increment(vocab_embeds.astype(dx.dtype))  # (V,)
        A = self.ewms.A_param().astype(dx.dtype) * jnp.asarray(a_scale, dtype=dx.dtype)
        # M/expm live on the EWS layer; P_cache (van-Loan segments) uses van_loan_expm, D_cache
        # (Chen flow exp(-Δθ A)) uses chen_expm: they differ (cayley vs exact) and mixing them up
        # compounds the flow error through the scan (the equivalence bug this path was built around).
        M, N = self.ewms.M, self.n
        expm, chen_expm = self.ewms.van_loan_expm, self.ewms.chen_expm
        leaf, sq = self.ewms.leaf, self.ewms.series_q
        P_cache, D_cache = build_token_cache(
            dx, dtheta, A, M, N, expm, chen_expm, leaf, sq
        )  # Δθ_g segments
        P_bp, _ = build_token_cache(
            dx, jnp.zeros_like(dtheta), A, M, N, expm, chen_expm, leaf, sq
        )  # Δθ=0 (basepoint)
        dim = A.shape[0]

        if self.basepoint:
            # intervals i=0..L-1 use token g_i; interval 0 is the Δθ=0 basepoint knot (M=I, P_bp).
            levels = _token_scan_with_basepoint(token_ids, mask, P_cache, D_cache, P_bp, dim, N)
        else:
            # intervals i=0..L-2 use token g_{i+1}; a zero level-row is prepended afterwards.
            pref = token_cached_prefix_scan(token_ids[1:], mask[1:], P_cache, D_cache, dim, N)
            zero = jnp.zeros((1, pref.shape[1]), dtype=pref.dtype)
            levels = jnp.concatenate([zero, pref], axis=0)
        return levels

    # -- per-sample forward: copy of EWSBlock._call_single with the levels line replaced ---------
    def _call_single(
        self,
        x,
        t,
        mask,
        clock_driver,
        key,
        inference,
        a_scale=1.0,
        token_ids=None,
        vocab_embeds=None,
    ):
        L = x.shape[0]
        if t is None:
            t = jnp.arange(L, dtype=x.dtype)
        no_drop = inference or self.dropout_rate == 0.0
        if not no_drop and key is None:
            raise ValueError("key is required when dropout is active.")
        k1, k2 = jr.split(key) if key is not None else (None, None)

        # branch_in is still needed for the gate + graded/channel path (identical to default)
        branch_in = self.norm1(x) if self.norm_placement == "pre" else x
        if self.path_mode == "increment":
            branch_in = self._to_increment_path(branch_in, mask)

        # >>> the only change vs EWSBlock._call_single: mixer levels from the token cache <<<
        levels = self._prescale(self._token_cached_levels(token_ids, vocab_embeds, mask, a_scale))

        r = jax.vmap(self.c_down)(self.graded(levels))
        if self.gate is not None:
            r = r * jax.nn.sigmoid(jax.vmap(self.gate)(branch_in))

        if self.norm_placement == "slice_post":
            y = x + r
            y = y + self._channel(y)
            y = jax.vmap(self.norm2)(y)
            return self.drop(y, key=k1, inference=no_drop)

        r = self.drop(r, key=k1, inference=no_drop)
        x = x + r
        xhat = self.norm2(x) if (self.norm_placement == "pre" and self.norm2 is not None) else x
        mch = self._channel(xhat)
        mch = self.drop(mch, key=k2, inference=no_drop)
        return x + mch

    def __call__(
        self,
        x,
        *,
        t=None,
        mask=None,
        clock_driver=None,
        key=None,
        inference=False,
        a_scale=1.0,
        token_ids=None,
        vocab_embeds=None,
    ):
        if token_ids is None or vocab_embeds is None:
            raise ValueError("TokenCachedEWSBlock requires token_ids and vocab_embeds.")
        if x.ndim == 2:
            return self._call_single(
                x, t, mask, clock_driver, key, inference, a_scale, token_ids, vocab_embeds
            )
        if x.ndim == 3:
            n_batch = x.shape[0]
            keys = jr.split(key, n_batch) if key is not None else None
            in_axes = (
                0,
                None,
                None if mask is None else 0,
                None,
                None if keys is None else 0,
                None,
                None,
                0,
                None,
            )
            return jax.vmap(self._call_single, in_axes=in_axes)(
                x, t, mask, clock_driver, keys, inference, a_scale, token_ids, vocab_embeds
            )
        raise ValueError(f"x must be 2D or 3D, got shape {x.shape}")


def _token_scan_with_basepoint(token_ids, mask, P_cache, D_cache, P_bp, dim, trunc):
    """token_cached_prefix_scan variant: position 0 uses (P_bp[g0], M=I); positions t>=1 use
    (P_cache[g_t], D_cache[g_t]); masked (pad) positions use (unit, I)."""
    import signax

    from ews.core.chen_scan import (
        _apply_M_to_signature,
        pack_signature_vector,
        unpack_signature_vector,
    )
    from ews.core.token_scan import _REMAT

    L = token_ids.shape[0]
    Sdim = P_cache.shape[1]
    I_d = jnp.eye(dim, dtype=D_cache.dtype)
    zeros_P = jnp.zeros((Sdim,), dtype=P_cache.dtype)
    is_first = jnp.arange(L) == 0

    def step(S, inp):
        gid, m, first = inp
        D = jnp.where(first, I_d, D_cache[gid])
        P = jnp.where(first, P_bp[gid], P_cache[gid])
        D = jnp.where(m, D, I_d)
        P = jnp.where(m, P, zeros_P)
        S_flowed = _apply_M_to_signature(unpack_signature_vector(S, dim, trunc), D, dim, trunc)
        S_res = signax.tensor_ops.mult(S_flowed, unpack_signature_vector(P, dim, trunc))
        S_new = pack_signature_vector(S_res, dim, trunc).astype(
            S.dtype
        )  # x64 mult upcasts; keep carry stable
        return S_new, S_new

    body = jax.checkpoint(step) if _REMAT else step
    _, prefixes = jax.lax.scan(body, zeros_P, (token_ids, mask, is_first))
    return prefixes


def _as_token_cached(block: EWSBlock) -> TokenCachedEWSBlock:
    """Retype an EWSBlock instance to TokenCachedEWSBlock (identical fields, method overrides only)."""
    obj = object.__new__(TokenCachedEWSBlock)
    for f in dataclasses.fields(block):
        object.__setattr__(obj, f.name, getattr(block, f.name))
    return obj


class TokenCachedStackedEWS(StackedEWS):
    """StackedEWS whose first block is token-cached. Precondition (hard error) checked at init;
    UEA/PPG never construct this (they don't use StackedEWS)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.tokens:
            raise ValueError("token_cache requires tokens=True (discrete vocabulary).")
        b0 = self.blocks[0]
        if b0.path_mode != "increment":
            raise ValueError(
                "token_cache requires block[0].path_mode == 'increment' (Δθ=f(token))."
            )
        if b0.clock.spec.source != "block_input":
            raise ValueError("token_cache requires block[0] clock source 'block_input'.")
        if b0.append_clock is not None:
            raise ValueError("token_cache does not support append_clock on block[0].")
        object.__setattr__(self, "blocks", (_as_token_cached(b0),) + tuple(self.blocks[1:]))

    def hidden(self, X, *, t=None, mask=None, raw_driver=None, key=None, inference=False):
        if not self._is_single(X):
            raise ValueError("hidden expects a single sample; use __call__ for batches.")
        token_ids = X.astype(jnp.int32)
        h = jax.vmap(self.embedding)(token_ids)
        vocab_embeds = self.embedding.weight  # (|V|, hidden_dim): the vocab table
        enc = h
        for i, block in enumerate(self.blocks):
            driver = (
                enc
                if block.clock.spec.source == "encoder_output"
                else (raw_driver if block.clock.spec.source == "raw_input" else None)
            )
            k_i = jr.fold_in(key, i) if key is not None else None
            a_scale = self._bounded_a_scale(block)
            if isinstance(block, TokenCachedEWSBlock):
                h = block(
                    h,
                    t=t,
                    mask=mask,
                    clock_driver=driver,
                    key=k_i,
                    inference=inference,
                    a_scale=a_scale,
                    token_ids=token_ids,
                    vocab_embeds=vocab_embeds,
                )
            else:
                h = block(
                    h,
                    t=t,
                    mask=mask,
                    clock_driver=driver,
                    key=k_i,
                    inference=inference,
                    a_scale=a_scale,
                )
        if self.final_norm is not None:
            h = self.final_norm(h)
        return h
