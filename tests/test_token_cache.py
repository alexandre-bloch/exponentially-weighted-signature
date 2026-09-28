"""The token-cached scan (ews/core/token_scan.py, TokenCachedStackedEWS in ews/block.py) equals the
default scan: forward pass and gradients within 1e-10 in float64, at depths one and two, with and
without a basepoint, on a batch of mixed lengths (padded rows) and on a sequence made of one
token. Also the precondition the cache relies on (the clock increment depends on the token only)
and the construction-time errors. The config uses different matrix exponentials for the Van Loan
segments and the Chen flow on purpose, so a build that used one method for both would fail.
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

from ews.block import StackedEWS, TokenCachedStackedEWS
from tests.block_kwargs import specs

VOCAB = 4  # data_dim (token vocabulary size)
HIDDEN = 16
LABEL = 3
DTYPE = jnp.float64
KEY = jr.PRNGKey(0)


def _block_kwargs(n, basepoint):
    """Token-cache-eligible block config: increment path + per-token (block_input) variation clock
    so Δθ = ||embed(g)||/c is a pure function of the token. van_loan != chen expm on purpose."""
    return dict(
        m=6,
        n=n,
        M=4,
        reweighting_mode="displacement",
        van_loan_expm="cayley",  # != chen_expm below: the regression-guard condition
        chen_expm="exact",
        basepoint=basepoint,
        path_mode="increment",
        clock_mode="blended_variation",
        clock_alpha=0.0,
        clock_normalize="constant",
        clock_norm_constant=8.0,
        clock_source="block_input",
        norm_placement="pre",
        graded_norm="live_rms",
        channel_stage="gelu_glu",
        dropout_rate=0.0,
    )


def _build_pair(n, basepoint):
    """StackedEWS (default path) and TokenCachedStackedEWS built with the same key => identical
    weights. Two token-cache-eligible blocks; only the first is retyped to the cache."""
    kw = [specs(_block_kwargs(n, basepoint)), specs(_block_kwargs(n, basepoint))]
    common = dict(
        tokens=True, head_mode="seq2seq", final_norm=False, per_block_kwargs=kw, dtype=DTYPE
    )
    ref = StackedEWS(2, VOCAB, HIDDEN, LABEL, key=KEY, **common)
    cached = TokenCachedStackedEWS(2, VOCAB, HIDDEN, LABEL, key=KEY, **common)
    return ref, cached


def _loss(model, X, mask):
    return jnp.sum(model(X, mask=mask, inference=True) ** 2)


def _max_grad_diff(ga, gb):
    la = jax.tree_util.tree_leaves(eqx.filter(ga, eqx.is_inexact_array))
    lb = jax.tree_util.tree_leaves(eqx.filter(gb, eqx.is_inexact_array))
    assert len(la) == len(lb) and la, "grad pytrees misaligned"
    return max(float(np.max(np.abs(np.asarray(x) - np.asarray(y)))) for x, y in zip(la, lb))


# ---- test batches (all valid tokens are 1..VOCAB-1; token 0 is the pad symbol) --------------
def _mixed_lengths():
    X = jnp.array([[1, 2, 3, 1, 2, 3], [2, 1, 3, 2, 0, 0], [3, 1, 0, 0, 0, 0]], jnp.int32)
    m = jnp.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]], bool)
    return X, m


def _all_one_token():
    X = jnp.array([[1, 1, 1, 1, 1, 1], [3, 3, 3, 3, 3, 3]], jnp.int32)
    return X, jnp.ones((2, 6), bool)


CASES = {"mixed_lengths": _mixed_lengths, "all_one_token": _all_one_token}


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("basepoint", [True, False])
@pytest.mark.parametrize("case", list(CASES))
def test_forward_and_gradient_equivalence(n, basepoint, case):
    ref, cached = _build_pair(n, basepoint)
    X, mask = CASES[case]()

    o_ref = np.asarray(ref(X, mask=mask, inference=True))
    o_cac = np.asarray(cached(X, mask=mask, inference=True))
    fwd_err = float(np.max(np.abs(o_ref - o_cac)))
    assert fwd_err < 1e-10, f"forward err {fwd_err:.2e} (n={n}, basepoint={basepoint}, {case})"

    g_ref = eqx.filter_grad(_loss)(ref, X, mask)
    g_cac = eqx.filter_grad(_loss)(cached, X, mask)
    grad_err = _max_grad_diff(g_ref, g_cac)
    assert grad_err < 1e-10, f"grad err {grad_err:.2e} (n={n}, basepoint={basepoint}, {case})"


def test_numerical_precondition_dtheta_is_function_of_token():
    """The whole path rests on Δθ = f(token): identical tokens must yield identical clock
    increments regardless of position and neighbours. Assert numerically (not by config field),
    by reading the block's own clock on a real sequence with repeated tokens in varied contexts."""
    ref, _ = _build_pair(n=1, basepoint=True)
    b0 = ref.blocks[0]
    X = jnp.array([1, 2, 1, 3, 2, 1, 3, 2], jnp.int32)  # each token recurs in different contexts
    mask = jnp.ones(X.shape[0], bool)
    x = jax.vmap(ref.embedding)(X)
    t = jnp.arange(X.shape[0], dtype=x.dtype)
    driver = b0._to_increment_path(b0._resolve_driver(x, None), mask)
    theta = b0.clock(t, driver, mask)
    dtheta = jnp.diff(theta)  # per-interval increment; interval i uses token X[i+1]

    # cache's per-token Δθ (raw-embedding variation form)
    vocab = ref.embedding.weight
    sq = jnp.sum(vocab.astype(theta.dtype) ** 2, axis=1)
    dtheta_tok = jnp.sqrt(sq + b0.clock.spec.variation_eps) / b0.clock.spec.norm_constant

    for i in range(1, X.shape[0]):  # interval i-1 (>=0) carries token X[i]
        g = int(X[i])
        got = float(dtheta[i - 1])
        want = float(dtheta_tok[g])
        assert abs(got - want) < 1e-12, f"pos {i} tok {g}: clock incr {got} != token Δθ {want}"

    # and: any two occurrences of the same token share the increment exactly
    for g in range(1, VOCAB):
        occ = [i - 1 for i in range(1, X.shape[0]) if int(X[i]) == g]
        if len(occ) >= 2:
            vals = np.asarray([float(dtheta[j]) for j in occ])
            assert float(np.max(np.abs(vals - vals[0]))) < 1e-12, f"token {g} incr varies: {vals}"


def test_precondition_hard_errors():
    """TokenCachedStackedEWS must hard-error (never silently fall back) when its assumptions break."""
    common = dict(tokens=True, head_mode="seq2seq", final_norm=False, dtype=DTYPE)

    # value path_mode (Δθ not a pure function of the token)
    kw = _block_kwargs(1, True)
    kw["path_mode"] = "value"
    with pytest.raises(ValueError, match="increment"):
        TokenCachedStackedEWS(
            1, VOCAB, HIDDEN, LABEL, key=KEY, per_block_kwargs=[specs(kw)], **common
        )

    # non-token input
    kw = _block_kwargs(1, True)
    with pytest.raises(ValueError, match="tokens"):
        TokenCachedStackedEWS(
            1,
            VOCAB,
            HIDDEN,
            LABEL,
            key=KEY,
            per_block_kwargs=[specs(kw)],
            **{**common, "tokens": False},
        )

    # append_clock on block 0 (extra channel breaks the per-token ΔX identity)
    kw = _block_kwargs(1, True)
    kw["append_clock_mode"] = "blended_variation"
    kw["append_clock_norm_constant"] = 8.0
    with pytest.raises(ValueError, match="append_clock"):
        TokenCachedStackedEWS(
            1, VOCAB, HIDDEN, LABEL, key=KEY, per_block_kwargs=[specs(kw)], **common
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
