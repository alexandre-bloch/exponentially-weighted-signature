"""Token-cached graded-EWS scan.

With the increment path and a finite vocabulary, a segment's displacement dX_g and clock increment dtheta_g depend
only on the current token g. The L - 1 per-position segments then reduce to |V| cached ones, a gather and a sequential
scan whose carry is one packed signature, in place of the (L - 1, m, m) carry of the associative scan. For the same
per-position (D_t, P_t) it reproduces `chen_associative_scan_prefix_packed_with_M` exactly; the precondition
dtheta = f(token) is checked by tests/test_token_cache.py.
"""

import os

import jax

jax.config.update("jax_enable_x64", True)  # match chen_scan.py's module-level setting

# Rematerialize the scan body in reverse-mode: recompute the |V|-entry gather (D_cache[gid],
# P_cache[gid]) in backward instead of storing the (m,m) flow matrix D at every step. This drops
# the reverse-mode tape from B*L*m^2 to B*L*m: the m^2 wall reappears in the autodiff tape of the
# sequential scan otherwise (fwd-only 213 MiB but fwd+bwd 18.9 GB at m=512/batch32). Measured to be
# a strict win: 18862->515 MiB at m=512/batch32, and faster at large m (95.6->76.3 ms) because
# storing/reloading the tape is bandwidth-bound while recomputing a 2-entry gather is not; neutral
# at small m. Equivalence verified with it on (fwd 8.9e-16, grad 2.1e-14). Default on; set
# TOKENSCAN_REMAT=0 to force the store-all path (e.g. to run Gate 2 with remat off).
_REMAT = os.environ.get("TOKENSCAN_REMAT", "1") == "1"

from functools import partial

import jax.numpy as jnp
import signax

from ews.core.chen_scan import (
    _apply_M_to_signature,
    pack_signature_vector,
    pack_signatures,
    unpack_signature_vector,
)
from ews.core.expm_approx import expm_matrix
from ews.core.series_leaf import assemble_from_letters, series_letters
from ews.core.signature import compute_reweighted_signatures
from ews.core.van_loan import reweighted_segment_van_loan


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8))
def build_token_cache(
    vocab_dx,
    vocab_dtheta,
    A,
    M,
    N,
    expm_method="exact",
    chen_expm_method=None,
    leaf="van_loan",
    series_q=None,
):
    """Per-token cache, differentiable in vocab_dx, vocab_dtheta and A (only the integer gather in the scan is not).

    P_cache uses `van_loan_expm` and D_cache `chen_expm`, as the default path does; using one method for both would break
    the equivalence whenever the two differ.

    Args:
        vocab_dx: (|V|, d) displacement of each token, already lifted. vocab_dtheta: (|V|,) clock increments.
        A: (d, d) effective drift, already scaled. M, N: Van Loan sub-steps and depth (static).
        expm_method, chen_expm_method: the layer's van_loan_expm and chen_expm (the second defaults to the first).

    Returns P_cache (|V|, sum_k d^k), the packed local signatures, and D_cache (|V|, d, d) = exp(-dtheta_g A).
    """
    d = vocab_dx.shape[1]
    dtype = A.dtype
    chen_method = expm_method if chen_expm_method is None else chen_expm_method
    # The leaf is computed once per vocab token here (|V| of them, not L), so this is cheap
    # either way, but it must follow the layer's  choice, or a token-cached run would
    # silently keep Van Loan while the rest of the model used the series.
    if leaf == "series":
        if series_q is None:
            raise ValueError("leaf='series' requires series_q")
        b = series_letters(vocab_dx, vocab_dtheta, A, int(series_q))  # (|V|, q+1, d)
        local = assemble_from_letters(b, N, int(series_q), dtype)  # level_k = (|V|,)+(d,)*k
    elif leaf == "van_loan":
        segs = jax.vmap(
            lambda dx, dt: reweighted_segment_van_loan(dx, dt, A, M, d, dtype, expm_method)
        )(vocab_dx, vocab_dtheta)  # (|V|, M+1, d)
        local = compute_reweighted_signatures(segs, N, dtype)  # tuple; level_k = (|V|, d^k)
    else:
        raise ValueError(f"leaf must be 'van_loan' or 'series', got {leaf!r}")
    P_cache = pack_signatures(local, d, N)  # (|V|, Σ d^k)
    D_cache = jax.vmap(lambda dt: expm_matrix(-dt * A, chen_method))(vocab_dtheta)  # (|V|, d, d)
    return P_cache, D_cache


@partial(jax.jit, static_argnums=(4, 5))
def token_cached_prefix_scan(token_ids, mask, P_cache, D_cache, dim, trunc):
    """Sequential prefix Chen scan over a token sequence, gathering the cache by token.

    S_t = mult(apply_M(S_{t-1}, D_t), P_t) with S_0 the unit, which is what the associative scan computes for the same
    (D, P). Padded steps apply D = I and the unit signature, matching the default path's masked clock.

    Args: token_ids (L,) ints; mask (L,) bool; P_cache and D_cache from build_token_cache; dim and trunc (static).
    Returns the (L, sum_k d^k) prefix signatures, in the layout of `pack_signatures`.
    """
    d = dim
    Sdim = P_cache.shape[1]
    S0 = jnp.zeros((Sdim,), dtype=P_cache.dtype)
    I_d = jnp.eye(d, dtype=D_cache.dtype)

    def step(S, inp):
        gid, m = inp
        D = jnp.where(m, D_cache[gid], I_d)  # pad -> identity flow
        P = jnp.where(m, P_cache[gid], jnp.zeros((Sdim,), dtype=P_cache.dtype))  # pad -> unit sig
        S_flowed = _apply_M_to_signature(unpack_signature_vector(S, d, trunc), D, d, trunc)
        S_res = signax.tensor_ops.mult(S_flowed, unpack_signature_vector(P, d, trunc))
        S_new = pack_signature_vector(S_res, d, trunc).astype(
            S.dtype
        )  # x64 mult upcasts; keep carry stable
        return S_new, S_new

    body = jax.checkpoint(step) if _REMAT else step
    _, prefixes = jax.lax.scan(body, S0, (token_ids, mask))
    return prefixes
