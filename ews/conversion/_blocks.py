"""SSM translation blocks: the vendored blocks with their SSM layer replaced by the EWSMixer.

LinOSSBlock, S5Block and LRUBlock share one forward up to the layer call (pre-norm BatchNorm, the layer, gelu, GLU,
dropout, residual), so one `EWSTranslationBlock` runs that forward with the mixer in the layer's slot. The constructors
convert the layer weights and reuse the vendored norm, GLU and dropout modules themselves, so any disagreement at the
block level comes from the mixer.
"""

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.random as jr

from ews.conversion._convert_core import linoss_im_to_ews, lru_to_ews, s5_to_ews
from ews.conversion._mixer import EWSMixer


class EWSTranslationBlock(eqx.Module):
    """Vendored SSM-block scaffold with the EWS mixer in the layer slot."""

    norm: eqx.nn.BatchNorm
    mixer: EWSMixer
    glu: eqx.Module
    drop: eqx.nn.Dropout

    def __call__(self, x, state, *, key):
        """Line-identical to the vendored blocks' forward, mixer swapped."""
        dropkey1, dropkey2 = jr.split(key, 2)
        skip = x
        x, state = self.norm(x.T, state)
        x = x.T
        x = self.mixer(x)
        x = self.drop(jax.nn.gelu(x), key=dropkey1)
        x = jax.vmap(self.glu)(x)
        x = self.drop(x, key=dropkey2)
        x = skip + x
        return x, state


def _from_block(block, layer_attr, convert_fn, N, mixer_kwargs):
    weights = convert_fn(getattr(block, layer_attr))
    return EWSTranslationBlock(
        norm=block.norm,
        mixer=EWSMixer(weights, N=N, **mixer_kwargs),
        glu=block.glu,
        drop=block.drop,
    )


def from_linoss_block(block, N=1, jordan=False, **mixer_kwargs):
    """LinOSSBlock -> EWSTranslationBlock (norm/glu/drop shared verbatim).

    jordan=True: rotate the converted weights into the canonical basis
    (similarity folded into B/C, exact) and make A a trainable JordanTiedA:
    32 generating parameters per block, tied re/im pairs, never the dense store.
    """
    if not jordan:
        return _from_block(block, "ssm", linoss_im_to_ews, N, mixer_kwargs)
    from ews.conversion._convert_core import jordanize
    from ews.conversion._mixer import EWSMixer, JordanTiedA

    w = jordanize(linoss_im_to_ews(block.ssm), copies=2)
    a_param = JordanTiedA(w["jordan_d"], w["jordan_w"], copies=2)
    return EWSTranslationBlock(
        norm=block.norm,
        mixer=EWSMixer(w, N=N, a_param=a_param, **mixer_kwargs),
        glu=block.glu,
        drop=block.drop,
    )


def _tied_planes_from_canonical(weights, tol=1e-12):
    """JordanTiedA (copies=1) from converted weights whose A is already the
    canonical plane stack [[d, w], [-w, d]] per 2x2 block (S5 and LRU convert
    to this form verbatim; LinOSS needs `jordanize` first). Asserted."""
    import numpy as np

    from ews.conversion._mixer import JordanTiedA

    A = np.asarray(weights["A"], dtype=np.float64)
    m = A.shape[0]
    d = A[np.arange(0, m, 2), np.arange(0, m, 2)]
    w = A[np.arange(0, m, 2), np.arange(1, m, 2)]
    blk = np.zeros_like(A)
    for j in range(m // 2):
        blk[2 * j : 2 * j + 2, 2 * j : 2 * j + 2] = [[d[j], w[j]], [-w[j], d[j]]]
    err = np.max(np.abs(A - blk)) / max(np.max(np.abs(A)), 1e-300)
    if err > tol:
        raise ValueError(f"converted A is not a canonical plane stack (rel dev {err:.2e})")
    return JordanTiedA(d, w, copies=1)


def _from_block_jordan(block, layer_attr, convert_fn, N, mixer_kwargs):
    w = convert_fn(getattr(block, layer_attr))
    return EWSTranslationBlock(
        norm=block.norm,
        mixer=EWSMixer(w, N=N, a_param=_tied_planes_from_canonical(w), **mixer_kwargs),
        glu=block.glu,
        drop=block.drop,
    )


def from_s5_block(block, N=1, jordan=False, **mixer_kwargs):
    """S5Block -> EWSTranslationBlock (norm/glu/drop shared verbatim).
    jordan=True: A as a trainable JordanTiedA (one plane per mode, copies=1):
    the converted A is canonical already, no similarity needed."""
    if jordan:
        return _from_block_jordan(block, "ssm", s5_to_ews, N, mixer_kwargs)
    return _from_block(block, "ssm", s5_to_ews, N, mixer_kwargs)


def from_lru_block(block, N=1, jordan=False, **mixer_kwargs):
    """LRUBlock -> EWSTranslationBlock (norm/glu/drop shared verbatim).
    jordan=True as for S5."""
    if jordan:
        return _from_block_jordan(block, "lru", lru_to_ews, N, mixer_kwargs)
    return _from_block(block, "lru", lru_to_ews, N, mixer_kwargs)


class RematBlock(eqx.Module):
    """Wraps a block so its forward is rematerialised in the backward pass
    (jax.checkpoint via eqx.filter_checkpoint): only one block's level-2 stream
    is live at a time. Field `block` keeps the wrapped module reachable."""

    block: eqx.Module

    def __call__(self, x, state, *, key):
        return eqx.filter_checkpoint(self.block)(x, state, key=key)


def remat_blocks(model):
    """model.blocks -> [RematBlock(b) ...] (tree surgery on the blocks list only)."""
    return eqx.tree_at(lambda m: m.blocks, model, [RematBlock(b) for b in model.blocks])


def swap_blocks(model, from_block_fn, N=1, **mixer_kwargs):
    """Vendored full model -> the same model with translation blocks.

    Pure tree surgery on `model.blocks`, nothing else: the encoder, the head
    (mean-pool/softmax classification or subsample/tanh regression), the
    classification/output_step flags, the per-block key splitting, and the
    vendored __call__ that runs them all are untouched: the translation
    model is their benchmark model, running their forward code; only the
    layer inside each block differs. This is the artifact their trained
    checkpoints load into.
    """
    new_blocks = [from_block_fn(b, N=N, **mixer_kwargs) for b in model.blocks]
    return eqx.tree_at(lambda m: m.blocks, model, new_blocks)
