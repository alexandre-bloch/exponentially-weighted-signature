"""The trained-checkpoint row: LinOSS-IM on Heartbeat, test metric before and after conversion.

A trained model, converted, should still score what it scored. The conversion is at depth 1, so the converted model
is mathematically the same function; any difference is arithmetic, not modelling.

`_blocks.swap_blocks(model, from_linoss_block, N=1)` replaces every block's LinOSS mixer with the converted EWS mixer,
leaving encoder, head, GLU and BatchNorm statistics alone.

Two baselines are reported: the stored f32-trained metric, and the model's own f64 evaluation. f32-vs-f64 arithmetic
can flip a borderline argmax by a sample or two, so comparing a f64 conversion against a f32 number would attribute
that to the conversion.

    python -m ews.conversion.trained_row --seeds 2345 3456 4567 5678 6789
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WARM = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import jax

jax.config.update("jax_enable_x64", True)
import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402

from ews.conversion._blocks import from_linoss_block, swap_blocks  # noqa: E402
from ews.conversion.reference.vendored.datasets import create_dataset  # noqa: E402
from ews.conversion.reference.vendored.linoss import LinOSS  # noqa: E402

# Trained LinOSS-IM Heartbeat checkpoints (about 6 MB of weights, not in the repository). Set LINOSS_HEARTBEAT_CKPT
# to their directory, and LNCDE_DATA_DIR to the processed data_dir/ of a github.com/Benjamin-Walker/log-neural-cdes
# clone. Either absent => skip.
_REPO = os.path.dirname(os.path.dirname(HERE))  # .../ews  (HERE is .../ews/ews/conversion)
CKPT = os.environ.get("LINOSS_HEARTBEAT_CKPT", "")
DATA_DIR = os.environ.get("LNCDE_DATA_DIR", "")
SKIP_REASON = (
    None
    if os.path.isdir(CKPT) and os.path.isdir(DATA_DIR)
    else "set LINOSS_HEARTBEAT_CKPT (trained LinOSS-IM Heartbeat checkpoints) and LNCDE_DATA_DIR"
)
CFG = dict(num_blocks=6, ssm_dim=16, hidden_dim=16, output_step=1, discretization="IM")


def _f64(m):
    return jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, m
    )


def load(seed):
    d = os.path.join(CKPT, f"seed_{seed}")
    best = json.load(open(os.path.join(d, "best.json")))
    dd, ld = best["config"]["data_dim"], best["config"]["label_dim"]
    skel = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float32) if eqx.is_inexact_array(x) else x,
        LinOSS(
            CFG["num_blocks"],
            dd,
            CFG["ssm_dim"],
            CFG["hidden_dim"],
            ld,
            True,
            CFG["output_step"],
            CFG["discretization"],
            key=jr.PRNGKey(0),
        ),
    )
    st_skel = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float32) if eqx.is_inexact_array(x) else x, eqx.nn.State(skel)
    )
    model = eqx.tree_deserialise_leaves(os.path.join(d, "model.eqx"), skel)
    st = eqx.tree_deserialise_leaves(os.path.join(d, "state.eqx"), st_skel)
    return _f64(model), _f64(st), best


def logits(model, st, loader, batch=32):
    """Raw logits over the split, BatchNorm frozen: the quantity behind the accuracy."""
    model = eqx.tree_at(
        lambda m: tuple(b.norm.inference for b in m.blocks), model, (True,) * CFG["num_blocks"]
    )
    out = []
    for X, _y in loader.loop_epoch(batch):
        o, _ = jax.vmap(model, axis_name="batch", in_axes=(0, None, None), out_axes=(0, None))(
            jnp.asarray(X, jnp.float64), st, jr.PRNGKey(0)
        )
        out.append(np.asarray(o, np.float64))
    return np.concatenate(out, axis=0)


def accuracy(model, st, loader, batch=32):
    """Test accuracy, BatchNorm frozen on its trained statistics."""
    model = eqx.tree_at(
        lambda m: tuple(b.norm.inference for b in m.blocks), model, (True,) * CFG["num_blocks"]
    )
    n = ok = 0
    for X, y in loader.loop_epoch(batch):
        out, _ = jax.vmap(model, axis_name="batch", in_axes=(0, None, None), out_axes=(0, None))(
            jnp.asarray(X, jnp.float64), st, jr.PRNGKey(0)
        )
        ok += int(jnp.sum(jnp.argmax(out, axis=1) == jnp.argmax(jnp.asarray(y), axis=1)))
        n += out.shape[0]
    return 100.0 * ok / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2345])
    a = ap.parse_args()
    rows = []
    for s in a.seeds:
        # the split is seed-dependent. create_dataset draws train/val/test from `key`, and the training run used
        # jr.split(jr.PRNGKey(seed), 4)[0]. Loading a checkpoint against a split built from a different key puts
        # samples it trained on into "test": with PRNGKey(0) this model scored 90.32 against its own recorded
        # 67.74, i.e. 56 of 62 correct instead of 42. Per-seed key, matching the training run's key split.
        datasetkey, _, _, _ = jr.split(jr.PRNGKey(s), 4)
        ds = create_dataset(
            DATA_DIR,
            "Heartbeat",
            stepsize=1,
            depth=1,
            include_time=True,
            T=1,
            use_idxs=False,
            use_presplit=False,
            key=datasetkey,
        )
        test = ds.raw_dataloaders["test"]
        model, st, best = load(s)
        before_f32 = 100.0 * float(best["test"])
        before_f64 = accuracy(model, st, test)
        conv = swap_blocks(model, from_linoss_block, N=1)
        after = accuracy(conv, st, test)
        # accuracy is a coarse instrument: it can hide a real drift until it flips an argmax. Report the max
        # absolute logit residual alongside it, which is the quantity the conversion actually controls.
        lg_before, lg_after = logits(model, st, test), logits(conv, st, test)
        lres = float(np.max(np.abs(lg_after - lg_before)))
        lrel = float(np.max(np.abs(lg_after - lg_before) / (np.abs(lg_before) + 1e-300)))
        rows.append(
            dict(
                seed=s,
                f32_stored=before_f32,
                f64_before=before_f64,
                f64_after=after,
                delta=after - before_f64,
                max_abs_logit_residual=lres,
                max_rel_logit_residual=lrel,
            )
        )
        print(
            f"  seed {s}: f64 before {before_f64:6.2f} | after {after:6.2f} | delta {after - before_f64:+.2f} "
            f"| max abs logit residual {lres:.3e}",
            flush=True,
        )
    json.dump(rows, open(os.path.join(HERE, "trained_row.json"), "w"), indent=1)
    print(
        f"\n  mean delta (converted - original, both f64): "
        f"{np.mean([r['delta'] for r in rows]):+.3f} accuracy points over {len(rows)} seeds"
    )


if __name__ == "__main__":
    main()
