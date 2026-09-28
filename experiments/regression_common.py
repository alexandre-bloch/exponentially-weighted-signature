"""Shared pieces of the SDE regression and SKF prediction trainers.

Both experiments train one EWS layer with a readout applied to every prefix of the path: the
signature parameters (the drift ``A`` and the optional lift ``B``) and the head get separate AdamW
groups, each with a linear warmup followed by a cosine decay, gradient clipping and weight decay;
the loss is the mean per-sequence RMSE plus an L1/L2 penalty on the head; the reported number is
the test RMSE at the best validation checkpoint. This module holds everything the two trainers
have in common; each trainer keeps only the alignment of predictions with its targets.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from ews.config import upgrade, upgrade_layer_model, validate
from ews.layer import EWS, B
from ews.params import build_ab_from_config, parse_a_config, parse_b_config

# ── Config ─────────────────────────────────────────────────────────────────────


@dataclass
class Config:
    data: Dict[str, Any]
    preprocessing: Dict[str, Any]
    model: Dict[str, Any]
    training: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.data,
            "preprocessing": self.preprocessing,
            "model": self.model,
            "training": self.training,
        }


def load_config(path: str) -> Config:
    """Read a config file, upgrade it to the sectioned schema and validate it."""
    with open(path) as f:
        d = json.load(f)
    d = upgrade(d)
    validate(d)
    return Config(
        data=d.get("dataset", {}),
        preprocessing=d.get("preprocessing", {}),
        model=d.get("model", {}),
        training=d.get("training", {}),
    )


def resolve_dtype(dtype_str: str) -> jnp.dtype:
    return jnp.float64 if dtype_str == "float64" else jnp.float32


# ── Model ──────────────────────────────────────────────────────────────────────


class HeadWrapper(eqx.Module):
    head: eqx.Module

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.head(x).squeeze(-1)


class PathModel(eqx.Module):
    """Optional lift, one EWS layer over every prefix, a readout per step."""

    b_layer: Optional[B]
    ews: EWS
    head: HeadWrapper
    dtype: jnp.dtype = eqx.field(static=True)

    def __call__(self, paths: jnp.ndarray, clocks: jnp.ndarray) -> jnp.ndarray:
        if self.b_layer is not None:
            if paths.ndim == 2:
                paths = self.b_layer(paths)
            else:
                paths = jax.vmap(self.b_layer)(paths)
        if paths.dtype != self.dtype:
            paths = paths.astype(self.dtype)
        if clocks.dtype != self.dtype:
            clocks = clocks.astype(self.dtype)
        sig = self.ews(paths, clocks, return_terminal=False, output_format="flat")
        if sig.ndim == 2:
            return jax.vmap(self.head)(sig)
        return jax.vmap(jax.vmap(self.head))(sig)


def build_model(cfg: Config, d_in: int, key: jax.random.PRNGKey) -> PathModel:
    model_cfg = upgrade_layer_model(cfg.model)
    comp = model_cfg.get("compute", {})
    dtype = resolve_dtype(cfg.preprocessing.get("dtype", "float32"))

    key, subkey = jr.split(key)
    a_param, b_layer, _w, _a_cfg, _b_cfg = build_ab_from_config(model_cfg, d_in, subkey, dtype)
    ews = EWS(
        A=a_param,
        N=int(model_cfg.get("depth", 2)),
        M=int(comp.get("M", 20)),
        dtype=dtype,
        reweighting_mode=comp.get("reweighting_mode", "displacement"),
        terminal_reduce=comp.get("terminal_reduce", "tree"),
        van_loan_expm=comp.get("van_loan_expm", "cayley"),
        chen_expm=comp.get("chen_expm", "exact"),
        leaf=comp.get("leaf", "van_loan"),
        series_q=comp.get("series_q"),
    )

    key, subkey = jr.split(key)
    head_type = model_cfg.get("head_type", "linear")
    in_size = ews.signature_dim + 1  # +1 for the level-0 constant of the flat signature
    if head_type == "mlp":
        hidden_dims = model_cfg.get("head_hidden_dims", [64, 32])
        head_core = eqx.nn.MLP(
            in_size=in_size,
            out_size=1,
            width_size=int(hidden_dims[0]) if hidden_dims else 64,
            depth=len(hidden_dims) if hidden_dims else 2,
            key=subkey,
        )
    else:
        head_core = eqx.nn.Linear(in_size, 1, key=subkey)

    head_init_scale = model_cfg.get("head_init_scale", None)
    if head_init_scale is not None and float(head_init_scale) != 1.0:
        scale = float(head_init_scale)
        head_core = jax.tree_util.tree_map(lambda x: x * scale if eqx.is_array(x) else x, head_core)

    return PathModel(b_layer=b_layer, ews=ews, head=HeadWrapper(head=head_core), dtype=dtype)


# ── Optimiser ──────────────────────────────────────────────────────────────────


def make_schedule(base_lr, min_lr, warmup_steps, total_steps):
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=base_lr,
        warmup_steps=max(int(warmup_steps), 0),
        decay_steps=max(int(total_steps) - max(int(warmup_steps), 0), 1),
        end_value=min_lr,
    )


def make_optimizer(model: PathModel, cfg: Config, total_steps: int):
    """Two AdamW groups (signature parameters, head) with clipping; frozen parameters get no update."""
    tc = cfg.training
    lr_f = float(tc.get("lr_features", 1e-4))
    lr_h = float(tc.get("lr_head", 1e-3))
    mlr_f = float(tc.get("min_lr_features", 0.0))
    mlr_h = float(tc.get("min_lr_head", 0.0))
    ws = int(tc.get("warmup_steps", 0))
    wd_f = float(tc.get("weight_decay_features", 0.0))
    wd_h = float(tc.get("weight_decay_head", 0.0))
    gc_f = float(tc.get("grad_clip_features", tc.get("grad_clip", 1.0)))
    gc_h = float(tc.get("grad_clip_head", tc.get("grad_clip", 1.0)))

    a_cfg = parse_a_config(cfg.model)
    b_cfg = parse_b_config(cfg.model)
    learn_A = bool(a_cfg.learn)
    learn_B = bool(b_cfg.learn and b_cfg.enabled)

    params = eqx.filter(model, eqx.is_array)
    label_tree = jax.tree_util.tree_map(lambda _: "frozen", params)
    if learn_A:
        label_tree = eqx.tree_at(
            lambda m: m.ews,
            label_tree,
            replace=jax.tree_util.tree_map(
                lambda _: "features", eqx.filter(model.ews, eqx.is_array)
            ),
        )
    if learn_B and model.b_layer is not None:
        label_tree = eqx.tree_at(
            lambda m: m.b_layer,
            label_tree,
            replace=jax.tree_util.tree_map(
                lambda _: "features", eqx.filter(model.b_layer, eqx.is_array)
            ),
        )
    label_tree = eqx.tree_at(
        lambda m: m.head,
        label_tree,
        replace=jax.tree_util.tree_map(lambda _: "head", eqx.filter(model.head, eqx.is_array)),
    )

    sched_f = make_schedule(lr_f, mlr_f, ws, total_steps)
    sched_h = make_schedule(lr_h, mlr_h, ws, total_steps)
    opt = optax.multi_transform(
        {
            "features": optax.chain(
                optax.clip_by_global_norm(gc_f), optax.adamw(sched_f, weight_decay=wd_f)
            ),
            "head": optax.chain(
                optax.clip_by_global_norm(gc_h), optax.adamw(sched_h, weight_decay=wd_h)
            ),
            "frozen": optax.set_to_zero(),
        },
        lambda _: label_tree,
    )
    return opt, opt.init(params)


# ── Loss, metrics, batches ─────────────────────────────────────────────────────


def hierarchical_rmse(preds: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Mean over sequences of the per-sequence RMSE."""
    return jnp.mean(jnp.sqrt(jnp.mean((targets - preds) ** 2, axis=1)))


def flatten_head_params(head: eqx.Module) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(eqx.filter(head, eqx.is_array))
    if not leaves:
        return jnp.array([0.0])
    return jnp.concatenate([leaf.reshape(-1) for leaf in leaves])


def group_by_length(paths, clocks, targets) -> List[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Stack the sequences of a split into one batch per distinct length (ascending)."""
    lengths = np.array([len(p) for p in paths])
    groups = []
    for length in np.unique(lengths):
        idx = np.where(lengths == length)[0]
        groups.append(
            (
                jnp.stack([paths[int(i)] for i in idx]),
                jnp.stack([clocks[int(i)] for i in idx]),
                jnp.stack([targets[int(i)] for i in idx]),
            )
        )
    return groups


def split_rmse(model: PathModel, groups, align: Callable[[jnp.ndarray], jnp.ndarray]) -> float:
    """Mean per-sequence RMSE over a split; `align` cuts the predictions to the target positions."""
    per_seq = []
    for pb, cb, tb in groups:
        pr = align(model(pb, cb))
        per_seq.append(jnp.sqrt(jnp.mean((tb - pr) ** 2, axis=1)))
    return float(jnp.mean(jnp.concatenate(per_seq)))


def create_batch_iterator(paths, clocks, targets, batch_size: int, key):
    """Shuffled mini-batches, one permutation per epoch drawn from `key`.

    Sequences of one common length are stacked once and gathered per batch; mixed lengths fall
    back to stacking each batch (which requires every batch to be single-length).
    """
    n = len(paths)
    uniform = len({len(p) for p in paths}) == 1

    def rows(arrays, batch):
        return jnp.stack([arrays[int(j)] for j in batch])

    if uniform:
        stacked = (jnp.stack(paths), jnp.stack(clocks), jnp.stack(targets))
    while True:
        key, subkey = jr.split(key)
        idx = jr.permutation(subkey, jnp.arange(n))
        idx_np = np.asarray(idx)
        for i in range(0, n, batch_size):
            batch = idx_np[i : i + batch_size]
            if uniform:
                yield tuple(arr[batch] for arr in stacked)
            else:
                yield rows(paths, batch), rows(clocks, batch), rows(targets, batch)


# ── Inspection and results ────────────────────────────────────────────────────


def extract_A(model: PathModel) -> Optional[jnp.ndarray]:
    ews = getattr(model, "ews", None)
    if ews is None or not hasattr(ews, "A_param"):
        return None
    return ews.A_param()


def extract_B(model: PathModel) -> Optional[jnp.ndarray]:
    b_layer = getattr(model, "b_layer", None)
    if b_layer is None:
        return None
    matrix = b_layer.matrix
    if not b_layer.preserve_first_channel:
        return matrix
    rows, cols = matrix.shape
    full = jnp.zeros((rows + 1, cols + 1), dtype=matrix.dtype)
    full = full.at[0, 0].set(1.0)
    full = full.at[1:, 1:].set(matrix)
    return full


def eigenvalues(A) -> List[List[float]]:
    """Eigenvalues of `A` as [real, imaginary] pairs, sorted by real part."""
    vals = np.linalg.eigvals(np.asarray(A, dtype=np.float64))
    vals = vals[np.argsort(vals.real)]
    return [[float(v.real), float(v.imag)] for v in vals]


def _tolist(x):
    return None if x is None else np.asarray(x, dtype=np.float64).tolist()


def save_run(
    out_dir: str,
    cfg: Config,
    seed: int,
    dataset_file: str,
    history: List[Dict[str, float]],
    best: Dict[str, float],
    model: PathModel,
    A_init=None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Write `results.json` (config, evaluation history, best checkpoint, learned operators) and `model.eqx`."""
    os.makedirs(out_dir, exist_ok=True)
    A_final = extract_A(model)
    record = {
        "seed": int(seed),
        "dataset": dataset_file,
        "config": cfg.as_dict(),
        "evals": history,
        "best": best,
        "A_init": _tolist(A_init),
        "A_final": _tolist(A_final),
        "eigenvalues": eigenvalues(A_final) if A_final is not None else None,
        "B_final": _tolist(extract_B(model)),
    }
    if extra:
        record.update(extra)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(record, f, indent=1)
    ckpt = os.path.join(out_dir, "model.eqx")
    eqx.tree_serialise_leaves(ckpt, model)
    return ckpt
