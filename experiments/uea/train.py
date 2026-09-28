"""Path-to-label classification trainer.

Loads a dataset in the NPZ format of experiments/uea/loader.py, builds an EWS model with an optional lift B and trains
it with cross-entropy.

The XLA switches are --autotune {on,off} and --deterministic-ops {on,off}, or training.xla_autotune and
training.deterministic_ops, with autotuning on and deterministic ops off by default. Each bank records its regime in
grid.json and uea_results/replicate.sh applies it.

    python experiments/uea/train.py --config uea_results/heartbeat/config.json --dataset_file <split.npz> --seed 0 --gpu 0
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from ews.clocks import Clock, make_clock

REPO_ROOT = Path(__file__).resolve().parents[2]

# The GPU and the memory settings are fixed before JAX is imported.
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--gpu", type=str, default=None)
_pre, _ = _parser.parse_known_args()
if _pre.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _pre.gpu
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# XLA switches (see ews.determinism).
from ews.determinism import configure_from_cli as _configure_xla

_configure_xla(default_autotune=True, default_deterministic_ops=False)

import json
import sys

sys.path.insert(0, str(REPO_ROOT))
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from ews.layer import EWS, B, LogEWS, log_level_slices
from ews.norms import FeatureNorm
from ews.params import build_ab_from_config, parse_a_config, parse_b_config
from experiments.uea.loader import load_path_classification_dataset

# Persistent XLA compilation cache: runs of the same model structure (the same config on several seeds) reuse the
# compiled executables across processes.
try:
    jax.config.update("jax_compilation_cache_dir", str(REPO_ROOT / "outputs" / ".jax_cache"))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
except Exception:
    pass


# ── Config ─────────────────────────────────────────────────────────────────────


@dataclass
class Config:
    data: Dict[str, Any]
    preprocessing: Dict[str, Any]
    model: Dict[str, Any]
    training: Dict[str, Any]


def load_config(path: str) -> Config:
    with open(path) as f:
        d = json.load(f)
    from ews.config import upgrade, validate

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


class PathLabelModel(eqx.Module):
    b_layer: Optional[B]
    ewms: eqx.Module
    head: eqx.Module
    feature_norm: eqx.Module
    clock: Optional[
        Clock
    ]  # a module only for clocks with parameters; static clocks are precomputed by the loader
    dtype: jnp.dtype = eqx.field(static=True)
    readout: str = eqx.field(static=True)
    clock_source: str = eqx.field(static=True, default="block_input")
    clock_skip: int = eqx.field(static=True, default=0)

    def __call__(
        self, paths: jnp.ndarray, clocks: jnp.ndarray, return_features: bool = False
    ) -> jnp.ndarray:
        raw = paths
        if self.b_layer is not None:
            if paths.ndim == 2:
                paths = self.b_layer(paths)
            else:
                paths = jax.vmap(self.b_layer)(paths)
        if paths.dtype != self.dtype:
            paths = paths.astype(self.dtype)
        if self.clock is not None:
            # A clock with parameters is evaluated here, on the signal its spec names. 'block_input' and
            # 'raw_input' are this model's input X with the clock channel dropped: the same signal the
            # loader builds a static clock from, so learnable and static clocks read the same thing.
            # 'encoder_output' is the lift B.X.
            if self.clock_source == "encoder_output":
                drv = paths
            else:
                drv = raw[..., self.clock_skip :]
                if drv.dtype != self.dtype:
                    drv = drv.astype(self.dtype)
            L = drv.shape[-2]
            t = jnp.arange(L, dtype=self.dtype)

            def one(p):
                return self.clock(t, p, None)

            clocks = one(drv) if drv.ndim == 2 else jax.vmap(one)(drv)
        if clocks.dtype != self.dtype:
            clocks = clocks.astype(self.dtype)
        sig = self.ewms(paths, clocks, readout=self.readout, output_format="flat")
        if return_features:
            return sig  # pre-normalization features (for 1/RMS gain init)
        sig = self.feature_norm(sig)
        if sig.ndim == 2:
            return jax.vmap(self.head)(sig)
        return self.head(sig)


def build_model(cfg: Config, d_in: int, n_classes: int, key: jax.random.PRNGKey) -> PathLabelModel:
    from ews.config import upgrade_layer_model

    build_key = key  # the untouched argument; `model.A.omega_log_uniform_hi` draws from it below
    model_cfg = upgrade_layer_model(cfg.model)
    comp = model_cfg.get("compute", {})
    dtype = resolve_dtype(cfg.preprocessing.get("dtype", "float32"))
    reweighting_mode = comp.get("reweighting_mode", "displacement")
    terminal_reduce = comp.get("terminal_reduce", "tree")
    van_loan_expm = comp.get("van_loan_expm", "cayley")
    leaf = comp.get("leaf", "van_loan")
    series_q = comp.get("series_q")
    chen_expm = comp.get("chen_expm", "exact")
    combine = comp.get("combine", "assoc")
    combine_chunk = int(comp.get("combine_chunk", 1024))
    combine_tile = int(comp.get("combine_tile", 4))

    readout = model_cfg.get("readout", "terminal")
    if readout not in {"terminal", "mean", "max"}:
        raise ValueError(f"model.readout must be 'terminal', 'mean', or 'max', got {readout}")

    feature_norm_mode = model_cfg.get("feature_norm", "none")

    # Uniformity guard: a non-uniform (variation-based) clock is incompatible
    # with the uniform Van Loan fast path. Auto-route to the general
    # 'displacement' kernel at build time and warn.
    from ews.paths import specs_from_config

    _, _clock_spec, _ = specs_from_config(cfg.preprocessing)
    clock_mode, clock_alpha = _clock_spec.mode, float(_clock_spec.alpha)
    clock_is_uniform = (clock_mode == "time") and (clock_alpha == 1.0)
    if not clock_is_uniform and reweighting_mode.endswith("uniform"):
        print(
            f"[guard] clock_mode={clock_mode} (alpha={clock_alpha}) is non-uniform; "
            f"overriding reweighting_mode '{reweighting_mode}' -> 'displacement'."
        )
        reweighting_mode = "displacement"

    N = int(model_cfg.get("depth", 2))
    M = int(comp.get("M", 20))

    # Feature kind: "signature" (default, unchanged behavior) or "logsignature" (log-EWS).
    feature_kind = model_cfg.get("feature", "signature")
    if feature_kind not in {"signature", "logsignature"}:
        raise ValueError(f"model.feature must be 'signature' or 'logsignature', got {feature_kind}")
    is_log = feature_kind == "logsignature"

    key, subkey = jr.split(key)
    a_param, b_layer, _w, _a_cfg, _b_cfg = build_ab_from_config(model_cfg, d_in, subkey, dtype)
    ews_kwargs = dict(
        A=a_param,
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
    )
    if is_log:
        # log(EWS) in the compressed Lyndon basis: no level-0 constant, Witt-sized level
        # blocks. LogEWS wraps EWS unchanged; only the feature dim + level layout differ.
        ewms = LogEWS(**ews_kwargs)
        feat_dim = ewms.log_signature_dim
        fn_slices = log_level_slices(ewms.d, ewms.N)
    else:
        ewms = EWS(**ews_kwargs)
        feat_dim = ewms.signature_dim + 1  # +1 = the level-0 constant prepended at flatten
        fn_slices = None

    key, subkey = jr.split(key)
    head_type = model_cfg.get("head_type", "mlp" if is_log else "linear")
    head_init_scale = model_cfg.get("head_init_scale", None)

    if head_type == "mlp":
        hidden_dims = model_cfg.get("head_hidden_dims", [64, 32])
        head = eqx.nn.MLP(
            in_size=feat_dim,
            out_size=n_classes,
            width_size=int(hidden_dims[0]) if hidden_dims else 64,
            depth=len(hidden_dims) if hidden_dims else 2,
            key=subkey,
        )
    else:
        head = eqx.nn.Linear(feat_dim, n_classes, key=subkey)

    if head_init_scale is not None:
        scale = float(head_init_scale)
        if scale != 1.0:
            head = jax.tree_util.tree_map(
                lambda x: x * scale if eqx.is_array(x) else x,
                head,
            )

    feature_norm = FeatureNorm(
        d=ewms.d, N=ewms.N, mode=feature_norm_mode, dtype=dtype, level_slices=fn_slices
    )
    # A parametrised clock reads the signal its spec names (see PathLabelModel.__call__): the model's input X
    # without the clock channel, or the lift B.X. Its gate weights are sized for whichever that is.
    clock_source = _clock_spec.source
    clock_skip = 1 if cfg.preprocessing.get("path", {}).get("clock_channel") == "main" else 0
    clock_dim = ewms.d if clock_source == "encoder_output" else int(d_in) - clock_skip

    clock = None
    if _clock_spec.has_parameters:
        key, subkey = jr.split(key)
        clock = make_clock(_clock_spec, clock_dim, key=subkey, dtype=dtype)

    model = PathLabelModel(
        b_layer=b_layer,
        ewms=ewms,
        head=head,
        feature_norm=feature_norm,
        clock=clock,
        dtype=dtype,
        readout=readout,
        clock_source=clock_source,
        clock_skip=clock_skip,
    )

    # `model.A.omega_log_uniform_hi`: the rotation of a complex `jordan_repeated` block is drawn log-uniformly in
    # [hi/100, hi] instead of from the A init's Gaussian, which draws a block's damping and its rotation from one
    # distribution and so cannot set a rotation scale on its own. The damping is left as the init drew it. A real
    # block has no rotation and is untouched. This runs after the model is built, on a fold of build_model's own
    # key, which is how the search that produced these configurations did it.
    omega_hi = float(model_cfg.get("A", {}).get("omega_log_uniform_hi", 0.0) or 0.0)
    if omega_hi > 0.0:
        A = model.ewms.A_param
        bp = getattr(A, "block_params", None)
        if bp is None:
            raise ValueError(
                "model.A.omega_log_uniform_hi needs a block parametrisation with block_params"
            )
        if not any(np.size(p) == 2 for p in bp):
            raise ValueError(
                "model.A.omega_log_uniform_hi needs at least one complex jordan_repeated block"
            )
        u = jr.uniform(
            jr.fold_in(build_key, 7000),
            (len(bp),),
            minval=math.log(omega_hi / 100.0),
            maxval=math.log(omega_hi),
        )
        new = tuple(
            jnp.stack([p[0], jnp.exp(u[j])]).astype(p.dtype) if np.size(p) == 2 else p
            for j, p in enumerate(bp)
        )
        model = eqx.tree_at(lambda m: m.ewms.A_param.block_params, model, new)

    return model


# ── Optimizer ──────────────────────────────────────────────────────────────────


def make_schedule(base_lr, min_lr, warmup_steps, total_steps):
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=base_lr,
        warmup_steps=max(int(warmup_steps), 0),
        decay_steps=max(int(total_steps) - max(int(warmup_steps), 0), 1),
        end_value=min_lr,
    )


def make_optimizer(model: PathLabelModel, cfg: Config, total_steps: int):
    tc = cfg.training
    mc = cfg.model

    lr_f = float(tc.get("lr_features", 1e-4))
    lr_h = float(tc.get("lr_head", 1e-3))
    mlr_f = float(tc.get("min_lr_features", 0.0))
    mlr_h = float(tc.get("min_lr_head", 0.0))
    ws = int(tc.get("warmup_steps", 0))
    wd_f = float(tc.get("weight_decay_features", 0.0))
    wd_h = float(tc.get("weight_decay_head", 0.0))
    gc_f = float(tc.get("grad_clip_features", 1.0))
    gc_h = float(tc.get("grad_clip_head", 1.0))

    a_cfg = parse_a_config(mc)
    b_cfg = parse_b_config(mc)

    learn_A = bool(a_cfg.learn)
    learn_B = bool(b_cfg.learn and b_cfg.enabled)

    params = eqx.filter(model, eqx.is_array)
    label_tree = jax.tree_util.tree_map(lambda _: "frozen", params)

    if learn_A:
        label_tree = eqx.tree_at(
            lambda m: m.ewms,
            label_tree,
            replace=jax.tree_util.tree_map(
                lambda _: "features", eqx.filter(model.ewms, eqx.is_array)
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
    if getattr(model, "feature_norm", None) is not None and model.feature_norm.mode != "none":
        label_tree = eqx.tree_at(
            lambda m: m.feature_norm,
            label_tree,
            replace=jax.tree_util.tree_map(
                lambda _: "features", eqx.filter(model.feature_norm, eqx.is_array)
            ),
        )
    label_tree = eqx.tree_at(
        lambda m: m.head,
        label_tree,
        replace=jax.tree_util.tree_map(lambda _: "head", eqx.filter(model.head, eqx.is_array)),
    )
    # A clock with parameters is trained unless `clock.learn` is false, which holds its parameters at their init while
    # gradients still flow through theta to A, B and the head; `clock.stop_gradient` cuts that path too. The clock
    # follows the features' schedule and clipping and never gets weight decay.
    clock_learn = (
        getattr(model, "clock", None) is not None
        and model.clock.spec.learn
        and not model.clock.spec.stop_gradient
        and len(jax.tree_util.tree_leaves(eqx.filter(model.clock, eqx.is_inexact_array))) > 0
    )
    if clock_learn:
        label_tree = eqx.tree_at(
            lambda m: m.clock,
            label_tree,
            replace=jax.tree_util.tree_map(
                lambda _: "clock", eqx.filter(model.clock, eqx.is_array)
            ),
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
            "clock": optax.chain(
                optax.clip_by_global_norm(gc_f), optax.adamw(sched_f, weight_decay=0.0)
            ),
            "frozen": optax.set_to_zero(),
        },
        lambda _: label_tree,
    )
    return opt, opt.init(params)


# ── Loss and metrics ───────────────────────────────────────────────────────────


def make_loss(task_type: str, label_smoothing: float = 0.0):
    eps = float(label_smoothing or 0.0)
    if task_type == "multiclass":

        def loss_fn(logits, labels):
            if eps > 0.0:
                n_classes = logits.shape[-1]
                one_hot = jax.nn.one_hot(labels, n_classes)
                smoothed = one_hot * (1.0 - eps) + (eps / n_classes)
                return jnp.mean(optax.softmax_cross_entropy(logits, smoothed))
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))

        return loss_fn

    def loss_fn(logits, labels):
        if eps > 0.0:
            labels_smoothed = labels * (1.0 - eps) + (0.5 * eps)
            return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels_smoothed))
        return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))

    return loss_fn


def compute_metrics(logits, labels, task_type: str, threshold: float = 0.5) -> Dict[str, float]:
    if task_type == "multiclass":
        preds = jnp.argmax(logits, axis=-1)
        return {"accuracy": float(jnp.mean(preds == labels))}

    probs = jax.nn.sigmoid(logits)
    preds = (probs >= threshold).astype(jnp.float32)
    labels_f = labels.astype(jnp.float32)

    hamming = float(jnp.mean(preds == labels_f))
    exact_match = float(jnp.mean(jnp.all(preds == labels_f, axis=1)))

    tp = jnp.sum(preds * labels_f, axis=0)
    fp = jnp.sum(preds * (1 - labels_f), axis=0)
    fn = jnp.sum((1 - preds) * labels_f, axis=0)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    macro_f1 = float(jnp.mean(f1))

    tp_micro = float(jnp.sum(tp))
    fp_micro = float(jnp.sum(fp))
    fn_micro = float(jnp.sum(fn))
    p_micro = tp_micro / (tp_micro + fp_micro + 1e-12)
    r_micro = tp_micro / (tp_micro + fn_micro + 1e-12)
    micro_f1 = 2 * p_micro * r_micro / (p_micro + r_micro + 1e-12)

    return {
        "hamming": hamming,
        "exact_match": exact_match,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
    }


def flatten_head_params(head: eqx.Module) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(eqx.filter(head, eqx.is_array))
    if not leaves:
        return jnp.array([0.0])
    return jnp.concatenate([leaf.reshape(-1) for leaf in leaves])


def extract_A(model: PathLabelModel):
    ewms = getattr(model, "ewms", None)
    if ewms is None:
        return None
    if hasattr(ewms, "A_param"):
        return ewms.A_param()
    return None


def extract_B(model: PathLabelModel):
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


def create_batch_iterator(paths: List, clocks: List, labels: jnp.ndarray, batch_size: int, key):
    n = len(paths)
    # Each batch is gathered on the host and moved to the device in one transfer.
    paths_np = np.stack([np.asarray(x) for x in paths])
    clocks_np = np.stack([np.asarray(x) for x in clocks])
    labels_np = np.asarray(labels)
    while True:
        key, subkey = jr.split(key)
        idx = np.asarray(jr.permutation(subkey, jnp.arange(n)))
        for i in range(0, n, batch_size):
            batch = idx[i : i + batch_size]
            yield (
                jnp.asarray(paths_np[batch]),
                jnp.asarray(clocks_np[batch]),
                jnp.asarray(labels_np[batch]),
            )


@eqx.filter_jit
def _eval_forward(model, paths, clocks):
    return model(paths, clocks)


@eqx.filter_jit
def _eval_features(model, paths, clocks):
    return model(paths, clocks, return_features=True)  # pre-normalisation features


def train_feature_stats(model, train_paths, train_clocks, chunk: int = 64):
    """Per-level BatchNorm statistics of the training set under `model` (model.feature_norm_eval = "train").

    `feature_norm='batchnorm'` keeps no running statistics; by default an evaluated set is normalised with its own
    statistics, so each prediction depends on the other evaluated samples. With these statistics every validation /
    test prediction is independent of the others. The forward is chunked for memory only."""
    n = train_paths.shape[0]
    feats = jnp.concatenate(
        [
            _eval_features(model, train_paths[i : i + chunk], train_clocks[i : i + chunk])
            for i in range(0, n, chunk)
        ]
    )
    return model.feature_norm.level_stats(feats)


def evaluate_model(model, paths, clocks, labels, task_type, loss_fn, threshold, norm_stats=None):
    # One jitted forward over the whole set: feature_norm='batchnorm' takes its statistics over the evaluated set, so
    # the set is not chunked.
    if (
        norm_stats is not None
    ):  # batchnorm with the training set's statistics (see train_feature_stats)
        feats = _eval_features(model, jnp.stack(paths, axis=0), jnp.stack(clocks, axis=0))
        logits = jax.vmap(model.head)(model.feature_norm.apply_with_stats(feats, norm_stats))
        loss = loss_fn(logits, labels)
        return float(loss), compute_metrics(logits, labels, task_type, threshold)
    logits = _eval_forward(model, jnp.stack(paths, axis=0), jnp.stack(clocks, axis=0))
    loss = loss_fn(logits, labels)
    metrics = compute_metrics(logits, labels, task_type, threshold)
    return float(loss), metrics


# ── Main ───────────────────────────────────────────────────────────────────────


def train(
    config_path: str,
    dataset_file: str | None = None,
    seed_override: int | None = None,
    out_ckpt: str | None = None,
):
    cfg = load_config(config_path)
    dtype = resolve_dtype(cfg.preprocessing.get("dtype", "float32"))
    if dtype == jnp.float64:
        jax.config.update("jax_enable_x64", True)

    file_path = dataset_file or cfg.data.get("file")
    if not file_path:
        raise ValueError("No dataset file specified.")
    if not os.path.isabs(file_path):
        file_path = str(REPO_ROOT / file_path)

    print(f"Loading dataset from {file_path} ...")
    dataset = load_path_classification_dataset(file_path, cfg.preprocessing)

    train_paths = dataset["train"]["paths"]
    train_clocks = dataset["train"]["clocks"]
    train_labels = dataset["train"]["labels"]
    val_paths = dataset["val"]["paths"]
    val_clocks = dataset["val"]["clocks"]
    val_labels = dataset["val"]["labels"]
    test_paths = dataset["test"]["paths"]
    test_clocks = dataset["test"]["clocks"]
    test_labels = dataset["test"]["labels"]

    meta = dataset["metadata"]
    d_in = meta["d_path"]
    n_classes = meta["n_classes"]
    task_type = meta["task_type"]

    print(f"  {meta['n_train']} train / {meta['n_val']} val / {meta['n_test']} test")
    print(f"  d_in={d_in}  task={task_type}  n_classes={n_classes}")

    tc = cfg.training
    num_steps = int(tc.get("num_steps", 15000))
    eval_every = int(tc.get("eval_every", 500))
    log_every = int(tc.get("log_every", 100))
    batch_size = int(tc.get("batch_size", 8))
    patience = int(tc.get("early_stopping_patience", 0))
    min_delta = float(tc.get("early_stopping_min_delta", 0.0))
    lambda1 = float(tc.get("lambda1", 0.0))
    lambda2 = float(tc.get("lambda2", 0.0))
    label_smoothing = float(tc.get("label_smoothing", 0.0))
    threshold = float(tc.get("threshold", 0.5))
    metric_name = tc.get("metric", "accuracy")

    seed = int(tc.get("seed", 42)) if seed_override is None else int(seed_override)
    key = jr.PRNGKey(seed)
    model = build_model(cfg, d_in, n_classes, key)

    # series-leaf guard: warn if series_q is too small for the observed
    # x = ||A||_op * max dtheta. Only fires when leaf="series" was selected.
    if getattr(model.ewms, "leaf", "van_loan") == "series":
        from ews.core.leaf import check_series_q

        _dth = jnp.concatenate([c_[1:] - c_[:-1] for c_ in train_clocks[:64]])
        check_series_q(
            np.asarray(model.ewms.A_param()),
            float(jnp.max(_dth)),
            int(model.ewms.N),
            int(model.ewms.series_q),
            where="build",
        )

    # learned-gain init: model.feature_norm_init in {"rms","ones"} (default "rms" =
    # the existing/long-standing behavior: per-level gains set to 1/RMS_k from a
    # training batch). "ones" leaves the gains at their constructor default of 1.0.
    feature_norm_init = cfg.model.get("feature_norm_init", "rms")
    if (
        getattr(model, "feature_norm", None) is not None
        and model.feature_norm.mode == "learned_gain"
        and feature_norm_init == "rms"
    ):
        k_init = min(64, len(train_paths))
        feats0 = model(
            jnp.stack(train_paths[:k_init]),
            jnp.stack(train_clocks[:k_init]),
            return_features=True,
        )
        rms = model.feature_norm.per_level_rms(feats0)
        gains0 = 1.0 / (rms + 1e-8)
        model = eqx.tree_at(lambda m: m.feature_norm.gains, model, gains0)
        print(f"feature_norm=learned_gain init=rms: per-level RMS={rms}  -> gains set to 1/RMS")

    opt, opt_state = make_optimizer(model, cfg, num_steps)
    loss_fn = make_loss(task_type, label_smoothing)

    A_init_snap = extract_A(model)
    B_init_snap = extract_B(model)
    if A_init_snap is not None:
        print(f"Initial A:\n{A_init_snap}")
    if B_init_snap is not None:
        print(f"Initial B:\n{B_init_snap}")

    batch_iter = create_batch_iterator(train_paths, train_clocks, train_labels, batch_size, key)

    # model.feature_norm_eval: which statistics a batchnorm feature norm uses at evaluation. "batch" (default) = the
    # evaluated set's own; "train" = the training set's, recomputed with the evaluated model at every evaluation, so
    # predictions are independent of each other. Training is identical in both (minibatch statistics).
    feature_norm_eval = cfg.model.get("feature_norm_eval", "batch")
    if feature_norm_eval not in {"batch", "train"}:
        raise ValueError(
            f"model.feature_norm_eval must be 'batch' or 'train', got {feature_norm_eval!r}"
        )
    norm_eval_train = feature_norm_eval == "train"
    if norm_eval_train and model.feature_norm.mode != "batchnorm":
        raise ValueError("model.feature_norm_eval='train' needs model.feature_norm='batchnorm'")
    if norm_eval_train:
        train_stack_p, train_stack_c = jnp.stack(train_paths), jnp.stack(train_clocks)

    @eqx.filter_jit
    def step(model, opt_state, batch_paths, batch_clocks, batch_labels):
        def loss_for_model(m):
            logits = m(batch_paths, batch_clocks)
            ce = loss_fn(logits, batch_labels)
            hp = flatten_head_params(m.head)
            return ce + lambda1 * jnp.sum(jnp.abs(hp)) + lambda2 * jnp.sum(hp**2)

        loss, grads = eqx.filter_value_and_grad(loss_for_model)(model)
        updates, opt_state = opt.update(
            eqx.filter(grads, eqx.is_array),
            opt_state,
            params=eqx.filter(model, eqx.is_array),
        )
        return eqx.apply_updates(model, updates), opt_state, loss

    print("=" * 70)
    print("Training — path classification (experiments/uea)")
    print("=" * 70)
    print(f"batch={batch_size}  steps={num_steps}  eval_every={eval_every}  metric={metric_name}")

    best_metric, best_model, best_step = None, None, None
    steps_no_improve = 0

    for step_idx in range(1, num_steps + 1):
        bp, bc, bl = next(batch_iter)
        model, opt_state, loss = step(model, opt_state, bp, bc, bl)
        jax.block_until_ready(loss)

        if step_idx % log_every == 0:
            print(f"Step {step_idx:5d}  train loss {float(loss):.6f}")

        if step_idx % eval_every == 0:
            stats = (
                train_feature_stats(model, train_stack_p, train_stack_c)
                if norm_eval_train
                else None
            )
            val_loss, val_metrics = evaluate_model(
                model, val_paths, val_clocks, val_labels, task_type, loss_fn, threshold, stats
            )
            test_loss, test_metrics = evaluate_model(
                model, test_paths, test_clocks, test_labels, task_type, loss_fn, threshold, stats
            )
            print(f"         val  loss={val_loss:.4f}  metrics={val_metrics}")
            print(f"         test loss={test_loss:.4f}  metrics={test_metrics}")

            current = val_metrics.get(metric_name, 0.0)
            improved = best_metric is None or current >= best_metric + min_delta
            if improved:
                best_metric, best_model, best_step = current, model, step_idx
                steps_no_improve = 0
            else:
                steps_no_improve += eval_every
                if patience > 0 and steps_no_improve >= patience:
                    print(f"Early stopping at step {step_idx}")
                    break

    if best_model is None:
        best_model = model

    A_final = extract_A(best_model)
    B_final = extract_B(best_model)
    if A_final is not None:
        print(f"Final A:\n{A_final}")
    if B_final is not None:
        print(f"Final B:\n{B_final}")

    final_stats = (
        train_feature_stats(best_model, train_stack_p, train_stack_c) if norm_eval_train else None
    )
    _, final_test_metrics = evaluate_model(
        best_model, test_paths, test_clocks, test_labels, task_type, loss_fn, threshold, final_stats
    )
    print("=" * 70)
    print(f"Best  step={best_step}  val_{metric_name}={best_metric:.6f}")
    print(f"Test metrics: {final_test_metrics}")
    print("=" * 70)

    out_dir = str(REPO_ROOT / "outputs")
    os.makedirs(out_dir, exist_ok=True)
    # One checkpoint path per run, so runs in parallel never overwrite each other.
    ckpt = out_ckpt or os.path.join(out_dir, "classification_best.eqx")
    os.makedirs(os.path.dirname(os.path.abspath(ckpt)), exist_ok=True)
    eqx.tree_serialise_leaves(ckpt, best_model)
    print(f"Saved best model → {ckpt}")

    return best_model, best_metric, final_test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Path-to-label classification trainer")
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "experiments" / "uea" / "configs" / "template.json"),
    )
    parser.add_argument("--dataset_file", type=str, default=None)
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    from ews.determinism import add_switches

    add_switches(parser)
    parser.add_argument(
        "--out_ckpt",
        type=str,
        default=None,
        help="checkpoint path (one per run when several run at once)",
    )
    args = parser.parse_args()

    train(args.config, args.dataset_file, args.seed, args.out_ckpt)
