"""Configuration schema: sectioned configs and the upgrade of older flat layouts.

A model config is organised by module:

    A, B           parametrisations of the drift and the lift (ews.core.ab_params)
    clock          the clock (ews.clocks.ClockSpec fields); append_clock: a second clock added as a channel
    path           the path stage (ews.paths.PathSpec fields)
    compute        how the signature is computed: leaf, series_q, van_loan_expm, chen_expm, M,
                   terminal_reduce, reweighting_mode, combine, combine_chunk, combine_tile
    block keys     m, depth (signature depth), and the block scaffold (norm_placement, second_norm,
                   graded_norm, graded_eps, graded_prescale, channel_stage, ff_mult, output_gate,
                   dropout_rate, norm_eps, angle_target)

Stacked models put these under `model.block` (defaults) and `model.blocks[i]` (per-block overrides).
Layer-only models put A, B, compute and the readout keys under `model`, and path/clock under
`preprocessing`. `upgrade` rewrites the older flat layouts into this one and is idempotent, so a
config may be passed through it any number of times.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

COMPUTE_KEYS = (
    "leaf",
    "series_q",
    "van_loan_expm",
    "chen_expm",
    "M",
    "terminal_reduce",
    "reweighting_mode",
    "combine",
    "combine_chunk",
    "combine_tile",
)
BLOCK_SCAFFOLD_KEYS = (
    "m",
    "depth",
    "norm_placement",
    "second_norm",
    "graded_norm",
    "graded_eps",
    "graded_prescale",
    "channel_stage",
    "ff_mult",
    "output_gate",
    "dropout_rate",
    "norm_eps",
    "angle_target",
)
BLOCK_SECTIONS = ("A", "B", "clock", "append_clock", "path", "compute")
STACK_MODEL_KEYS = (
    "num_blocks",
    "hidden_dim",
    "head",
    "tokens",
    "final_norm",
    "cosine_head_tau",
    "input_token_ids",
    "token_cache",
    "graded_eps_init",
    "dtype",
    "block",
    "blocks",
)
LAYER_MODEL_KEYS = (
    "A",
    "B",
    "compute",
    "depth",
    "readout",
    "feature",
    "feature_norm",
    "feature_norm_init",
    "feature_norm_eval",
    "head_type",
    "head_hidden_dims",
    "head_init_scale",
)
PREPROCESSING_KEYS = (
    "dtype",
    "path",
    "clock",
    "append_clock",
    "normalize_targets",
    "target_range",  # regression: target scaling
    "predict_channel",
)  # prediction: which channel is predicted
XLA_KEYS = ("xla_autotune", "deterministic_ops")
STACK_TRAINING_KEYS = (
    "seed",
    "batch_size",
    "num_steps",
    "print_steps",
    "learning_rate",
    "final_lr",
    "warmup_fraction",
    "schedule",
    "weight_decay_embedding",
    "weight_decay_others",
    "weight_decay_A",
    "weight_decay_A_damp",
    "vf_A_norm_lambda",
    "label_smoothing",
    "grad_clip",
    "early_stop_threshold",
    "checkpoint_every",
    "val_batch_size",
    "adam_b1",
    "adam_b2",
    "adam_eps",
    "lr_mult_A",
    "lr_mult_B",
    "fix_embedding_norm",
    "futility_step",
    "futility_min_acc",
    "max_train_steps",
    "init_from",
) + XLA_KEYS
LAYER_TRAINING_KEYS = (
    "seed",
    "batch_size",
    "num_steps",
    "eval_every",
    "log_every",
    "warmup_steps",
    "lr_features",
    "lr_head",
    "min_lr_features",
    "min_lr_head",
    "weight_decay_features",
    "weight_decay_head",
    "grad_clip_features",
    "grad_clip_head",
    "lambda1",
    "lambda2",
    "label_smoothing",
    "metric",
    "threshold",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "test_mode",
    "test_batch_size",
) + XLA_KEYS
_REMOVED = ("combine", "combine_q", "combine_K")
_LEGACY_PATH_KEYS = (
    "path_derivative",
    "savgol_window",
    "savgol_polyorder",
    "normalize_paths",
    "normalize_mode",
    "normalize_range",
    "per_sample_normalize",
    "per_sample_norm_stat",
    "add_basepoint",
    "augment_with_clock",
    "clock_mode",
    "clock_alpha",
    "clock_normalize",
    "append_clock_mode",
    "append_clock_alpha",
)


def upgrade_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """Flat block keys -> sectioned block (idempotent)."""
    b = copy.deepcopy(block)
    for k in _REMOVED:
        v = b.pop(k, None)
        if k == "combine" and v not in (None, "assoc.build"):
            raise ValueError(f"combine={v!r} is no longer supported")
    if "path_mode" in b or "basepoint" in b:
        path = dict(b.get("path", {}))
        if "path_mode" in b:
            path.setdefault("representation", b.pop("path_mode"))
        if "basepoint" in b:
            path.setdefault("basepoint", b.pop("basepoint"))
        b["path"] = path
    if any(k in b for k in COMPUTE_KEYS):
        comp = dict(b.get("compute", {}))
        for k in COMPUTE_KEYS:
            if k in b:
                comp.setdefault(k, b.pop(k))
        b["compute"] = comp
    if "n" in b:
        b.setdefault("depth", b.pop("n"))
    return b


def upgrade_stack_model(model: Dict[str, Any]) -> Dict[str, Any]:
    m = copy.deepcopy(model)
    if "block" in m:
        m["block"] = upgrade_block(m["block"])
    if "blocks" in m:
        m["blocks"] = [upgrade_block(x) for x in m["blocks"]]
    return m


def upgrade_layer_model(model: Dict[str, Any]) -> Dict[str, Any]:
    m = copy.deepcopy(model)
    if "van_loan_mode" in m:
        m.setdefault("reweighting_mode", m.pop("van_loan_mode"))
    if any(k in m for k in COMPUTE_KEYS):
        comp = dict(m.get("compute", {}))
        for k in COMPUTE_KEYS:
            if k in m:
                comp.setdefault(k, m.pop(k))
        m["compute"] = comp
    if "ews_depth" in m:
        m.setdefault("depth", m.pop("ews_depth"))
    return m


def upgrade_preprocessing(pp: Dict[str, Any]) -> Dict[str, Any]:
    """Flat preprocessing keys -> `path` / `clock` / `append_clock` sections (idempotent)."""
    p = copy.deepcopy(pp)
    if not any(k in p for k in _LEGACY_PATH_KEYS):
        return p
    from ews.paths import legacy_specs

    path, clock, append = legacy_specs(p)
    out = {k: v for k, v in p.items() if k not in _LEGACY_PATH_KEYS}
    out["path"] = {k: getattr(path, k) for k in path.__dataclass_fields__}
    out["clock"] = {
        k: getattr(clock, k)
        for k in clock.__dataclass_fields__
        if getattr(clock, k) != type(clock)().__getattribute__(k) or k in ("mode",)
    }
    if append is not None:
        out["append_clock"] = {
            k: getattr(append, k)
            for k in append.__dataclass_fields__
            if getattr(append, k) != type(append)().__getattribute__(k) or k in ("mode",)
        }
    return out


def upgrade(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Upgrade a whole config (layer-only or stacked) to the sectioned schema."""
    c = copy.deepcopy(cfg)
    if "model" in c:
        if "block" in c["model"] or "blocks" in c["model"]:
            c["model"] = upgrade_stack_model(c["model"])
        else:
            c["model"] = upgrade_layer_model(c["model"])
    if "preprocessing" in c:
        c["preprocessing"] = upgrade_preprocessing(c["preprocessing"])
    return c


def validate(cfg: Dict[str, Any]) -> None:
    """Reject unknown keys in the model and preprocessing sections."""
    from ews.clocks import ClockSpec
    from ews.paths import PathSpec

    m = cfg.get("model", {})
    stacked = "block" in m or "blocks" in m
    allowed = set(STACK_MODEL_KEYS if stacked else LAYER_MODEL_KEYS)
    unknown = set(m) - allowed
    if unknown:
        raise ValueError(f"unknown model keys: {sorted(unknown)}")

    def check_block(b):
        bad = set(b) - set(BLOCK_SCAFFOLD_KEYS) - set(BLOCK_SECTIONS)
        if bad:
            raise ValueError(f"unknown block keys: {sorted(bad)}")
        bad = set(b.get("compute", {})) - set(COMPUTE_KEYS)
        if bad:
            raise ValueError(f"unknown compute keys: {sorted(bad)}")
        ClockSpec.from_config(b.get("clock"))
        if b.get("append_clock") is not None:
            ClockSpec.from_config(b["append_clock"])
        if "path" in b:
            PathSpec.from_config({k: v for k, v in b["path"].items()})

    if stacked:
        if "block" in m:
            check_block(m["block"])
        for x in m.get("blocks", []):
            check_block(x)
    else:
        bad = set(m.get("compute", {})) - set(COMPUTE_KEYS)
        if bad:
            raise ValueError(f"unknown compute keys: {sorted(bad)}")
    tr = cfg.get("training")
    if tr is not None:
        allowed_tr = set(STACK_TRAINING_KEYS if stacked else LAYER_TRAINING_KEYS)
        bad = set(tr) - allowed_tr
        if bad:
            raise ValueError(f"unknown training keys: {sorted(bad)}")
    pp = cfg.get("preprocessing")
    if pp is not None:
        bad = set(pp) - set(PREPROCESSING_KEYS)
        if bad:
            raise ValueError(f"unknown preprocessing keys: {sorted(bad)}")
        if "path" in pp:
            PathSpec.from_config(pp["path"])
        if "clock" in pp:
            ClockSpec.from_config(pp["clock"])
