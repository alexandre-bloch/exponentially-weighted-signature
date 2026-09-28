"""Formal-language trainer for stacked EWS blocks, under Walker's length-generalisation protocol.

AdamW (no weight decay on the embedding), a linear warm-up over the first 10% of the steps and then cosine decay;
cross-entropy over the masked positions plus `vf_A_norm_lambda` times each block's ||A||_F; masked per-token validation
accuracy every `print_steps`, a checkpoint on improvement and an early stop above `early_stop_threshold`. The seed is
`training.seed + --run`. Clock denominators and graded-norm floors are calibrated on one training batch with a probe
built from the same key, so the parameters are the same as without calibration. The clock's time denominator defaults
to the validation padding length minus 1. XLA switches: --autotune and --deterministic-ops (by default autotuning off
and deterministic ops off, the regime of the banked runs).

    python experiments/fl/train.py --config formal_language_results/cycle_nav/config.json --run 0 --gpu 0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# GPU and memory settings, before JAX is imported. A caller that has set CUDA_VISIBLE_DEVICES keeps it; only a
# standalone run parses --gpu.
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--gpu", type=str, default="0")
if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    _pre, _ = _parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = _pre.gpu
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# XLA switches (see ews.determinism). The banked formal-language results were produced with
# autotuning off and deterministic ops off, so those are the defaults here.
from ews.determinism import configure_from_cli as _configure_xla

_configure_xla(default_autotune=False, default_deterministic_ops=False)
# Determinism gate hook: if GATE_DUMP_LOSS=N is set, dump full-precision per-step loss for the first N
# steps (repr of the float32 value) so two runs can be diffed bit-for-bit. Off (0) in normal training.
_GATE_DUMP = int(os.environ.get("GATE_DUMP_LOSS", "0") or "0")

import json
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(REPO_ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from ews.block import StackedEWS
from experiments.fl.dataloading import (
    eval_batches,
    make_task_loaders,
    validity_mask,
)

# Persistent XLA compilation cache (as in the classification harness, but kept outside the repository).
try:
    # The compile cache grows to several GB (it once filled a quota-limited home directory and crashed runs with
    # "Disk quota exceeded"), so it goes to the temporary directory, not the repository. Set
    # JAX_COMPILATION_CACHE_DIR to put it elsewhere.
    import tempfile

    jax.config.update(
        "jax_compilation_cache_dir",
        os.environ.get("JAX_COMPILATION_CACHE_DIR")
        or os.path.join(tempfile.gettempdir(), "jax_cache_ews"),
    )
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
except Exception:
    pass


# -- Config --------------------------------------------------------------------


@dataclass
class Config:
    dataset: Dict[str, Any]
    model: Dict[str, Any]
    training: Dict[str, Any]


def load_config(path_or_name: str) -> Config:
    p = Path(path_or_name)
    if not p.exists():
        name = path_or_name if path_or_name.endswith(".json") else path_or_name + ".json"
        p = REPO_ROOT / "experiments" / "fl" / "configs" / name
    with open(p) as f:
        d = json.load(f)
    from ews.config import upgrade, validate

    d = upgrade(d)
    validate(d)
    return Config(dataset=d["dataset"], model=d["model"], training=d["training"])


_CLOCK_KEY_MAP = {
    "mode": "clock_mode",
    "alpha": "clock_alpha",
    "scaling": "clock_scaling",
    "normalize": "clock_normalize",
    "norm_constant": "clock_norm_constant",
    "time_denominator": "clock_time_denominator",
    "source": "clock_source",
    "stop_gradient": "clock_stop_gradient",
    "variation_eps": "clock_variation_eps",
    "gate_init_scale": "clock_gate_init_scale",
    "gate_init_bias": "clock_gate_init_bias",
    "gate_K": "clock_gate_K",
}

_BLOCK_PASSTHROUGH = [
    "m",
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
]


def _block_kwargs_from_cfg(block_cfg: dict) -> dict:
    """Sectioned block config -> EWSBlock keyword arguments (flat clock_* keys for the calibrations)."""
    from ews.config import upgrade_block

    block_cfg = upgrade_block(block_cfg)
    kw: Dict[str, Any] = {}
    for k in _BLOCK_PASSTHROUGH:
        if k in block_cfg:
            kw[k] = block_cfg[k]
    if "depth" in block_cfg:
        kw["n"] = block_cfg["depth"]
    for k, v in block_cfg.get("compute", {}).items():
        kw[k] = v
    path = block_cfg.get("path", {})
    if "representation" in path:
        kw["path_mode"] = path["representation"]
    if "basepoint" in path:
        kw["basepoint"] = path["basepoint"]
    if "A" in block_cfg:
        kw["a_config"] = block_cfg["A"]
    if "B" in block_cfg:
        kw["b_config"] = block_cfg["B"]
    for k, v in block_cfg.get("clock", {}).items():
        kw[_CLOCK_KEY_MAP[k]] = v
    ap = block_cfg.get("append_clock") or {}
    if ap.get("mode"):
        kw["append_clock_mode"] = ap["mode"]
        if "alpha" in ap:
            kw["append_clock_alpha"] = ap["alpha"]
        if "norm_constant" in ap:
            kw["append_clock_norm_constant"] = ap["norm_constant"]
    return kw


def resolve_dtype(s: str) -> jnp.dtype:
    return jnp.float64 if s == "float64" else jnp.float32


# -- Model build + init calibrations ------------------------------------------


def _per_block_kwargs(cfg: Config) -> List[dict]:
    base = _block_kwargs_from_cfg(cfg.model.get("block", {}))
    overrides = cfg.model.get("blocks")
    n_blocks = int(cfg.model["num_blocks"])
    out = []
    for i in range(n_blocks):
        kw = dict(base)
        if overrides:
            kw.update(_block_kwargs_from_cfg(overrides[i]))
        out.append(kw)
    return out


def _resolve_harness_defaults(kwargs_list: List[dict], val_pad: int) -> None:
    """Task-level clock defaults (in place): time denominator and the a-priori
    one-hot constant. Learned-driver constants are calibrated separately."""
    for kw in kwargs_list:
        kw.setdefault("clock_time_denominator", float(val_pad - 1))
        needs_const = (
            kw.get("clock_normalize", "per_sample") == "constant"
            and kw.get("clock_norm_constant") is None
            and kw.get("clock_mode", "time") != "time"
        )
        if needs_const and kw.get("clock_source", "block_input") == "raw_input":
            kw["clock_norm_constant"] = float(np.sqrt(2.0) * (val_pad - 1))


_CLOCK_KW = (
    "clock_mode",
    "clock_alpha",
    "clock_scaling",
    "clock_normalize",
    "clock_norm_constant",
    "clock_time_denominator",
    "clock_source",
    "clock_stop_gradient",
    "clock_variation_eps",
    "clock_gate_init_scale",
    "clock_gate_init_bias",
    "clock_gate_K",
)
_APPEND_KW = ("append_clock_mode", "append_clock_alpha", "append_clock_norm_constant")


def _block_specs(kw: dict) -> dict:
    """Turn the flat clock_* / append_clock_* keys of a block kwargs dict into ClockSpec objects.

    The flat keys are what the harness calibrations read and write; the block itself takes specs."""
    from dataclasses import replace as _replace

    from ews.clocks import ClockSpec

    kw = dict(kw)
    clock_kwargs = {k[len("clock_") :]: kw.pop(k) for k in _CLOCK_KW if k in kw}
    spec = ClockSpec.from_config(clock_kwargs)
    app_mode = kw.pop("append_clock_mode", None)
    app_alpha = kw.pop("append_clock_alpha", 1.0)
    app_const = kw.pop("append_clock_norm_constant", None)
    kw["clock"] = spec
    kw["append_clock"] = (
        None
        if app_mode is None
        else _replace(spec, mode=app_mode, alpha=float(app_alpha), norm_constant=app_const)
    )
    return kw


def _build(
    cfg: Config,
    data_dim: int,
    label_dim: int,
    key,
    kwargs_list: List[dict],
    force_default: bool = False,
) -> StackedEWS:
    if cfg.model.get("token_cache", False) and not force_default:
        # Fast path: first-block token cache, valid only under
        # increment mode + a discrete vocab. Same constructor args as StackedEWS; forward/gradient
        # equivalence is checked by tests/test_token_cache.py. Lazy import so the default path (flag
        # absent) never loads the module.
        from ews.block import TokenCachedStackedEWS

        return TokenCachedStackedEWS(
            int(cfg.model["num_blocks"]),
            data_dim,
            int(cfg.model["hidden_dim"]),
            label_dim,
            tokens=bool(cfg.model.get("tokens", True)),
            head_mode={"seq2seq": "seq2seq", "mean_pool": "mean_pool"}[
                cfg.model.get("head", "seq2seq")
            ],
            final_norm=bool(cfg.model.get("final_norm", False)),
            cosine_head_tau=cfg.model.get("cosine_head_tau"),
            input_token_ids=cfg.model.get("input_token_ids"),
            per_block_kwargs=[
                _block_specs({k: v for k, v in kw.items() if k != "angle_target"})
                for kw in kwargs_list
            ],
            dtype=resolve_dtype(cfg.model.get("dtype", "float32")),
            key=key,
        )
    return StackedEWS(
        int(cfg.model["num_blocks"]),
        data_dim,
        int(cfg.model["hidden_dim"]),
        label_dim,
        tokens=bool(cfg.model.get("tokens", True)),
        head_mode={"seq2seq": "seq2seq", "mean_pool": "mean_pool"}[
            cfg.model.get("head", "seq2seq")
        ],
        final_norm=bool(cfg.model.get("final_norm", False)),
        cosine_head_tau=cfg.model.get("cosine_head_tau"),
        input_token_ids=cfg.model.get("input_token_ids"),
        # angle_target is a calibration-only meta-field (A is rescaled after build);
        # strip it here so it never reaches EWSBlock.__init__, which rejects it.
        per_block_kwargs=[
            _block_specs({k: v for k, v in kw.items() if k != "angle_target"}) for kw in kwargs_list
        ],
        dtype=resolve_dtype(cfg.model.get("dtype", "float32")),
        key=key,
    )


def _embed_batch(model: StackedEWS, X: jnp.ndarray) -> jnp.ndarray:
    if model.tokens:
        return jax.vmap(lambda x: jax.vmap(model.embedding)(x.astype(jnp.int32)))(X)
    return jax.vmap(jax.vmap(model.embedding))(X)


def _gated_total_variation(driver: np.ndarray, valid: np.ndarray, quad: bool) -> np.ndarray:
    dx = np.diff(driver, axis=1)
    sq = (dx * dx).sum(axis=-1)
    incr = sq if quad else np.sqrt(sq)
    pair = (valid[:, 1:] & valid[:, :-1]).astype(incr.dtype)
    return (incr * pair).sum(axis=1)  # (B,)


def _batched_mixer_levels(block, h, valid, driver):
    if driver is None:
        return jax.vmap(lambda x, m: block.mixer_levels(x, mask=m))(h, valid)
    return jax.vmap(lambda x, m, d: block.mixer_levels(x, mask=m, clock_driver=d))(h, valid, driver)


def _warn_series_q(model, X0, valid0):
    """Warn per block if leaf="series" and series_q is too low for the measured x."""
    from ews.core.leaf import check_series_q

    if not any(getattr(b.ewms, "leaf", "van_loan") == "series" for b in model.blocks):
        return
    try:
        thetas = _block_clocks(model, jnp.asarray(X0), jnp.asarray(valid0))
    except Exception:
        return  # diagnostics must never break a build
    for i, block in enumerate(model.blocks):
        if getattr(block.ewms, "leaf", "van_loan") != "series":
            continue
        A = np.asarray(block.ewms.A_param()) * float(model._bounded_a_scale(block))
        if not np.all(np.isfinite(A)):
            print(f"block {i}: A is non-finite, skipping the series-order check", flush=True)
            continue
        dth = np.diff(np.asarray(thetas[i]), axis=1)
        dth = dth[np.isfinite(dth)]
        if dth.size:
            check_series_q(
                A,
                float(np.max(dth)),
                int(block.ewms.N),
                int(block.ewms.series_q),
                where=f"block {i}",
            )


def _block_clocks(model, X, mask):
    """Per-block clocks, replicating the stack's forward loop."""

    def one(x_row, mask_row):
        out = []
        h = jax.vmap(model.embedding)(x_row.astype(jnp.int32) if model.tokens else x_row)
        enc = h
        for block in model.blocks:
            driver = enc if block.clock.spec.source == "encoder_output" else h
            if block.path_mode == "increment" and driver is not None:
                driver = block._to_increment_path(driver, mask_row)
            t = jnp.arange(h.shape[0], dtype=h.dtype)
            out.append(block.clock(t, driver, mask_row))
            h = block(
                h,
                t=None,
                mask=mask_row,
                clock_driver=None,
                inference=True,
                a_scale=model._bounded_a_scale(block),
            )
        return out

    return jax.vmap(one)(X, mask)


def calibrate_and_build(
    cfg: Config,
    data_dim: int,
    label_dim: int,
    key,
    X0: np.ndarray,
    valid0: np.ndarray,
    raw_driver0: Optional[np.ndarray],
    val_pad: int,
):
    """Build the model, running the init calibrations if the config asks for
    them. Probe models share the PRNG key (and therefore all parameters) with
    the final model; only static clock/eps fields differ."""
    kwargs_list = _per_block_kwargs(cfg)
    _resolve_harness_defaults(kwargs_list, val_pad)
    calibration: Dict[str, Any] = {"time_denominator": kwargs_list[0]["clock_time_denominator"]}

    needs_clock_cal = [
        kw.get("clock_normalize", "per_sample") == "constant"
        and kw.get("clock_norm_constant") is None
        and kw.get("clock_mode", "time") != "time"
        for kw in kwargs_list
    ]
    if any(needs_clock_cal):
        probe_kwargs = []
        for kw in kwargs_list:
            pk = dict(kw)
            pk["clock_mode"] = "time"  # driver-independent probe forward
            pk.pop("clock_norm_constant", None)
            # the probe only measures the driver; drop the appended channel so
            # its (not-yet-calibrated) constant is never required
            pk["append_clock_mode"] = None
            pk.pop("append_clock_norm_constant", None)
            probe_kwargs.append(pk)
        # token_cache blocks need token_ids to advance h; the clock probe advances h with
        # clock_driver=None, so build the probe on the default path (equivalent forward) to
        # avoid the "requires token_ids" error. The measured clock constant is path-independent.
        probe = _build(
            cfg,
            data_dim,
            label_dim,
            key,
            probe_kwargs,
            force_default=bool(cfg.model.get("token_cache", False)),
        )
        h = _embed_batch(probe, jnp.asarray(X0))
        enc = h
        vj = jnp.asarray(valid0)
        constants = []
        for i, block in enumerate(probe.blocks):
            if block.clock.spec.source == "encoder_output":
                driver = enc
            elif block.clock.spec.source == "raw_input":
                driver = jnp.asarray(raw_driver0)
            else:
                driver = h
            quad = kwargs_list[i].get("clock_mode") == "blended_quad_variation"
            tot = _gated_total_variation(np.asarray(driver), valid0, quad)
            constants.append(float(np.median(tot)))
            h = block(h, mask=vj, clock_driver=None, inference=True)
        for i, kw in enumerate(kwargs_list):
            if needs_clock_cal[i]:
                kw["clock_norm_constant"] = constants[i]
            if (
                kw.get("append_clock_mode")
                and kw.get("clock_normalize", "per_sample") == "constant"
                and kw.get("append_clock_norm_constant") is None
            ):
                kw["append_clock_norm_constant"] = constants[i]
        calibration["clock_norm_constants"] = constants

    if cfg.model.get("graded_eps_init", "constant") == "calibrated":
        # The graded-eps floor is a median-RMS init statistic from the default block forward.
        # token_cache blocks need tokens in __call__ and their default-path mixer_levels OOMs at
        # large m, so for token_cache configs calibrate on a plain StackedEWS (equivalence => the
        # same levels) over a small subsample. Non-token_cache configs are byte-unchanged.
        tc = bool(cfg.model.get("token_cache", False))
        ncal = min(8, int(np.asarray(X0).shape[0])) if tc else int(np.asarray(X0).shape[0])
        probe = _build(cfg, data_dim, label_dim, key, kwargs_list, force_default=tc)
        h = _embed_batch(probe, jnp.asarray(X0)[:ncal])
        enc = h
        vcal = np.asarray(valid0)[:ncal]
        vj = jnp.asarray(vcal)
        eps_record = []
        for i, block in enumerate(probe.blocks):
            if block.clock.spec.source == "encoder_output":
                driver = enc
            elif block.clock.spec.source == "raw_input":
                driver = jnp.asarray(raw_driver0)
            else:
                driver = None
            if block.graded.variant == "live_rms":
                levels = _batched_mixer_levels(block, h, vj, driver)
                rms = np.asarray(block.graded.per_level_rms(levels))  # (B, L, N)
                eps_i = []
                for k in range(block.n):
                    vals = rms[..., k][vcal & (rms[..., k] > 0)]
                    med = float(np.median(vals)) if vals.size else 1e-1
                    eps_i.append(1e-2 * med)
                kwargs_list[i]["graded_eps"] = tuple(eps_i)
                eps_record.append(eps_i)
            else:
                eps_record.append(None)
            drv = (
                jnp.asarray(raw_driver0)
                if block.clock.spec.source == "raw_input"
                else (enc if block.clock.spec.source == "encoder_output" else None)
            )
            h = block(h, mask=vj, clock_driver=drv, inference=True)
        calibration["graded_eps"] = eps_record

    model = _build(cfg, data_dim, label_dim, key, kwargs_list)

    # angle-init calibration: rescale each block's A so the median per-token
    # realised rotation angle hits block config `angle_target` (radians).
    # Realised angle under cayley with M substeps: psi(phi) = 2M*arctan(phi/2M)
    # -> phi* = 2M*tan(psi*/2M); under pade/exact psi = phi.
    targets = [kw.get("angle_target") for kw in kwargs_list]
    if any(t is not None for t in targets):
        emb = np.asarray(model.embedding.weight)
        angle_record = []
        for i, (blk, tgt) in enumerate(zip(model.blocks, targets)):
            if tgt is None:
                angle_record.append(None)
                continue
            c = float(kwargs_list[i].get("clock_norm_constant") or 1.0)
            dth = np.median(np.linalg.norm(emb, axis=1)) / c
            A = np.asarray(blk.ewms.A_param())
            if not np.all(np.isfinite(A)):
                angle_record.append({"error": "non-finite A; calibration skipped"})
                continue
            w_med = float(np.median(np.abs(np.linalg.eigvals(A).imag)))
            phi_med = dth * max(w_med, 1e-12)
            M_sub = int(kwargs_list[i].get("M", 4))
            if kwargs_list[i].get("van_loan_expm", "cayley") == "cayley":
                phi_star = 2 * M_sub * np.tan(float(tgt) / (2 * M_sub))
            else:
                phi_star = float(tgt)
            factor = phi_star / phi_med
            model = eqx.tree_at(
                lambda t, i=i: t.blocks[i].ewms,
                model,
                replace=jax.tree_util.tree_map(
                    lambda x: x * factor if eqx.is_inexact_array(x) else x, model.blocks[i].ewms
                ),
            )
            angle_record.append(
                {
                    "target_psi": float(tgt),
                    "phi_star": float(phi_star),
                    "phi_med_before": float(phi_med),
                    "scale": float(factor),
                }
            )
        calibration["angle_init"] = angle_record
    # series-leaf guard: warn if series_q is too small for the observed
    # x = ||A_eff||_op * max dtheta. Only fires when a block actually selected leaf="series";
    # q is fixed for a run, so this is a warning to act on between runs, not something the
    # training loop can correct.
    _warn_series_q(model, X0, valid0)
    return model, calibration


# -- Loss / metrics ------------------------------------------------------------


def masked_ce(
    logits: jnp.ndarray, y: jnp.ndarray, loss_mask: jnp.ndarray, smoothing: float = 0.0
) -> jnp.ndarray:
    if smoothing > 0.0:
        n = logits.shape[-1]
        target = jax.nn.one_hot(y, n) * (1.0 - smoothing) + smoothing / n
        ce = optax.softmax_cross_entropy(logits, target)
    else:
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    w = loss_mask.astype(ce.dtype)
    return (ce * w).sum() / jnp.maximum(w.sum(), 1.0)


def a_penalty(model: StackedEWS) -> jnp.ndarray:
    total = 0.0
    for block in model.blocks:
        A = block.ewms.A_param()
        total = total + jnp.sqrt(jnp.sum(A * A))  # unsquared Frobenius, per block
    return total


def param_diagnostics(model: StackedEWS):
    """Host-side per-block parameter statistics for the eval log/results.

    Cheap at block widths <= ~64 (dense eigendecomposition per block)."""
    diag = {"A_frob": [], "A_radius": [], "B_frob": [], "gains": []}
    for block in model.blocks:
        A = np.asarray(block.ewms.A_param())
        if np.all(np.isfinite(A)):
            diag["A_frob"].append(round(float(np.linalg.norm(A)), 3))
            diag["A_radius"].append(round(float(np.abs(np.linalg.eigvals(A)).max()), 3))
        else:
            # Diverged. Record it and keep going: np.linalg would raise LinAlgError here and
            # lose the pre-divergence best_acc that the run had already earned.
            diag["A_frob"].append(float("nan"))
            diag["A_radius"].append(float("nan"))
        diag["B_frob"].append(
            round(float(np.linalg.norm(np.asarray(block.b_layer.matrix))), 3)
            if block.b_layer is not None
            else None
        )
        g = block.graded.gains if block.graded.gains is not None else block.graded.lam
        diag["gains"].append([round(float(x), 3) for x in np.atleast_1d(np.asarray(g))])
    short = f"A_F={diag['A_frob']} rad={diag['A_radius']} B_F={diag['B_frob']} gain={diag['gains']}"
    return diag, short


# -- Training ------------------------------------------------------------------


def train(cfg: Config, run: int, args, *, eval_callback=None, save_checkpoints: bool = True):
    tc = cfg.training
    seed = int(tc["seed"]) + run
    batch_size = int(args.batch_size or tc["batch_size"])
    num_steps = int(args.num_steps or tc["num_steps"])
    # Optuna search speedup: the schedule / LR curve below stays computed on num_steps (the full
    # horizon), but the training loop may stop early at max_steps, so the learning-rate value at
    # every step is identical to a full run, we just don't run the tail. Default = num_steps
    # (byte-identical when unset). Set via args.max_steps or training.max_train_steps.
    max_steps = int(getattr(args, "max_steps", None) or tc.get("max_train_steps") or num_steps)
    print_steps = int(args.print_steps or tc["print_steps"])
    lam = float(tc.get("vf_A_norm_lambda", 0.001))
    futility_step = (
        args.futility_step if args.futility_step is not None else tc.get("futility_step")
    )
    futility_min_acc = float(
        args.futility_acc if args.futility_acc is not None else tc.get("futility_min_acc", 0.0)
    )

    td = make_task_loaders(
        cfg.dataset["task"],
        batch_size,
        seed,
        length=cfg.dataset.get("length"),
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        val_batch_size=tc.get("val_batch_size"),
    )
    dense = td.loss_mask_is_dense
    print(
        f"Task: {td.task}, data_dim: {td.data_dim}, label_dim: {td.label_dim}, "
        f"train_pad: {td.train_pad}, val_pad: {td.val_pad}, seed: {seed}"
    )

    kwargs_list_probe = _per_block_kwargs(cfg)
    needs_raw = any(
        kw.get("clock_source") == "raw_input" and kw.get("clock_mode", "time") != "time"
        for kw in kwargs_list_probe
    )

    def raw_of(X: np.ndarray) -> Optional[np.ndarray]:
        if not needs_raw:
            return None
        return np.eye(td.data_dim, dtype=np.float32)[X]

    # first batch feeds the init calibrations, then training proper
    first = next(td.train_iter)
    X0 = first[0]
    m0 = first[2]
    valid0 = validity_mask(m0, dense)

    key = jr.PRNGKey(seed)
    key, model_key = jr.split(key)
    model, calibration = calibrate_and_build(
        cfg, td.data_dim, td.label_dim, model_key, X0, valid0, raw_of(X0), td.val_pad
    )
    init_from = getattr(args, "init_from", None) or tc.get("init_from")
    if init_from:
        # weight warm-start: array leaves come from the checkpoint; static
        # fields (clock constants, graded_eps, time_denominator) stay as
        # resolved for this task/length
        model = eqx.tree_deserialise_leaves(init_from, model)
        calibration["init_from"] = str(init_from)
        print(f"Initialized weights from checkpoint: {init_from}")
    n_params = sum(
        x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    )
    print(f"Parameters: {n_params}")

    params, static = eqx.partition(model, eqx.is_inexact_array)
    labels = jax.tree_util.tree_map(lambda _: "other", params)
    labels = eqx.tree_at(
        lambda t: t.embedding,
        labels,
        replace=jax.tree_util.tree_map(lambda _: "embedding", params.embedding),
    )
    # optional per-group lr multipliers for the mixer core (A drift / B lift);
    # with multipliers 1.0 the update math is identical to the 2-group setup
    lr_mult_A = float(tc.get("lr_mult_A", 1.0))
    lr_mult_B = float(tc.get("lr_mult_B", 1.0))
    for i, blk in enumerate(params.blocks):
        labels = eqx.tree_at(
            lambda t, i=i: t.blocks[i].ewms,
            labels,
            replace=jax.tree_util.tree_map(lambda _: "ews_A", blk.ewms),
        )
        if blk.b_layer is not None:
            labels = eqx.tree_at(
                lambda t, i=i: t.blocks[i].b_layer,
                labels,
                replace=jax.tree_util.tree_map(lambda _: "ews_B", blk.b_layer),
            )
    # training.weight_decay_A_damp (default off): give the A damping
    # vector v its own weight decay, separate from weight_decay_A which then applies to the
    # frequency vector u alone. Motivation: on mod_arith, wd on A is what pulls a plane's
    # damping toward the softplus(0) = 0.693 capture zone, but the same wd on u is not
    # implicated: one knob currently cannot express "decay the frequencies, leave the
    # damping alone". Only the rotation classes carry a `v` (bounded/unbounded_rotation have
    # no damping vector at all), hence the getattr guard.
    # absent key -> `labels` is not touched and the optimizer dict below is unchanged, so
    # every existing config trains byte-identically. Verified by ma_wdsplit_smoke.py.
    wd_A_damp = tc.get("weight_decay_A_damp")
    if wd_A_damp is not None:
        for i, blk in enumerate(params.blocks):
            if getattr(blk.ewms.A_param, "v", None) is not None:
                labels = eqx.tree_at(
                    lambda t, i=i: t.blocks[i].ewms.A_param.v, labels, replace="ews_A_damp"
                )
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=float(tc["learning_rate"]),
        warmup_steps=int(float(tc.get("warmup_fraction", 0.1)) * num_steps),
        decay_steps=num_steps,
        end_value=float(tc.get("final_lr", 1e-5)),
    )

    def group(wd, mult: float = 1.0):
        chain = []
        gc = float(tc.get("grad_clip", 0.0))
        if gc > 0:
            chain.append(optax.clip_by_global_norm(gc))
        sched = schedule if mult == 1.0 else (lambda s: schedule(s) * mult)
        chain.append(
            optax.adamw(
                sched,
                b1=float(tc.get("adam_b1", 0.9)),
                b2=float(tc.get("adam_b2", 0.999)),
                eps=float(tc.get("adam_eps", 1e-8)),
                weight_decay=wd,
            )
        )
        return optax.chain(*chain)

    wd_others = float(tc.get("weight_decay_others", 0.01))
    wd_A = float(tc.get("weight_decay_A", wd_others))
    _groups = {
        "embedding": group(float(tc.get("weight_decay_embedding", 0.0))),
        "other": group(wd_others),
        "ews_A": group(wd_A, lr_mult_A),
        "ews_B": group(wd_others, lr_mult_B),
    }
    # Fifth group only exists when the split is requested; see the labels block above.
    if wd_A_damp is not None:
        _groups["ews_A_damp"] = group(float(wd_A_damp), lr_mult_A)
    opt = optax.multi_transform(
        _groups,
        # Note: passed as a callable returning the prebuilt tree; an eqx.Module
        # pytree is callable, and optax would otherwise invoke the model itself.
        lambda _: labels,
    )
    opt_state = opt.init(params)
    fix_emb = bool(tc.get("fix_embedding_norm", False))
    emb_target = float(np.sqrt(int(cfg.model["hidden_dim"])))

    def _renorm_emb(p):
        # hard constraint: embedding rows kept at norm sqrt(H); removes the
        # (emb-scale x omega) flat direction entirely
        def fix(w):
            n = jnp.linalg.norm(w, axis=-1, keepdims=True)
            return w * (emb_target / jnp.maximum(n, 1e-8))

        return eqx.tree_at(lambda t: t.embedding.weight, p, replace_fn=fix)

    label_smooth = float(tc.get("label_smoothing", 0.0))

    @eqx.filter_jit
    def train_step(params, opt_state, X, y, lm, vm, key):
        def loss_fn(p):
            mdl = eqx.combine(p, static)
            logits = mdl(X, mask=vm, key=key, inference=False)
            return masked_ce(logits, y, lm, label_smooth) + lam * a_penalty(mdl)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        if fix_emb:
            params = _renorm_emb(params)
        return params, opt_state, loss

    @eqx.filter_jit
    def eval_step(params, X, y, lm, vm):
        mdl = eqx.combine(params, static)
        logits = mdl(X, mask=vm, inference=True)
        pred = jnp.argmax(logits, axis=-1)
        pos_hits = ((pred == y) & lm).sum(axis=0)  # (T,)
        pos_tot = lm.sum(axis=0)  # (T,)
        return pos_hits, pos_tot

    def evaluate(params):
        """Aggregate accuracy + per-position (token) accuracy over the loss mask.

        per_token lists positions with any supervised labels, in order (the
        histogram of final positions under final-position masks)."""
        pos_hits = pos_tot = None
        for Xv, yv, mv in eval_batches(td.val_loader):
            vv = validity_mask(mv, dense)
            h, t = eval_step(params, Xv, yv, mv, vv)
            h, t = np.asarray(h), np.asarray(t)
            if pos_hits is None:
                pos_hits, pos_tot = h.astype(np.int64), t.astype(np.int64)
            else:
                pos_hits += h
                pos_tot += t
        acc = pos_hits.sum() / max(pos_tot.sum(), 1)
        per_token = [round(float(hh) / float(tt), 4) for hh, tt in zip(pos_hits, pos_tot) if tt > 0]
        return float(acc), per_token

    tag = f"{td.task}_ews_run{run}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    out_dir = REPO_ROOT / "outputs" / "fl" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar = {
        "config": {"dataset": cfg.dataset, "model": cfg.model, "training": cfg.training},
        "run": run,
        "seed": seed,
        "data_dim": td.data_dim,
        "label_dim": td.label_dim,
        "calibration": calibration,
        "n_params": n_params,
    }
    with open(out_dir / "sidecar.json", "w") as f:
        json.dump(sidecar, f, indent=1)

    def batch_arrays(batch):
        X, y, lm = batch
        return (X, y, lm, validity_mask(lm, dense))

    results: List[Dict[str, Any]] = []
    best_acc = -1.0
    window_losses: List[float] = []
    t0 = time.time()
    early_stop = False
    futility = False

    # consume the calibration batch as training step 0's data
    pending = first
    for step_idx in range(max_steps):  # schedule uses num_steps; loop may cap early at max_steps
        if step_idx % print_steps == 0:
            acc, per_token = evaluate(params)
            dt = time.time() - t0
            avg = float(np.mean(window_losses)) if window_losses else float("nan")
            diag, diag_short = param_diagnostics(eqx.combine(params, static))
            print(
                f"Step: {step_idx}, Loss: {avg:.4f}, Validation Accuracy: {acc:.4f}, "
                f"Time: {dt:.2f}s | tok={per_token} | {diag_short}",
                flush=True,
            )
            window_losses = []
            results.append(
                {
                    "step": step_idx,
                    "val_acc": acc,
                    "avg_loss": avg,
                    "time_s": dt,
                    "per_token": per_token,
                    "diagnostics": diag,
                }
            )
            with open(out_dir / "results.json", "w") as f:
                json.dump(results, f, indent=1)
            if acc > best_acc:
                best_acc = acc
                if save_checkpoints:
                    eqx.tree_serialise_leaves(
                        out_dir / "checkpoint.eqx", eqx.combine(params, static)
                    )
            # training.checkpoint_every: also snapshot the
            # raw params at every eval step divisible by the given interval, as
            # checkpoint_step<N>.eqx. Eval-side only;
            # exists for trajectory probes (basin-commitment analysis), not for model selection.
            _ck_every = tc.get("checkpoint_every")
            if _ck_every and save_checkpoints and step_idx > 0 and step_idx % int(_ck_every) == 0:
                eqx.tree_serialise_leaves(
                    out_dir / f"checkpoint_step{step_idx}.eqx", eqx.combine(params, static)
                )
            if eval_callback is not None:
                eval_callback(step_idx, acc)
            if acc > float(tc.get("early_stop_threshold", 1.1)):
                early_stop = True
                break
            if futility_step and step_idx >= int(futility_step) and acc < futility_min_acc:
                futility = True
                print(
                    f"Futility stop at step {step_idx}: acc {acc:.4f} < {futility_min_acc}",
                    flush=True,
                )
                break
        batch = pending if pending is not None else next(td.train_iter)
        pending = None
        key, sk = jr.split(key)
        arrs = batch_arrays(batch)
        params, opt_state, loss = train_step(params, opt_state, *arrs, sk)
        window_losses.append(float(loss))
        if _GATE_DUMP and step_idx < _GATE_DUMP:
            print(f"GATE_LOSS {step_idx} {float(loss)!r}", flush=True)

    acc, per_token = evaluate(params)
    dt = time.time() - t0
    avg = float(np.mean(window_losses)) if window_losses else float("nan")
    diag, diag_short = param_diagnostics(eqx.combine(params, static))
    print(
        f"Step: {max_steps if not (early_stop or futility) else step_idx}, Loss: {avg:.4f}, "
        f"Validation Accuracy: {acc:.4f}, Time: {dt:.2f}s | tok={per_token} | {diag_short}",
        flush=True,
    )
    results.append(
        {
            "step": max_steps if not (early_stop or futility) else step_idx,
            "val_acc": acc,
            "avg_loss": avg,
            "time_s": dt,
            "per_token": per_token,
            "early_stop": early_stop,
            "futility_stop": futility,
            "diagnostics": diag,
        }
    )
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=1)
    if acc > best_acc:
        best_acc = acc
        if save_checkpoints:
            eqx.tree_serialise_leaves(out_dir / "checkpoint.eqx", eqx.combine(params, static))
    print(
        f"Best Validation Accuracy: {best_acc:.4f}, Early stop: {early_stop}, Futility stop: {futility}",
        flush=True,
    )
    print(f"Artifacts: {out_dir}", flush=True)
    threshold = float(tc.get("early_stop_threshold", 1.1))
    steps_to_threshold = next((r["step"] for r in results if r["val_acc"] > threshold), None)
    return {
        "best_acc": best_acc,
        "early_stop": early_stop,
        "futility": futility,
        "steps_to_threshold": steps_to_threshold,
        "out_dir": str(out_dir),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(parents=[_parser])
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run", type=int, default=0)
    from ews.determinism import add_switches

    add_switches(parser)
    # smoke-test conveniences (override the config / protocol sample counts)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--print-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--futility-step", type=int, default=None)
    parser.add_argument("--futility-acc", type=float, default=None)
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="checkpoint.eqx to warm-start weights from (overrides training.init_from)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, args.run, args)


if __name__ == "__main__":
    main()
