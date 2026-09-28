"""Same-time regression on paths: a target value at every step, read from the EWS prefix signature.

Usage:
    python experiments/sde/train.py --config sde_results/oscillator/ews/config.json --gpu 0
    python experiments/sde/train.py --config <config> --dataset_file data/sde_a/sde_a_seed3.npz --seed 103 --out_dir <dir>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# GPU and XLA configuration from the command line, before JAX is imported
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--gpu", type=str, default="2")
_pre, _ = _parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _pre.gpu
from ews.determinism import configure_from_cli as _configure_xla

_configure_xla(default_autotune=True, default_deterministic_ops=False)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys

sys.path.insert(0, str(REPO_ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from ews.paths import specs_from_config
from experiments.regression_common import (
    Config,
    build_model,
    create_batch_iterator,
    extract_A,
    extract_B,
    flatten_head_params,
    group_by_length,
    hierarchical_rmse,
    load_config,
    make_optimizer,
    resolve_dtype,
    save_run,
    split_rmse,
)
from experiments.sde.loader import load_path_regression_dataset

__all__ = ["Config", "build_model", "load_config", "train"]


def _aligner(skip_first: bool):
    """With a basepoint the model emits one extra leading prediction (basepoint to t0)."""
    if skip_first:
        return lambda preds: preds[:, 1:]
    return lambda preds: preds


def compute_val_rmse(model, paths, clocks, targets, skip_first: bool) -> float:
    return split_rmse(model, group_by_length(paths, clocks, targets), _aligner(skip_first))


def train(
    config_path: str,
    dataset_file: str | None = None,
    seed: int | None = None,
    out_dir: str | None = None,
):
    cfg = load_config(config_path)

    dtype = resolve_dtype(cfg.preprocessing.get("dtype", "float32"))
    if dtype == jnp.float64:
        jax.config.update("jax_enable_x64", True)

    file_path = dataset_file or cfg.data.get("file")
    if not file_path:
        raise ValueError("No dataset file: set dataset.file in the config or pass --dataset_file.")
    if not os.path.isabs(file_path):
        file_path = str(REPO_ROOT / file_path)

    print(f"Loading dataset from {file_path} ...")
    dataset = load_path_regression_dataset(file_path, cfg.preprocessing)
    train_paths = dataset["train"]["paths"]
    train_clocks = dataset["train"]["clocks"]
    train_targets = dataset["train"]["targets"]
    val_paths, val_clocks, val_targets = (dataset["val"][k] for k in ("paths", "clocks", "targets"))
    test_paths, test_clocks, test_targets = (
        dataset["test"][k] for k in ("paths", "clocks", "targets")
    )
    meta = dataset["metadata"]
    d_in = meta["d_path"]
    target_scaler = meta["target_scaler"]

    # With a basepoint the targets start at t1: drop the target at t0 and the prediction for it.
    skip_first = specs_from_config(cfg.preprocessing)[0].basepoint
    if skip_first:
        train_targets = [t[1:] for t in train_targets]
        val_targets = [t[1:] for t in val_targets]
        test_targets = [t[1:] for t in test_targets]
    align = _aligner(skip_first)
    val_groups = group_by_length(val_paths, val_clocks, val_targets)
    test_groups = group_by_length(test_paths, test_clocks, test_targets)

    print(f"  {meta['n_train']} train / {meta['n_val']} val / {meta['n_test']} test")
    print(
        f"  d_in={d_in},  path length={train_paths[0].shape[0]},  target length={train_targets[0].shape[0]}"
    )

    tc = cfg.training
    num_steps = int(tc.get("num_steps", 2000))
    eval_every = int(tc.get("eval_every", 500))
    log_every = int(tc.get("log_every", 100))
    batch_size = int(tc.get("batch_size", 32))
    patience = int(tc.get("early_stopping_patience", 0))
    min_delta = float(tc.get("early_stopping_min_delta", 0.0))
    lambda1 = float(tc.get("lambda1", 0.0))
    lambda2 = float(tc.get("lambda2", 0.0))
    seed = int(tc.get("seed", 42)) if seed is None else int(seed)

    key = jr.PRNGKey(seed)
    model = build_model(cfg, d_in, key)
    opt, opt_state = make_optimizer(model, cfg, num_steps)

    A_init_snap = extract_A(model)
    B_init_snap = extract_B(model)
    if A_init_snap is not None:
        print(f"Initial A:\n{A_init_snap}")
    if B_init_snap is not None:
        print(f"Initial B:\n{B_init_snap}")
    batch_iter = create_batch_iterator(train_paths, train_clocks, train_targets, batch_size, key)

    @eqx.filter_jit
    def step(model, opt_state, batch_paths, batch_clocks, batch_targets):
        def loss_fn(m):
            preds = align(m(batch_paths, batch_clocks))
            rmse = hierarchical_rmse(preds, batch_targets)
            hp = flatten_head_params(m.head)
            return rmse + lambda1 * jnp.sum(jnp.abs(hp)) + lambda2 * jnp.sum(hp**2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = opt.update(
            eqx.filter(grads, eqx.is_array),
            opt_state,
            params=eqx.filter(model, eqx.is_array),
        )
        return eqx.apply_updates(model, updates), opt_state, loss

    print("=" * 70)
    print("Training: same-time regression (experiments/sde)")
    print("=" * 70)
    print(f"batch={batch_size}  steps={num_steps}  eval_every={eval_every}  seed={seed}")

    history = []
    best_val, best_test, best_step, best_model = None, None, None, None
    steps_no_improve = 0

    for step_idx in range(1, num_steps + 1):
        batch_paths, batch_clocks, batch_targets = next(batch_iter)
        model, opt_state, loss = step(model, opt_state, batch_paths, batch_clocks, batch_targets)
        jax.block_until_ready(loss)

        if step_idx % log_every == 0:
            print(f"Step {step_idx:5d}  train RMSE {float(loss):.6f}")

        if step_idx % eval_every == 0:
            val_rmse = split_rmse(model, val_groups, align)
            test_rmse = split_rmse(model, test_groups, align)
            print(f"         val  RMSE {val_rmse:.6f}   test RMSE {test_rmse:.6f}")
            history.append(
                {
                    "step": step_idx,
                    "train_loss": float(loss),
                    "val_rmse": val_rmse,
                    "test_rmse": test_rmse,
                }
            )

            A_now = extract_A(model)
            if A_now is not None and A_init_snap is not None:
                diff = A_now - A_init_snap
                print(
                    f"         A: ||A||={float(jnp.linalg.norm(A_now)):.4g}  "
                    f"||ΔA||={float(jnp.linalg.norm(diff)):.4g}"
                )

            improved = best_val is None or val_rmse < best_val - min_delta
            if improved:
                best_val, best_test, best_step, best_model = val_rmse, test_rmse, step_idx, model
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

    print("=" * 70)
    print(f"Best  step={best_step}  val={best_val:.6f}  test={best_test:.6f}  (normalised)")
    target_scale = None
    if target_scaler is not None:
        target_scale = float(target_scaler.data_max - target_scaler.data_min)
        print(
            f"      (raw units)  val={best_val * target_scale:.6f}  test={best_test * target_scale:.6f}"
        )
    print("=" * 70)

    if out_dir is None:
        out_dir = str(REPO_ROOT / "outputs" / "sde" / Path(config_path).stem / f"seed{seed}")
    ckpt = save_run(
        out_dir,
        cfg,
        seed,
        file_path,
        history,
        {"step": best_step, "val_rmse": best_val, "test_rmse": best_test},
        best_model,
        A_init=A_init_snap,
        extra={"target_scale": target_scale},
    )
    print(f"Saved best model → {ckpt}")

    return best_model, best_val, best_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Same-time regression trainer")
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "experiments" / "sde" / "configs" / "template.json"),
    )
    parser.add_argument("--dataset_file", type=str, default=None, help="Override dataset.file")
    parser.add_argument("--gpu", type=str, default="2")
    parser.add_argument("--seed", type=int, default=None, help="Override training.seed")
    parser.add_argument(
        "--out_dir", type=str, default=None, help="Where results.json and model.eqx are written"
    )
    from ews.determinism import add_switches

    add_switches(parser)
    args = parser.parse_args()

    train(args.config, args.dataset_file, args.seed, args.out_dir)
