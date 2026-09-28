"""Train one model of the oscillatory-kernel experiment on one target.

    python experiments/sde/oscillatory_kernel/train.py --config <json> --seed 0 --gpu 0 --out_dir <dir>
    JAX_PLATFORMS=cpu python experiments/sde/oscillatory_kernel/train.py --config <json> --gpu "" --steps 50

Config: the standard sectioned layout (dataset, preprocessing, model, training; validated by
ews.config) plus an `experiment` section:
    init.rate_range      log-uniform range of every rate (EFM lambda, EWS damping)
    init.freq_range_hz   log-uniform range of the EWS frequencies, in Hz (sorted per model)
    lr_lift              learning rate of B (A uses training.lr_features, the readout training.lr_head)
    grad_clip_global     one global-norm clip over all parameters
    eval_batch_size, max_run_minutes, reduced_num_steps (a run projected to exceed max_run_minutes is cut to
    reduced_num_steps and flagged)

Loss: T1/T2 mean over outputs of MSE / Var_train(y_k); T3 sigmoid binary cross-entropy. Adam, cosine
decay to the min learning rates, early stopping on the validation loss, test at the best validation
checkpoint. Writes results.json (config, history, best checkpoint metrics, init and learned generator
parameters, wall clock) and model.eqx.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", type=str, default="")
_pre_args, _ = _pre.parse_known_args()
if _pre_args.gpu == "":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _pre_args.gpu
sys.path.insert(0, str(REPO_ROOT))
from ews.determinism import add_switches, configure_from_cli  # noqa: E402

configure_from_cli(default_autotune=True, default_deterministic_ops=False)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from ews.config import upgrade, validate  # noqa: E402
from ews.core.leaf import check_series_q  # noqa: E402
from ews.paths import prepare_paths_np, specs_from_config  # noqa: E402
from experiments.sde.oscillatory_kernel.data import (  # noqa: E402
    TARGET_DIMS,
    TARGET_KINDS,
    feature_count,
)
from experiments.sde.oscillatory_kernel.model import (  # noqa: E402
    OscModel,
    a_params_summary,
    build_model,
    model_kind,
    override_a_init,
    resolve_dtype,
    series_regime_x,
)

# ── config and data ───────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    cfg = upgrade(cfg)
    validate(cfg)
    if "experiment" not in cfg:
        raise ValueError("config needs an `experiment` section")
    return cfg


def load_dataset(file_path: str, target: str, preprocessing: dict, dtype):
    raw = np.load(file_path)
    path_spec, clock_spec, append_spec = specs_from_config(preprocessing)
    out = {}
    for split in ("train", "val", "test"):
        paths = [p for p in raw[f"{split}_paths"]]
        clocks = [c for c in raw[f"{split}_clocks"]]
        paths_j, clocks_j, _, d_path = prepare_paths_np(
            paths,
            clocks,
            path_spec,
            clock_spec,
            append_spec,
            scaler=None,
            fit_scaler=False,
            dtype=dtype,
        )
        if d_path != 2:
            raise ValueError(f"expected a 2-channel path, got {d_path}")
        out[split] = {
            "paths": jnp.stack(paths_j),
            "clocks": jnp.stack(clocks_j),
            "targets": jnp.asarray(raw[f"{split}_{target}"], dtype=dtype),
        }
    return out


# ── loss and metrics ──────────────────────────────────────────────────────────


def make_loss(kind: str, inv_var):
    if kind == "regression":

        def loss(pred, y):
            return jnp.mean((pred - y) ** 2 * inv_var)

    else:

        def loss(pred, y):
            return jnp.mean(optax.sigmoid_binary_cross_entropy(pred, y))

    return loss


def split_metrics(model: OscModel, split: dict, kind: str, inv_var, batch: int) -> dict:
    fn = eqx.filter_jit(lambda m, p, c: m(p, c))
    preds = []
    n = split["paths"].shape[0]
    for i in range(0, n, batch):
        preds.append(
            np.asarray(
                fn(model, split["paths"][i : i + batch], split["clocks"][i : i + batch]),
                dtype=np.float64,
            )
        )
    pred = np.concatenate(preds, axis=0)
    y = np.asarray(split["targets"], dtype=np.float64)
    if kind == "regression":
        mse = ((pred - y) ** 2).reshape(-1, y.shape[-1]).mean(axis=0)
        var_self = y.reshape(-1, y.shape[-1]).var(axis=0)
        return {
            "loss": float((mse * np.asarray(inv_var)).mean()),  # MSE / Var_train, the training loss
            "nmse": float((mse / var_self).mean()),  # MSE / Var of this split, the reported metric
            "rmse": float(np.sqrt(mse.mean())),
            "nmse_per_output": (mse / var_self).tolist(),
        }
    p = 1.0 / (1.0 + np.exp(-pred))
    bce = -(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))).mean()
    hat = (p >= 0.5).astype(np.float64)
    tp = float((hat * y).sum())
    fp = float((hat * (1 - y)).sum())
    fn_ = float(((1 - hat) * y).sum())
    f1 = 2 * tp / max(2 * tp + fp + fn_, 1.0)
    return {"loss": float(bce), "bce": float(bce), "acc": float((hat == y).mean()), "f1": float(f1)}


# ── optimiser ─────────────────────────────────────────────────────────────────


def make_optimizer(model: OscModel, cfg: dict, total_steps: int):
    tc, ec = cfg["training"], cfg["experiment"]
    params = eqx.filter(model, eqx.is_array)
    labels = jax.tree_util.tree_map(lambda _: "frozen", params)
    learn_A = bool(cfg["model"].get("A", {}).get("learn", True))
    learn_B = bool(cfg["model"].get("B", {}).get("learn", True))
    if learn_A and jax.tree_util.tree_leaves(eqx.filter(model.ews, eqx.is_array)):
        labels = eqx.tree_at(
            lambda m: m.ews,
            labels,
            replace=jax.tree_util.tree_map(
                lambda _: "generator", eqx.filter(model.ews, eqx.is_array)
            ),
        )
    if learn_B:
        labels = eqx.tree_at(
            lambda m: m.lift,
            labels,
            replace=jax.tree_util.tree_map(lambda _: "lift", eqx.filter(model.lift, eqx.is_array)),
        )
    labels = eqx.tree_at(
        lambda m: m.head,
        labels,
        replace=jax.tree_util.tree_map(lambda _: "readout", eqx.filter(model.head, eqx.is_array)),
    )
    warm = int(tc.get("warmup_steps", 0))

    def sched(peak, end):
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0 if warm > 0 else float(peak),
            peak_value=float(peak),
            warmup_steps=warm,
            decay_steps=max(total_steps - warm, 1),
            end_value=float(end),
        )

    lr_gen = float(tc.get("lr_features", 1e-3))
    lr_head = float(tc.get("lr_head", 2e-3))
    lr_lift = float(ec.get("lr_lift", lr_head))
    end_gen = float(tc.get("min_lr_features", 0.0))
    end_head = float(tc.get("min_lr_head", 0.0))
    opt = optax.chain(
        optax.clip_by_global_norm(float(ec.get("grad_clip_global", 1.0))),
        optax.multi_transform(
            {
                "generator": optax.adam(sched(lr_gen, end_gen)),
                "lift": optax.adam(sched(lr_lift, end_head)),
                "readout": optax.adam(sched(lr_head, end_head)),
                "frozen": optax.set_to_zero(),
            },
            lambda _: labels,
        ),
    )
    return opt, opt.init(params)


# ── training ──────────────────────────────────────────────────────────────────


def train(
    config_path: str, seed=None, out_dir=None, dataset_file=None, target=None, steps=None
) -> dict:
    t_start = time.time()
    cfg = load_config(config_path)
    tc, ec = cfg["training"], cfg["experiment"]
    target = target or cfg["dataset"].get("target", "T1")
    kind = TARGET_KINDS[target]
    n_out = TARGET_DIMS[target]
    seed = int(tc.get("seed", 0)) if seed is None else int(seed)
    num_steps = int(tc.get("num_steps", 12000)) if steps is None else int(steps)
    eval_every = min(int(tc.get("eval_every", 500)), num_steps)
    log_every = int(tc.get("log_every", 500))
    batch_size = int(tc.get("batch_size", 64))
    patience = int(tc.get("early_stopping_patience", 0))
    min_delta = float(tc.get("early_stopping_min_delta", 0.0))
    eval_batch = int(ec.get("eval_batch_size", 250))
    dtype = resolve_dtype(cfg["preprocessing"].get("dtype", "float32"))

    file_path = dataset_file or cfg["dataset"]["file"]
    if not os.path.isabs(file_path):
        file_path = str(REPO_ROOT / file_path)
    data = load_dataset(file_path, target, cfg["preprocessing"], dtype)
    train_split = data["train"]
    n_train = train_split["paths"].shape[0]
    dtheta = float(train_split["clocks"][0, 1] - train_split["clocks"][0, 0])
    y_train = np.asarray(train_split["targets"], dtype=np.float64).reshape(-1, n_out)
    inv_var = jnp.asarray(1.0 / y_train.var(axis=0), dtype=dtype) if kind == "regression" else None

    key = jr.PRNGKey(seed)
    k_model, k_init, k_batch = jr.split(key, 3)
    model = build_model(
        cfg["model"], n_out, k_model, dtype, readout_init=ec.get("readout_init", "zero")
    )
    model = override_a_init(model, ec["init"], k_init)
    init_params = a_params_summary(model)
    m = model.ews.d
    depth = model.ews.N
    D = feature_count(m, depth)
    kind_name = model_kind(cfg["model"])
    q = cfg["model"].get("compute", {}).get("series_q")

    opt, opt_state = make_optimizer(model, cfg, num_steps)
    loss_fn = make_loss(kind, inv_var)

    @eqx.filter_jit
    def step(model, opt_state, bp, bc, by):
        def objective(mdl):
            return loss_fn(mdl(bp, bc), by)

        loss, grads = eqx.filter_value_and_grad(objective)(model)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), opt_state, loss

    def batches():
        k = k_batch
        while True:
            k, sub = jr.split(k)
            perm = np.asarray(jr.permutation(sub, n_train))
            for i in range(0, n_train - batch_size + 1, batch_size):
                idx = perm[i : i + batch_size]
                yield (
                    train_split["paths"][idx],
                    train_split["clocks"][idx],
                    train_split["targets"][idx],
                )

    def regime(mdl):
        x = series_regime_x(mdl, dtheta)
        bound = None
        if q is not None and x > 0:
            _, bound, _ = check_series_q(
                mdl.ews.A_param(),
                dtheta,
                depth,
                int(q),
                tol=1e-6,
                where=f"{kind_name} n{depth} m{m}",
            )
            bound = float(bound)
        return x, bound

    print(
        f"target={target} ({kind})  model={kind_name}  n={depth}  m={m}  D={D}  seed={seed}  steps={num_steps}"
    )
    print(f"init A: {init_params}")
    x0, b0 = regime(model)
    print(f"series regime x={x0:.3f}  bound={b0}")

    history = []
    best = None
    best_model = model
    steps_no_improve = 0
    reduced = False
    it = batches()
    t_train = time.time()
    t_after_first = None  # set once the first (compiling) step has finished
    s = 0
    while s < num_steps:  # num_steps may be lowered by the kill rule below
        s += 1
        bp, bc, by = next(it)
        model, opt_state, loss = step(model, opt_state, bp, bc, by)
        if s == 1:
            jax.block_until_ready(loss)
            t_after_first = time.time()
            print(f"first step (compile) {t_after_first - t_train:.1f} s")
        if s % log_every == 0:
            jax.block_until_ready(loss)
            print(
                f"step {s:6d}  train loss {float(loss):.4e}  ({(time.time() - t_after_first) / max(s - 1, 1) * 1000:.1f} ms/step)"
            )
        if s % eval_every == 0 or s == num_steps:
            val = split_metrics(model, data["val"], kind, inv_var, eval_batch)
            test = split_metrics(model, data["test"], kind, inv_var, eval_batch)
            x, bound = regime(model)
            rec = {
                "step": s,
                "train_loss": float(loss),
                "val": val,
                "test": test,
                "series_x": x,
                "series_bound": bound,
                "elapsed_s": time.time() - t_start,
            }
            history.append(rec)
            print(f"   eval step {s}: val {val}  test {test}  x={x:.3f}")
            improved = best is None or val["loss"] < best["val"]["loss"] - min_delta
            if improved:
                best = {"step": s, "val": val, "test": test, "series_x": x}
                best_model = model
                steps_no_improve = 0
            else:
                steps_no_improve += eval_every
                if patience > 0 and steps_no_improve >= patience:
                    print(f"early stopping at step {s}")
                    break
            # kill rule: a run projected past max_run_minutes is cut to reduced_num_steps
            # (projected from the steady-state rate after the compiling first step, evals included)
            if not reduced and s == eval_every and steps is None:
                projected = (time.time() - t_after_first) / max(s - 1, 1) * num_steps / 60.0
                if projected > float(ec.get("max_run_minutes", 10.0)):
                    num_steps = min(num_steps, int(ec.get("reduced_num_steps", 6000)))
                    reduced = True
                    print(
                        f"projected {projected:.1f} min > limit: steps reduced to {num_steps} (flagged)"
                    )

    wall = time.time() - t_start
    learned = a_params_summary(best_model)
    out_dir = out_dir or str(
        REPO_ROOT
        / "outputs"
        / "oscillatory_kernel"
        / target
        / f"{kind_name}_n{depth}_m{m}"
        / f"seed{seed}"
    )
    os.makedirs(out_dir, exist_ok=True)
    record = {
        "target": target,
        "kind": kind,
        "model": kind_name,
        "n": depth,
        "m": m,
        "D": D,
        "seed": seed,
        "num_steps_planned": int(tc.get("num_steps", 12000)) if steps is None else int(steps),
        "num_steps_run": history[-1]["step"] if history else 0,
        "steps_reduced": reduced,
        "dataset": file_path,
        "config": cfg,
        "evals": history,
        "best": best,
        "A_init": init_params,
        "A_learned": learned,
        "B_learned": np.asarray(best_model.lift.matrix, dtype=np.float64).tolist(),
        "wall_time_s": wall,
        "compile_s": (t_after_first - t_train) if t_after_first else None,
        "ms_per_step": (time.time() - (t_after_first or t_train))
        / max((history[-1]["step"] if history else 1) - 1, 1)
        * 1000,
        "device": str(jax.devices()[0].platform),
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(record, f, indent=1)
    eqx.tree_serialise_leaves(os.path.join(out_dir, "model.eqx"), best_model)
    print(f"best step {best['step']}  val {best['val']}  test {best['test']}")
    print(f"learned A: {learned}")
    print(f"wall {wall / 60:.1f} min  ->  {out_dir}")
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Oscillatory-kernel trainer")
    p.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "experiments/sde/oscillatory_kernel/configs/template.json"),
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--target", type=str, default=None, choices=[None, "T1", "T2", "T3"])
    p.add_argument("--dataset_file", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--steps", type=int, default=None, help="override num_steps (smoke tests)")
    p.add_argument("--gpu", type=str, default="", help="CUDA device; empty = CPU")
    add_switches(p)
    a = p.parse_args()
    train(a.config, a.seed, a.out_dir, a.dataset_file, a.target, a.steps)
