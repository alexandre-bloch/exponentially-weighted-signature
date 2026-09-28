"""Weather forecasting trainer (Informer weather, 720 -> 720) under the D-LinOSS training protocol.

MSE training loss (MAE with `training.loss = "l1"`), validation every `print_steps`, selection and early stopping on
validation MSE (MAE with `training.select = "val_mae"`), the best-validation checkpoint tested, AdamW, a step budget.
Three parameter groups: A (lr x `lr_mult_A`, linear warm-up over `a_warmup` steps, weight decay `wd_A`), the clock
(lr x `lr_mult_clock`) and the rest (lr, weight decay `wd`), each with global-norm clipping. By default autotuning is
off and deterministic ops are off, the regime of the banked runs.

    python experiments/weather/train.py --config weather_results/config.json --seed 2345 --gpu 0

Outputs go to <out>/seed_<seed>/: train.log, history.json, best.json (with the resolved config) and model.eqx.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", type=str, default=None)
_p, _ = _pre.parse_known_args()
if _p.gpu is not None:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = _p.gpu
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from ews.determinism import configure_from_cli as _configure_xla  # noqa: E402

_configure_xla(default_autotune=False, default_deterministic_ops=False)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)  # the ews core forces x64; training arrays stay f32

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from experiments.weather import model as M  # noqa: E402
from experiments.weather.data import create_weather  # noqa: E402


def _inner(b):
    return b.block if hasattr(b, "block") else b


def label_tree(model):
    """'A' for every block's ewms leaves, 'clock' for the clock leaves, 'rest' otherwise."""
    params = eqx.filter(model, eqx.is_inexact_array)
    labels = jax.tree_util.tree_map(lambda _: "rest", params)
    for i in range(len(model.blocks)):
        labels = eqx.tree_at(
            lambda t, i=i: _inner(t.blocks[i]).ewms,
            labels,
            replace=jax.tree_util.tree_map(lambda _: "A", _inner(params.blocks[i]).ewms),
        )
        if jax.tree_util.tree_leaves(
            eqx.filter(_inner(params.blocks[i]).clock, eqx.is_inexact_array)
        ):
            labels = eqx.tree_at(
                lambda t, i=i: _inner(t.blocks[i]).clock,
                labels,
                replace=jax.tree_util.tree_map(lambda _: "clock", _inner(params.blocks[i]).clock),
            )
    return labels


def load_config(path, overrides):
    cfg = json.load(open(path))
    for o in overrides:
        k, v = o.split("=", 1)
        sec, name = k.split(".", 1)
        cfg[sec][name] = json.loads(v) if v not in ("true", "false") else v == "true"
    return cfg


def run(cfg, seed, out_dir, eval_batches=None, log=print):
    d, mc, tc = cfg["data"], cfg["model"], cfg["training"]
    if mc["omega_scale"] > math.pi + 1e-6:
        log(
            f"WARNING omega_scale {mc['omega_scale']} > pi: planes above pi rad/step alias onto lower frequencies"
        )
    model_kw = dict(
        clock_mode=mc["clock"],
        a_param=mc["a_param"],
        omega_scale=mc["omega_scale"],
        damp_init_mean=mc["damp_init"],
        sigma_range=mc["sigma_range"],
        path_mode=mc["path_mode"],
        basepoint=mc["basepoint"],
    )
    is_jordan = mc["a_param"] == "jordan_repeated"
    if is_jordan:  # optional keys: a rotation-plane config needs none of them
        model_kw.update(
            jordan=mc["jordan"],
            init_variance=float(mc.get("init_variance", 1.0)),
            omega_log_uniform_hi=float(mc.get("omega_log_uniform_hi", 0.0)),
        )
    os.makedirs(out_dir, exist_ok=True)
    key = jr.PRNGKey(seed)
    calkey, modelkey, trainkey, key = jr.split(key, 4)
    splits, info = create_weather(d["seq_len"], d["pred_len"], d["include_time"], marks=d["marks"])
    Xc, _ = next(splits["train"].loop(32, key=calkey))
    c, cstats = M.calibrate_norm_constant(
        info["d_in"],
        info["d_out"],
        mc["hidden"],
        mc["num_blocks"],
        mc["m"],
        mc["depth"],
        Xc,
        modelkey,
        **model_kw,
    )
    log(f"clock norm_constant {c:.4g} (raw increments {cstats})")
    series_q = None
    if (
        mc["leaf"] == "series" and not is_jordan
    ):  # a Jordan A runs on the closed-form Jordan leaf, which has no order
        from ews.core.leaf import suggest_series_q

        series_q = int(
            min(
                mc["series_q_max"],
                suggest_series_q(3.0 * cstats["x_max_init"], mc["depth"], tol=1e-8),
            )
        )
    model = M.build(
        info["d_in"],
        info["d_out"],
        mc["hidden"],
        mc["num_blocks"],
        mc["m"],
        mc["depth"],
        c,
        mc["dropout"],
        modelkey,
        remat=True,
        leaf=mc["leaf"],
        series_q=series_q,
        **model_kw,
    )
    if not is_jordan:  # the plane kernel engine exists for rotation planes only
        model = M.swap_in_kernel_ews(model, chunk_target=mc["chunk"])
    n_params = sum(
        int(np.prod(x.shape))
        for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    )
    labels = label_tree(model)
    n_A = sum(
        int(np.prod(x.shape))
        for x, l in zip(
            jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array)),
            jax.tree_util.tree_leaves(labels),
        )
        if l == "A"
    )
    loss_kind, select = (
        tc.get("loss", "mse"),
        tc.get("select", "val_mse"),
    )  # optional keys: absent = MSE loss, val-MSE selection
    assert loss_kind in ("mse", "l1") and select in ("val_mse", "val_mae"), (loss_kind, select)
    log(
        f"params {n_params} (A group {n_A}); lr {tc['lr']:.3g}, lr_A x{tc['lr_mult_A']:g} (warmup {tc['a_warmup']}), "
        f"wd {tc['wd']} / wd_A {tc['wd_A']}, batch {tc['batch_size']}, loss {loss_kind}, select {select}, "
        f"patience {tc['patience']}"
    )

    def group(lr, wd, warmup=0):
        sched = (
            optax.linear_schedule(0.0, lr, int(warmup)) if warmup > 0 else lr
        )  # linear warm-up then constant
        base = (
            optax.adamw(learning_rate=sched, weight_decay=wd)
            if wd > 0
            else optax.adam(learning_rate=sched)
        )
        return (
            optax.chain(optax.clip_by_global_norm(tc["grad_clip"]), base)
            if tc["grad_clip"] > 0
            else base
        )

    opt = optax.multi_transform(
        {
            "rest": group(tc["lr"], tc["wd"]),
            "A": group(tc["lr"] * tc["lr_mult_A"], tc["wd_A"], tc["a_warmup"]),
            "clock": group(tc["lr"] * tc["lr_mult_clock"], 0.0),
        },
        lambda _: labels,
    )
    spec = jax.tree_util.tree_map(eqx.is_inexact_array, model)
    opt_state = opt.init(eqx.filter(model, spec))

    @eqx.filter_jit
    def step(model, opt_state, X, y, key):
        diff, static = eqx.partition(model, spec)

        def loss_fn(dd):
            mdl = eqx.combine(dd, static)
            pred = mdl(X, key=key, inference=False)[:, -d["pred_len"] :]
            if loss_kind == "l1":  # L1 / MAE training (the D-LinOSS authors' weather loss)
                return jnp.mean(jnp.abs(pred - y))
            return jnp.mean((pred - y) ** 2)

        value, grads = eqx.filter_value_and_grad(loss_fn)(diff)
        gl = jax.tree_util.tree_leaves(grads)
        ll = jax.tree_util.tree_leaves(labels)
        norms = {
            g: optax.global_norm([x for x, l in zip(gl, ll) if l == g])
            if any(l == g for l in ll)
            else jnp.zeros(())
            for g in ("A", "clock", "rest")
        }
        updates, opt_state = opt.update(grads, opt_state, params=diff)
        return eqx.apply_updates(model, updates), opt_state, value, norms

    resolved = dict(
        copy.deepcopy(cfg),
        seed=seed,
        norm_constant=c,
        params=n_params,
        series_q=series_q,
        clock_calibration=cstats,
        d_in=info["d_in"],
        d_out=info["d_out"],
    )
    batchkey, trainkey = jr.split(trainkey)
    running, history, best = 0.0, [], {}
    best_val, no_improve = float("inf"), 0
    gnorm_acc = {g: 0.0 for g in ("A", "clock", "rest")}
    start = time.time()
    train_s = 0.0  # training-only time of the current interval (eval excluded)
    status = "ok"
    eb = tc["batch_size"]
    for it, (X, y) in zip(
        range(tc["num_steps"]), splits["train"].loop(tc["batch_size"], key=batchkey)
    ):
        t0 = time.time()
        sk, trainkey = jr.split(trainkey)
        model, opt_state, value, norms = step(model, opt_state, jnp.asarray(X), jnp.asarray(y), sk)
        running += float(value)  # host read-back: blocks on the step
        train_s += time.time() - t0
        for g in gnorm_acc:
            gnorm_acc[g] += float(norms[g])
        if not np.isfinite(float(value)):
            log(f"NON-FINITE loss at step {it + 1}")
            status = "diverged"
            break
        if (it + 1) % tc["print_steps"] == 0:
            tr_mae, tr_mse, trainkey = M.eval_split(
                model, splits["train"], eb, trainkey, max_batches=50
            )
            va_mae, va_mse, trainkey = M.eval_split(
                model, splits["val"], eb, trainkey, max_batches=eval_batches
            )
            el = time.time() - start
            gn = {g: v / tc["print_steps"] for g, v in gnorm_acc.items()}
            log(
                f"step {it + 1} loss {running / tc['print_steps']:.4f} train {tr_mae:.4f}/{tr_mse:.4f} val {va_mae:.4f}/{va_mse:.4f} "
                f"(train {1e3 * train_s / tc['print_steps']:.1f} ms/step, {el:.0f}s total)"
            )
            rec = dict(
                step=it + 1,
                loss=running / tc["print_steps"],
                train_mae=tr_mae,
                train_mse=tr_mse,
                val_mae=va_mae,
                val_mse=va_mse,
                grad_norms=gn,
                train_s=train_s,
            )
            gnorm_acc = {g: 0.0 for g in gnorm_acc}
            crit = va_mae if select == "val_mae" else va_mse
            if crit < best_val:
                best_val, no_improve = crit, 0
                te_mae, te_mse, trainkey = M.eval_split(
                    model, splits["test"], eb, trainkey, max_batches=eval_batches
                )
                eqx.tree_serialise_leaves(os.path.join(out_dir, "model.eqx"), model)
                best = dict(
                    step=it + 1,
                    val_mae=va_mae,
                    val_mse=va_mse,
                    test_mae=te_mae,
                    test_mse=te_mse,
                    seed=seed,
                    config=resolved,
                )
                with open(os.path.join(out_dir, "best.json"), "w") as f:
                    json.dump(best, f, indent=2, default=str)
                rec.update(test_mae=te_mae, test_mse=te_mse)
                log(f"  test mae/mse {te_mae:.4f}/{te_mse:.4f}")
            else:
                no_improve += 1
            running = 0.0
            history.append(rec)
            with open(os.path.join(out_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=1)
            train_s = 0.0
            if (
                no_improve >= tc["patience"]
            ):  # D-LinOSS train.py: stop when the counter reaches patience
                log(f"early stop at {it + 1}")
                break
    best["status"] = status
    best.setdefault("seed", seed)
    best.setdefault("config", resolved)
    try:
        peak = jax.local_devices()[0].memory_stats()["peak_bytes_in_use"]
    except Exception:  # noqa: BLE001
        peak = float("nan")
    best["peak_bytes_in_use"] = peak
    with open(os.path.join(out_dir, "best.json"), "w") as f:
        json.dump(best, f, indent=2, default=str)
    log(
        f"FINAL status {status} test mae {best.get('test_mae')} mse {best.get('test_mse')} (best step {best.get('step')}, "
        f"val {best.get('val_mae')}/{best.get('val_mse')}) peak_gb {peak / 2**30:.2f}"
    )
    return best


def main(argv=None):
    from ews.determinism import add_switches

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True, help="JSON config (data / model / training sections)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--gpu", type=str, default=None)
    p.add_argument(
        "--out", default=None, help="output directory (default outputs/weather/<config stem>)"
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override a config value",
    )
    p.add_argument(
        "--eval_batches", type=int, default=None, help="smoke only: cap val/test batches"
    )
    add_switches(p)
    a = p.parse_args(argv)
    cfg = load_config(a.config, a.set)
    out = a.out or str(REPO_ROOT / "outputs" / "weather" / Path(a.config).stem)
    out_dir = os.path.join(out, f"seed_{a.seed}")
    tag = f"[weather seed {a.seed}]"
    run(
        cfg,
        a.seed,
        out_dir,
        eval_batches=a.eval_batches,
        log=lambda s: print(f"{tag} {s}", flush=True),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
