"""EWS on PPG-DaLiA (heart-rate regression) under the LinOSS / D-LinOSS training protocol.

    python experiments/ppg/train.py --config ppg_results/config.json --seed 2345 --gpu 0 [--out DIR] [--set a.b=v]

A stack of EWS blocks (experiments/weather/model.py) on the 49,920-step windows of 6 channels plus a time channel;
Walker's head reads every 128th position through tanh to give the 390 heart-rate targets, and the loss is their MSE.
Adam at a constant learning rate, a 100,000-step budget, validation every 1,000 steps on the whole validation set,
early stopping after 10 evaluations without improvement, the best-validation model tested. Outputs (best.json with
the resolved config, history.json, model.eqx) go to --out/seed_<seed>/.
"""

import argparse
import json
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)  # the ews core forces x64; training arrays stay f32

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from experiments.ppg.data import create_ppg  # noqa: E402
from experiments.weather.model import (  # noqa: E402
    build,
    calibrate_norm_constant,
    swap_in_kernel_ews,
)

DEFAULTS = {
    "data": {"variant": "walker", "include_time": True, "T": 1.0, "output_step": 128},
    "model": {
        "depth": 1,
        "m": 64,
        "hidden": 16,
        "num_blocks": 4,
        "a_param": "damped_rotation",
        "clock": "clamped_gate",
        "omega_scale": 1.0,
        "damp_init": -2.0,
        "sigma_range": 0.25,
        "leaf": "series",
        "series_q_max": 48,
        "engine": "kernel",
        "chunk": 1024,
        "dropout": 0.05,
        "remat": True,
    },
    "training": {
        "lr": 1e-3,
        "wd": 0.0,
        "schedule": "constant",
        "warmup": 0.1,
        "grad_clip": 0.0,
        "batch": 8,
        "micro_batch": 0,
        "num_steps": 100_000,
        "print_steps": 1000,
        "patience": 10,
        "eval_batches": None,
    },
}


def load_config(path, overrides, cli):
    cfg = json.loads(json.dumps(DEFAULTS))
    for sec, vals in json.load(open(path)).items():
        cfg.setdefault(sec, {}).update(vals)
    for kv in overrides:
        k, v = kv.split("=", 1)
        sec, key = k.split(".", 1)
        old = cfg[sec].get(key)
        cfg[sec][key] = (
            type(old)(v)
            if isinstance(old, (int, float)) and not isinstance(old, bool)
            else v.lower() in ("1", "true", "yes")
            if isinstance(old, bool)
            else v
        )
    for key in ("num_steps", "print_steps", "batch", "eval_batches"):
        if getattr(cli, key) is not None:
            cfg["training"][key] = getattr(cli, key)
    return cfg


def series_regime(model, X, series_q, depth):
    """x_now = max_j |z_j| over the 2x2 planes (no eigendecomposition) x max block-1 clock step on a batch; returns
    (x, error bound (2x)^(q+1)/(q+1)! e^(2x)): a check that the series leaf still holds during training."""
    from ews.core.leaf import series_error_bound

    inner = lambda b: b.block if hasattr(b, "block") else b  # noqa: E731
    a_norm = 0.0
    for b in model.blocks:
        A = np.asarray(inner(b).ewms.A_param(), np.float64)
        ii = np.arange(0, A.shape[0], 2)
        off = A.copy()
        off[ii, ii] = 0
        off[ii, ii + 1] = 0
        off[ii + 1, ii] = 0
        off[ii + 1, ii + 1] = 0
        a_norm = max(
            a_norm,
            float(np.max(np.sqrt(A[ii, ii] ** 2 + A[ii, ii + 1] ** 2)))
            if np.max(np.abs(off)) < 1e-12
            else float(np.linalg.norm(A)),
        )
    enc = jax.vmap(jax.vmap(model.embedding))(jnp.asarray(X, jnp.float32))
    blk = inner(model.blocks[0])
    t = jnp.arange(X.shape[1], dtype=jnp.float32)
    theta = jax.vmap(lambda drv: blk.clock(t, drv, None))(enc)
    dmax = float(jnp.max(jnp.diff(theta, axis=1)))
    x = a_norm * dmax
    return x, (
        float(series_error_bound(x, int(depth), int(series_q))) if series_q else float("nan")
    )


def run(cfg, seed, out_dir, log=print):
    d, m, t = cfg["data"], cfg["model"], cfg["training"]
    os.makedirs(out_dir, exist_ok=True)
    model_kw = dict(
        clock_mode=m["clock"],
        a_param=m["a_param"],
        omega_scale=m["omega_scale"],
        damp_init_mean=m["damp_init"],
        sigma_range=m["sigma_range"],
    )
    if m["a_param"] == "jordan_repeated":
        # uniform Jordan chains (model.jordan: "complex4", "real2", ...) on the generic engine, with the block
        # kernel's flows in closed form
        if m["engine"] == "kernel":
            raise ValueError(
                "a Jordan A needs model.engine = 'generic' (the kernel engine is rotation planes only)"
            )
        model_kw.update(
            jordan=m["jordan"],
            init_variance=float(m.get("init_variance", 1.0)),
            omega_log_uniform_hi=float(m.get("omega_log_uniform_hi", 0.0)),
            jordan_flows=True,
        )
    key = jr.PRNGKey(seed)
    calkey, modelkey, trainkey, key = jr.split(key, 4)
    loaders, info = create_ppg(variant=d["variant"], include_time=d["include_time"], T=d["T"])
    out_step = d["output_step"]

    @eqx.filter_jit
    def forward(model, X, key, inference):
        out = model(X, key=key, inference=inference)  # (B, L, 1)
        return jnp.tanh(out[:, out_step - 1 :: out_step, 0])  # (B, 390), Walker's head

    def epoch_mse(model, loader, batch, key, max_batches=None):
        preds, labels = [], []
        for bi, (X, y) in enumerate(loader.loop_epoch(batch)):
            if max_batches is not None and bi >= max_batches:
                break
            key, sk = jr.split(key)
            preds.append(np.asarray(forward(model, jnp.asarray(X), sk, True), np.float64))
            labels.append(y.astype(np.float64))
        preds, labels = np.concatenate(preds), np.concatenate(labels)
        return float(np.mean((preds - labels) ** 2)), key

    Xc, _ = next(loaders["train"].loop(4, key=calkey))
    c, cstats = calibrate_norm_constant(
        info["d_in"], 1, m["hidden"], m["num_blocks"], m["m"], m["depth"], Xc, modelkey, **model_kw
    )
    tag = f"[ppg ews n={m['depth']} m={m['m']} seed {seed}]"
    log(
        f"{tag} L={info['L']} d_in={info['d_in']} clock norm_constant {c:.4g} (raw increments {cstats})"
    )
    if m["leaf"] == "series":
        from ews.core.leaf import suggest_series_q

        series_q = int(
            min(
                m["series_q_max"],
                suggest_series_q(3.0 * cstats["x_max_init"], m["depth"], tol=1e-8),
            )
        )
        log(f"{tag} series leaf q={series_q} (x_max at init {cstats['x_max_init']:.3g}, margin 3x)")
    else:
        series_q = None
    model = build(
        info["d_in"],
        1,
        m["hidden"],
        m["num_blocks"],
        m["m"],
        m["depth"],
        c,
        m["dropout"],
        modelkey,
        remat=bool(m["remat"]),
        leaf=m["leaf"],
        series_q=series_q,
        **model_kw,
    )
    if m["engine"] == "kernel":
        model = swap_in_kernel_ews(model, chunk_target=int(m["chunk"]))
        log(
            f"{tag} engine = kernel (plane closed forms + fused block kernel, chunk target {m['chunk']})"
        )
    n_params = sum(
        int(np.prod(x.shape))
        for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    )
    log(f"{tag} params {n_params}")
    spec = jax.tree_util.tree_map(eqx.is_inexact_array, model)
    lr = t["lr"]
    if t["schedule"] == "cosine":
        lr = optax.warmup_cosine_decay_schedule(
            0.0, t["lr"], int(t["warmup"] * t["num_steps"]), t["num_steps"], 1e-5
        )
    base_opt = (
        optax.adamw(learning_rate=lr, weight_decay=t["wd"])
        if t["wd"] > 0
        else optax.adam(learning_rate=lr)
    )
    opt = (
        optax.chain(optax.clip_by_global_norm(t["grad_clip"]), base_opt)
        if t["grad_clip"] > 0
        else base_opt
    )
    opt_state = opt.init(eqx.filter(model, spec))

    @eqx.filter_jit
    @eqx.filter_value_and_grad
    def loss_fn(diff, static, X, y, key):
        mdl = eqx.combine(diff, static)
        out = mdl(X, key=key, inference=False)
        pred = jnp.tanh(out[:, out_step - 1 :: out_step, 0])
        return jnp.mean(jnp.mean((pred - y) ** 2, axis=1))

    micro = int(t["micro_batch"] or 0)
    n_micro = max(1, t["batch"] // micro) if micro else 1

    @eqx.filter_jit
    def step(model, opt_state, X, y, key):
        diff, static = eqx.partition(model, spec)
        if n_micro == 1:
            value, grads = loss_fn(diff, static, X, y, key)
        else:
            value, grads = 0.0, None
            for i, k in enumerate(jr.split(key, n_micro)):
                sl = slice(i * micro, (i + 1) * micro)
                v_i, g_i = loss_fn(diff, static, X[sl], y[sl], k)
                value = value + v_i / n_micro
                grads = (
                    g_i if grads is None else jax.tree_util.tree_map(lambda a, b: a + b, grads, g_i)
                )
            grads = jax.tree_util.tree_map(lambda g: g / n_micro, grads)
        updates, opt_state = opt.update(grads, opt_state, params=diff)
        return eqx.apply_updates(model, updates), opt_state, value

    batch = int(t["batch"])
    eb = micro or batch
    batchkey, trainkey = jr.split(trainkey)
    running, history, best = 0.0, [], {}
    best_val, no_improve = float("inf"), 0
    resolved = dict(
        data=d,
        model=m,
        training=t,
        seed=seed,
        norm_constant=c,
        params=n_params,
        series_q=series_q,
        clock_calibration=cstats,
        d_in=info["d_in"],
        L=info["L"],
    )
    start = time.time()
    train_s = 0.0  # training-only seconds of the current interval (evaluations excluded)
    for it, (X, y) in zip(range(int(t["num_steps"])), loaders["train"].loop(batch, key=batchkey)):
        t0 = time.time()
        sk, trainkey = jr.split(trainkey)
        model, opt_state, value = step(model, opt_state, jnp.asarray(X), jnp.asarray(y), sk)
        running += float(value)  # host read-back: blocks on the step
        train_s += time.time() - t0
        if not np.isfinite(float(value)):
            log(f"{tag} NON-FINITE loss at step {it + 1}")
            break
        if (it + 1) % int(t["print_steps"]) == 0:
            log(
                f"TrainTime: step {it + 1} secs {train_s:.4f}"
            )  # training-only seconds of this interval (the timing harness reads it)
            train_s = 0.0
            tr, trainkey = epoch_mse(
                model, loaders["train"], eb, trainkey, max_batches=t["eval_batches"]
            )
            va, trainkey = epoch_mse(
                model, loaders["val"], eb, trainkey, max_batches=t["eval_batches"]
            )
            el = time.time() - start
            log(
                f"{tag} step {it + 1} loss {running / t['print_steps']:.4f} train mse {tr:.4f} val mse {va:.4f} "
                f"({el:.0f}s, {1e3 * el / t['print_steps']:.0f} ms/step incl. eval)"
            )
            start = time.time()
            rec = dict(step=it + 1, loss=running / t["print_steps"], train_mse=tr, val_mse=va)
            if series_q:
                x_now, bound = series_regime(model, X, series_q, m["depth"])
                rec.update(series_x=x_now, series_bound=bound)
                if bound > 1e-6:
                    log(
                        f"{tag}   WARNING series regime drifted: x={x_now:.3g} bound {bound:.1e} > 1e-6 at q={series_q}"
                    )
            if va <= best_val:
                best_val, no_improve = va, 0
                te, trainkey = epoch_mse(
                    model, loaders["test"], eb, trainkey, max_batches=t["eval_batches"]
                )
                eqx.tree_serialise_leaves(os.path.join(out_dir, "model.eqx"), model)
                best = dict(step=it + 1, val_mse=va, test_mse=te, seed=seed, config=resolved)
                with open(os.path.join(out_dir, "best.json"), "w") as f:
                    json.dump(best, f, indent=2, default=str)
                rec["test_mse"] = te
                log(f"{tag}   test mse {te:.4f} (x1e-2: {100 * te:.2f})")
            else:
                no_improve += 1
            running = 0.0
            history.append(rec)
            with open(os.path.join(out_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=1)
            if no_improve > int(t["patience"]):
                log(f"{tag} early stop at {it + 1}")
                break
    log(
        f"{tag} FINAL test mse {best.get('test_mse')} (x1e-2 {100 * best.get('test_mse', float('nan')):.2f}) "
        f"best step {best.get('step')} val {best.get('val_mse')}"
    )
    return best


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument(
        "--out",
        default=None,
        help="output root (default outputs/ppg/<config stem>); the run goes to <out>/seed_<seed>",
    )
    p.add_argument("--num_steps", type=int, default=None)
    p.add_argument("--print_steps", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--eval_batches", type=int, default=None)
    p.add_argument(
        "--set", action="append", default=[], help="override a config value, e.g. --set model.m=32"
    )
    a = p.parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(a.gpu))
    os.environ.setdefault(
        "XLA_FLAGS", "--xla_gpu_autotune_level=0"
    )  # the regime of the banked runs (deterministic losses)
    cfg = load_config(a.config, a.set, a)
    out = a.out or os.path.join(
        REPO, "outputs", "ppg", os.path.splitext(os.path.basename(a.config))[0]
    )
    run(cfg, a.seed, os.path.join(out, f"seed_{a.seed}"), log=lambda s: print(s, flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
