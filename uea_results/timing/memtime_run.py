"""One speed and memory measurement: one model on one UEA dataset, on its own config and its own code, alone on the card.

Only the run length changes: the first published seed, a short fixed number of steps (print_steps as published), no
early stopping and a fresh output directory. Time is training only: the first interval, which holds the compilation,
is dropped and the figure is the median of the rest, in seconds per 1000 steps. Our trainer has no timer, so each
"Step N train loss" line is timestamped as it is written (the trainer blocks on every step) and intervals holding an
evaluation are dropped; the JAX baselines report their own training-only "Time:", and the torch ones get an added
"TrainTime:" line. Memory is the in-process allocator peak, written here, and the nvidia-smi process peak, sampled by
the caller and added to the same JSON.

The EWS needs only this repository: the dataset's uea_results/<dataset>/config.json (or MEMTIME_EWS_CONFIG /
MEMTIME_EWS_CONFIG_DIR) and data/<stem>/splits/<stem>_seed0.npz. The baselines run their authors' code from clones
under MEMTIME_ROOT: lncde/ (log-neural-cdes @ 5612f000), linoss/ (tk-rusch/linoss @ 05a83535) and dlinoss/
(jaredbmit/damped-linoss @ a396997).

    python memtime_run.py --family ews --model ews --dataset Heartbeat --steps 1100 --out DIR
    MEMTIME_ROOT=<dir> python memtime_run.py --family lncde --model S5 --dataset Heartbeat --steps 4000 --out DIR
"""

import argparse
import json
import os
import re
import statistics
import sys
import time

EWS = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # this repository's root
ROOT = os.environ.get(
    "MEMTIME_ROOT"
)  # the baselines' code bases (see above); not needed for --family ews
STEM = {
    "EigenWorms": "eigenworms_walker",
    "EthanolConcentration": "ethanolconcentration_walker",
    "Heartbeat": "heartbeat_walker",
    "MotorImagery": "motorimagery_walker",
    "SelfRegulationSCP1": "selfregulationscp1_walker",
    "SelfRegulationSCP2": "selfregulationscp2_walker",
}
BANK = {
    "EigenWorms": "eigenworms",
    "EthanolConcentration": "ethanol",
    "Heartbeat": "heartbeat",
    "MotorImagery": "motor",
    "SelfRegulationSCP1": "scp1",
    "SelfRegulationSCP2": "scp2",
}

ap = argparse.ArgumentParser()
ap.add_argument("--family", required=True, choices=["lncde", "torch", "linoss", "dlinoss", "ews"])
ap.add_argument("--model", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--steps", type=int, required=True)
ap.add_argument("--out", required=True, help="fresh directory for this measurement")
ap.add_argument("--gpu", default="0")
ap.add_argument("--dry", action="store_true", help="write the config and stop")
ap.add_argument("--print_steps", type=int, default=None, help="override print_steps (smoke runs)")
ap.add_argument(
    "--batch",
    type=int,
    default=None,
    help="override every model's batch size (one standardised batch per dataset, so the time and memory "
    "columns compare models rather than batch sizes)",
)
args = ap.parse_args()
if args.family != "ews" and not ROOT:
    ap.error(
        "MEMTIME_ROOT is not set: it must name the directory holding the baselines' code (see this file's header)"
    )
args.out = os.path.abspath(
    args.out
)  # the run changes directory, so the output path must be absolute
os.makedirs(args.out, exist_ok=True)


class Tee:
    """Pass stdout through unchanged and remember every completed line with the time it was written."""

    def __init__(self, real):
        self.real, self.buf, self.lines = real, "", []

    def write(self, s):
        self.real.write(s)
        self.real.flush()
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self.lines.append((time.time(), line))
        return len(s)

    def flush(self):
        self.real.flush()

    def __getattr__(self, name):
        return getattr(self.real, name)


def build_config():
    """Return (cfg, where_written) for the family; only run length / seed / output dir / early stopping change."""
    fam, m, ds, n = args.family, args.model, args.dataset, args.steps
    if fam in ("lncde", "torch"):
        cfg = json.load(open(f"{ROOT}/lncde/experiment_configs/repeats/{m}/{ds}.json"))
        cfg.update(
            seeds=[cfg["seeds"][0]],
            num_steps=n,
            early_stopping_steps=10**9,
            output_parent_dir=os.path.join(args.out, "run") + "/",
        )
        if fam == "torch":
            cfg["data_dir"] = (
                "data_dir_np1"  # numpy-1.26 re-pickles of the same arrays (Walker's torch pin)
            )
        p = f"{ROOT}/lncde/experiment_configs/memtime/{m}/{ds}.json"
    elif fam == "linoss":
        if ds == "SelfRegulationSCP2":
            # LinOSS-IM on SCP2 was never committed (only IMEX). Paper Table 4, IM row: lr 1e-4, hidden 128,
            # state 64, 6 blocks, include_time False; parameter count 399,106 vs paper 399,112 (same -6 offset as
            # the committed SCP1 / Heartbeat configs, so the reconstruction is exact).
            cfg = json.load(open(f"{ROOT}/linoss/experiment_configs/repeats/LinOSS/{ds}.json"))
            cfg.update(
                linoss_discretization="IM",
                lr="0.0001",
                hidden_dim="128",
                ssm_dim="64",
                num_blocks="6",
                time="False",
            )
        else:
            cfg = json.load(open(f"{ROOT}/linoss/experiment_configs/repeats/LinOSS/{ds}.json"))
            assert cfg["linoss_discretization"] == "IM", cfg
        cfg.update(
            seeds=[cfg["seeds"][0]],
            num_steps=n,
            output_parent_dir=os.path.join(args.out, "run") + "/",
        )
        p = f"{ROOT}/linoss/experiment_configs/memtime/LinOSS/{ds}.json"
    elif fam == "dlinoss":
        cfg = json.load(open(f"{ROOT}/dlinoss/config/repeats/LinOSS/{ds}/config_000.json"))
        assert cfg.get("damping") is True, cfg
        cfg.update(seeds=[cfg["seeds"][0]], num_steps=n)
        p = f"{ROOT}/dlinoss/config/memtime/LinOSS/{ds}/config_000.json"
    else:  # ews: the dataset's selected config, read in place and copied
        # MEMTIME_EWS_CONFIG: time a different EWS config
        # MEMTIME_EWS_CONFIG_DIR: a directory of <Dataset>.json configs to use instead of the selected ones
        cdir = os.environ.get("MEMTIME_EWS_CONFIG_DIR")
        cfg = json.load(
            open(
                os.environ.get("MEMTIME_EWS_CONFIG")
                or (
                    os.path.join(cdir, f"{ds}.json")
                    if cdir
                    else os.path.join(EWS, "uea_results", BANK[ds], "config.json")
                )
            )
        )
        tc = cfg.setdefault("training", {})
        # MEMTIME_EVAL_EVERY > n puts the evaluation outside the timing window (pure training-step time)
        tc.update(
            num_steps=n,
            eval_every=int(os.environ.get("MEMTIME_EVAL_EVERY", 500)),
            log_every=100,
            early_stopping_patience=0,
        )
        p = os.path.join(args.out, "config.json")
    if args.batch is not None:  # standardised batch: the same for every model on a dataset
        if fam == "ews":
            cfg["training"]["batch_size"] = int(args.batch)
        else:
            cfg["batch_size"] = int(args.batch)
    if args.print_steps is not None:  # gates only; real measurements keep the published print_steps
        if fam == "ews":
            cfg["training"].update(eval_every=args.print_steps, log_every=1)
        else:
            cfg["print_steps"] = args.print_steps
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(cfg, open(p, "w"), indent=2)
    return cfg, p


cfg, cfg_path = build_config()
json.dump(
    {"config": cfg, "config_path": cfg_path},
    open(os.path.join(args.out, "config_used.json"), "w"),
    indent=2,
)
if args.dry:
    print(f"DRY {args.family} {args.model} {args.dataset}: config written to {cfg_path}")
    sys.exit(0)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ.setdefault(
    "XLA_PYTHON_CLIENT_PREALLOCATE", "false"
)  # a caller may ask for the preallocated arena
tee = Tee(sys.stdout)
sys.stdout = tee
t0 = time.time()
status = "ok"
try:
    if args.family in ("lncde", "torch"):
        os.chdir(f"{ROOT}/lncde")
        sys.path.insert(0, f"{ROOT}/lncde")
        from run_experiment import run_experiments

        run_experiments(
            [args.model], [args.dataset], "experiment_configs/memtime", args.family == "torch"
        )
    elif args.family == "linoss":
        os.chdir(f"{ROOT}/linoss")
        sys.path.insert(0, f"{ROOT}/linoss")
        from run_experiment import run_experiments

        run_experiments(["LinOSS"], [args.dataset], "experiment_configs/memtime")
    elif args.family == "dlinoss":
        os.chdir(f"{ROOT}/dlinoss")
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "dl_run", f"{ROOT}/dlinoss/scripts/run/run_experiment.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.run_experiments(["LinOSS"], [args.dataset], "config/memtime")
    else:
        import runpy

        import jax

        # Our trainer turns on a persistent XLA compilation cache (ews/outputs/.jax_cache). A cache hit skips compilation
        # and its autotuning scratch buffers, which lowers the allocator peak (seen: 807 MB warm vs 1286 MB cold on
        # SCP1); Walker's trainers have no such cache and always compile cold. For a like-for-like measurement every
        # model compiles cold, so the cache settings are dropped here (train.py itself is not touched).
        _jax_update = jax.config.update

        def _update_without_persistent_cache(name, value):
            if name.startswith("jax_compilation_cache") or name.startswith("jax_persistent_cache"):
                return None
            return _jax_update(name, value)

        jax.config.update = _update_without_persistent_cache
        os.chdir(EWS)
        split = os.path.join(
            EWS, "data", STEM[args.dataset], "splits", f"{STEM[args.dataset]}_seed0.npz"
        )
        sys.argv = [
            "experiments/uea/train.py",
            "--config",
            cfg_path,
            "--gpu",
            args.gpu,
            "--seed",
            "0",
            "--dataset_file",
            split,
            "--out_ckpt",
            os.path.join(args.out, "ckpt.eqx"),
        ]
        runpy.run_path(f"{EWS}/experiments/uea/train.py", run_name="__main__")
except BaseException as e:  # noqa: BLE001 - record any failure, then re-raise after writing the result
    status = f"FAILED: {type(e).__name__}: {e}"
wall = time.time() - t0
sys.stdout = tee.real

# ---------------- timing
lines = tee.lines
intervals = []  # (steps covered, training seconds, label)
their_time = []  # torch only: their own Time: (includes train+val evaluation)
P = cfg.get("print_steps") if args.family != "ews" else 100
if args.family in ("lncde", "linoss", "dlinoss"):
    for _, ln in lines:
        mm = re.search(r"Step: (\d+), Loss: .*Time: ([0-9.eE+-]+)", ln)
        if mm:
            intervals.append((int(P), float(mm.group(2)), int(mm.group(1))))
elif args.family == "torch":
    for _, ln in lines:
        mm = re.search(r"TrainTime: step (\d+) secs ([0-9.eE+-]+)", ln)
        if mm:
            s = int(mm.group(1))
            intervals.append((1 if s == 0 else int(P), float(mm.group(2)), s))
        mm = re.search(r"Step: (\d+), .*Time: ([0-9.eE+-]+)", ln)
        if mm:
            their_time.append((int(mm.group(1)), float(mm.group(2))))
else:
    prev = None
    evald = False
    for t, ln in lines:
        mm = re.match(r"Step\s+(\d+)\s+train loss", ln)
        if mm:
            s = int(mm.group(1))
            if prev is not None and not evald:
                intervals.append((s - prev[0], t - prev[1], s))
            prev, evald = (s, t), False
        elif "val  loss=" in ln or "test loss=" in ln:
            evald = True
full = [iv for iv in intervals if iv[0] == (P if args.family != "ews" else 100)]
steady = full[1:]  # the first full interval holds compilation / warm-up
per1k = [1000.0 * sec / n for n, sec, _ in steady]
res = {
    "family": args.family,
    "model": args.model,
    "dataset": args.dataset,
    "status": status,
    "steps_requested": args.steps,
    "print_steps": P,
    "batch_size": (
        cfg.get("batch_size") if args.family != "ews" else cfg["training"].get("batch_size")
    ),
    "intervals": intervals,
    "steady_intervals": len(steady),
    "s_per_1k_median": statistics.median(per1k) if per1k else None,
    "s_per_1k_mean": statistics.mean(per1k) if per1k else None,
    "s_per_1k_min": min(per1k) if per1k else None,
    "s_per_1k_max": max(per1k) if per1k else None,
    "first_interval_s": full[0][1] if full else None,
    "their_time_incl_eval": their_time,
    "wall_s": wall,
    "env": {
        k: os.environ.get(k)
        for k in (
            "CUDA_VISIBLE_DEVICES",
            "XLA_PYTHON_CLIENT_PREALLOCATE",
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
            "XLA_FLAGS",
        )
    },
    "python": sys.executable,
}
# ---------------- in-process allocator peak
try:
    if args.family == "torch":
        import torch

        res["mem"] = {
            "torch_max_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
            "torch_max_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
        }
        res["versions"] = {"torch": torch.__version__}
    else:
        import jax

        st = jax.local_devices()[0].memory_stats() or {}
        res["mem"] = {k: (v / 2**20 if "bytes" in k or "size" in k else v) for k, v in st.items()}
        res["mem"]["jax_peak_mb"] = st.get("peak_bytes_in_use", 0) / 2**20
        res["versions"] = {"jax": jax.__version__}
except Exception as e:  # noqa: BLE001
    res["mem_error"] = repr(e)
json.dump(res, open(os.path.join(args.out, "result.json"), "w"), indent=2)
print(
    f"MEMTIME {args.family} {args.model} {args.dataset}: {status}; "
    f"{res['s_per_1k_median'] if res['s_per_1k_median'] is None else round(res['s_per_1k_median'], 2)} s/1k "
    f"(median of {len(steady)}), in-process peak "
    f"{res.get('mem', {}).get('jax_peak_mb', res.get('mem', {}).get('torch_max_reserved_mb'))} MB, wall {wall:.0f}s"
)
sys.stdout.flush()
sys.stderr.flush()
os._exit(
    0 if status == "ok" else 1
)  # skip interpreter teardown: JAX-on-CPU + OpenBLAS can abort there (glibc)
