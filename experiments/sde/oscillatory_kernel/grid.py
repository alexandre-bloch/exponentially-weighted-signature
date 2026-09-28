"""Run the grid of the oscillatory-kernel experiment and collect results.csv.

    python experiments/sde/oscillatory_kernel/grid.py --out outputs/oscillatory_kernel --gpus 0,1 --par 6
    python experiments/sde/oscillatory_kernel/grid.py --out <dir> --cpu --par 4 --steps 30 --only "ews:1:8,efm:2:8" --targets T1 --seeds 0
    python experiments/sde/oscillatory_kernel/grid.py --out <dir> --dry-run          # list the runs
    python experiments/sde/oscillatory_kernel/grid.py --out <dir> --collect-only     # rebuild results.csv

Grid, feature count D = sum_{k<=n} m^k capped at D <= 5000:
    sig  n in {1,2,3,4}  m in {2,4,8,16}
    efm  n in {1,2,3}    m in {2,4,8,16,32,64}
    ews  n in {1,2}      m in {8,16,32}
times targets {T1, T2, T3} times seeds {0, 1, 2}. A run whose results.json exists is skipped.
Writes per-run configs from the template, launches train.py with `--par` concurrent runs per GPU,
then writes results.csv and config.yaml (every option used).
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
HERE = Path(__file__).resolve().parent

from experiments.sde.oscillatory_kernel.data import BANK_HZ, DAMPING, feature_count  # noqa: E402

GRID = {
    "sig": {"n": [1, 2, 3, 4], "m": [2, 4, 8, 16]},
    "efm": {"n": [1, 2, 3], "m": [2, 4, 8, 16, 32, 64]},
    "ews": {"n": [1, 2], "m": [8, 16, 32]},
}
A_OF_MODEL = {
    "sig": {"parametrization": "sig", "learn": False},
    "efm": {"parametrization": "efm", "positivity": "softplus", "learn": True},
    "ews": {"parametrization": "damped_rotation", "omega_scale": 314.1592653589793, "learn": True},
}
D_CAP = 5000
TARGETS = ["T1", "T2", "T3"]
SEEDS = [0, 1, 2]


def grid_points(d_cap: int = D_CAP):
    for model, g in GRID.items():
        for n, m in itertools.product(g["n"], g["m"]):
            D = feature_count(m, n)
            if D <= d_cap:
                yield model, n, m, D


def run_config(template: dict, model: str, n: int, m: int, target: str, seed: int) -> dict:
    cfg = json.loads(json.dumps(template))
    cfg["dataset"]["target"] = target
    cfg["model"]["A"] = dict(A_OF_MODEL[model])
    cfg["model"]["B"]["out_dim"] = m
    cfg["model"]["depth"] = n
    cfg["training"]["seed"] = seed
    return cfg


def run_dir(out: str, model: str, n: int, m: int, target: str, seed: int) -> str:
    return os.path.join(out, "runs", target, f"{model}_n{n}_m{m}", f"seed{seed}")


def memory_gb(model: str, n: int, m: int) -> float:
    """Rough GPU memory of a run (batch 64, T = 256), measured on an H100 with 12 co-located runs:
    depth-4 runs 12-18 GB, depth-3 runs with m >= 8 about 10 GB, the D > 1000 depth-2 runs
    about 10 GB, everything else a few GB."""
    D = feature_count(m, n)
    if n >= 4:
        return 18.0
    if n == 3:
        return 10.0 if m >= 8 else 4.0
    if D > 1000:
        return 10.0
    return 3.0


def launch(jobs, gpus, par, py, cpu, steps, extra, mem_budget_gb):
    """Run `jobs` = [(cmd_args, run_dir, mem_gb)]: at most `par` concurrent processes per GPU and
    at most `mem_budget_gb` of estimated memory per GPU, first fit over the queue so small runs
    are admitted beside big ones."""
    running = {}  # proc -> (gpu, mem)
    queue = list(jobs)
    t0 = time.time()
    done = 0
    while queue or running:
        started = True
        while queue and started:
            started = False
            for gpu in gpus:
                on_gpu = [(g, mem) for g, mem in running.values() if g == gpu]
                if len(on_gpu) >= par:
                    continue
                used = sum(mem for _, mem in on_gpu)
                pick = next((j for j in queue if j[2] + used <= mem_budget_gb or not on_gpu), None)
                if pick is None:
                    continue
                queue.remove(pick)
                args, d, mem = pick
                slot = (gpu, mem)
                started = True
                os.makedirs(d, exist_ok=True)
                env = dict(os.environ)
                env["PYTHONUNBUFFERED"] = "1"
                cmd = [py, str(HERE / "train.py")] + args + ["--out_dir", d]
                if cpu:
                    env["JAX_PLATFORMS"] = "cpu"
                    cmd += ["--gpu", ""]
                else:
                    cmd += ["--gpu", str(gpu)]
                if steps is not None:
                    cmd += ["--steps", str(steps)]
                cmd += extra
                log = open(os.path.join(d, "train.log"), "w")
                proc = subprocess.Popen(
                    cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(REPO_ROOT)
                )
                running[proc] = slot
                print(
                    f"[{len(running)} running, {len(queue)} queued] started {d} ({mem:.0f} GB est.) on {'cpu' if cpu else 'gpu ' + str(gpu)}",
                    flush=True,
                )
        time.sleep(2.0)
        for proc in list(running):
            if proc.poll() is not None:
                running.pop(proc)
                done += 1
                status = "ok" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
                print(f"[{done} done, {(time.time() - t0) / 60:.1f} min] {status}", flush=True)


def collect(out: str) -> str:
    rows = []
    for p in sorted(glob.glob(os.path.join(out, "runs", "*", "*", "seed*", "results.json"))):
        rows.append(result_row(p))
    path = os.path.join(out, "results.csv")
    write_rows(rows, path)
    print(f"collected {len(rows)} runs -> {path}")
    return path


def write_rows(rows, path):
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def result_row(p: str) -> dict:
    r = json.load(open(p))
    best = r["best"] or {}
    row = {
        "target": r["target"],
        "model": r["model"],
        "n": r["n"],
        "m": r["m"],
        "D": r["D"],
        "init": "spread",
        "seed": r["seed"],
        "best_step": best.get("step"),
        "val_loss": best.get("val", {}).get("loss"),
        "test_loss": best.get("test", {}).get("loss"),
        "val_metric": best.get("val", {}).get("nmse", best.get("val", {}).get("acc")),
        "test_metric": best.get("test", {}).get("nmse", best.get("test", {}).get("acc")),
        "test_f1": best.get("test", {}).get("f1"),
        "test_nmse_per_output": json.dumps(best.get("test", {}).get("nmse_per_output")),
        "freq_dist": r["config"]["experiment"]["init"].get("freq_dist", "log_uniform"),
        "steps_run": r["num_steps_run"],
        # the flag alone is not enough: the first grid's trainer set it without shortening the
        # loop, so a run counts as cut only if it stopped at the reduced count and not by early
        # stopping (an early stop leaves at least `patience` steps between the best and the last)
        "steps_reduced": bool(r["steps_reduced"])
        and int(r["num_steps_run"]) == int(r["config"]["experiment"].get("reduced_num_steps", 6000))
        and int(r["num_steps_run"]) - int((r["best"] or {}).get("step", 0))
        < int(r["config"]["training"].get("early_stopping_patience", 0) or 10**9),
        "series_x_best": best.get("series_x"),
        "ms_per_step": r.get("ms_per_step"),
        "wall_min": r["wall_time_s"] / 60.0,
        "device": r.get("device"),
        "learned_params": json.dumps(r["A_learned"]),
        "init_params": json.dumps(r["A_init"]),
        "path": p,
    }
    return row


def write_config_yaml(out: str, template: dict, points, targets, seeds, steps):
    import yaml

    doc = {
        "experiment": "oscillatory_kernel",
        "data": {
            "file": template["dataset"]["file"],
            "T": 256,
            "bank_hz": list(BANK_HZ),
            "damping": DAMPING,
            "train_val_test": [8000, 1000, 1000],
            "data_seed": 0,
        },
        "path": template["preprocessing"]["path"],
        "clock": template["preprocessing"]["clock"],
        "dtype": template["preprocessing"]["dtype"],
        "models": {k: v for k, v in A_OF_MODEL.items()},
        "lift_B": template["model"]["B"],
        "compute": template["model"]["compute"],
        "readout": "linear on the flat levels 1..n (level 0 dropped); logistic for T3",
        "grid": [{"model": mo, "n": n, "m": m, "D": D} for mo, n, m, D in points],
        "D_cap": D_CAP,
        "targets": targets,
        "seeds": seeds,
        "training": {
            **template["training"],
            "num_steps": steps or template["training"]["num_steps"],
        },
        "experiment_section": template["experiment"],
    }
    with open(os.path.join(out, "config.yaml"), "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)


def main():
    p = argparse.ArgumentParser(description="Oscillatory-kernel grid")
    p.add_argument("--out", type=str, default=str(REPO_ROOT / "outputs/oscillatory_kernel"))
    p.add_argument("--template", type=str, default=str(HERE / "configs/template.json"))
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--par", type=int, default=4, help="concurrent runs per GPU (or on the CPU)")
    p.add_argument(
        "--mem-budget",
        type=float,
        default=70.0,
        help="estimated GB of runs admitted per GPU at once",
    )
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--py", type=str, default=sys.executable)
    p.add_argument("--steps", type=int, default=None, help="override num_steps (smoke tests)")
    p.add_argument("--targets", type=str, default=",".join(TARGETS))
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    p.add_argument(
        "--only", type=str, default=None, help='subset "model:n:m,..." e.g. "ews:1:8,efm:2:8"'
    )
    p.add_argument("--skip", type=str, default=None, help='drop grid points "model:n:m,..."')
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--collect-only", action="store_true")
    p.add_argument("--extra", type=str, default="", help="extra args passed to train.py")
    a = p.parse_args()

    template = json.load(open(a.template))
    targets = a.targets.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    points = list(grid_points())
    if a.only:
        keep = {tuple(x.split(":")) for x in a.only.split(",")}
        points = [pt for pt in points if (pt[0], str(pt[1]), str(pt[2])) in keep]
    if a.skip:
        drop = {tuple(x.split(":")) for x in a.skip.split(",")}
        points = [pt for pt in points if (pt[0], str(pt[1]), str(pt[2])) not in drop]

    os.makedirs(a.out, exist_ok=True)
    if a.collect_only:
        collect(a.out)
        return

    jobs = []
    for target in targets:
        for model, n, m, D in points:
            for seed in seeds:
                d = run_dir(a.out, model, n, m, target, seed)
                if os.path.exists(os.path.join(d, "results.json")):
                    continue
                os.makedirs(d, exist_ok=True)
                cfg_path = os.path.join(d, "config.json")
                with open(cfg_path, "w") as f:
                    json.dump(run_config(template, model, n, m, target, seed), f, indent=1)
                jobs.append(
                    (
                        ["--config", cfg_path, "--seed", str(seed), "--target", target],
                        d,
                        memory_gb(model, n, m),
                    )
                )
    print(
        f"{len(points)} grid points x {len(targets)} targets x {len(seeds)} seeds; {len(jobs)} runs to do"
    )
    for model, n, m, D in points:
        print(f"  {model:4s} n={n} m={m:3d} D={D}")
    write_config_yaml(a.out, template, points, targets, seeds, a.steps)
    if a.dry_run:
        return
    gpus = [g for g in a.gpus.split(",") if g != ""] if not a.cpu else ["cpu"]
    extra = a.extra.split() if a.extra else []
    t0 = time.time()
    launch(jobs, gpus, a.par, a.py, a.cpu, a.steps, extra, a.mem_budget if not a.cpu else 1e9)
    print(f"grid done in {(time.time() - t0) / 60:.1f} min")
    collect(a.out)


if __name__ == "__main__":
    main()
