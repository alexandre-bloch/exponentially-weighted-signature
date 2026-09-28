"""The PPG selection grid of ppg_results/grid.json with experiments/ppg/train.py, then its report.

Stage 1 runs the cells of grid.json on seeds 2345/3456/4567; stage 2 runs the best `top_for_stage2` of them by
three-seed mean validation MSE (ties at the cut kept) on 5678/6789; the selection is the lowest five-seed mean
validation MSE. Claim-safe (a claim directory per run), so several drivers, one per GPU or several per GPU, can
share the grid. A run counts once its log ends with FINAL: the trainer rewrites best.json at every improvement, so
a run stopped part-way leaves a partial one.

    python ppg_results/run_grid.py --gpu 0 --slots 3 --out outputs/ppg_grid      # a driver: 3 runs at a time on GPU 0
    python ppg_results/run_grid.py --report --out outputs/ppg_grid               # the table and the selection
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEF = json.load(open(os.path.join(HERE, "grid.json")))["definition"]
KEYS = ("lr", "hidden", "m", "num_blocks")
SEEDS1, SEEDS2 = tuple(DEF["seeds_stage1"]), tuple(DEF["seeds_stage2"])
ALL = SEEDS1 + SEEDS2


def name_of(c):
    return f"lr{c['lr']:.4g}_h{c['hidden']}_m{c['m']}_b{c['num_blocks']}"


def cells():
    return [{k: c[k] for k in KEYS} for c in DEF["cells_stage1"]]


def best(out, c, s):
    d = os.path.join(out, name_of(c), f"seed_{s}")
    p, log = os.path.join(d, "best.json"), os.path.join(d, "train.log")
    if (
        not os.path.exists(p)
        or not os.path.exists(log)
        or "FINAL" not in open(log, errors="ignore").read()
    ):
        return None
    return json.load(open(p))


def one(out, c, s, gpu):
    if best(out, c, s):
        return
    try:
        os.makedirs(os.path.join(out, ".claims", f"{name_of(c)}_s{s}"))
    except FileExistsError:
        return
    d = os.path.join(out, name_of(c))
    os.makedirs(os.path.join(d, f"seed_{s}"), exist_ok=True)
    sets = [
        f"training.lr={c['lr']}",
        f"model.hidden={c['hidden']}",
        f"model.m={c['m']}",
        f"model.num_blocks={c['num_blocks']}",
    ]
    cmd = [
        sys.executable,
        os.path.join(ROOT, "experiments", "ppg", "train.py"),
        "--config",
        os.path.join(HERE, "config.json"),
        "--seed",
        str(s),
        "--gpu",
        str(gpu),
        "--out",
        d,
    ]
    for kv in sets:
        cmd += ["--set", kv]
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    t0 = time.time()
    with open(os.path.join(d, f"seed_{s}", "train.log"), "w") as f:
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    b = best(out, c, s)
    val = b and round(100 * b["val_mse"], 3)
    print(f"{name_of(c)} seed {s}: val {val} ({round(time.time() - t0)} s)", flush=True)


def mean_val(out, c, seeds):
    rs = [best(out, c, s) for s in seeds]
    return statistics.mean(100 * r["val_mse"] for r in rs) if all(rs) else None


def stage2_cells(out):
    ranked = sorted(
        (v, i) for i, c in enumerate(cells()) if (v := mean_val(out, c, SEEDS1)) is not None
    )
    cut = ranked[min(DEF["top_for_stage2"], len(ranked)) - 1][0]
    return [cells()[i] for v, i in ranked if v <= cut + 1e-12]


def drive(a):
    jobs = [(c, s) for s in SEEDS1 for c in cells()]
    with ThreadPoolExecutor(a.slots) as ex:
        list(ex.map(lambda j: one(a.out, *j, a.gpu), jobs))
    while not all(best(a.out, c, s) for c in cells() for s in SEEDS1):
        time.sleep(120)  # other drivers are still on stage 1
    jobs = [(c, s) for s in SEEDS2 for c in stage2_cells(a.out)]
    with ThreadPoolExecutor(a.slots) as ex:
        list(ex.map(lambda j: one(a.out, *j, a.gpu), jobs))


def report(a):
    rows = []
    for c in cells():
        runs = {s: best(a.out, c, s) for s in ALL}
        runs = {s: (100 * r["val_mse"], 100 * r["test_mse"]) for s, r in runs.items() if r}
        rows.append((name_of(c), runs))
    print("| cell | seeds | validation MSE x1e-2 | test MSE x1e-2 |\n|---|---|---|---|")
    for name, runs in sorted(
        rows, key=lambda r: (-len(r[1]), statistics.mean([v for v, _ in r[1].values()] or [9e9]))
    ):
        if runs:
            v, t = [x for x, _ in runs.values()], [x for _, x in runs.values()]
            print(
                f"| {name} | {len(v)} | {statistics.mean(v):.3f} | {statistics.mean(t):.3f} ± {statistics.pstdev(t):.3f} |"
            )
    five = [(name, runs) for name, runs in rows if len(runs) == len(ALL)]
    if five:
        name, runs = min(five, key=lambda r: statistics.mean(v for v, _ in r[1].values()))
        t = [x for _, x in runs.values()]
        print(
            f"\nselected: {name}, test MSE {statistics.mean(t):.2f} ± {statistics.pstdev(t):.2f} (x1e-2)"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "ppg_grid"))
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report(a) if a.report else (os.makedirs(a.out, exist_ok=True), drive(a))
