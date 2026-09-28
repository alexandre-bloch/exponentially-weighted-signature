"""The weather selection grid of weather_results/grid.json with experiments/weather/train.py, then its report.

lr {2.5e-5, 5e-5, 1e-4} x hidden {32, 64, 128} x m {16, 32, 64} x blocks {1, 2} = 54 cells, every other setting
from config.json. Stage 1: the cells of grid.json on seeds 2345, 3456, 4567. Stage 2: the best `top_for_stage2` by
three-seed mean validation MAE (ties at the cut kept) on seeds 5678, 6789. Selection: lowest five-seed mean
validation MAE; within the per-seed spread of the best (the larger of the two population stds), fewer parameters.
Runs are claim-safe (a claim directory per run), so several drivers can share it.

    python weather_results/run_grid.py --gpu 0 --slots 4 --out outputs/weather_grid      # a driver (both stages)
    python weather_results/run_grid.py --report --out outputs/weather_grid [--write]     # table + selection
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
SEEDS1, SEEDS2 = tuple(DEF["seeds_stage1"]), tuple(DEF["seeds_stage2"])
ALL = SEEDS1 + SEEDS2
KEYS = ("lr", "hidden", "m", "num_blocks")


def name_of(c):
    return f"lr{c['lr']:.4g}_h{c['hidden']}_m{c['m']}_b{c['num_blocks']}"


def cells():
    return [{k: c[k] for k in KEYS} for c in DEF["cells_stage1"]]


def run_dir(out, c, s):
    return os.path.join(out, name_of(c), f"seed_{s}")


def best(out, c, s):
    p = os.path.join(run_dir(out, c, s), "best.json")
    if not os.path.exists(p):
        return None
    b = json.load(open(p))
    return b if b.get("status") == "ok" and b.get("val_mae") is not None else None


def one(out, c, s, gpu):
    if best(out, c, s):
        return
    claim = os.path.join(out, ".claims", f"{name_of(c)}_s{s}")
    try:
        os.makedirs(claim)
    except FileExistsError:
        return
    d = os.path.join(out, name_of(c))
    os.makedirs(os.path.join(d, f"seed_{s}"), exist_ok=True)
    cmd = [
        sys.executable,
        os.path.join(ROOT, "experiments", "weather", "train.py"),
        "--config",
        os.path.join(HERE, "config.json"),
        "--seed",
        str(s),
        "--gpu",
        str(gpu),
        "--out",
        d,
        "--set",
        f"training.lr={c['lr']!r}",
        "--set",
        f"model.hidden={c['hidden']}",
        "--set",
        f"model.m={c['m']}",
        "--set",
        f"model.num_blocks={c['num_blocks']}",
    ]
    with open(os.path.join(d, f"seed_{s}", "train.log"), "w") as f:
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    b = best(out, c, s)
    print(
        f"[{time.strftime('%H:%M:%S')}] {name_of(c)} s{s}: val MAE {b and b['val_mae']} test MAE {b and b['test_mae']}",
        flush=True,
    )


def mean_val(out, c, seeds):
    rs = [best(out, c, s) for s in seeds]
    return statistics.mean(r["val_mae"] for r in rs) if all(rs) else None


def stage2_cells(out):
    ranked = sorted(
        (v, i) for i, c in enumerate(cells()) if (v := mean_val(out, c, SEEDS1)) is not None
    )
    cut = ranked[min(DEF["top_for_stage2"], len(ranked)) - 1][0]
    return [cells()[i] for v, i in ranked if v <= cut + 1e-12]


def drive(a):
    jobs = [(c, s) for s in SEEDS1 for c in cells()]
    with ThreadPoolExecutor(a.slots) as ex:
        list(ex.map(lambda j: one(a.out, j[0], j[1], a.gpu), jobs))
    while not all(best(a.out, c, s) for c in cells() for s in SEEDS1):
        time.sleep(60)  # other drivers still on stage 1
    jobs = [(c, s) for s in SEEDS2 for c in stage2_cells(a.out)]
    with ThreadPoolExecutor(a.slots) as ex:
        list(ex.map(lambda j: one(a.out, j[0], j[1], a.gpu), jobs))


def n_params(out, c):
    for s in ALL:
        p = os.path.join(run_dir(out, c, s), "train.log")
        if os.path.exists(p):
            for line in open(p):
                if " params " in line:
                    return int(line.split(" params ")[1].split()[0])
    return None


def report(a):
    rows = []
    for c in cells():
        runs = {str(s): best(a.out, c, s) for s in ALL}
        runs = {
            s: dict(
                val_mae=r["val_mae"],
                val_mse=r["val_mse"],
                test_mae=r["test_mae"],
                test_mse=r["test_mse"],
                best_step=r["step"],
            )
            for s, r in runs.items()
            if r
        }
        rows.append(dict(config=c, name=name_of(c), params=n_params(a.out, c), runs=runs))
    five = [r for r in rows if len(r["runs"]) == len(ALL)]
    for r in five:
        r["val_mae_5"] = statistics.mean(x["val_mae"] for x in r["runs"].values())
    five.sort(key=lambda r: r["val_mae_5"])
    b = five[0]
    sd = lambda r: statistics.pstdev(x["val_mae"] for x in r["runs"].values())  # noqa: E731
    near = [r for r in five if r["val_mae_5"] - b["val_mae_5"] < max(sd(b), sd(r))]
    sel = sorted(near, key=lambda r: (r["params"] or 10**9, r["val_mae_5"]))[0]
    print(
        "| cell | params | seeds | val MAE (mean) | test MAE (mean +- pop sd) |\n|---|---|---|---|---|"
    )
    for r in sorted(
        rows,
        key=lambda r: (-len(r["runs"]), statistics.mean(x["val_mae"] for x in r["runs"].values())),
    ):
        t = [x["test_mae"] for x in r["runs"].values()]
        print(
            f"| {r['name']} | {r['params']} | {len(r['runs'])} | {statistics.mean(x['val_mae'] for x in r['runs'].values()):.4f} | "
            f"{statistics.mean(t):.4f} +- {statistics.pstdev(t):.4f} |"
        )
    print(
        f"\nselected: {sel['name']} (val MAE {sel['val_mae_5']:.4f}, {sel['params']} params)"
        f"{'' if sel is b else ' by the parameter tie rule over ' + b['name']}"
    )
    if a.write:
        g = json.load(open(os.path.join(HERE, "grid.json")))
        g["cells"] = rows
        g["selected"] = dict(
            sel["config"], name=sel["name"], val_mae_5=sel["val_mae_5"], params=sel["params"]
        )
        json.dump(g, open(os.path.join(HERE, "grid.json"), "w"), indent=1)
        print("wrote grid.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "weather_grid"))
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--slots", type=int, default=4, help="runs at a time for this driver")
    ap.add_argument("--report", action="store_true")
    ap.add_argument(
        "--write",
        action="store_true",
        help="with --report: write the cells and the selection into grid.json",
    )
    a = ap.parse_args()
    report(a) if a.report else drive(a)


if __name__ == "__main__":
    main()
