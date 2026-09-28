"""UEA signature-depth ablation on a paired random grid: one random draw of configurations, each trained at depths 1,
2 and 3 on the same width, so the two members of a pair differ in depth alone.

With a separate search per depth, one depth might simply get the better search; here every configuration is its own
control. Two results come from one set of runs: the best configuration per depth and, per configuration, how often and
by how much the deeper model wins. The grid axes, the draw seed and the drawn configurations are recorded in each
dataset's configs.json (configuration 0 is the champion); everything else stays at the champion
(<Dataset>/champion_config.json), under Walker's protocol. Every depth uses depth 3's width ladder {6, 8, 12, 16}
(width 8 on EigenWorms), since depth 3 cannot take larger widths.

    python uea_results/ablation/uea_paired.py build  --dataset Heartbeat          # draw the configurations
    python uea_results/ablation/uea_paired.py run    --dataset Heartbeat --gpu 0 --seeds 0,1,2             # every cell
    python uea_results/ablation/uea_paired.py run    --dataset Heartbeat --gpu 0 --seeds 0,1,2,3,4 --top_k 5
    python uea_results/ablation/uea_paired.py report --dataset Heartbeat

`run` workers claim cells per seed with mkdir under outputs/uea_ablation/paired/<Dataset>/claims, so several can
share a dataset.
"""

import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import OUT as RUN_OUT  # noqa: E402  (config building + trainer call)
from common import STEM, blended, champion, run_one

OUT = os.path.join(RUN_OUT, "paired")
# UEA_PAIRED_TAG: a separate namespace for an ablation of a different champion (UEA_CHAMPION_CONFIG), so its cells
# never mix with the dataset champion's.
PAIRED_TAG = os.environ.get("UEA_PAIRED_TAG", "")
N_CONFIGS = 32
SEARCH_SEEDS = (0, 1)
SEARCH_STEPS = 6000
STALE_H = 3.0  # a claim older than this with no result is taken over (dead worker)
DEPTHS = (1, 2, 3)
# Width ladders by depth, recorded in configs.json as `d_ladder`. A drawn width level is mapped through depth 3's row
# at every depth (params_for), so the pair differs only in depth; the rows for depths 1 and 2 are the per-depth
# widths of the secondary free-width arm (free_width.json), which the paper does not use.
D_LADDER = {1: [8, 16, 32, 48], 2: [8, 16, 24, 32], 3: [6, 8, 12, 16]}
# EigenWorms: depth 3 fits at width 8 only (width 12 and 16 run out of memory on a 96 GB card at any batch size; the
# limiting buffer is the level-3 state inside the per-sample scan). Depth 3's row is [8] written three times so the
# width level still indexes it, which makes the matched arm width 8 everywhere.
D_LADDER_BY_DS = {"EigenWorms": {1: [8, 12, 16], 2: [8, 12, 16], 3: [8, 8, 8]}}
DAMP_MULT = [
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
]  # champion damping init mean x this, the same shape as LR_MULT
A_VAR = [0.003, 0.01, 0.03, 0.1, 0.3]
HEAD = [0.05, 0.1, 0.2, 0.35, 0.7]
READOUT = ["mean", "max", "terminal"]
LAMBDA2 = [0.0, 0.003, 0.01, 0.03]
LABEL_SMOOTH = [0.0, 0.1]
ALPHA = [0.0, 0.5]  # 0 = full variation
LR_MULT = [0.1, 0.3, 1.0, 3.0, 10.0]
NICE = sorted(k * 10.0**e for e in range(-7, 2) for k in (1, 2, 5))
SEED = 20260912


def lr_axis(ds):
    import numpy as np

    lr0 = champion(ds)["training"]["lr_features"]
    base = min(NICE, key=lambda v: abs(np.log(v / lr0)))
    return [base * m for m in LR_MULT]


def ladder(ds):
    return D_LADDER_BY_DS.get(ds, D_LADDER)


def stable_block_champion(ds):
    return champion(ds)["model"]["A"].get("parametrization") == "stable_block"


def axes(ds):
    lad = ladder(ds)
    a = dict(lr=lr_axis(ds), d_level=list(range(len(lad[3]))))
    if stable_block_champion(ds):
        # the init variance is inert for stable_block; the damping init mean is the live A-scale knob, and lambda2 /
        # label smoothing are out of the EigenWorms space by decision
        d0 = float(champion(ds)["model"]["A"]["damp_init_mean"])
        a["damp"] = [d0 * m for m in DAMP_MULT]
        a.update(head=HEAD, readout=READOUT)
    else:
        a.update(
            a_var=A_VAR, head=HEAD, readout=READOUT, lambda2=LAMBDA2, label_smoothing=LABEL_SMOOTH
        )
    if blended(champion(ds)):
        a["alpha"] = ALPHA
    return a


def build(ds):
    import numpy as np

    rng = np.random.RandomState(SEED + sum(map(ord, ds)))
    ax = axes(ds)
    n_cells = int(np.prod([len(v) for v in ax.values()]))
    seen, cfgs = set(), []
    c = champion(ds)
    champ = dict(
        lr=c["training"]["lr_features"],
        d_level=None,
        d_champ=int(c["model"]["B"]["out_dim"]),
        head=c["model"]["head_init_scale"],
        readout=c["model"].get("readout"),
        champion=True,
    )
    if stable_block_champion(ds):
        champ["damp"] = float(c["model"]["A"]["damp_init_mean"])
    else:
        champ.update(
            a_var=c["model"]["A"]["init"]["variance"],
            lambda2=c["training"]["lambda2"],
            label_smoothing=c["training"]["label_smoothing"],
        )
    if blended(c):
        pp = c["preprocessing"]
        champ["alpha"] = float(pp.get("clock_alpha", pp.get("clock", {}).get("alpha", 0.0)))
    cfgs.append(champ)
    while len(cfgs) < N_CONFIGS + 1:
        p = {k: v[rng.randint(len(v))] for k, v in ax.items()}
        key = json.dumps(p, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        p["champion"] = False
        cfgs.append(p)
    os.makedirs(os.path.join(OUT, ds + PAIRED_TAG), exist_ok=True)
    json.dump(
        dict(
            dataset=ds,
            seed=SEED + sum(map(ord, ds)),
            n_grid_cells=n_cells,
            axes=ax,
            d_ladder=ladder(ds),
            depths=DEPTHS,
            seeds=SEARCH_SEEDS,
            steps=SEARCH_STEPS,
            configs=cfgs,
        ),
        open(os.path.join(OUT, ds + PAIRED_TAG, "configs.json"), "w"),
        indent=1,
    )
    print(
        f"{ds}: grid {n_cells:,} cells -> drew {len(cfgs) - 1} configurations + the champion; "
        f"{len(cfgs)} x {len(DEPTHS)} depths"
    )


def params_for(ds, cfg, depth):
    """A drawn configuration -> the knobs make_config expects. The width is taken from depth 3's ladder whatever the
    depth (the champion keeps its own width, capped at that ladder's top), so depth is the only difference within a
    pair; `depth` is accepted for symmetry and does not change the result."""
    lad = ladder(ds)[3]
    d = cfg["d_champ"] if cfg.get("champion") else lad[cfg["d_level"]]
    p = dict(
        lr=float(cfg["lr"]), d=min(int(d), lad[-1]), head=float(cfg["head"]), readout=cfg["readout"]
    )
    if "damp" in cfg:
        p["damp"] = float(cfg["damp"])
    else:
        p.update(
            a_var=float(cfg["a_var"]),
            lambda2=float(cfg["lambda2"]),
            label_smoothing=float(cfg["label_smoothing"]),
        )
    if "alpha" in cfg:
        p["alpha"] = float(cfg["alpha"])
    return p


def cell_dir(ds, i, depth):
    return os.path.join(OUT, ds + PAIRED_TAG, f"c{i:02d}_n{depth}")


def seed_results(cd):
    """{seed: (val, test)} for a cell."""
    out = {}
    if os.path.isdir(cd):
        for f in os.listdir(cd):
            if f.startswith("seed") and f.endswith(".json"):
                r = json.load(open(os.path.join(cd, f)))
                out[int(r["seed"])] = (r["val"], r["test"])
    return out


def write_aggregate(ds, i, depth, p, champion_flag):
    cd = cell_dir(ds, i, depth)
    r = seed_results(cd)
    seeds = sorted(r)
    json.dump(
        dict(
            config_index=i,
            depth=depth,
            params=p,
            champion=champion_flag,
            seeds=seeds,
            val_per_seed=[r[sd][0] for sd in seeds],
            test_per_seed=[r[sd][1] for sd in seeds],
            val=statistics.mean(r[sd][0] for sd in seeds) if seeds else None,
            test=statistics.mean(r[sd][1] for sd in seeds) if seeds else None,
            n_seeds=len(seeds),
        ),
        open(os.path.join(cd, "result.json"), "w"),
        indent=1,
    )


def top_cells(ds, spec, k, min_seeds):
    """The top-k configurations per depth by mean validation, among cells with at least min_seeds seeds."""
    keep = set()
    for n in DEPTHS:
        scored = []
        for i in range(len(spec["configs"])):
            r = seed_results(cell_dir(ds, i, n))
            if len(r) >= min_seeds:
                scored.append((statistics.mean(v[0] for v in r.values()), i))
        keep.update((i, n) for _, i in sorted(scored, reverse=True)[:k])
    return keep


def _claim(path):
    """mkdir claim; a claim older than STALE_H with no result is taken over (its worker died)."""
    try:
        os.mkdir(path)
        return True
    except FileExistsError:
        if (time.time() - os.path.getmtime(path)) / 3600 < STALE_H:
            return False
        os.utime(path, None)
        return True


def run(ds, gpu, only_depth=None, deadline=None, seeds=SEARCH_SEEDS, top_k=None):
    """Run every (config, depth, seed) that is still missing, claiming per SEED so seeds can be added to finished cells
    without re-running them (the protocol: 3 seeds on every cell, then the top 5 per depth to 5 seeds).
    top_k restricts the cells to the top-k per depth by mean val among cells with >= 3 seeds."""
    spec = json.load(open(os.path.join(OUT, ds + PAIRED_TAG, "configs.json")))
    claims = os.path.join(OUT, ds + PAIRED_TAG, "claims")
    os.makedirs(claims, exist_ok=True)
    depths = DEPTHS if only_depth is None else [only_depth]
    order = [(i, n) for n in depths for i in range(len(spec["configs"]))]
    if top_k:
        keep = top_cells(ds, spec, top_k, min_seeds=3)
        order = [c for c in order if c in keep]
    for i, depth in order:
        if deadline and time.time() > deadline:
            print(f"[uea_paired] deadline reached; stopping before c{i:02d} n{depth}", flush=True)
            return
        cd = cell_dir(ds, i, depth)
        have = seed_results(cd)
        missing = [sd for sd in seeds if sd not in have]
        if not missing:
            continue
        os.makedirs(cd, exist_ok=True)
        p = params_for(ds, spec["configs"][i], depth)
        for sd in missing:
            if not _claim(os.path.join(claims, f"c{i:02d}_n{depth}_s{sd}")):
                continue
            t0 = time.time()
            try:
                v, t, _ = run_one(ds, depth, p, sd, SEARCH_STEPS, f"paired_c{i:02d}", gpu)
                json.dump(
                    dict(seed=sd, val=v, test=t, params=p, wall_s=time.time() - t0),
                    open(os.path.join(cd, f"seed{sd}.json"), "w"),
                    indent=1,
                )
                print(
                    f"[uea_paired] {ds} c{i:02d} n{depth} seed {sd}: val {v:.2f} test {t:.2f}",
                    flush=True,
                )
            except BaseException as e:  # noqa: BLE001 (any failure is recorded and its claim released)
                json.dump(
                    dict(seed=sd, error=str(e)[:300]),
                    open(os.path.join(cd, f"failed_seed{sd}.json"), "w"),
                )
                print(
                    f"[uea_paired] {ds} c{i:02d} n{depth} seed {sd} FAILED: {str(e)[:120]}",
                    flush=True,
                )
                try:  # release the claim so a later pass retries it (a kept claim
                    os.rmdir(
                        os.path.join(claims, f"c{i:02d}_n{depth}_s{sd}")
                    )  # would make an OOM hole permanent)
                except OSError:
                    pass
            write_aggregate(ds, i, depth, p, spec["configs"][i].get("champion", False))


def load(ds, min_seeds=1):
    spec = json.load(open(os.path.join(OUT, ds + PAIRED_TAG, "configs.json")))
    res = {}
    for i in range(len(spec["configs"])):
        for n in DEPTHS:
            r = seed_results(cell_dir(ds, i, n))
            if len(r) >= min_seeds:
                seeds = sorted(r)
                res[(i, n)] = dict(
                    val=statistics.mean(r[sd][0] for sd in seeds),
                    test=statistics.mean(r[sd][1] for sd in seeds),
                    n_seeds=len(seeds),
                    params=params_for(ds, spec["configs"][i], n),
                    champion=spec["configs"][i].get("champion", False),
                )
    return spec, res


def report(ds):
    spec, res = load(ds)
    ns = sorted({r["n_seeds"] for r in res.values()})
    print(
        f"# {ds}: paired depth ablation, matched width "
        f"({len(res)} cells with results of {len(spec['configs']) * len(DEPTHS)}; seeds per cell {ns})\n"
    )
    print("| depth | cells | best val | seeds | its test | d | lr | readout | config |")
    print("|---|---|---|---|---|---|---|---|---|")
    for n in DEPTHS:
        at = [(k, r) for k, r in res.items() if k[1] == n]
        if not at:
            print(f"| {n} | 0 | | | | | | | |")
            continue
        (i, _), b = max(at, key=lambda kr: kr[1]["val"])
        print(
            f"| {n} | {len(at)} | {b['val']:.2f} | {b['n_seeds']} | {b['test']:.2f} | {b['params']['d']} | {b['params']['lr']:.3g} | "
            f"{b['params']['readout']} | c{i:02d}{' (champion)' if b['champion'] else ''} |"
        )
    print("\n| pair | configs with both | deeper better | median gain (val) |")
    print("|---|---|---|---|")
    for lo, hi in ((1, 2), (2, 3), (1, 3)):
        both = [
            (res[(i, lo)]["val"], res[(i, hi)]["val"])
            for i in range(len(spec["configs"]))
            if (i, lo) in res and (i, hi) in res
        ]
        if not both:
            print(f"| n{lo} -> n{hi} | 0 | | |")
            continue
        diffs = [h - l for l, h in both]
        print(
            f"| n{lo} -> n{hi} | {len(both)} | {sum(d > 0 for d in diffs)}/{len(both)} | "
            f"{statistics.median(diffs):+.2f} |"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run", "report"])
    ap.add_argument("--dataset", required=True, choices=sorted(STEM))
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--depth", type=int, choices=[1, 2, 3])
    ap.add_argument(
        "--stop_at",
        default="",
        help="epoch seconds; the worker stops claiming new cells after this",
    )
    ap.add_argument(
        "--seeds",
        default="0,1",
        help="split seeds every cell should have, e.g. 0,1,2; only missing ones run",
    )
    ap.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="restrict to the top-k configs per depth (cells with >= 3 seeds)",
    )
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.dataset)
    elif a.cmd == "run":
        run(
            a.dataset,
            a.gpu,
            a.depth,
            float(a.stop_at) if a.stop_at else None,
            seeds=tuple(int(x) for x in a.seeds.split(",")),
            top_k=a.top_k,
        )
    else:
        report(a.dataset)
