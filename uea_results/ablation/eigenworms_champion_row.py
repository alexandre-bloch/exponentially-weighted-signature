"""The champion row of the EigenWorms depth ablation: the banked configuration at depths 1, 2 and 3, everything else the
same, at the ablation's 6,000 steps and width 8 (depth 3's level-3 state does not fit above it), five split seeds per
depth. At depth 2 this is the banked run, so its five seeds reproduce `../eigenworms/result.md`.

Two computational settings differ across the row: depth 3 uses the `series` leaf (the closed form exists for depths 1
and 2 only) and the `assoc` combine (the block kernel compiles too slowly there). The clock's `norm_constant` is not
re-calibrated per depth: a clock reading the input X is sized by X's channels, so its increments at initialisation do
not depend on depth.

    python uea_results/ablation/eigenworms_champion_row.py --gpu 0      # 3 depths x 5 seeds = 15 runs
    python uea_results/ablation/eigenworms_champion_row.py --report
"""

import argparse
import copy
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import OUT as RUN_OUT  # noqa: E402
from common import leaf_of, run_one, set_depth, use_series_leaf

EWS = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(RUN_OUT, "eigenworms_champion_row")
SEEDS = (0, 1, 2, 3, 4)
DEPTHS = (1, 2, 3)
STEPS = 6000
WIDTH = 8  # EigenWorms' matched ladder is width 8 at every depth
CLOSED_FORM_LEAVES = ("plane_closed", "block_series", "jordan_closed", "jordan_closed_tables")


def make(cfg_path, depth):
    c = copy.deepcopy(json.load(open(cfg_path)))
    m = c["model"]
    set_depth(m, depth)
    m["B"]["out_dim"] = WIDTH
    if depth > 2:
        if leaf_of(m) in CLOSED_FORM_LEAVES:
            use_series_leaf(
                m
            )  # the closed forms exist for depths 1-2 only; series needs an explicit order
        m.setdefault("compute", {})["combine"] = (
            "assoc"  # the block kernel does not compile at depth 3 in reasonable time
        )
    c["training"].update(
        num_steps=STEPS, log_every=STEPS, eval_every=1000, early_stopping_patience=10**9
    )
    return c


def run(cfg_path, gpu, depths):
    os.makedirs(os.path.join(OUT, "claims"), exist_ok=True)
    json.dump(
        json.load(open(cfg_path)), open(os.path.join(OUT, "champion_config.json"), "w"), indent=2
    )
    for depth in depths:
        cfg = make(cfg_path, depth)
        for sd in SEEDS:
            out = os.path.join(OUT, f"n{depth}_s{sd}.json")
            if os.path.exists(out):
                continue
            try:
                os.mkdir(os.path.join(OUT, "claims", f"n{depth}_s{sd}"))
            except FileExistsError:
                continue
            try:
                v, t, best = run_one(
                    "EigenWorms",
                    depth,
                    {"d": WIDTH},
                    sd,
                    STEPS,
                    f"champrow_n{depth}",
                    gpu,
                    config=cfg,
                )
                json.dump(
                    dict(
                        depth=depth,
                        width=WIDTH,
                        seed=sd,
                        val=v,
                        test=t,
                        best_step=best,
                        leaf=leaf_of(cfg["model"]),
                    ),
                    open(out, "w"),
                    indent=1,
                )
                print(
                    f"[champ row] depth {depth} seed {sd}: val {v:.2f} test {t:.2f} (best step {best})",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[champ row] depth {depth} seed {sd} FAILED: {str(e)[:200]}", flush=True)
                os.rmdir(os.path.join(OUT, "claims", f"n{depth}_s{sd}"))


def report():
    print(
        "# EigenWorms: the banked configuration at each depth (width 8, 6,000 steps, five split seeds)\n"
    )
    print(
        "| depth | leaf | seeds | mean val | mean test | test pop. sd | per-seed val | per-seed test | best step per seed |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for depth in DEPTHS:
        got = [
            json.load(open(os.path.join(OUT, f"n{depth}_s{s}.json")))
            for s in SEEDS
            if os.path.exists(os.path.join(OUT, f"n{depth}_s{s}.json"))
        ]
        if not got:
            continue
        v = [g["val"] for g in got]
        t = [g["test"] for g in got]
        print(
            f"| {depth} | {got[0].get('leaf', '?')} | {len(got)} | {statistics.mean(v):.2f} | {statistics.mean(t):.2f} | "
            f"{statistics.pstdev(t) if len(t) > 1 else 0:.2f} | {', '.join(f'{x:.2f}' for x in v)} | "
            f"{', '.join(f'{x:.2f}' for x in t)} | {', '.join(str(g['best_step']) for g in got)} |"
        )
    p = os.path.join(OUT, "champion_config.json")
    if os.path.exists(p):
        c = json.load(open(p))
        m = c["model"]
        print(
            f"\nConfiguration: depth {m.get('depth', m.get('ews_depth'))}, width {m['B']['out_dim']}, "
            f"{m['A']['parametrization']}, {c['preprocessing']['clock']['mode']} clock, {m['readout']} readout, "
            f"lr {c['training']['lr_features']:.4g}, batch {c['training']['batch_size']} "
            f"(the ablation runs it at {STEPS} steps and width {WIDTH})."
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", default=os.path.join(EWS, "uea_results", "eigenworms", "config.json")
    )
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--depths", type=int, nargs="+", default=list(DEPTHS))
    a = ap.parse_args()
    report() if a.report else run(a.config, a.gpu, tuple(a.depths))
