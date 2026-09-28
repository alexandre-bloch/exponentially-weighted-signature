"""The champion row of the Ethanol depth ablation: the banked configuration trained at depth 1, 2 and 3 with everything
else identical, at the ablation's own horizon (6,000 steps) and on the matched width ladder (cap 16), five split seeds
per depth. It works for any config file, so it describes whichever configuration is banked.

    python uea_results/ablation/ethanol_champion_row.py --gpu 0      # 3 depths x 5 seeds = 15 runs
    python uea_results/ablation/ethanol_champion_row.py --report
--config defaults to EthanolConcentration/champion_row_config.json next to this file (the banked Ethanol
configuration, identical to uea_results/ethanol/config.json).
"""

import argparse
import copy
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (  # noqa: E402
    GATED,
    calibrate_norm_constant,
    leaf_of,
    run_one,
    set_depth,
    use_series_leaf,
)
from common import OUT as RUN_OUT

OUT = os.path.join(RUN_OUT, "ethanol_champion_row")
SEEDS = (0, 1, 2, 3, 4)
DEPTHS = (1, 2, 3)
STEPS = 6000  # the ablation's horizon, so the row is comparable with the 33 drawn configurations
WIDTH_CAP = 16  # the depth-3 ladder's top: the matched-width convention of the ablation


def make(cfg_path, depth):
    c = copy.deepcopy(json.load(open(cfg_path)))
    m = c["model"]
    width = min(int(m["B"]["out_dim"]), WIDTH_CAP)
    set_depth(m, depth)
    m["B"]["out_dim"] = width
    m["A"]["blocks"] = [2] * (width // 2)
    if depth > 2 and leaf_of(m) in ("plane_closed", "block_series"):
        use_series_leaf(m)  # the fast leaves exist for depths 1-2 only
    m.setdefault("compute", {})["combine"] = "assoc"
    c["training"].update(num_steps=STEPS, log_every=STEPS, early_stopping_patience=10**9)
    if c["preprocessing"].get("clock", {}).get("mode") in GATED:
        c["preprocessing"]["clock"]["norm_constant"] = float(
            calibrate_norm_constant("EthanolConcentration", c)
        )
    return c, width


def run(cfg_path, gpu):
    os.makedirs(os.path.join(OUT, "claims"), exist_ok=True)
    json.dump(
        json.load(open(cfg_path)), open(os.path.join(OUT, "champion_config.json"), "w"), indent=2
    )
    for depth in DEPTHS:
        cfg, width = make(cfg_path, depth)
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
                    "EthanolConcentration",
                    depth,
                    {"d": width},
                    sd,
                    STEPS,
                    f"champrow_n{depth}",
                    gpu,
                    config=cfg,
                )
                json.dump(
                    dict(depth=depth, width=width, seed=sd, val=v, test=t, best_step=best),
                    open(out, "w"),
                    indent=1,
                )
                print(f"[champ row] depth {depth} seed {sd}: val {v:.2f} test {t:.2f}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[champ row] depth {depth} seed {sd} FAILED: {str(e)[:160]}", flush=True)
                os.rmdir(os.path.join(OUT, "claims", f"n{depth}_s{sd}"))


def report():
    print(
        "# Ethanol: the banked configuration at each depth (matched width, 6,000 steps, five split seeds)\n"
    )
    print("| depth | width | seeds | mean val | mean test | test sd | per-seed test |")
    print("|---|---|---|---|---|---|---|")
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
            f"| {depth} | {got[0]['width']} | {len(got)} | {statistics.mean(v):.2f} | {statistics.mean(t):.2f} | "
            f"{statistics.pstdev(t) if len(t) > 1 else 0:.2f} | {', '.join(f'{x:.2f}' for x in t)} |"
        )
    p = os.path.join(OUT, "champion_config.json")
    if os.path.exists(p):
        c = json.load(open(p))
        m = c["model"]
        print(
            f"\nConfiguration: depth {m.get('depth', m.get('ews_depth'))}, width {m['B']['out_dim']}, "
            f"{m['A']['parametrization']}, {c['preprocessing']['clock']['mode']} clock, {m['readout']} readout, "
            f"lr {c['training']['lr_features']:.4g}, batch {c['training']['batch_size']} "
            f"(the ablation runs it at {STEPS} steps and width <= {WIDTH_CAP})."
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", default=os.path.join(HERE, "EthanolConcentration", "champion_row_config.json")
    )
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report() if a.report else run(a.config, a.gpu)
