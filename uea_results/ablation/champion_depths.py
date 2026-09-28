"""The banked champion of each dataset trained at depths 1, 2 and 3, everything else the same.

Each cell loads `uea_results/<dataset>/config.json` and changes only the signature depth: the bank's width, learning
rate, clock, batch, step budget, evaluation cadence, split files and regime (`--autotune off --deterministic-ops on`).
Ethanol's leaf `plane_closed` exists for depths 1 and 2 only, so its depth-3 cells use `van_loan`, which computes the
same local signatures. EigenWorms' row comes from `eigenworms_champion_row.py`.

    python uea_results/ablation/champion_depths.py --gpu 0                      # every dataset, depths 1-3, 5 seeds
    python uea_results/ablation/champion_depths.py --gpu 0 --datasets ethanol --depths 3
"""

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATASETS = {
    "ethanol": "ethanolconcentration_walker",
    "heartbeat": "heartbeat_walker",
    "motor": "motorimagery_walker",
    "scp1": "selfregulationscp1_walker",
    "scp2": "selfregulationscp2_walker",
}
LEAF_AT_DEPTH_3 = {"ethanol": "van_loan"}
SEEDS = (0, 1, 2, 3, 4)
_BEST = re.compile(r"Best\s+step=(\d+)\s+val_accuracy=([0-9.]+)")
_TEST = re.compile(r"Test metrics: \{'accuracy': ([0-9.]+)")


def cfg_for(ds, depth):
    c = copy.deepcopy(json.load(open(os.path.join(ROOT, "uea_results", ds, "config.json"))))
    key = "ews_depth" if "ews_depth" in c["model"] else "depth"
    c["model"][key] = int(depth)
    if int(depth) == 3 and ds in LEAF_AT_DEPTH_3:
        c["model"].setdefault("compute", {})["leaf"] = LEAF_AT_DEPTH_3[ds]
    return c


def run(ds, depth, seed, gpu, out):
    tag = f"{ds}_n{depth}_s{seed}"
    rec = os.path.join(out, f"{tag}.json")
    if os.path.exists(rec):
        return json.load(open(rec))
    os.makedirs(out, exist_ok=True)
    cp = os.path.join(out, f"{tag}_config.json")
    json.dump(cfg_for(ds, depth), open(cp, "w"), indent=1)
    stem = DATASETS[ds]
    split = os.path.join(ROOT, "data", stem, "splits", f"{stem}_seed{seed}.npz")
    env = dict(os.environ, PYTHONPATH=ROOT, PYTHONUNBUFFERED="1")
    cmd = [
        sys.executable,
        os.path.join(ROOT, "experiments", "uea", "train.py"),
        "--config",
        cp,
        "--gpu",
        str(gpu),
        "--seed",
        str(seed),
        "--dataset_file",
        split,
        "--out_ckpt",
        os.path.join(out, f"{tag}.eqx"),
        "--autotune",
        "off",
        "--deterministic-ops",
        "on",
    ]
    log = os.path.join(out, f"{tag}.log")
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    txt = open(log).read()
    mb, mt = _BEST.search(txt), _TEST.search(txt)
    row = {
        "dataset": ds,
        "depth": int(depth),
        "seed": seed,
        "wall_s": round(time.time() - t0, 1),
        "rc": rc,
    }
    if mb and mt:
        row.update(
            val=round(100 * float(mb.group(2)), 4),
            test=round(100 * float(mt.group(1)), 4),
            best_step=int(mb.group(1)),
        )
    else:
        row.update(failed=True, tail=" | ".join(txt.splitlines()[-3:])[:400])
    for p in (os.path.join(out, f"{tag}.eqx"),):
        if os.path.exists(p):
            os.remove(p)
    json.dump(row, open(rec, "w"), indent=1)
    print(f"{tag}: val {row.get('val')} test {row.get('test')} ({row['wall_s']} s)", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out", default=os.path.join(HERE, "champion_depths_runs"))
    ap.add_argument("--datasets", nargs="+", default=sorted(DATASETS))
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    a = ap.parse_args()
    for ds in a.datasets:
        for d in a.depths:
            for s in a.seeds:
                run(ds, d, s, a.gpu, a.out)


if __name__ == "__main__":
    main()
