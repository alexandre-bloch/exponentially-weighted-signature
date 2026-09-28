"""Walker's (log-NCDE) per-seed train/val/test splits of the UEA datasets, reproduced exactly, so the EWS is evaluated
on the partitions of his benchmark.

Walker's split (use_presplit=False): the train and test sets are pooled and deduplicated (np.unique over the features,
the first occurrence kept), and for seed s

    datasetkey = jr.split(jr.PRNGKey(s), 4)[0];  permkey = jr.split(datasetkey)[0];  idxs = jr.permutation(permkey, N)
    train, val, test = idxs[:int(0.7 N)], idxs[int(0.7 N):int(0.85 N)], idxs[int(0.85 N):]

(a plain jr.permutation(jr.PRNGKey(s), N) gives different splits). Raw paths and linspace clocks are stored and the
loader scales them. File i holds Walker's seed WALKER_SEEDS[i], recorded as `walker_seed` in its json:
data/<name>_walker/splits/<name>_walker_seed{0..4}.npz.

    python experiments/uea/make_walker_splits.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import jax.random as jr
import numpy as np

from experiments.uea.make_uea_splits import _clocks, _counts, build_pool

WALKER_SEEDS = [2345, 3456, 4567, 5678, 6789]
DATASETS = [
    "EigenWorms",
    "EthanolConcentration",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "Heartbeat",
    "MotorImagery",
]
SHARED_RAW = REPO_ROOT / "data" / "eigenworms" / "raw"


def walker_indices(seed: int, N: int):
    """Reproduce log-NCDE's exact permutation key chain -> (train_idx, val_idx, test_idx)."""
    key = jr.PRNGKey(seed)
    datasetkey = jr.split(key, 4)[0]  # train.py:406  -> [datasetkey, modelkey, trainkey, key]
    permkey = jr.split(datasetkey)[0]  # datasets.py:125  permkey, key = jr.split(key)
    idxs = np.asarray(jr.permutation(permkey, N))
    b1, b2 = int(N * 0.7), int(N * 0.85)  # datasets.py:126-127 (not n_tr+n_va; flooring differs)
    return idxs[:b1], idxs[b1:b2], idxs[b2:]


def write_walker_split(pool_p, pool_l, class_names, dataset, idx, wseed, out) -> bool:
    """`wseed` (Walker's seed) drives the partition; `idx` (0..4) is the local file label / our
    model-init seed. The data partition is Walker's; the training RNG label is arbitrary."""
    N = len(pool_p)
    tr, va, te = walker_indices(wseed, N)

    def P(i):
        return pool_p[i].astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(
        out,
        train_paths=P(tr),
        val_paths=P(va),
        test_paths=P(te),
        train_clocks=_clocks(len(tr), pool_p.shape[1]),
        val_clocks=_clocks(len(va), pool_p.shape[1]),
        test_clocks=_clocks(len(te), pool_p.shape[1]),
        train_labels=pool_l[tr].astype(np.int32),
        val_labels=pool_l[va].astype(np.int32),
        test_labels=pool_l[te].astype(np.int32),
    )
    nclass = len(class_names)
    ok = all(len(np.unique(pool_l[i])) == nclass for i in (tr, va, te))
    meta = {
        "dataset": dataset,
        "task_type": "multiclass",
        "n_classes": nclass,
        "class_names": [str(c) for c in class_names],
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
        "T": int(pool_p.shape[1]),
        "d": int(pool_p.shape[2]),
        "seed": int(idx),
        "walker_seed": int(wseed),
        "N_pool": int(N),
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "deduplicated": True,
        "split": "walker_jax_70_15_15",
        "protocol": "walker_exact_jax_rng",
        "non_degenerate": bool(ok),
    }
    json.dump(meta, open(out.replace(".npz", ".json"), "w"), indent=2)
    flag = "" if ok else "   *** DEGENERATE (a class missing from a split) ***"
    print(
        f"  seed{idx} (walker {wseed}): {len(tr)}/{len(va)}/{len(te)}  N={N}  "
        f"test-classes={_counts(pool_l[te])}{flag}"
    )
    return ok


def process(dataset, raw_dir, seeds):
    pool_p, pool_l, class_names = build_pool(dataset, raw_dir)
    ds = dataset.lower() + "_walker"
    out_dir = REPO_ROOT / "data" / ds / "splits"
    degenerate = []
    for idx, wseed in enumerate(seeds):
        out = str(out_dir / f"{ds}_seed{idx}.npz")
        if not write_walker_split(pool_p, pool_l, class_names, dataset, idx, wseed, out):
            degenerate.append(idx)
    # main npz (= idx 0, Walker seed seeds[0]) so any config dataset.file resolves
    main = str(REPO_ROOT / "data" / ds / f"{ds}.npz")
    write_walker_split(pool_p, pool_l, class_names, dataset, 0, seeds[0], main)
    print(
        f"Done: {dataset} -> data/{ds}/ ({len(seeds)} walker splits, idx 0..{len(seeds) - 1} "
        f"= walker seeds {seeds})  degenerate={degenerate or 'none'}\n"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Walker-EXACT (JAX-RNG) per-seed splits for the UEA datasets"
    )
    ap.add_argument(
        "--dataset",
        help="UEA name (EigenWorms / EthanolConcentration / SelfRegulationSCP1 / SelfRegulationSCP2)",
    )
    ap.add_argument("--all", action="store_true", help="process all 4 Walker datasets")
    ap.add_argument("--raw_dir", default=str(SHARED_RAW))
    ap.add_argument("--seeds", default=",".join(map(str, WALKER_SEEDS)))
    args = ap.parse_args()
    if not args.all and not args.dataset:
        ap.error("pass --dataset <Name> or --all")
    seeds = [int(x) for x in args.seeds.split(",")]
    for d in DATASETS if args.all else [args.dataset]:
        process(d, args.raw_dir, seeds)


if __name__ == "__main__":
    main()
