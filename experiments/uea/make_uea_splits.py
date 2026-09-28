"""Pool, deduplicate and split a UEA dataset with Walker's preprocessing: the train and test sets pooled, np.unique
over the features (first occurrence kept), then 70/15/15 per seed. This script's own command line draws the split with
numpy's RandomState; make_walker_splits.py reuses the pooling and draws Walker's JAX splits, which the banked results
use.

    python experiments/uea/make_uea_splits.py --dataset SelfRegulationSCP1 --seeds 0-9
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from experiments.uea.eigenworms.download_eigenworms import (
    deduplicate,
    download_uea,
    load_arff,
)

SHARED_RAW = REPO_ROOT / "data" / "eigenworms" / "raw"  # archive has every UEA dataset


def build_pool(dataset: str, raw_dir: str):
    arff_dir = download_uea(raw_dir)
    ds_dir = os.path.join(arff_dir, dataset)
    train_file = os.path.join(ds_dir, f"{dataset}_TRAIN.arff")
    test_file = os.path.join(ds_dir, f"{dataset}_TEST.arff")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"ARFF not found for {dataset}: {train_file}")
    print(f"Loading {dataset} ARFF ...")
    tr_p, te_p, tr_l, te_l, class_names = load_arff(train_file, test_file)
    all_p = np.concatenate([tr_p, te_p], axis=0)
    all_l = np.concatenate([tr_l, te_l], axis=0)
    pool_p, pool_l, _ = deduplicate(all_p, all_l)  # np.unique keep-first == Walker
    print(f"  pool: {pool_p.shape}  classes={class_names} ({len(class_names)})")
    return pool_p, pool_l, class_names


def _clocks(n: int, t_len: int) -> np.ndarray:
    clk = np.linspace(0.0, 1.0, t_len, dtype=np.float32)
    return np.broadcast_to(clk[None, :], (n, t_len)).copy()


def _counts(lab):
    u, c = np.unique(lab, return_counts=True)
    return {int(k): int(v) for k, v in zip(u, c)}


def write_split(
    pool_p, pool_l, class_names, dataset, seed, out, train_ratio=0.7, val_ratio=0.15
) -> bool:
    N = len(pool_p)
    n_tr = int(N * train_ratio)
    n_va = int(N * val_ratio)
    rng = np.random.RandomState(seed)  # RandomState (our chosen RNG; Walker uses JAX)
    perm = rng.permutation(N)
    tr, va, te = perm[:n_tr], perm[n_tr : n_tr + n_va], perm[n_tr + n_va :]

    def P(idx):
        return pool_p[idx].astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(
        out,
        train_paths=P(tr),
        val_paths=P(va),
        test_paths=P(te),
        train_clocks=_clocks(len(tr), pool_p.shape[1]),
        val_clocks=_clocks(len(va), pool_p.shape[1]),
        test_clocks=_clocks(len(te), pool_p.shape[1]),
        train_labels=pool_l[tr],
        val_labels=pool_l[va],
        test_labels=pool_l[te],
    )
    nclass = len(class_names)
    tr_ok = len(np.unique(pool_l[tr])) == nclass
    te_ok = len(np.unique(pool_l[te])) == nclass
    va_ok = len(np.unique(pool_l[va])) == nclass
    ok = tr_ok and te_ok and va_ok
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
        "seed": int(seed),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "deduplicated": True,
        "split": "random_70_15_15",
        "protocol": "walker_same_protocol_repo_rng",
        "non_degenerate": ok,
    }
    json.dump(meta, open(out.replace(".npz", ".json"), "w"), indent=2)
    flag = "" if ok else "   *** DEGENERATE (a class missing from train/val/test) ***"
    print(f"  seed={seed}: {len(tr)}/{len(va)}/{len(te)}  test-classes={_counts(pool_l[te])}{flag}")
    return ok


def parse_seeds(s: str):
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return out


def main():
    p = argparse.ArgumentParser(description="Walker-protocol per-seed splits for any UEA dataset")
    p.add_argument(
        "--dataset",
        required=True,
        help="UEA name (EigenWorms / EthanolConcentration / SelfRegulationSCP1 / SelfRegulationSCP2 / ...)",
    )
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--raw_dir", default=str(SHARED_RAW))
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument(
        "--verify",
        action="store_true",
        help="(EigenWorms only) check seed split matches the committed eigenworms split",
    )
    args = p.parse_args()

    seeds = parse_seeds(args.seeds)
    pool_p, pool_l, class_names = build_pool(args.dataset, args.raw_dir)
    out_dir = REPO_ROOT / "data" / args.dataset.lower() / "splits"
    degenerate = []
    for s in seeds:
        out = str(out_dir / f"{args.dataset.lower()}_seed{s}.npz")
        if not write_split(
            pool_p, pool_l, class_names, args.dataset, s, out, args.train_ratio, args.val_ratio
        ):
            degenerate.append(s)
        if args.verify and args.dataset == "EigenWorms":
            ref = REPO_ROOT / "data" / "eigenworms" / "splits" / f"eigenworms_seed{s}.npz"
            if ref.exists() and os.path.abspath(out) != os.path.abspath(str(ref)):
                a, b = np.load(out), np.load(ref)
                same = all(np.array_equal(a[k], b[k]) for k in a.files if k in b.files)
                print(f"  [verify] seed {s} matches existing eigenworms split: {same}")

    print(f"\nDone: {args.dataset} -> {len(seeds)} splits in {out_dir}")
    print(f"  degenerate seeds (exclude from eval): {degenerate or 'none'}")


if __name__ == "__main__":
    main()
