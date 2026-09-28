"""Download the UEA multivariate archive (if absent) and write EigenWorms as an NPZ.

The ARFF train and test files are pooled and deduplicated (Walker's preprocessing, which matters for EigenWorms), the
original train set is split 85/15 into train and validation, and the original test set is kept. The NPZ holds
{train,val,test}_paths (n, T, d) unscaled, {train,val,test}_clocks (n, T) uniform and {train,val,test}_labels (n,) in
0..4. The downloaded archive also serves the other UEA datasets (make_walker_splits.py).

    python experiments/uea/eigenworms/download_eigenworms.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import warnings
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

try:
    from sktime.datasets import load_from_arff_to_dataframe
except ImportError:
    raise ImportError("sktime required: pip install sktime")


UEA_URL = "http://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_arff.zip"
DATASET = "EigenWorms"


def download_uea(raw_dir: str) -> str:
    arff_dir = os.path.join(raw_dir, "Multivariate_arff")
    if os.path.exists(arff_dir):
        print(f"UEA archive already present at {arff_dir}")
        return arff_dir

    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "uea.zip")
    print(f"Downloading UEA archive from {UEA_URL} ...")
    urllib.request.urlretrieve(UEA_URL, zip_path)
    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)
    os.remove(zip_path)
    print(f"Extracted to {arff_dir}")
    return arff_dir


def load_arff(train_file: str, test_file: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        train_df, train_labels = load_from_arff_to_dataframe(train_file)
        test_df, test_labels = load_from_arff_to_dataframe(test_file)

    def to_array(df):
        expanded = df.map(lambda x: x.values).values
        return np.stack([np.vstack(x).T for x in expanded])  # (n, T, d)

    train_paths = to_array(train_df)
    test_paths = to_array(test_df)

    encoder = LabelEncoder().fit(train_labels)
    train_labels_int = encoder.transform(train_labels).astype(np.int32)
    test_labels_int = encoder.transform(test_labels).astype(np.int32)

    return train_paths, test_paths, train_labels_int, test_labels_int, list(encoder.classes_)


def deduplicate(paths: np.ndarray, labels: np.ndarray):
    _, first_idx = np.unique(paths, axis=0, return_index=True)
    keep = np.sort(first_idx)
    n_removed = len(paths) - len(keep)
    if n_removed > 0:
        print(f"  Removed {n_removed} duplicate samples ({n_removed / len(paths) * 100:.1f}%)")
    return paths[keep], labels[keep], keep


def generate(
    output: str | None = None,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    raw_dir: str | None = None,
) -> str:
    if output is None:
        output = str(REPO_ROOT / "data" / "eigenworms" / "eigenworms.npz")
    if raw_dir is None:
        raw_dir = str(REPO_ROOT / "data" / "eigenworms" / "raw")

    # 1. Download
    arff_dir = download_uea(raw_dir)

    # 2. Load ARFF
    ds_dir = os.path.join(arff_dir, DATASET)
    train_file = os.path.join(ds_dir, f"{DATASET}_TRAIN.arff")
    test_file = os.path.join(ds_dir, f"{DATASET}_TEST.arff")
    print(f"\nLoading {DATASET} ARFF files ...")
    train_paths, test_paths, train_labels, test_labels, class_names = load_arff(
        train_file, test_file
    )
    print(f"  Train: {train_paths.shape}  Test: {test_paths.shape}")
    print(f"  Classes: {class_names}")

    # 3. Concatenate and deduplicate (matches Walker / original eigenworms loader)
    all_paths = np.concatenate([train_paths, test_paths], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)
    all_paths_dedup, all_labels_dedup, _ = deduplicate(all_paths, all_labels)

    # 4. Random 70/15/15 split on all deduplicated data (matches original use_original_split=false)
    n_total = len(all_paths_dedup)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    rng = np.random.RandomState(seed)  # RandomState to match original numpy RNG
    perm = rng.permutation(n_total)

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    final_train_paths = all_paths_dedup[train_idx].astype(np.float32)
    final_train_labels = all_labels_dedup[train_idx]
    final_val_paths = all_paths_dedup[val_idx].astype(np.float32)
    final_val_labels = all_labels_dedup[val_idx]
    final_test_paths = all_paths_dedup[test_idx].astype(np.float32)
    final_test_labels = all_labels_dedup[test_idx]

    print(
        f"\n  Split: {len(final_train_paths)} train / {len(final_val_paths)} val / {len(final_test_paths)} test"
    )

    def make_clocks(n: int, t_len: int) -> np.ndarray:
        clock = np.linspace(0.0, 1.0, t_len, dtype=np.float32)
        return np.broadcast_to(clock[None, :], (n, t_len)).copy()

    train_clocks = make_clocks(len(final_train_paths), final_train_paths.shape[1])
    val_clocks = make_clocks(len(final_val_paths), final_val_paths.shape[1])
    test_clocks = make_clocks(len(final_test_paths), final_test_paths.shape[1])

    # 5. Save NPZ
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    np.savez_compressed(
        output,
        train_paths=final_train_paths,
        val_paths=final_val_paths,
        test_paths=final_test_paths,
        train_clocks=train_clocks,
        val_clocks=val_clocks,
        test_clocks=test_clocks,
        train_labels=final_train_labels,
        val_labels=final_val_labels,
        test_labels=final_test_labels,
    )

    meta = {
        "dataset": "EigenWorms",
        "task_type": "multiclass",
        "n_classes": len(class_names),
        "class_names": class_names,
        "n_train": int(len(final_train_paths)),
        "n_val": int(len(final_val_paths)),
        "n_test": int(len(final_test_paths)),
        "T": int(final_train_paths.shape[1]),
        "d": int(final_train_paths.shape[2]),
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "deduplicated": True,
        "split": "random_70_15_15",
    }
    meta_path = output.replace(".npz", ".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {output}")
    print(f"  paths shape: {final_train_paths.shape}  (n_train, T, d)")
    print(f"  labels shape: {final_train_labels.shape}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare EigenWorms → NPZ")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument(
        "--raw_dir", type=str, default=None, help="Where to store the downloaded UEA archive"
    )
    args = parser.parse_args()

    generate(
        output=args.output,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        raw_dir=args.raw_dir,
    )
