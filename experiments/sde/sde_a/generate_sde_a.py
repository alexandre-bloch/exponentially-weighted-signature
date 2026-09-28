"""Generate a coupled-oscillator dataset (simulator.py) as an NPZ of raw paths, clocks and targets; the config's
preprocessing is applied at load time. Seed s draws the Brownian paths of sequence i from numpy generator s + i and
shuffles the 70/15/15 split with generator s.

Keys: {train,val,test}_paths (n, T, 2) = [W1, W2], {train,val,test}_clocks (n, T), and {train,val,test}_targets
(n, T), X1 at every step.

    python experiments/sde/sde_a/generate_sde_a.py --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

import numpy as np

from experiments.sde.sde_a.simulator import simulate_sde


def generate(
    n_sequences: int = 750,
    T: float = 4.0,
    n_steps: int = 10000,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 0,
    output: str | None = None,
) -> str:
    if output is None:
        output = str(REPO_ROOT / "data" / "sde_a" / f"sde_a_seed{seed}.npz")

    print(f"Simulating {n_sequences} SDE-A sequences (T={T}, n_steps={n_steps})...")
    paths_list, clocks_list, targets_list = simulate_sde(n_sequences, T, n_steps, seed)

    n_total = len(paths_list)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    rng = np.random.default_rng(seed=seed)
    indices = rng.permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_paths = np.stack([paths_list[i] for i in train_idx]).astype(np.float32)
    val_paths = np.stack([paths_list[i] for i in val_idx]).astype(np.float32)
    test_paths = np.stack([paths_list[i] for i in test_idx]).astype(np.float32)
    train_clocks = np.stack([clocks_list[i] for i in train_idx]).astype(np.float32)
    val_clocks = np.stack([clocks_list[i] for i in val_idx]).astype(np.float32)
    test_clocks = np.stack([clocks_list[i] for i in test_idx]).astype(np.float32)
    train_targets = np.stack([targets_list[i] for i in train_idx]).astype(np.float32)
    val_targets = np.stack([targets_list[i] for i in val_idx]).astype(np.float32)
    test_targets = np.stack([targets_list[i] for i in test_idx]).astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    np.savez_compressed(
        output,
        train_paths=train_paths,
        val_paths=val_paths,
        test_paths=test_paths,
        train_clocks=train_clocks,
        val_clocks=val_clocks,
        test_clocks=test_clocks,
        train_targets=train_targets,
        val_targets=val_targets,
        test_targets=test_targets,
    )

    meta = {
        "dataset": "sde_a",
        "n_sequences": n_total,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "T": T,
        "n_steps": n_steps,
        "d": int(train_paths.shape[2]),
        "seed": seed,
    }
    meta_path = output.replace(".npz", ".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved to {output}")
    print(f"  paths shape:   {train_paths.shape}   (n_train, T, d)")
    print(f"  clocks shape:  {train_clocks.shape}  (n_train, T)")
    print(f"  targets shape: {train_targets.shape}  (n_train, T)")
    print(f"  split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a coupled-oscillator dataset")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--n_sequences", type=int, default=750)
    parser.add_argument("--T", type=float, default=4.0)
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    generate(
        n_sequences=args.n_sequences,
        T=args.T,
        n_steps=args.n_steps,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        output=args.output,
    )
