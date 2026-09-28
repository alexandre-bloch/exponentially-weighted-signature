"""Generate an expressivity-gap dataset: Brownian paths and a depth-2 EWS target with a fixed operator.

The target is one component of the depth-2 EWS of the time-augmented path (t, W1, W2) under a
fixed operator A_true (EWS: a full matrix with eigenvalues -0.5 +/- 5.2i and 0.8; EFM:
diag(0.5, 0.3, 0.8); Sig: A = 0), computed in float64 with exact matrix exponentials. By default
the paths are used raw (time in its original units); with --normalize_paths the target is computed
on paths scaled to [0, 1] on the training set instead.

Usage:
    python experiments/sde/sde_d/generate_sde_d.py --a_true EWS --seed 0 --gpu 0
    python experiments/sde/sde_d/generate_sde_d.py --a_true EFM --seed 0 --normalize_paths

Output: data/sde_d/sde_d_<a_true>_seed<seed>.npz (raw paths, clocks and targets) plus a .json
with the generation parameters. The loader applies the config's preprocessing at training time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# GPU configuration before JAX is imported
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--gpu", type=str, default="0")
_pre, _ = _parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _pre.gpu
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np

sys.path.insert(0, str(REPO_ROOT))

from ews.clocks import ClockSpec
from ews.paths import PathSpec, prepare_paths_np
from experiments.sde.sde_d.simulator import compute_ews_targets, get_A_true, simulate_bm_paths


def generate(
    a_true: str = "EWS",
    seed: int = 0,
    normalize_paths: bool = False,
    target_depth: int = 2,
    target_component: int = 11,
    M: int = 10,
    n_sequences: int = 750,
    T: float = 5.0,
    n_steps: int = 10000,
    path_batch_size: int = 3,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    target_batch_size: int = 50,
    van_loan_expm: str = "exact",
    chen_expm: str = "exact",
    output: str | None = None,
) -> str:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    if output is None:
        suffix = "_norm" if normalize_paths else ""
        output = str(REPO_ROOT / "data" / "sde_d" / f"sde_d_{a_true}{suffix}_seed{seed}.npz")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    print(f"SDE-D: a_true={a_true} seed={seed} normalize_paths={normalize_paths}")
    paths_np, clocks_np = simulate_bm_paths(n_sequences, T, n_steps, seed, path_batch_size)

    # sequential split (no shuffling)
    n_total = len(paths_np)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    splits = {
        "train": (paths_np[:n_train], clocks_np[:n_train]),
        "val": (paths_np[n_train : n_train + n_val], clocks_np[n_train : n_train + n_val]),
        "test": (paths_np[n_train + n_val :], clocks_np[n_train + n_val :]),
    }

    # the path the target is computed on: (t, W1, W2), raw or scaled to [0, 1] on the training set
    path_spec = PathSpec(
        normalize=normalize_paths,
        normalize_mode="train_only",
        normalize_range="[0,1]",
        basepoint=False,
        clock_channel="main",
    )
    clock_spec = ClockSpec(mode="time", scaling="[0,1]" if normalize_paths else "none")
    A_np = get_A_true(a_true)
    print(f"  A_true:\n{A_np}\n  ||A_true||_F = {np.linalg.norm(A_np):.6f}")

    scaler = None
    targets = {}
    for split, (paths, clocks) in splits.items():
        paths_pp, clocks_pp, scaler, _ = prepare_paths_np(
            paths,
            clocks,
            path_spec,
            clock_spec,
            None,
            scaler=scaler,
            fit_scaler=split == "train",
            dtype=jnp.float32,
        )
        print(f"  targets ({split}) ...")
        targets[split] = compute_ews_targets(
            [np.array(p) for p in paths_pp],
            [np.array(c) for c in clocks_pp],
            A_np,
            target_component,
            N=target_depth,
            M=M,
            batch_size=target_batch_size,
            van_loan_expm=van_loan_expm,
            chen_expm=chen_expm,
        )

    arrays = {}
    for split, (paths, clocks) in splits.items():
        arrays[f"{split}_paths"] = np.stack([np.asarray(p, dtype=np.float32) for p in paths])
        arrays[f"{split}_clocks"] = np.stack([np.asarray(c, dtype=np.float32) for c in clocks])
        arrays[f"{split}_targets"] = np.stack(
            [np.asarray(t, dtype=np.float32) for t in targets[split]]
        )
    np.savez_compressed(output, **arrays)

    meta = {
        "a_true": a_true,
        "seed": seed,
        "normalize_paths": normalize_paths,
        "target_depth": target_depth,
        "target_component": target_component,
        "M": M,
        "van_loan_expm": van_loan_expm,
        "chen_expm": chen_expm,
        "n_sequences": n_total,
        "T": T,
        "n_steps": n_steps,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_total - n_train - n_val,
        "d_path": 3,
    }
    with open(output.replace(".npz", ".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  saved {output}")
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Generate an expressivity-gap dataset")
    parser.add_argument("--a_true", type=str, default="EWS", choices=["EWS", "EFM", "Sig"])
    parser.add_argument("--seed", type=int, default=0, help="Brownian path seed")
    parser.add_argument(
        "--normalize_paths",
        action="store_true",
        help="compute the target on paths scaled to [0, 1] instead of raw paths",
    )
    parser.add_argument("--target_depth", type=int, default=2)
    parser.add_argument(
        "--target_component",
        type=int,
        default=11,
        help="index in the flat depth-2 signature (11 = the W2 x W1 cross term)",
    )
    parser.add_argument("--M", type=int, default=10, help="Van Loan subdivisions for the target")
    parser.add_argument("--n_sequences", type=int, default=750)
    parser.add_argument("--T", type=float, default=5.0)
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--path_batch_size", type=int, default=3)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--target_batch_size", type=int, default=50)
    parser.add_argument("--van_loan_expm", type=str, default="exact")
    parser.add_argument("--chen_expm", type=str, default="exact")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--gpu", type=str, default="0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(
        a_true=args.a_true,
        seed=args.seed,
        normalize_paths=args.normalize_paths,
        target_depth=args.target_depth,
        target_component=args.target_component,
        M=args.M,
        n_sequences=args.n_sequences,
        T=args.T,
        n_steps=args.n_steps,
        path_batch_size=args.path_batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        target_batch_size=args.target_batch_size,
        van_loan_expm=args.van_loan_expm,
        chen_expm=args.chen_expm,
        output=args.output,
    )
