"""Generate the oscillatory-kernel dataset: signals, path, clock and the three exact targets.

    python experiments/sde/oscillatory_kernel/generate.py --seed 0
    python experiments/sde/oscillatory_kernel/generate.py --seed 0 --n_train 200 --n_val 50 --n_test 50 --output <file>

Output NPZ keys (float32 except the flags):
    {split}_paths    (n, T + 1, 2)   X_i = (t_i, sum_{k<=i} x_k dt), X_0 = (0, 0)
    {split}_clocks   (n, T + 1)      t_i, t_0 = 0
    {split}_signal   (n, T)          x_i
    {split}_T1       (n, T, 8)       (c_j, s_j) per bank frequency
    {split}_T2       (n, T, 1)       high-band energy
    {split}_T3       (n, T, 1)       burst-active flag
plus a .json with the generation parameters, the oracle check and the oscillation count.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

from experiments.sde.oscillatory_kernel.data import (
    BANK_HZ,
    DAMPING,
    NOISE_SIGMA,
    T1_NAMES,
    T_DEFAULT,
    generate_signals,
    make_clocks,
    make_paths,
    make_targets,
    oracle_check,
    oscillation_count,
)


def generate(
    seed: int = 0,
    n_train: int = 8000,
    n_val: int = 1000,
    n_test: int = 1000,
    T: int = T_DEFAULT,
    n_oracle: int = 20,
    output: str | None = None,
) -> str:
    if output is None:
        output = str(
            REPO_ROOT / "data" / "oscillatory_kernel" / f"oscillatory_kernel_seed{seed}.npz"
        )
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    rng = np.random.default_rng(seed)
    arrays = {}
    meta_splits = {}
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        sig = generate_signals(n, rng, T)
        targets = make_targets(sig)
        arrays[f"{split}_paths"] = make_paths(sig).astype(np.float32)
        arrays[f"{split}_clocks"] = make_clocks(sig).astype(np.float32)
        arrays[f"{split}_signal"] = sig.x.astype(np.float32)
        for name, arr in targets.items():
            arrays[f"{split}_{name}"] = arr.astype(np.float32)
        meta_splits[split] = {
            "n": int(n),
            "active_fraction": float(sig.active.mean()),
            "no_burst_fraction": float((~sig.active.any(axis=1)).mean()),
            "T1_std_per_output": [float(v) for v in targets["T1"].reshape(-1, 8).std(axis=0)],
            "T2_std": float(targets["T2"].std()),
        }
        if split == "train":
            oracle = oracle_check(sig.x, n_oracle, rng)
    np.savez_compressed(output, **arrays)

    meta = {
        "dataset": "oscillatory_kernel",
        "seed": int(seed),
        "T": int(T),
        "dt": 1.0 / T,
        "bank_hz": list(BANK_HZ),
        "damping": DAMPING,
        "noise_sigma": NOISE_SIGMA,
        "T1_outputs": T1_NAMES,
        "splits": meta_splits,
        "oracle": oracle,
        "oscillations": oscillation_count(),
    }
    with open(output.replace(".npz", ".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved {output}")
    for split, m in meta_splits.items():
        print(f"  {split}: n={m['n']}  active fraction {m['active_fraction']:.3f}")
    print(f"  oracle max relative error {oracle['max_rel_err']:.2e}  pass={oracle['pass']}")
    if not oracle["pass"]:
        raise SystemExit("oracle check FAILED (max relative error above 1e-6)")
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate the oscillatory-kernel dataset")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_train", type=int, default=8000)
    p.add_argument("--n_val", type=int, default=1000)
    p.add_argument("--n_test", type=int, default=1000)
    p.add_argument("--T", type=int, default=T_DEFAULT)
    p.add_argument("--n_oracle", type=int, default=20)
    p.add_argument("--output", type=str, default=None)
    a = p.parse_args()
    generate(a.seed, a.n_train, a.n_val, a.n_test, a.T, a.n_oracle, a.output)
