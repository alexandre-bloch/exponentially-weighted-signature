"""Sanity checks of the oscillatory-kernel experiment, run before the grid.

    JAX_PLATFORMS=cpu python experiments/sde/oscillatory_kernel/sanity.py --dataset data/oscillatory_kernel/oscillatory_kernel_seed0.npz

1. Planted depth-1 solution: EWS (n, m) = (1, 8) with A = the bank, hand-set B and readout;
   normalised MSE on T1 (test split) must be <= 1e-6.
2. Planted depth-2 solution: the same at n = 2 with the level-2 diagonal readout; normalised MSE
   on T2 must be <= 1e-6.
3. Oracle check: copied from the dataset's .json (computed at generation).
4. Oscillation count and the implied EFM bound.

Each planted check runs in float64 and float32, with the series leaf at the configured q and, for
reference, with the Van Loan leaf at M = 8. Writes sanity.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from experiments.sde.oscillatory_kernel.data import oscillation_count
from experiments.sde.oscillatory_kernel.model import (
    default_compute,
    normalised_mse,
    planted_model,
    predict_in_batches,
    series_regime_x,
    van_loan_compute,
)


def run(dataset: str, out: str, q: int, n_samples: int) -> dict:
    raw = np.load(dataset)
    meta = json.load(open(dataset.replace(".npz", ".json")))
    paths = raw["test_paths"][:n_samples]
    clocks = raw["test_clocks"][:n_samples]
    dt = float(clocks[0, 1] - clocks[0, 0])
    result = {"dataset": dataset, "n_samples": int(paths.shape[0]), "series_q": q, "planted": {}}
    for depth, target in ((1, "T1"), (2, "T2")):
        y = raw[f"test_{target}"][:n_samples].astype(np.float64)
        for dtype_name, dtype in (("float64", jnp.float64), ("float32", jnp.float32)):
            for solver_name, comp in (
                ("series", default_compute(q)),
                ("van_loan_M8", van_loan_compute(8)),
            ):
                model = planted_model(depth, comp, dtype)
                pred = predict_in_batches(model, paths, clocks)
                nmse = normalised_mse(pred, y)
                key = f"{target}_n{depth}_{solver_name}_{dtype_name}"
                result["planted"][key] = {
                    "nmse": nmse,
                    "max_abs_err": float(np.abs(pred - y).max()),
                    "max_abs_target": float(np.abs(y).max()),
                    "pass": bool(nmse <= 1e-6),
                    "series_regime_x": series_regime_x(model, dt),
                }
                print(
                    f"  {key:32s} nmse {nmse:.3e}  max|err| {np.abs(pred - y).max():.3e}  pass={nmse <= 1e-6}"
                )
    result["oracle"] = meta["oracle"]
    result["oscillations"] = oscillation_count()
    gate = {
        "planted_T1_series_float64": result["planted"]["T1_n1_series_float64"]["pass"],
        "planted_T2_series_float64": result["planted"]["T2_n2_series_float64"]["pass"],
        "oracle": bool(meta["oracle"]["pass"]),
    }
    result["gate"] = gate
    result["all_pass"] = all(gate.values())
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"oracle max relative error {meta['oracle']['max_rel_err']:.2e}")
    print(
        f"oscillations {result['oscillations']['oscillations_in_window']}  EFM bound nm >= {result['oscillations']['efm_bound_nm_at_least']}"
    )
    print(f"gate {gate}  ->  {'ALL PASS' if result['all_pass'] else 'FAILED'}")
    print(f"wrote {out}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Oscillatory-kernel sanity checks")
    p.add_argument(
        "--dataset",
        type=str,
        default=str(REPO_ROOT / "data/oscillatory_kernel/oscillatory_kernel_seed0.npz"),
    )
    p.add_argument(
        "--out", type=str, default=str(REPO_ROOT / "outputs/oscillatory_kernel/sanity.json")
    )
    p.add_argument("--q", type=int, default=12)
    p.add_argument("--n_samples", type=int, default=200)
    a = p.parse_args()
    r = run(a.dataset, a.out, a.q, a.n_samples)
    sys.exit(0 if r["all_pass"] else 1)
