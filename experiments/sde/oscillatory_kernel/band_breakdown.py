"""Post-hoc per-output test metrics for finished runs (from their model.eqx checkpoints).

    python experiments/sde/oscillatory_kernel/band_breakdown.py --out outputs/oscillatory_kernel --gpu 0

For every results.json under <out>/runs/T1/ (or the given --target) whose best test metrics lack
`nmse_per_output`, rebuild the model from the run's config, load model.eqx, predict the test split
and write `nmse_per_output` (MSE / Var of the test split, per output) into best.test. Runs trained
after this field was added to the trainer already have it and are skipped.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", type=str, default="")
_a, _ = _pre.parse_known_args()
if _a.gpu == "":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _a.gpu
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
sys.path.insert(0, str(REPO_ROOT))

import equinox as eqx  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402

from experiments.sde.oscillatory_kernel.data import TARGET_DIMS  # noqa: E402
from experiments.sde.oscillatory_kernel.model import (  # noqa: E402
    build_model,
    predict_in_batches,
    resolve_dtype,
)
from experiments.sde.oscillatory_kernel.train import load_config, load_dataset  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default=str(REPO_ROOT / "outputs/oscillatory_kernel"))
    p.add_argument("--target", type=str, default="T1")
    p.add_argument("--gpu", type=str, default="")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    files = sorted(glob.glob(os.path.join(a.out, "runs", a.target, "*", "seed*", "results.json")))
    cache = {}
    n_done = 0
    for f in files:
        r = json.load(open(f))
        if not r["best"] or (not a.force and "nmse_per_output" in r["best"]["test"]):
            continue
        cfg = load_config(os.path.join(os.path.dirname(f), "config.json"))
        dtype = resolve_dtype(cfg["preprocessing"].get("dtype", "float32"))
        dkey = (r["dataset"], a.target)
        if dkey not in cache:
            cache[dkey] = load_dataset(r["dataset"], a.target, cfg["preprocessing"], dtype)["test"]
        test = cache[dkey]
        model = build_model(
            cfg["model"], TARGET_DIMS[a.target], jr.PRNGKey(0), dtype, readout_init="zero"
        )
        model = eqx.tree_deserialise_leaves(os.path.join(os.path.dirname(f), "model.eqx"), model)
        pred = predict_in_batches(model, np.asarray(test["paths"]), np.asarray(test["clocks"]))
        y = np.asarray(test["targets"], dtype=np.float64)
        k = y.shape[-1]
        mse = ((pred.reshape(-1, k) - y.reshape(-1, k)) ** 2).mean(axis=0)
        var = y.reshape(-1, k).var(axis=0)
        per = (mse / var).tolist()
        stored = r["best"]["test"]["nmse"]
        if abs(float(np.mean(per)) - stored) > 0.02 * max(
            1e-6, stored
        ):  # float32 GPU vs CPU kernels differ slightly
            print(f"  WARNING {f}: recomputed nmse {np.mean(per):.4e} vs stored {stored:.4e}")
        r["best"]["test"]["nmse_per_output"] = per
        with open(f, "w") as fh:
            json.dump(r, fh, indent=1)
        n_done += 1
        print(
            f"{r['model']} n{r['n']} m{r['m']} seed{r['seed']}: per-output nmse {np.round(per, 4).tolist()}"
        )
    print(f"updated {n_done} of {len(files)} runs")


if __name__ == "__main__":
    main()
