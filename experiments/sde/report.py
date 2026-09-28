"""Write the result tables of sde_results/ from the banked runs.

Reads every sde_results/<experiment>/<entry>/runs/seed<k>/results.json and writes the entry's
result.md (per-seed numbers, protocol, reproduce command) and sde_results/README.md (the two
summary tables and the learned spectra of the oscillator models).

    python experiments/sde/report.py            # default: sde_results/ at the repo root
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

LEARNER = {"ews": "EWS", "efm": "EFM", "sig": "Signature"}
TARGET = {"ews": "EWS", "efm": "EFM", "sig": "Signature"}
PROTOCOL = {
    "expressivity": (
        "750 time-augmented Brownian paths on [0, 5] with 10,000 steps, split 525/112/113 in order; the "
        "target is the (W2, W1) component of the depth-2 EWS of (t, W1, W2) under the target operator, "
        "scaled to [0, 1] on the training set; raw paths; depth-2 learners with M = 10 and exact "
        "matrix exponentials; AdamW with 500 warmup steps and a cosine decay, batch 32, 30,000 steps, "
        "evaluation every 500 steps, test RMSE at the best validation checkpoint; ten seeds, seed k "
        "using Brownian paths drawn with data seed k and model seed 100 + k."
    ),
    "oscillator": (
        "750 trajectories of the coupled oscillatory SDE on [0, 4] with 10,000 Euler-Maruyama steps, "
        "shuffled 70/15/15 split; inputs (t, W1, W2) scaled to [0, 1] on the training set with a "
        "basepoint (0, 0, 0) prepended, target X1 scaled to [0, 1]; depth-2 learners with M = 10 and "
        "exact matrix exponentials; AdamW with 500 warmup steps and a cosine decay, batch 32, "
        "15,000 steps, evaluation every 500 steps, RMSE at the best validation checkpoint; ten seeds, "
        "seed k using data seed k and model seed 100 + k."
    ),
}
TUNING = {
    "expressivity": "Optuna (TPE), 15 trials of 15,000 steps per target-learner pair (50 for the EWS learner on the EWS target), on the seed-321 data.",
    "oscillator": "Optuna (TPE), 100 trials of 15,000 steps per learner, on the seed-42 data.",
}


def fmt(x: float) -> str:
    """Scientific notation with two decimals, e.g. 4.96e-04."""
    return f"{x:.2e}"


def load_entry(entry_dir: str):
    runs = {}
    for p in sorted(glob.glob(f"{entry_dir}/runs/seed*/results.json")):
        k = int(re.search(r"seed(\d+)", p).group(1))
        runs[k] = json.load(open(p))
    return runs


def stats(values):
    a = np.asarray(values, dtype=float)
    return a.mean(), a.std()


def learner_of(entry: str) -> str:
    return entry.split("_")[-1]


def target_of(entry: str) -> str:
    return entry.split("_")[0]


def write_entry(root: str, experiment: str, entry: str) -> dict | None:
    edir = f"{root}/{experiment}/{entry}"
    runs = load_entry(edir)
    if not runs:
        return None
    cfg = json.load(open(f"{edir}/config.json"))
    seeds = sorted(runs)
    val = [runs[k]["best"]["val_rmse"] for k in seeds]
    test = [runs[k]["best"]["test_rmse"] for k in seeds]
    vm, vs = stats(val)
    tm, ts = stats(test)
    learner = LEARNER[learner_of(entry)]
    if experiment == "expressivity":
        title = f"{TARGET[target_of(entry)]} target, {learner} learner"
    else:
        title = f"Coupled oscillator, {learner} learner"
    a = cfg["model"]["A"]
    a_desc = {
        "general": "full 3 x 3 matrix",
        "efm": "positive diagonal (theta**2)",
        "sig": "A = 0 (fixed)",
    }[a["parametrization"]]
    lines = [
        f"# {title}",
        "",
        f"**Test RMSE {fmt(tm)} ± {fmt(ts)}** (validation {fmt(vm)} ± {fmt(vs)}); mean ± population std over {len(seeds)} seeds, normalised targets.",
        "",
        "| seed | " + " | ".join(str(k) for k in seeds) + " |",
        "|---|" + "---|" * len(seeds),
        "| val | " + " | ".join(fmt(v) for v in val) + " |",
        "| test | " + " | ".join(fmt(t) for t in test) + " |",
        "",
        f"**Learner:** `A` {a_desc}; `config.json` holds the resolved config with the tuned hyperparameters (seed 100 is seed 0, `--seed 100 + k` with the seed-k data file gives seed k).",
        "",
        f"**Protocol:** {PROTOCOL[experiment]}",
        "",
        f"**Hyperparameters:** {TUNING[experiment]}",
        "",
        "**Runs:** `runs/seed<k>/results.json` holds each banked run's evaluation history, best checkpoint, and the learned `A` with its eigenvalues.",
        "",
        "## Reproduce",
        "",
        "```bash",
        f"PY=<your-python> GPUS=0 bash sde_results/replicate.sh {experiment}/{entry}   # all ten seeds",
        f"python experiments/sde/train.py --config sde_results/{experiment}/{entry}/config.json --dataset_file <seed-k data> --seed <100 + k> --gpu <g>   # one seed",
        "```",
        "",
    ]
    with open(f"{edir}/result.md", "w") as f:
        f.write("\n".join(lines))
    spectra = None
    if a["parametrization"] != "sig":
        eig = [runs[k]["eigenvalues"] for k in seeds if runs[k].get("eigenvalues")]
        spectra = spectrum_summary(eig, a["parametrization"])
    return {"val": (vm, vs), "test": (tm, ts), "n": len(seeds), "spectra": spectra}


def spectrum_summary(eig_per_seed, parametrization: str):
    """Aggregate the learned eigenvalues over seeds: complex pairs (real part, |imag|) and real modes."""
    reals, pair_re, pair_im = [], [], []
    for eig in eig_per_seed:
        for re_, im_ in eig:
            if abs(im_) > 1e-9:
                if im_ > 0:
                    pair_re.append(re_)
                    pair_im.append(abs(im_))
            else:
                reals.append(re_)
    out = {}
    if pair_re:
        out["complex pair real part"] = stats(pair_re)
        out["complex pair imaginary magnitude"] = stats(pair_im)
    if reals:
        if parametrization == "efm":
            # the diagonal entries sorted per seed, reported per rank
            per_seed = [sorted((re_ for re_, _ in eig), reverse=True) for eig in eig_per_seed]
            for i in range(len(per_seed[0])):
                out[f"real eigenvalue {i + 1}"] = stats([s[i] for s in per_seed])
        else:
            out["real eigenvalues"] = stats(reals)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Write the sde_results tables from the banked runs"
    )
    parser.add_argument("--root", type=str, default=str(REPO_ROOT / "sde_results"))
    args = parser.parse_args()
    root = args.root

    table = {}
    for experiment in ("expressivity", "oscillator"):
        for edir in sorted(glob.glob(f"{root}/{experiment}/*/config.json")):
            entry = os.path.basename(os.path.dirname(edir))
            res = write_entry(root, experiment, entry)
            if res is not None:
                table[(experiment, entry)] = res
                print(
                    f"{experiment}/{entry}: test {fmt(res['test'][0])} ± {fmt(res['test'][1])} ({res['n']} seeds)"
                )

    lines = [
        "# sde_results: EWS, EFM and the classical signature on two synthetic regression tasks",
        "",
        "Banked runs of the two synthetic experiments. Each `<experiment>/<entry>/` holds:",
        "- `config.json`: the exact config (tuned hyperparameters; seed 100 is seed 0)",
        "- `result.md`: the banked numbers, protocol, reproduce command",
        "- `runs/seed<k>/results.json`: the banked run's evaluation history, best checkpoint, learned `A` and eigenvalues",
        "",
        "All numbers are RMSE on normalised targets at the best validation checkpoint, mean ± population std over ten seeds.",
        "",
        "## Expressivity gap (depth 2): test RMSE per target (rows) and learner (columns)",
        "",
        "| target \\ learner | EWS | EFM | Signature |",
        "|---|---|---|---|",
    ]
    for t in ("ews", "efm", "sig"):
        cells = []
        for learner in ("ews", "efm", "sig"):
            r = table.get(("expressivity", f"{t}_{learner}"))
            cells.append(
                f"[{fmt(r['test'][0])} ± {fmt(r['test'][1])}](expressivity/{t}_{learner}/result.md)"
                if r
                else "—"
            )
        lines.append(f"| **{TARGET[t]}** | " + " | ".join(cells) + " |")
    lines += [
        "",
        "The target is the (W2, W1) component of the depth-2 EWS of a time-augmented Brownian path under a",
        "fixed operator: a full matrix (EWS; built as the real part of a unitary similarity of diag(−0.5 ± 5.2i, 0.8),",
        "which gives the eigenvalues −0.44 ± 3.22i and 0.68), diag(0.5, 0.3, 0.8) (EFM) or zero (Signature).",
        "Each learner is trained at the same depth with an unconstrained matrix (EWS), a",
        "positive diagonal (EFM) or a fixed zero (Signature).",
        "",
        "## Coupled oscillator: validation and test RMSE per learner",
        "",
        "| learner | validation | test |",
        "|---|---|---|",
    ]
    for learner in ("ews", "efm", "sig"):
        r = table.get(("oscillator", learner))
        if r:
            lines.append(
                f"| [{LEARNER[learner]}](oscillator/{learner}/result.md) | {fmt(r['val'][0])} ± {fmt(r['val'][1])} | {fmt(r['test'][0])} ± {fmt(r['test'][1])} |"
            )
    lines += [
        "",
        "Same-time regression of X1 from the driving path (t, W1, W2) of dX1 = (3 sin X2 − 0.5 X1) dt + 0.4 dW1,",
        "dX2 = (3 cos X1 − 0.5 X2) dt + 0.4 dW2.",
        "",
        "## Learned spectra on the oscillator (mean ± std over seeds)",
        "",
        "| | EWS | EFM |",
        "|---|---|---|",
    ]
    ews_s = (table.get(("oscillator", "ews")) or {}).get("spectra") or {}
    efm_s = (table.get(("oscillator", "efm")) or {}).get("spectra") or {}

    def cell(d, key):
        return f"{d[key][0]:.2f} ± {d[key][1]:.2f}" if key in d else "–"

    lines.append(f"| real part (complex pair) | {cell(ews_s, 'complex pair real part')} | – |")
    lines.append(f"| imaginary magnitude | {cell(ews_s, 'complex pair imaginary magnitude')} | – |")
    efm_reals = (
        ", ".join(cell(efm_s, k) for k in sorted(efm_s) if k.startswith("real eigenvalue")) or "–"
    )
    lines.append(f"| real eigenvalues | {cell(ews_s, 'real eigenvalues')} | {efm_reals} |")
    osc_table = f"{root}/oscillatory_kernel/table_matched_budget.md"
    if os.path.exists(osc_table):
        table_lines = [ln for ln in open(osc_table).read().splitlines() if ln.startswith("|")]
        lines += [
            "",
            "## Oscillatory kernel: causal windowed Fourier coefficients (test metric, mean ± std over 3 seeds)",
            "",
            "Three per-timestep targets of a synthetic signal (T1: the eight windowed Fourier coefficients at",
            "{2, 4, 22, 27} Hz with damping 5; T2: the high-band energy; T3: the burst flag), read from the",
            "prefix EWS of the integral path by a dense lift, one layer at depth n and width m and a linear",
            "readout. Matched budget (n, m) = (2, 8) is 72 features for every model; best is each model's validation pick over its",
            "(n, m) grid (D <= 5000). Normalised MSE for T1 and T2, accuracy for T3.",
            "",
        ]
        lines += table_lines[:5]  # the main table: header, rule, T1, T2, T3
        lines += [
            "",
            "The per-band split of T1, the frequency-recovery figure and the sanity checks are in",
            "[oscillatory_kernel/](oscillatory_kernel/README.md) (figures, `results.csv`, per-run `results.json`).",
        ]
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "PY=<your-python> GPUS=<list> bash sde_results/replicate.sh <experiment>/<entry>   # e.g. expressivity/ews_efm, oscillator/ews",
        "```",
        "",
        "The runs used XLA autotuning on and deterministic ops off.",
        "`python experiments/sde/report.py` rewrites these tables from `runs/`.",
        "",
    ]
    with open(f"{root}/README.md", "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {root}/README.md")


if __name__ == "__main__":
    main()
