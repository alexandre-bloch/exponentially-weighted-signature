"""Bank the oscillatory-kernel deliverables into sde_results/oscillatory_kernel/.

    python experiments/sde/oscillatory_kernel/bank.py --src outputs/oscillatory_kernel --dst sde_results/oscillatory_kernel

Copies the deliverables (config.yaml, sanity.json, results.csv, the figures, table_matched_budget.md) and every
run's config.json and results.json (not model.eqx or the logs) under runs/, then writes a README.md with the protocol
and the reproduce commands. Run `python experiments/sde/report.py` afterwards to refresh sde_results/README.md.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

DELIVERABLES = (
    "config.yaml",
    "sanity.json",
    "results.csv",
    "fig_T1.png",
    "fig_T2.png",
    "fig_T3.png",
    "fig_T1_bands.png",
    "fig_freq_recovery.png",
    "table_matched_budget.md",
)

README = """# Oscillatory kernel: causal windowed Fourier coefficients (EWS vs EFM vs signature)

The third synthetic experiment of the SDE family. Code: `experiments/sde/oscillatory_kernel/`. Three per-timestep targets of a synthetic signal
(T1: the eight causal exponentially-windowed Fourier coefficients at the bank {2, 4, 22, 27} Hz with
damping 5; T2: the high-band energy; T3: the burst-active flag) are read from the prefix EWS of the
time-augmented integral path by a bare model: dense lift, one layer at depth n and width m, linear
readout. Three generator families: `sig` (A = 0), `efm` (positive diagonal) and `ews`
(damped-rotation planes), on a grid over (n, m) capped at 5000 features, three seeds, one data seed.

| file | contents |
|---|---|
| `table_matched_budget.md` | the matched budget (n, m) = (2, 8) and each model's best cell (chosen on validation), per target, plus the T1 per-band split |
| `fig_T1.png`, `fig_T2.png`, `fig_T3.png` | test metric against feature count, one curve per (model, depth), error bars over seeds |
| `fig_T1_bands.png` | T1 split into the low band (2, 4 Hz) and the high band (22, 27 Hz) |
| `fig_freq_recovery.png` | learned (frequency, damping) of the EWS T1 planes against the bank |
| `sanity.json` | the planted-solution errors at (1, 8) and (2, 8), the quadrature oracle, the oscillation count |
| `config.yaml` | every option used |
| `results.csv`, `runs/<target>/<model>_n<n>_m<m>/seed<k>/` | one row / one `results.json` + `config.json` per run (evaluation history, best checkpoint, init and learned generator parameters) |

## Reproduce

```bash
PY=<your-python>
JAX_PLATFORMS=cpu $PY experiments/sde/oscillatory_kernel/generate.py --seed 0            # data/oscillatory_kernel/oscillatory_kernel_seed0.npz
JAX_PLATFORMS=cpu $PY experiments/sde/oscillatory_kernel/sanity.py --out <out>/sanity.json  # must print ALL PASS
$PY experiments/sde/oscillatory_kernel/grid.py   --out <out> --gpus <g> --par 12 --mem-budget 70
$PY experiments/sde/oscillatory_kernel/band_breakdown.py --out <out> --target T1 --gpu <g>
$PY experiments/sde/oscillatory_kernel/grid.py --out <out> --collect-only
$PY experiments/sde/oscillatory_kernel/report.py --out <out>
```

The runs used XLA autotuning on and deterministic ops off. Twelve runs shared one H100; the per-run wall clocks in
`results.csv` reflect that sharing.
"""


def copy_runs(src_root: str, dst_root: str, pattern: str) -> int:
    n = 0
    for p in glob.glob(os.path.join(src_root, pattern)):
        rel = os.path.relpath(p, src_root)
        d = os.path.join(dst_root, rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy2(p, d)
        n += 1
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=str, default=str(REPO_ROOT / "outputs/oscillatory_kernel"))
    p.add_argument("--dst", type=str, default=str(REPO_ROOT / "sde_results/oscillatory_kernel"))
    a = p.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    for f in DELIVERABLES:
        s = os.path.join(a.src, f)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(a.dst, f))
        else:
            print(f"missing deliverable: {f}")
    n = copy_runs(a.src, a.dst, "runs/*/*/seed*/results.json") + copy_runs(
        a.src, a.dst, "runs/*/*/seed*/config.json"
    )
    with open(os.path.join(a.dst, "README.md"), "w") as f:
        f.write(README)
    print(f"banked {len(DELIVERABLES)} deliverables and {n} run files into {a.dst}")


if __name__ == "__main__":
    main()
