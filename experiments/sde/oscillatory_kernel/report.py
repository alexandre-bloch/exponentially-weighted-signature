"""Figures and table of the oscillatory-kernel experiment from results.csv.

    python experiments/sde/oscillatory_kernel/report.py --out outputs/oscillatory_kernel

Reads <out>/results.csv and writes fig_T1.png, fig_T2.png, fig_T3.png, fig_T1_bands.png, fig_freq_recovery.png and
table_matched_budget.md into <out>.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from experiments.sde.oscillatory_kernel.data import BANK_HZ, feature_count  # noqa: E402

MODELS = ("ews", "efm", "sig")
LABEL = {"ews": "EWS", "efm": "EFM", "sig": "sig"}
COLOR = {"ews": "#1f77b4", "efm": "#d62728", "sig": "#7f7f7f"}
LINESTYLE = {1: "-", 2: "--", 3: "-.", 4: ":"}
MARKER = {1: "o", 2: "s", 3: "^", 4: "D"}
METRIC_NAME = {
    "T1": "normalised MSE (test)",
    "T2": "normalised MSE (test)",
    "T3": "accuracy (test)",
}
MINIMAL_EWS = {"T1": (1, 8), "T2": (2, 8)}
MATCHED = (2, 8)  # D = 72


def load_rows(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in ("n", "m", "D", "seed", "steps_run"):
                r[k] = int(float(r[k]))
            for k in (
                "val_metric",
                "test_metric",
                "test_f1",
                "val_loss",
                "test_loss",
                "wall_min",
                "ms_per_step",
                "series_x_best",
            ):
                r[k] = float(r[k]) if r.get(k) not in (None, "", "None") else np.nan
            r["steps_reduced"] = r["steps_reduced"] == "True"
            rows.append(r)
    return rows


def group(rows, target):
    g = defaultdict(list)
    for r in rows:
        if r["target"] == target:
            g[(r["model"], r["n"], r["m"])].append(r)
    return g


def stat(values):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return np.nan, np.nan, 0
    return float(v.mean()), float(v.std()), int(v.size)


def fmt(x, target):
    if not np.isfinite(x):
        return "—"
    return f"{x:.3f}" if target == "T3" else f"{x:.2e}"


def fig_metric(rows, target, out):
    g = group(rows, target)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for model in MODELS:
        depths = sorted({k[1] for k in g if k[0] == model})
        for n in depths:
            pts = sorted(
                (k[2], stat([r["test_metric"] for r in v]))
                for k, v in g.items()
                if k[0] == model and k[1] == n
            )
            if not pts:
                continue
            D = [feature_count(m, n) for m, _ in pts]
            mean = [s[0] for _, s in pts]
            std = [s[1] for _, s in pts]
            ax.errorbar(
                D,
                mean,
                yerr=std,
                color=COLOR[model],
                linestyle=LINESTYLE.get(n, "-"),
                marker=MARKER.get(n, "o"),
                markersize=5,
                capsize=3,
                label=f"{LABEL[model]} n={n}",
            )
    if target in MINIMAL_EWS:
        n, m = MINIMAL_EWS[target]
        ax.axvline(
            feature_count(m, n),
            color=COLOR["ews"],
            alpha=0.25,
            linewidth=8,
            label=f"EWS minimal (n,m)=({n},{m})",
        )
    ax.set_xscale("log")
    if target != "T3":
        ax.set_yscale("log")
    ax.set_xlabel("feature count D = sum_k m^k")
    ax.set_ylabel(METRIC_NAME[target])
    ax.set_title(
        f"{target}: {METRIC_NAME[target]} vs feature count, mean ± std over seeds", fontsize=10
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = os.path.join(out, f"fig_{target}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


LOW_OUTPUTS = (0, 1, 2, 3)  # (c, s) of 2 Hz and 4 Hz
HIGH_OUTPUTS = (4, 5, 6, 7)  # (c, s) of 22 Hz and 27 Hz


def band_metric(r, outputs):
    per = json.loads(r.get("test_nmse_per_output") or "null")
    if not per:
        return np.nan
    return float(np.mean([per[i] for i in outputs]))


def fig_t1_bands(rows, out):
    """T1 split into the low band (2, 4 Hz) and the high band (22, 27 Hz): the Chebyshev bound in action."""
    g = group(rows, "T1")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (name, outputs) in zip(
        axes,
        (
            ("low band: 2 and 4 Hz coefficients", LOW_OUTPUTS),
            ("high band: 22 and 27 Hz coefficients", HIGH_OUTPUTS),
        ),
    ):
        for model in MODELS:
            depths = sorted({k[1] for k in g if k[0] == model})
            for n in depths:
                pts = sorted(
                    (k[2], stat([band_metric(r, outputs) for r in v]))
                    for k, v in g.items()
                    if k[0] == model and k[1] == n
                )
                pts = [(m, s) for m, s in pts if np.isfinite(s[0])]
                if not pts:
                    continue
                D = [feature_count(m, n) for m, _ in pts]
                ax.errorbar(
                    D,
                    [s[0] for _, s in pts],
                    yerr=[s[1] for _, s in pts],
                    color=COLOR[model],
                    linestyle=LINESTYLE.get(n, "-"),
                    marker=MARKER.get(n, "o"),
                    markersize=5,
                    capsize=3,
                    label=f"{LABEL[model]} n={n}",
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("feature count D")
        ax.set_title(name, fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("normalised MSE (test), band average")
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle(
        "T1 by band: EFM fits the slow coefficients (about one oscillation per window) and not the fast ones (13-16 per window)",
        fontsize=10,
    )
    fig.tight_layout()
    path = os.path.join(out, "fig_T1_bands.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_freq(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, m in zip(axes, (8, 16)):
        runs = [
            r
            for r in rows
            if r["target"] == "T1" and r["model"] == "ews" and r["n"] == 1 and r["m"] == m
        ]
        for f in BANK_HZ:
            ax.axvline(f, color="k", alpha=0.3, linestyle="--")
        for r in runs:
            lp = json.loads(r["learned_params"])
            ip = json.loads(r["init_params"])
            ax.scatter(ip["freq_hz"], ip["damping"], marker="x", s=25, color="#bbbbbb")
            ax.scatter(
                lp["freq_hz"], lp["damping"], marker="o", s=30, alpha=0.8, label=f"seed {r['seed']}"
            )
        ax.axhline(5.0, color="k", alpha=0.3, linestyle=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("learned frequency omega_j / 2 pi  [Hz]")
        ax.set_title(f"EWS T1, n=1, m={m} ({m // 2} planes); x = init, o = learned")
        ax.grid(True, which="both", alpha=0.3)
        if runs:
            ax.legend(fontsize=7)
    axes[0].set_ylabel("learned damping a_j")
    fig.suptitle("Frequency recovery: bank {2, 4, 22, 27} Hz (dashed), damping 5 (dotted)")
    fig.tight_layout()
    path = os.path.join(out, "fig_freq_recovery.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def best_over_grid(g, target):
    """(key, mean, std, n_seeds) of the cell with the best mean validation metric, reported on test.

    Lower is better for T1/T2 (normalised MSE), higher for T3 (accuracy).
    """
    best = None
    for k, v in g.items():
        val, _, _ = stat([r["val_metric"] for r in v])
        mean, std, n = stat([r["test_metric"] for r in v])
        if not np.isfinite(val) or not np.isfinite(mean):
            continue
        if best is None or (val > best[4] if target == "T3" else val < best[4]):
            best = (k, mean, std, n, val)
    return None if best is None else best[:4]


def table_matched(rows, out):
    lines = [
        "# Matched budget and the best cell of each model (chosen on validation; test metric, mean ± std over seeds)",
        "",
    ]
    lines += [
        f"Matched budget: (n, m) = {MATCHED}, D = {feature_count(MATCHED[1], MATCHED[0])}.",
        "",
    ]
    lines += [
        "| target | metric | EWS (2,8) | EFM (2,8) | sig (2,8) | best EWS | best EFM | best sig |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for target in ("T1", "T2", "T3"):
        g = group(rows, target)
        cells = []
        for model in MODELS:
            mean, std, n = stat([r["test_metric"] for r in g.get((model,) + MATCHED, [])])
            cells.append(f"{fmt(mean, target)} ± {fmt(std, target)}" if n else "—")
        for model in MODELS:
            sub = {k: v for k, v in g.items() if k[0] == model}
            b = best_over_grid(sub, target)
            cells.append(
                f"{fmt(b[1], target)} ± {fmt(b[2], target)} at (n,m)=({b[0][1]},{b[0][2]})"
                if b
                else "—"
            )
        lines.append(f"| {target} | {METRIC_NAME[target]} | " + " | ".join(cells) + " |")
    if any(r["target"] == "T3" for r in rows):
        lines += [
            "",
            "T3 F1 (test) at the matched budget and at each model's best-accuracy configuration:",
            "",
        ]
        g = group(rows, "T3")
        lines += [
            "| | EWS (2,8) | EFM (2,8) | sig (2,8) | best EWS | best EFM | best sig |",
            "|---|---|---|---|---|---|---|",
        ]
        cells = []
        for model in MODELS:
            mean, std, n = stat([r["test_f1"] for r in g.get((model,) + MATCHED, [])])
            cells.append(f"{mean:.3f} ± {std:.3f}" if n else "—")
        for model in MODELS:
            sub = {k: v for k, v in g.items() if k[0] == model}
            b = best_over_grid(sub, "T3")
            if b:
                mean, std, n = stat([r["test_f1"] for r in g[b[0]]])
                cells.append(f"{mean:.3f} ± {std:.3f}")
            else:
                cells.append("—")
        lines.append("| F1 | " + " | ".join(cells) + " |")
    g1 = group(rows, "T1")
    if any(np.isfinite(band_metric(r, LOW_OUTPUTS)) for v in g1.values() for r in v):
        lines += [
            "",
            "T1 by band (normalised MSE over the band's four outputs, test), matched budget (2,8) and each model's best cell:",
            "",
        ]
        lines += [
            "| band | EWS (2,8) | EFM (2,8) | sig (2,8) | best EWS | best EFM | best sig |",
            "|---|---|---|---|---|---|---|",
        ]
        best_t1 = {
            model: best_over_grid({k: v for k, v in g1.items() if k[0] == model}, "T1")
            for model in MODELS
        }
        for band, outputs in (
            ("low, 2 and 4 Hz", LOW_OUTPUTS),
            ("high, 22 and 27 Hz", HIGH_OUTPUTS),
        ):
            cells = []
            for model in MODELS:
                mean, std, n = stat(
                    [band_metric(r, outputs) for r in g1.get((model,) + MATCHED, [])]
                )
                cells.append(f"{fmt(mean, 'T1')} ± {fmt(std, 'T1')}" if n else "—")
            for model in MODELS:
                b = best_t1[model]
                mean, std, n = (
                    stat([band_metric(r, outputs) for r in g1[b[0]]]) if b else (np.nan, np.nan, 0)
                )
                cells.append(
                    f"{fmt(mean, 'T1')} ± {fmt(std, 'T1')} at ({b[0][1]},{b[0][2]})" if n else "—"
                )
            lines.append(f"| {band} | " + " | ".join(cells) + " |")
    path = os.path.join(out, "table_matched_budget.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main():
    p = argparse.ArgumentParser(description="Oscillatory-kernel report")
    p.add_argument("--out", type=str, default=str(REPO_ROOT / "outputs/oscillatory_kernel"))
    a = p.parse_args()
    rows = load_rows(os.path.join(a.out, "results.csv"))
    for target in ("T1", "T2", "T3"):
        if any(r["target"] == target for r in rows):
            print("wrote", fig_metric(rows, target, a.out))
    print("wrote", fig_freq(rows, a.out))
    if any(
        json.loads(r.get("test_nmse_per_output") or "null") for r in rows if r["target"] == "T1"
    ):
        print("wrote", fig_t1_bands(rows, a.out))
    print("wrote", table_matched(rows, a.out))


if __name__ == "__main__":
    main()
