"""Collate the UEA memory/time cells into RESULTS.md (and results.csv) next to Walker's own published measurements.

Time  = training-only seconds per 1000 steps (median of the steady intervals; eval excluded, compile excluded).
Mem   = process-level peak from nvidia-smi (CUDA context included; Walker's figures look like this), and the
        in-process allocator peak (JAX peak_bytes_in_use / torch max_memory_reserved) next to it.
Walker = results/memory_time_results.json of github.com/Benjamin-Walker/log-neural-cdes (his card; his torch times
include evaluation), read from the lncde/ clone under MEMTIME_ROOT (see memtime_run.py); rows are left out if absent.

    MEMTIME_ROOT=<dir> python collate_memtime.py          # writes <dir>/RESULTS.md and <dir>/results.csv
"""

import csv
import glob
import json
import os
import statistics

R = os.environ.get("MEMTIME_ROOT")
if not R:
    raise SystemExit(
        "MEMTIME_ROOT is not set: it must name the lane's working directory (see run_memtime_claims.sh)"
    )
OUTDIR = os.environ.get(
    "OUTDIR", f"{R}/out"
)  # standardised-batch redo collates from its own directory
SUFFIX = os.environ.get("RESULTS_SUFFIX", "")
WALKER = os.path.join(R, "lncde", "results", "memory_time_results.json")
DS = [
    "EigenWorms",
    "EthanolConcentration",
    "Heartbeat",
    "MotorImagery",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
]
SHORT = {
    "EigenWorms": "EW",
    "EthanolConcentration": "EC",
    "Heartbeat": "HB",
    "MotorImagery": "MI",
    "SelfRegulationSCP1": "SCP1",
    "SelfRegulationSCP2": "SCP2",
}
ROWS = [
    ("ews", "ews", "EWS (ours, selected)", None),
    ("linoss", "LinOSS", "LinOSS-IM", None),
    ("dlinoss", "LinOSS", "D-LinOSS", None),
    ("lncde", "S5", "S5", "S5"),
    ("lncde", "lru", "LRU", "lru"),
    ("torch", "S6", "S6", "S6"),
    ("torch", "mamba", "Mamba", "mamba"),
    ("lncde", "bd_linear_ncde", "BD-SLiCE", "bd_linear_ncde"),
    ("lncde", "diagonal_linear_ncde", "D-SLiCE", "diagonal_linear_ncde"),
    ("lncde", "log_ncde", "Log-NCDE", "log_ncde"),
]

walker = json.load(open(WALKER)) if os.path.exists(WALKER) else {"time": {}, "memory": {}}
res = {}
for p in glob.glob(f"{OUTDIR}/*/result.json"):
    r = json.load(open(p))
    res[(r["family"], r["model"], r["dataset"])] = r


def f(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


gpus = sorted({str(r.get("env", {}).get("CUDA_VISIBLE_DEVICES")) for r in res.values()})
lines = [
    f"# UEA memory and time, one card (GPU {', '.join(gpus) or '?'}, H100 NVL), solo, no holder",
    "",
    "Training-only seconds per 1000 steps (median of steady intervals; evaluation and compilation excluded) / "
    "process-level peak MB (nvidia-smi, CUDA context included). Walker = his published measurement on his card "
    "(his S6/Mamba times include evaluation). `!` = SHARED card during the cell, `x` = failed.",
    "",
]
hdr = "| model | " + " | ".join(SHORT[d] for d in DS) + " | mean s/1k | mean MB |"
lines += ["## Time (s / 1000 steps) and process peak (MB)", "", hdr, "|" + "---|" * (len(DS) + 3)]
csvrows = []
for fam, m, name, wkey in ROWS:
    cells, ts, ms = [], [], []
    for d in DS:
        r = res.get((fam, m, d))
        if r is None:
            cells.append("pending")
            continue
        flag = ("x" if r["status"] != "ok" else "") + ("!" if r.get("shared") else "")
        t, mb = r.get("s_per_1k_median"), r.get("proc_peak_mb")
        if r["status"] == "ok" and t is not None:
            ts.append(t)
            ms.append(mb or 0)
        cells.append(f"{f(t)} / {mb if mb else '—'}{flag}")
        mem = r.get("mem", {})
        csvrows.append(
            {
                "model": name,
                "dataset": d,
                "status": r["status"],
                "shared": r.get("shared"),
                "s_per_1k_median": t,
                "s_per_1k_min": r.get("s_per_1k_min"),
                "s_per_1k_max": r.get("s_per_1k_max"),
                "steady_intervals": r.get("steady_intervals"),
                "proc_peak_mb": mb,
                "jax_peak_mb": mem.get("jax_peak_mb"),
                "torch_max_reserved_mb": mem.get("torch_max_reserved_mb"),
                "torch_max_allocated_mb": mem.get("torch_max_allocated_mb"),
                "batch_size": r.get("batch_size"),
                "walker_s_per_1k": walker["time"].get(wkey, {}).get(d) if wkey else None,
                "walker_mb": walker["memory"].get(wkey, {}).get(d) if wkey else None,
            }
        )
    mean_t = f(statistics.mean(ts)) if len(ts) == len(DS) else "—"
    mean_m = f(statistics.mean(ms), 0) if len(ms) == len(DS) else "—"
    lines.append(f"| {name} | " + " | ".join(cells) + f" | {mean_t} | {mean_m} |")
    if wkey and wkey in walker["time"]:
        wt = [walker["time"][wkey].get(d) for d in DS]
        wm = [walker["memory"][wkey].get(d) for d in DS]
        lines.append(
            f"| ↳ Walker {name} | "
            + " | ".join(f"{f(a)} / {b}" for a, b in zip(wt, wm))
            + f" | {f(statistics.mean(wt))} | {f(statistics.mean(wm), 0)} |"
        )

lines += [
    "",
    "## In-process allocator peak (MB): JAX peak_bytes_in_use, torch max_memory_reserved",
    "",
    hdr.replace(" mean s/1k | mean MB |", ""),
    "|" + "---|" * (len(DS) + 1),
]
for fam, m, name, _ in ROWS:
    cells = []
    for d in DS:
        r = res.get((fam, m, d))
        mem = (r or {}).get("mem", {})
        v = mem.get("jax_peak_mb", mem.get("torch_max_reserved_mb"))
        cells.append("pending" if r is None else f(v, 0))
    lines.append(f"| {name} | " + " | ".join(cells) + " |")
open(f"{R}/RESULTS{SUFFIX}.md", "w").write("\n".join(lines) + "\n")
if csvrows:
    with open(f"{R}/results{SUFFIX}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csvrows[0]))
        w.writeheader()
        w.writerows(csvrows)
print(f"collated {len(res)} cells from {OUTDIR} -> {R}/RESULTS{SUFFIX}.md")
