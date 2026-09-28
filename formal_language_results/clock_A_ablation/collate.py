"""The FL clock / A-structure ablation table: best validation accuracy per run, mean +- population std over the three
seeds, how each run ended, next to the banked champions.

    python formal_language_results/clock_A_ablation/collate.py

Every run in runs/ went to 100,000 steps without a futility stop, except even_pairs "efm A + champion clocks" run 0,
which solved the task and stopped early under an earlier 40k-step cap and was not re-run. A run whose training loss
went NaN after step 0 is diverged: its value is the best validation accuracy before the divergence, marked with an
asterisk.
"""

import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
CHAMP = {"parity": 99.995, "even_pairs": 100.0, "cycle_nav": 99.985, "mod_arith_no_brack": 96.71}
CHANCE = {"parity": 50, "even_pairs": 50, "cycle_nav": 20, "mod_arith_no_brack": 20}
ARMS = [
    ("champion A + uniform clocks", "{t}_uniform"),
    ("efm A + champion clocks", "{t}_efm"),
    ("efm A + uniform clocks", "{t}_efm_uniform"),
]
EARLIER = {("even_pairs", "efm A + champion clocks", 0): "even_pairs_efm_40k_r0.log"}


def final(path):
    t = open(path).read()
    m = re.search(
        r"Best Validation Accuracy: ([0-9.]+), Early stop: (\w+), Futility stop: (\w+)", t
    )
    if not m:
        return None
    evals = [
        (int(s), l, float(a))
        for s, l, a in re.findall(
            r"Step: (\d+), Loss: ([0-9.na]+), Validation Accuracy: ([0-9.]+)", t
        )
    ]
    nan_steps = [s for s, l, _ in evals if l == "nan" and s > 0]
    if nan_steps:
        first = min(nan_steps)
        best_before = max([a for s, _, a in evals if 0 < s < first] or [0.0])
        return 100 * best_before, f"DIVERGED at {first // 1000}k"
    return 100 * float(m.group(1)), "solved" if m.group(2) == "True" else (
        "futility@40k" if m.group(3) == "True" else "budget"
    )


def table(folder, suffix, rule):
    print("| task | arm | rule | runs | mean +- std | ended by | champion |")
    print("|---|---|---|---|---|---|---|")
    for t in CHAMP:
        for arm, stem in ARMS:
            got = []
            for r in (0, 1, 2):
                p = os.path.join(HERE, folder, stem.format(t=t) + f"{suffix}_r{r}.log")
                if os.path.exists(p) and final(p):
                    got.append(final(p))
                elif (t, arm, r) in EARLIER:
                    txt = open(os.path.join(HERE, folder, EARLIER[(t, arm, r)])).read()
                    best = float(re.search(r"Best Validation Accuracy: ([0-9.]+)", txt).group(1))
                    got.append((100 * best, "solved (40k rule)"))
            if not got:
                continue
            div = [g for g in got if g[1].startswith("DIVERGED")]
            fin = [g for g in got if not g[1].startswith("DIVERGED")]
            vals = [g[0] for g in got]
            if fin and not div:
                mean = f"{st.mean(vals):.2f} +- {st.pstdev(vals):.2f}"
            elif fin:
                mean = f"{st.mean(g[0] for g in fin):.2f}* +- {st.pstdev([g[0] for g in fin]) if len(fin) > 1 else 0:.2f} ({len(fin)} of {len(got)} finished)"
            else:
                mean = f"{st.mean(vals):.2f}** +- {st.pstdev(vals):.2f} (best before divergence, all {len(got)} seeds)"
            runs = ", ".join(f"{g[0]:.2f}{'*' if g[1].startswith('DIVERGED') else ''}" for g in got)
            print(
                f"| {t} | {arm} | {rule} | {runs} | {mean} | {', '.join(g[1] for g in got)} | {CHAMP[t]} (chance {CHANCE[t]}) |"
            )


print(
    "# FL clock / A-structure ablation (best validation accuracy %, runs 0-2 = seeds 1234-1236; champions = banked 5-seed rows)\n"
)
table("runs", "_100k", "100k")
print(
    "\n\\* diverged seed: value = best validation accuracy before the training loss went NaN; \\*\\* every seed diverged."
)
