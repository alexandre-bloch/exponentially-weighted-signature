# formal_language_results: EWS on Walker's formal-language benchmark

Banked EWS results on four length-generalization tasks from `structured-linear-cdes` (train
lengths 3–40, validate 40–256; batch 256, 100k steps, eval every 5000; metric = best validation
accuracy over the run; early stop at > 0.9995; 5 seeds 1234–1238, one run per seed).

Each `<task>/` holds:
- `config.json`: the selected config, fully resolved (`--run r` gives seed 1234 + r)
- `result.md`: the banked numbers, the protocol and the command that reproduces them
- `runs/seed<S>/`: the banked runs' `results.json` (every evaluation) and `sidecar.json`

## Results (best validation accuracy %, mean ± population std over 5 seeds)

| task | result | solved (≥ 99.95) |
|---|---|---|
| [parity](parity/result.md) | **99.995 ± 0.006** | 5/5 |
| [even_pairs](even_pairs/result.md) | **100.000 ± 0.000** | 5/5 |
| [cycle_nav](cycle_nav/result.md) | **99.985 ± 0.014** | 5/5 |
| [mod_arith_no_brack](mod_arith_no_brack/result.md) | **96.71 ± 0.75** | 0/5 |

Deviations from Walker's training defaults are listed per task in `result.md` (dropout 0 on all four; mod_arith
additionally trains A without weight decay under an ‖A‖_F penalty and calibrates block 1's init scale).

## Clock / A-structure ablation

`clock_A_ablation/` asks which ingredient carries these results: the banked config with only the clock made
uniform, only A made a positive diagonal (efm), or both, 3 seeds per cell at the full 100k steps. Only the
champion cell solves parity, cycle_nav or mod_arith; even_pairs is solved by efm with the champion clocks and not
by the rotation A with a uniform clock. The table and the logs are in `clock_A_ablation/`.

## Reproduce

```bash
GPU=<n> bash formal_language_results/replicate.sh <task>   # all five seeds
```

The runs use XLA autotuning off and deterministic ops off, which is not bit-reproducible, so a replicate lands near the
banked values rather than on them.
