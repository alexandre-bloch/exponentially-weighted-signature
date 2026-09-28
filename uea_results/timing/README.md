# UEA speed and memory

Training speed and memory of the six selected UEA models, each measured alone on an H100 NVL.

Time is training-only seconds per 1000 steps. The run prints every 100 steps; the first interval (which contains the
compilation) and the evaluation intervals are dropped, and the median of the rest is taken. Memory is the process
peak in `nvidia-smi`, sampled every second with the CUDA context included, and the in-process allocator peak (JAX
`peak_bytes_in_use`). Every cell runs 1,100 steps with `XLA_PYTHON_CLIENT_PREALLOCATE=false` and the persistent
compilation cache off, so each one compiles cold.

**Standardised batch** (32; EthanolConcentration 16, EigenWorms 8): seconds per 1000 steps / `nvidia-smi` peak MB.

| EigenWorms | Ethanol | Heartbeat | Motor | SCP1 | SCP2 |
|---|---|---|---|---|---|
| 19.7 / 4,788 | 27.8 / 5,964 | 14.6 / 2,764 | 83.6 / 34,506 | 10.2 / 2,752 | 29.5 / 9,540 |

Allocator peak (MB):

| EigenWorms | Ethanol | Heartbeat | Motor | SCP1 | SCP2 |
|---|---|---|---|---|---|
| 3,117 | 3,972 | 913 | 16,144 | 1,282 | 4,113 |

**Own batch** (each model at the batch of its config: 4 / 8 / 8 / 16 / 32 / 8), seconds per 1000 steps /
`nvidia-smi` peak MB; the Ethanol model was measured at the standardised batch only.

| EigenWorms | Ethanol | Heartbeat | Motor | SCP1 | SCP2 |
|---|---|---|---|---|---|
| 14.2 / 4,788 | — | 6.2 / 1,740 | 46.6 / 18,122 | 10.2 / 2,752 | 11.1 / 9,540 |

Allocator peak (MB):

| EigenWorms | Ethanol | Heartbeat | Motor | SCP1 | SCP2 |
|---|---|---|---|---|---|
| 2,933 | — | 689 | 10,027 | 1,282 | 3,079 |

`memtime_run.py` measures one cell: it trains the dataset's banked `config.json` on its seed-0 split and writes the
intervals and memory peaks to `result.json`. `collate_memtime.py` turns a folder of cells into these tables.

```bash
python uea_results/timing/memtime_run.py --family ews --model ews --dataset Heartbeat --steps 1100 --out <dir> --gpu 0 --batch 32
```
