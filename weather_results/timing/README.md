# Weather speed and memory

The selected model (`../config.json`) on its own trainer, one cell at a time on an H100 NVL: 1440-step sequences, 4000
steps with a print every 1000, early stopping off, XLA autotuning on, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, seed 2345.
Time is training only, the median of the steady intervals after the first (which contains the compilation). Memory is
the in-process allocator peak and the process peak in `nvidia-smi`, polled every second.

| batch | params | s / 1000 steps | min-max | allocator peak | nvidia-smi peak |
|---|---|---|---|---|---|
| 16 | 18,669 | 8.95 | 8.74-9.05 | 364.5 MB | 1722 MB |
| 4 | 18,669 | 7.74 | 7.74-8.04 | 288.9 MB | 1210 MB |

A second batch-16 pass gave 9.22 s / 1000 steps with the same memory. `cells.json` holds the per-cell records.
