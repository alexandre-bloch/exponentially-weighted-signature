# Weather forecasting (Informer weather, 720 → 720)

**Test MAE 0.4733 ± 0.0016** (mean ± population std over seeds 2345, 3456, 4567, 5678, 6789), 18,669 parameters,
trained with an L1 loss and selected by validation MAE from a 54-cell grid.

| seed | 2345 | 3456 | 4567 | 5678 | 6789 |
|---|---|---|---|---|---|
| test MAE | 0.4722 | 0.4761 | 0.4739 | 0.4725 | 0.4717 |
| validation MAE | 0.5952 | 0.5939 | 0.5963 | 0.5962 | 0.5949 |
| best step | 23,000 | 26,000 | 38,000 | 22,000 | 22,000 |

Each run's `best.json` also records the MSE of the same checkpoint.

## Task and model

Informer's weather data (`data/weather/WTH.csv`, 35,064 hourly rows of 12 climate channels), multivariate: the input
is the last 720 hours of all 12 channels and the target the next 720 hours. `experiments/weather/data.py` transcribes
the S4 / Informer loader: a 70 / 10 / 20 split by time, standard scaling fitted on the training slice, metrics in the
scaled space. The inputs are the 12 channels, a time channel and Informer's four calendar marks over the 1440-step
window, with the forecast half of the climate channels set to 0.

The model (`config.json`, `experiments/weather/model.py`) is one EWS block: a linear encoder to width 64, an EWS layer
at depth 1 with m = 64 on the time clock, a gelu-GLU channel stage with a pre-norm residual and dropout 0.15, and a
per-position linear head whose last 720 positions are the forecast. A is 16 complex Jordan chains of size 4, each
[[C, I], [0, C]] with C a damped rotation; all of A is trained. It runs on the closed-form Jordan leaf and the fused
block kernel.

## Training protocol

D-LinOSS's: the L1 loss, validation MAE every 1000 steps, the best-validation checkpoint tested, early stopping after
10 evaluations without improvement, Adam at one constant learning rate (5e-5 here) with no weight decay and no
gradient clipping, a 100,000-step budget, five seeds. It differs from D-LinOSS's published run in the batch (16
here, 4 there) and in the inputs (theirs are the 12 channels with the forecast half set to −1).

## Selection

The grid is lr ∈ {2.5e-5, 5e-5, 1e-4} × hidden ∈ {32, 64, 128} × m ∈ {16, 32, 64} × blocks ∈ {1, 2} = 54 cells.
Seventeen of them ran on seeds 2345, 3456 and 4567, and the four best by mean validation MAE on 5678 and 6789. The
rule is the lowest five-seed mean validation MAE, and within the per-seed spread of the best, the model with fewer
parameters.

| lr | hidden | m | blocks | params | validation MAE (5 seeds) | test MAE |
|---|---|---|---|---|---|---|
| **5e-05** | **64** | **64** | **1** | **18,669** | **0.5953** | **0.4733 ± 0.0016** |
| 1e-04 | 128 | 64 | 1 | 53,677 | 0.5963 | 0.4777 ± 0.0031 |
| 5e-05 | 32 | 64 | 1 | 7,309 | 0.5985 | 0.4736 ± 0.0016 |
| 2.5e-05 | 128 | 32 | 1 | 45,469 | 0.5985 | 0.4755 ± 0.0010 |

`grid.json` lists every cell with each seed's validation and test metrics; `grid_runs.tar.gz` holds every grid run's
`best.json`, `history.json` and training log.

At batch 16 the model trains at 8.95 s per 1000 steps with a 365 MB allocator peak (`timing/`).

## Files and replication

| file | contents |
|---|---|
| `config.json` | the selected configuration |
| `grid.json`, `grid_runs.tar.gz`, `run_grid.py` | the grid, its runs and the script that ran it |
| `search.csv` | the TPE search run before the grid (50 trials, objective: the validation MSE of one seed) |
| `runs/seed_<s>/` | the five banked runs (`best.json`, `history.json`, `train.log`, `model.eqx`) |
| `timing/` | speed and memory |

```bash
bash weather_results/replicate.sh                          # the five seeds on GPU 0 (GPU=<n>, PAR=<runs in parallel>)
python weather_results/run_grid.py --gpu 0 --slots 4 --out outputs/weather_grid
python weather_results/run_grid.py --report --out outputs/weather_grid
```
