# PPG-DaLiA heart-rate regression (49,920 → 390)

**Test MSE 5.90 ± 0.73 (×10⁻²)** (mean ± population std over seeds 2345, 3456, 4567, 5678, 6789), 11,109 parameters,
a stack of four EWS blocks at depth 1, selected by validation MSE from a 54-cell grid.

| seed | 2345 | 3456 | 4567 | 5678 | 6789 |
|---|---|---|---|---|---|
| test MSE ×10⁻² | 6.55 | 6.42 | 4.91 | 5.11 | 6.51 |
| validation MSE ×10⁻² | 4.14 | 4.36 | 3.64 | 3.97 | 3.67 |
| best step | 34,000 | 15,000 | 6,000 | 3,000 | 24,000 |

## Task and data

PPG-DaLiA (UCI archive 495): 15 subjects, wrist PPG + accelerometer + EDA + temperature at 64 Hz with ECG-derived
heart rate every 2 s. Walker's processing (`data_dir/process_ppg.py` of Benjamin-Walker/log-neural-cdes, as used by
LinOSS / D-LinOSS): per-subject min–max scaling to [−1, 1], windows of 49,920 input steps × 6 channels (ACC ×3, BVP,
EDA, TEMP) with 390 heart-rate targets (one per 128 steps), split in time within each subject (contiguous 70 / 15 / 15
blocks, one of six block orders per subject, drawn by the published script) into train 1,020 / val 106 / test 106
windows. The model sees the 6 channels plus a uniform time channel (7 inputs). Data files:
`data/ppg/processed_walker/{X,y}_{train,val,test}.npy` (see `data/README.md`); the loader is `experiments/ppg/data.py`
(Walker's `Dataloader`, transcribed).

## Model (`config.json`)

A linear encoder to width 16, four EWS blocks at depth 1 with m = 64, and a linear head; every 128th position through
tanh gives the 390 outputs (Walker's head). In each block A is 16 complex Jordan chains of size 4, each [[C, I], [0, C]]
with C = [[a, −b], [b, a]] a damped rotation: a = softplus(v) with v ~ N(−4.671, 1), and b drawn log-uniformly in
[0.0234, 2.341] rad per step; all of A is trained. The clock is a clamped gate (four clipped units of the block input,
calibrated so that the mean increment is 1 at initialisation: norm constant 4.0). Each block has a gelu-GLU channel
stage, a pre-norm residual and dropout 0.2. The local terms come from the closed-form Jordan leaf; the prefix comes from
the fused block kernel, with the chains' flows in closed form, in chunks of 4,992 steps.

## Training protocol (LinOSS / D-LinOSS's)

MSE training loss; validation MSE every 1,000 steps over the whole validation set; the best-validation checkpoint is
the one tested; early stop once 10 consecutive evaluations bring no improvement; Adam at a constant learning rate
2.5e-4 with no weight decay and no gradient clipping, batch 8, a 100,000-step budget; seeds 2345–6789. These are the
settings of D-LinOSS's published PPG run (damped-linoss a396997, `config/repeats/LinOSS/ppg/config_000.json`:
`print_steps` 1000, `num_steps` 100,000, the same five seeds) with its trainer's patience of 10 evaluations; the batch
(8 here, 4 there) and the learning rate differ.

## Selection (`grid.json`)

The grid is lr ∈ {2.5e-4, 5e-4, 1e-3} × hidden ∈ {8, 16, 32} × m ∈ {32, 64} × blocks ∈ {2, 4, 6} = 54 cells. Sixteen
of them ran on seeds 2345, 3456 and 4567 and the four below on all five; the rule is the lowest five-seed mean
validation MSE.

| lr | hidden | m | blocks | params | validation MSE ×10⁻² (5 seeds) | test MSE ×10⁻² |
|---|---|---|---|---|---|---|
| **2.5e-4** | **16** | **64** | **4** | **11,109** | **3.955** | **5.90 ± 0.73** |
| 1e-3 | 16 | 64 | 4 | 11,109 | 3.966 | 6.02 ± 0.86 |
| 5e-4 | 32 | 64 | 4 | 26,165 | 4.098 | 5.82 ± 0.68 |
| 2.5e-4 | 32 | 32 | 6 | 26,719 | 4.186 | 6.77 ± 0.49 |

`grid.json` lists every cell with each seed's validation and test MSE; `grid_runs.tar.gz` holds every grid run's
`best.json`, `history.json` and training log.

## Files and replication

| file | contents |
|---|---|
| `config.json` | the selected configuration |
| `grid.json`, `grid_runs.tar.gz`, `run_grid.py` | the grid, its runs and the script that runs it |
| `search.csv` | the TPE search run before the grid (43 trials, 34 of them complete; objective: the validation MSE of seed 2345) |
| `runs/seed_<s>/` | the five banked runs (`best.json`, `history.json`, `train.log`, `model.eqx`) |

```bash
bash ppg_results/replicate.sh                                        # the five seeds on GPU 0 (GPU=<n>, PAR=<runs in parallel>)
python ppg_results/run_grid.py --gpu 0 --slots 3 --out outputs/ppg_grid
python ppg_results/run_grid.py --report --out outputs/ppg_grid
```
