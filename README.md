# Extending SSMs with the Exponentially Weighted Signature

Code for the paper [Extending SSMs with the Exponentially Weighted Signature](https://arxiv.org/abs/2603.19198).

The exponentially weighted signature (EWS) is a continuous-time model that computes iterated integrals of a path, each
increment weighted by the matrix exponential of a learnable generator over elapsed clock time. It solves a linear
controlled differential equation, keeps the group-like structure and universality of the signature, and satisfies a
modified Chen identity, enabling a parallel scan. At depth one the EWS is a state-space model (SSM), and linear
time-invariant SSMs, Mamba channels and Mamba-2 heads map to it in closed form. The EWS extends SSMs through an
arbitrary matrix generator, a clock that generalises the step size to causal functionals of the input, and higher
truncation depths that are non-linear in the path within a single layer.

This repository is a JAX implementation of the EWS and the experiments of the paper.

## Requirements

Python 3.12. In a fresh virtual environment:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # pinned: JAX 0.9.0, signax 0.2.1, equinox 0.13.2, ...
pip install -e .
```

The requirements pin the CUDA 12 GPU wheels; for CPU only, install `jax[cpu]==0.9.0` instead. The package is `ews/`:
`paths.py` (the path stage), `clocks.py` (the clocks), `params.py` (the parametrisations of `A` and of the lift `B`),
`core/` (the local terms and the Chen scan), `layer.py` (`EWS`, `LogEWS`), `block.py` (`EWSBlock`, `StackedEWS`),
`norms.py`, `config.py` (the config schema) and `determinism.py` (the XLA switches). The documentation is in
[`docs/`](docs/overview.md).

## Data

Dataset files are not tracked. [`data/README.md`](data/README.md) gives the command for each: the UEA archive and
Walker's per-seed splits, the generated synthetic data, the Informer weather file and the PPG-DaLiA windows. The
formal-language tasks are generated on the fly.

## Experiments

| family | data | model |
|---|---|---|
| `experiments/uea/` | six UEA classification datasets | one EWS layer |
| `experiments/fl/` | parity, even_pairs, cycle_nav, mod_arith_no_brack | stacked EWS blocks |
| `experiments/sde/` | the expressivity study, the coupled oscillator, the oscillatory kernel | one EWS layer |
| `experiments/weather/` | Informer weather, 720 steps ahead | stacked EWS blocks |
| `experiments/ppg/` | PPG-DaLiA heart rate | stacked EWS blocks |

```bash
python experiments/uea/train.py --config uea_results/eigenworms/config.json --seed 0 --dataset_file <split.npz> --gpu 0
python experiments/fl/train.py --config formal_language_results/parity/config.json --run 0 --gpu 0
python experiments/sde/train.py --config sde_results/oscillator/ews/config.json --seed 100 --gpu 0
python experiments/weather/train.py --config weather_results/config.json --seed 2345 --gpu 0
python experiments/ppg/train.py --config ppg_results/config.json --seed 2345 --gpu 0
```

The config keys are documented in [`docs/configs.md`](docs/configs.md), the trainers and their protocols in
[`docs/training.md`](docs/training.md).

## Results

**UEA classification** (`uea_results/`): test accuracy (%), mean ± population standard deviation over the five split
seeds, each configuration selected on validation from the grid in its `grid.json`.

| dataset | test accuracy | per seed |
|---|---|---|
| EigenWorms | 93.33 ± 1.36 | 91.67, 94.44, 94.44, 91.67, 94.44 |
| EthanolConcentration | 71.90 ± 4.76 | 77.22, 64.56, 73.42, 75.95, 68.35 |
| Heartbeat | 80.00 ± 4.52 | 72.58, 79.03, 79.03, 83.87, 85.48 |
| MotorImagery | 61.05 ± 5.01 | 59.65, 57.89, 64.91, 54.39, 68.42 |
| SelfRegulationSCP1 | 87.06 ± 2.47 | 88.24, 89.41, 87.06, 82.35, 88.24 |
| SelfRegulationSCP2 | 56.49 ± 5.81 | 66.67, 50.88, 50.88, 56.14, 57.89 |

```bash
bash uea_results/replicate.sh motor    # or eigenworms, ethanol, heartbeat, scp1, scp2
```

**Formal languages** (`formal_language_results/`): best validation accuracy (%) under Walker's length-generalisation
protocol, five seeds.

| task | best validation accuracy | per seed |
|---|---|---|
| parity | 100.0 | 100.0, 100.0, 100.0, 100.0, 100.0 |
| even_pairs | 100.0 | 100.0, 100.0, 100.0, 100.0, 100.0 |
| cycle_nav | 100.0 | 100.0, 100.0, 100.0, 100.0, 100.0 |
| mod_arith_no_brack | 96.7 | 97.1, 96.6, 97.4, 95.3, 97.1 |

```bash
GPU=0 bash formal_language_results/replicate.sh parity    # or even_pairs, cycle_nav, mod_arith_no_brack
```

**Synthetic regression** (`sde_results/`): the expressivity study (nine target-learner pairs) and the coupled
oscillator (three learners), ten seeds each, and the oscillatory kernel (EWS against the EFM and the signature over a
depth-width grid, three seeds). Its README has the tables.

```bash
GPUS=0 bash sde_results/replicate.sh oscillator/ews
```

**Weather** (`weather_results/`): test MAE 0.4733 ± 0.0016 over the five seeds, 18,669 parameters, selected on
validation MAE from a 54-cell grid.

**PPG-DaLiA** (`ppg_results/`): test MSE 5.90 ± 0.73 (×10⁻²) over the five seeds, 11,109 parameters, selected on
validation from a 54-cell grid.

```bash
GPU=0 bash weather_results/replicate.sh
GPU=0 bash ppg_results/replicate.sh
```

**TUAB** (`tuab_results/`): eval AUROC 0.8824 ± 0.0028 and balanced accuracy 79.90 ± 0.45 over five seeds, 20,250
parameters, selected on validation AUROC from an 18-cell grid. The folder holds the configuration and the grid; the
corpus is distributed under its own data-use agreement.

**SSM conversion** (`ews/conversion/`): S5, LRU, LinOSS-IM/IMEX, D-LinOSS, S4D, S4 and Mamba-1/2 written as depth-one
EWS layers, with the check that each reproduces the model's own state (`python -m ews.conversion.verify`).

## Citation

```bibtex
@misc{bloch2026ews,
  title  = {Extending {SSMs} with the Exponentially Weighted Signature},
  author = {Bloch, Alexandre and Cohen, Samuel N. and Lyons, Terry and Mouterde, Jo{\"e}l and Walker, Benjamin},
  year   = {2026},
  url    = {https://arxiv.org/abs/2603.19198},
}
```

## License

The code in this repository is released under the MIT License. See [LICENSE](LICENSE) for details. The files in
`ews/conversion/reference/vendored/` and `experiments/fl/dataloaders.py` are copied from other MIT-licensed
repositories; their sources and copyright notices are in
[`ews/conversion/reference/vendored/README.md`](ews/conversion/reference/vendored/README.md).

## Forks

[ews-log-ncde-fork](https://github.com/alexandre-bloch/ews-log-ncde-fork) and
[ews-slices-fork](https://github.com/alexandre-bloch/ews-slices-fork) run the UEA and the formal-language experiments
inside Walker's [log-neural-cdes](https://github.com/Benjamin-Walker/log-neural-cdes) and
[structured-linear-cdes](https://github.com/Benjamin-Walker/structured-linear-cdes), for consistency with the models
there. They add no new method code, and every result in them can also be reproduced from this repository.
