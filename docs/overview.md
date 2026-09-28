# EWS: overview

Every model in this repository is the same five stages, each with its own module and its own
config section:

| stage | module | config | what it decides |
|---|---|---|---|
| path | `ews/paths.py` | `path` | the raw values, a cumulative sum of increments, or a derivative; normalisation, basepoint, a clock channel |
| clock | `ews/clocks.py` | `clock` | time, variation of a driver, or a learnable gate on the driver's increments |
| parametrisation | `ews/params.py` | `A`, `B` | the structural family of the drift A and the linear lift B, and their init |
| computation | `ews/core/`, `ews/layer.py` | `compute` | the leaf (per-interval local signatures) and the combine (Chen's associative scan) |
| block | `ews/block.py` | block keys | the residual scaffold around the layer, the stack, the token cache |

The order of operations is fixed: the representation is formed, the clock reads it before any
normalisation, then normalisation, basepoint and the optional clock channel are applied, then the
layer computes the signature. In a layer-only model the first two stages are constants of the data
and are precomputed once; inside a block they are evaluated in the forward pass because the driver
is learnable.

## Two model shapes

- **Layer-only** (`experiments/uea`, `experiments/sde`): path and clock are
  precomputed by the loader from `preprocessing`, the `EWS` layer maps (path, θ) to the flattened
  signature, a feature normalisation and a head follow. A clock with parameters can also be used
  here; it is then evaluated in the forward pass on the model input.
- **Stacked** (`experiments/fl`, `experiments/weather`, `experiments/ppg`): an encoder, `num_blocks` `EWSBlock`s and a head. Each
  block owns its clock, its path choice, its computation options and its parametrisations, with
  defaults in `model.block` and per-block overrides in `model.blocks[i]`.

## Where to look

| document | contents |
|---|---|
| [configs.md](configs.md) | every config section and key |
| [paths.md](paths.md), [clocks.md](clocks.md) | the two input stages |
| [params.md](params.md) | the A and B families and their initialisation |
| [computation.md](computation.md) | leaf, combine, exponentials, depth |
| [layer.md](layer.md), [block.md](block.md) | the layer API, the block scaffold, the stack, the token cache |
| [training.md](training.md) | the trainers, protocols, reproducibility, banked results |
| [../data/README.md](../data/README.md) | datasets and how to regenerate them |
