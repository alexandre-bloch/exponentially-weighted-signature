# The EWS layer (`ews/layer.py`) and the layer-only model

`EWS(A_param, N, M, ...)` maps one path and one clock to signature features. Its call signature is
`layer(path, clock, readout, output_format, a_scale)`:

| argument | meaning |
|---|---|
| `path` | (L, m) array, the lifted path X |
| `clock` | (L,) non-decreasing array θ |
| `readout` | `terminal` (the signature at the end), `prefix` (at every position), `mean` or `max` over positions |
| `output_format` | `flat` (one vector per readout position, level 0 first) or `list` (level tensors) |
| `a_scale` | a scalar multiplier applied to A at forward time, used by the block for the rotation families |

Batches are vmapped inside. The layer holds only the parametrisation of A and the computation
options; it knows nothing about how the path or the clock were produced. `B` is the linear lift
applied before the layer; `FeatureNorm` normalises the flattened feature level by level, either a
learned gain per level, batch statistics, or nothing; `LogEWS` wraps the layer and takes the
log-signature at the readout.

## The layer-only model (`experiments/uea/train.py`)

`PathLabelModel` = lift, layer, feature normalisation, head. The loader precomputes the path and
the clock from the `preprocessing` section, so the model receives constant arrays. The one
exception is a clock with parameters: then the model holds the clock module, and θ is evaluated
in the forward pass from the model's input path, with a placeholder time clock from the loader.
Everything else about the model is unchanged when the clock is static.

`model` keys for this shape: `A`, `B`, `compute`, `depth`, `readout`, `feature` (`signature` or
`logsignature`), `feature_norm` (`none`, `learned_gain`, `batchnorm`), `feature_norm_init` (`rms`
sets the learned gains to 1/RMS of a training batch at init, `ones` leaves them at one),
`head_type` (`linear`, `mlp`), `head_hidden_dims`, `head_init_scale`.

The regression and prediction trainers use the same layer with their own heads and losses.
