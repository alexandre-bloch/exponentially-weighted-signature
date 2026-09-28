# Configuration reference

Every experiment is driven by one JSON config. Configs are organised by module, so a section name
tells you which part of the code reads it:

| section | read by | contents |
|---|---|---|
| `A`, `B` | `ews/params.py` | parametrisation and init of the drift `A` and the lift `B` |
| `clock` | `ews/clocks.py` | the clock θ |
| `append_clock` | `ews/clocks.py` | a second clock added to the path as a channel (optional) |
| `path` | `ews/paths.py` | what enters the signature as X |
| `compute` | `ews/layer.py` | how the signature is computed |
| block keys | `ews/block.py` | width, depth and the block scaffold (stacked models) |
| `training` | the trainer | optimiser, schedule, protocol, XLA switches |

Two model shapes exist. A **stacked model** (formal languages) puts the block sections under
`model.block` as defaults and `model.blocks[i]` as per-block overrides. A **layer-only model**
(UEA, SDE regression) puts `A`, `B`, `compute` and the readout keys directly under
`model`, and `path` / `clock` under `preprocessing`, because there they are constants of the data
and are computed once at load time.

Unknown keys raise at load. Older flat layouts are rewritten by `ews.config.upgrade`, which every
trainer applies before validating, so a config may be passed through it any number of times.
The banked configs under `uea_results/` and `formal_language_results/` are complete examples;
`experiments/uea/configs/template.json` and `experiments/fl/configs/template.json`
are copies of two of them to start from.

## `path`

| key | values | default | meaning |
|---|---|---|---|
| `representation` | `value`, `increment`, `derivative` | `value` | X_i = x_i; X = cumulative sum of the rows (the discrete integral of an increment stream, used by the formal-language blocks); the path replaced by its Savitzky-Golay derivative |
| `derivative_order`, `derivative_window`, `derivative_polyorder` | ints | 0, 15, 3 | the derivative, when `representation` is `derivative` |
| `normalize`, `normalize_mode`, `normalize_range` | bool; `train_only` / `all_data`; `[0,1]` / `[-1,1]` | true, `train_only`, `[0,1]` | channel normalisation fitted on the training split or on all splits |
| `per_sample_normalize`, `per_sample_norm_stat` | bool; `std` / `rms_dx` | false, `std` | one scalar per instance, applied before anything fitted |
| `basepoint` | bool | false | prepend a zero row so the first increment is the first value |
| `clock_channel` | `null`, `main`, `separate` | `null` | add a clock as channel 0: the main clock θ itself, or the `append_clock` |

Order of application: representation, per-sample normalisation, then the clock is built from the
representation, then channel normalisation, basepoint, clock channel. The clock never sees the
normalised path.

## `clock`

| key | values | default | meaning |
|---|---|---|---|
| `mode` | `time`, `blended_variation`, `blended_quad_variation`, `blended_learn`, `token_gated`, `clamped_gate` | `time` | θ = t; α·t + (1−α)·V with V the cumulative L2 increment norm; the same with squared increments; the same with α trainable (`alpha` is then its initialisation); cumsum relu(w·Δx − b)/c; cumsum Σ_k clip(w_k·Δx − b_k, 0, 1)/c |
| `alpha` | float in [0, 1] | 1.0 | blend weight of the time component (blended modes) |
| `scaling` | `none`, `[0,1]`, `[-1,1]`, `steps` | `[0,1]` | range of each component before blending; `steps` (mode `time` only) is the sample counter, theta_i = i, not normalised by length, and counts the basepoint step |
| `normalize`, `norm_constant` | `per_sample` / `constant`; float | `per_sample`, null | divide the variation by its own total, or by a fixed constant (required by the gated modes) |
| `time_denominator` | float | null | scale of the time component when not per sample |
| `variation_eps` | float | 1e-8 | inside the variation square root |
| `stop_gradient` | bool | false | detach θ from the loss |
| `gate_init_scale`, `gate_init_bias`, `gate_K` | float; float or list; int | null, −2.0, 4 | the gated modes' parameters: weight scale, threshold(s), number of clamped terms |
| `gate_in_scale` | float > 0 | 1.0 | fixed factor on the driver before a gated clock |
| `alpha_parametrisation` | `clip`, `sigmoid` | `clip` | `blended_learn` only: α clipped to [0, 1], or α = sigmoid(raw) initialised at logit(α) |
| `learn` | bool | true | train the clock's parameters |
| `source` | `block_input`, `encoder_output`, `raw_input` | `block_input` | stacked models only: which signal the block hands to its clock |

`time` and the two fixed blended modes have no parameters; `blended_learn` and the gated modes are trained unless
`learn` is false. In a layer-only model a
gated clock is evaluated in the forward pass on the model input. `append_clock` takes the same keys
and, in a stacked model, inherits the main clock's settings except `mode`, `alpha` and
`norm_constant`.

## `A` and `B`

| key | values | default | meaning |
|---|---|---|---|
| `parametrization` | see below | `general` | structural family of `A` |
| `init.type` | `sig`, or a parametrisation name | `sig` | starting matrix: `sig` is the zero matrix for the linear families and the family's own random init for the rotation families |
| `init.mean`, `init.variance` | floats | 0.0, 1.0 | Gaussian init of the parameters |
| `init.sigma_range` | float | 0.0 | rotation families: frequencies start uniform in ±logit(0.5 + sigma_range) |
| `omega_scale` | float | 1.0 | rotation families: frequency range multiplier |
| `damp_init_mean`, `damp_init_std` | floats | −2.0, 1.0 | damped rotation: init of the damping logits |
| `positivity` | `softplus`, `square` | `softplus` | efm: how the diagonal is kept positive |
| `learn` | bool | true | train `A` |
| `blocks`, `kinds` | list of int; list of `real` / `complex` | null | `block_diagonal`, `unconstrained_rotation` and `jordan_repeated` |
| `epsilon` | float | 0.0 | added to the diagonal for the definite families |

Parametrisations: `sig` (zero, the classical signature), `efm` (positive diagonal), `general`,
`skew_symmetric`, `symmetric`, `positive_definite`, `definite_plus_skew`, `triangular_upper`,
`triangular_lower`, `block_diagonal`, `jordan_repeated` (true Jordan blocks, one repeated
eigenvalue per block), and the rotation families `bounded_rotation`, `unconstrained_rotation`,
`damped_rotation`, `relu_damped_rotation`, `clamped_rotation`, `unbounded_rotation` (axis-aligned
planes with sigmoid-bounded frequencies and, for the damped ones, a per-plane contraction rate),
and `stable_block` (those planes plus a bounded shear, the free 2 × 2 block confined to the region
that cannot expand; `shear_init_mean` / `shear_init_std` init its shear fraction).

`B` (`enabled`, `out_dim`, `parametrization` `general`, `init.type` `general` / `identity_block` /
`sig`, `init.mean`, `init.variance`, `learn`, `_preserve_first_channel`) is the linear lift applied
before the signature; `_preserve_first_channel` passes channel 0 through unmixed, which is what an
appended clock channel needs.

## `compute`

| key | values | default | meaning |
|---|---|---|---|
| `leaf` | `van_loan`, `series`, `plane_closed`, `block_series`, `jordan_closed`, `jordan_closed_tables` | `van_loan` | per-interval local signatures by Van Loan's augmented exponential, by the closed-form series, by the rotation-plane closed forms, by the blockwise series (block-diagonal A), or by the Jordan-chain closed forms (`jordan_repeated` with uniform chains; `_tables` is their float64-table cross-check) |
| `series_q` | int | null | order of the series leaves, fixed for a run |
| `combine` | `assoc`, `lean`, `block_kernel`, `dense_kernel` | `assoc` | how the intervals are composed: the original associative scan, the lean scan on precomputed discounts, the fused block-diagonal kernel, or the buffered dense kernel (see computation.md) |
| `combine_chunk`, `combine_tile` | int | 1024, 4 | tuning knobs of the fused kernels; no effect on the result |
| `van_loan_expm`, `chen_expm` | `exact`, `cayley`, `pade`, `euler` | `cayley`, `exact` | matrix-exponential methods in the leaf and in the Chen combination |
| `M` | int | 20 (layer), 4 (block) | sub-steps per interval in the leaf |
| `terminal_reduce` | `tree`, `assoc` | `tree` | how the terminal signature is reduced |
| `reweighting_mode` | `displacement`, `displacement_uniform`, `displacement_uniform_basepoint` | `displacement` | the reweighting kernel; the uniform variants need a uniform clock and are auto-routed to `displacement` otherwise |

The combination of intervals is the associative scan unless `combine` selects one of the kernels.

## Stacked models (`model`)

| key | values | default | meaning |
|---|---|---|---|
| `num_blocks`, `hidden_dim` | ints | | the stack |
| `head` | `seq2seq`, `mean_pool` | `seq2seq` | per-position head, or masked mean pool then head |
| `tokens` | bool | true | embedding encoder for token inputs, linear encoder otherwise |
| `token_cache` | bool | false | per-token cache for the first block over a discrete vocabulary (needs `path.representation` `increment`, the block-input clock source and no appended clock on that block) |
| `input_token_ids` | list of int | null | vocabulary ids that occur as inputs (bounded-rotation scale excludes the others) |
| `graded_eps_init` | `constant`, `calibrated` | `constant` | init of the graded-norm floors |
| `final_norm`, `cosine_head_tau`, `dtype` | bool; float; `float32` / `float64` | false, null, `float32` | |
| `block` | dict | | defaults for every block |
| `blocks` | list of dict | | per-block overrides, same keys as `block` |

Block keys: the sections `A`, `B`, `clock`, `append_clock`, `path` (`representation` and
`basepoint` only), `compute`, plus `m` (signature width), `depth` (signature depth),
`norm_placement` (`pre`, `slice_post`, `none`), `second_norm`, `graded_norm` (`live_rms`,
`static_gain`, `dilation`), `graded_eps`, `graded_prescale` (`null`, `binomial`), `channel_stage`
(`mlp`, `tanh`, `glu`, `gelu_glu`), `ff_mult`, `output_gate`, `dropout_rate`, `norm_eps`, and
`angle_target`, the init-scale calibration: the block's `A` is rescaled once at init so the median
per-token rotation angle equals this value in radians.

## Layer-only models (`model` and `preprocessing`)

`preprocessing`: `dtype`, `path`, `clock`, `append_clock`; regression adds `normalize_targets` and
`target_range`, prediction adds `predict_channel`. `model`: `A`, `B`, `compute`, `depth`
(signature depth), `readout` (`terminal`, `mean`, `max`), `feature` (`signature`, `logsignature`),
`feature_norm` (`none`, `learned_gain`, `batchnorm`), `feature_norm_init` (`rms`, `ones`) and
`feature_norm_eval` (`batch`, `train`; `batchnorm` keeps no running statistics: `batch` normalises an evaluated set
with its own statistics, `train` with the training set's, recomputed with the evaluated model, so predictions are
independent of each other),
`head_type` (`linear`, `mlp`), `head_hidden_dims`, `head_init_scale`.

## `training`

Stacked models (formal languages, Walker's protocol): `seed` (run r uses seed + r), `batch_size`,
`num_steps`, `print_steps` (eval cadence), `learning_rate`, `final_lr`, `warmup_fraction`,
`schedule` (`cosine`), `weight_decay_embedding`, `weight_decay_others`, `weight_decay_A`,
`weight_decay_A_damp`, `vf_A_norm_lambda` (penalty λ·Σ‖A‖_F), `label_smoothing`, `grad_clip`,
`early_stop_threshold`, `checkpoint_every`, `val_batch_size`, `adam_b1`, `adam_b2`,
`adam_eps`, `lr_mult_A`, `lr_mult_B`, `fix_embedding_norm`, `futility_step`, `futility_min_acc`,
`max_train_steps`, `init_from`.

Layer-only models: `seed`, `batch_size`, `num_steps`, `eval_every`, `log_every`, `warmup_steps`,
`lr_features`, `lr_head`, `min_lr_features`, `min_lr_head`, `weight_decay_features`,
`weight_decay_head`, `grad_clip_features`, `grad_clip_head`, `lambda1`, `lambda2`,
`label_smoothing`, `metric`, `threshold`, `early_stopping_patience`, `early_stopping_min_delta`,
and for prediction `test_mode`, `test_batch_size`.

Weather (`experiments/weather/train.py`, flat `model` and `training` sections; see `weather_results/config.json`):
`lr`, `lr_mult_A`, `a_warmup`, `lr_mult_clock`, `wd`, `wd_A`, `grad_clip`, `batch_size`, `num_steps`,
`print_steps`, `patience`, `loss` (`mse`, `l1`; default `mse`) and `select` (`val_mse`, `val_mae`; default
`val_mse`), the training loss and the validation metric that picks the checkpoint and drives early stopping.

Both: `xla_autotune` and `deterministic_ops` (booleans) set the two XLA switches; the command-line
`--autotune {on,off}` and `--deterministic-ops {on,off}` take precedence (see [training.md](training.md)).
Deterministic ops make a run bit-reproducible and slower; autotuning makes it faster and not reproducible.

## Running

```bash
python experiments/uea/train.py --config uea_results/scp1/config.json --seed 0 --dataset_file <split.npz> --gpu 0
python experiments/fl/train.py --config formal_language_results/parity/config.json --run 0 --gpu 0
python experiments/sde/train.py --config sde_results/oscillator/ews/config.json --seed 100 --gpu 0
```
