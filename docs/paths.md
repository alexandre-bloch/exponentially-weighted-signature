# The path stage (`ews/paths.py`, config section `path`)

The path stage decides what enters the signature as X. `PathSpec` holds the choices and
`prepare_paths_np` applies them to a batch of numpy paths together with the clock; that is the
precomputation used whenever the path and the clock are constants of the data. Inside a block the
same choices are applied per forward pass, with `to_increment_path_jnp` as the in-model counterpart
of the increment representation.

## Options

| key | values | default | meaning |
|---|---|---|---|
| `representation` | `value` | `value` | X_i = x_i, the raw series |
| | `increment` | | the rows are per-step increments and X is their cumulative sum, the discrete integral of an increment stream. This is how token embeddings enter the formal-language blocks |
| | `derivative` | | the path is replaced by its Savitzky-Golay derivative of order `derivative_order`, window `derivative_window`, polynomial order `derivative_polyorder`; length-preserving |
| `per_sample_normalize`, `per_sample_norm_stat` | bool; `std`, `rms_dx` | false, `std` | divide each instance by one scalar of its own, the standard deviation over all entries or the RMS of its increments |
| `normalize`, `normalize_mode`, `normalize_range` | bool; `train_only`, `all_data`; `[0,1]`, `[-1,1]` | true, `train_only`, `[0,1]` | min-max scaling of the channels, fitted on the training split or on all splits pooled |
| `basepoint` | bool | false | prepend a zero row, so the first increment is the first value and the signature becomes translation-invariant |
| `clock_channel` | `null`, `main`, `separate` | `null` | add a clock as channel 0 of the path: the main clock θ itself, or the separately configured `append_clock`, scaled like a path channel |

## Order of application

1. representation (`increment` cumulative sum, or the derivative)
2. per-sample normalisation
3. the clock is built from the result, so it never sees the channel normalisation
4. channel normalisation
5. basepoint, which duplicates θ's first entry so the clock stays aligned
6. the clock channel

## Notes

- A clock channel raises the channel count by one; the lift `B` can keep it unmixed with
  `_preserve_first_channel`.
- `all_data` normalisation uses validation and test statistics for the scaling only, which is a
  disclosed choice on the one dataset that uses it.
