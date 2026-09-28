# Clocks (`ews/clocks.py`, config sections `clock` and `append_clock`)

The clock θ is the monotone reparametrisation that drives the reweighting e^{-(θ_t − θ_u)A}. A clock
maps a time base t and a driver, the signal whose variation it measures, to a non-decreasing array
of the same length. Every mode implements one interface, `Clock`: `theta = clock(t, driver,
mask)`, plus `per_token_increment(embeds)` for the token cache and `is_static`.

## Modes

| mode | θ | parameters |
|---|---|---|
| `time` | the scaled time base | none |
| `blended_variation` | α·t + (1 − α)·V, V the cumulative L2 norm of the driver's increments | none |
| `blended_quad_variation` | α·t + (1 − α)·Q, Q the cumulative squared increments | none |
| `blended_learn` | α·t + (1 − α)·V with α trained | α (one scalar per block) |
| `token_gated` | cumsum(relu(w·Δx − b)) / c | w (driver width), b |
| `clamped_gate` | cumsum(Σ_k clip(w_k·Δx − b_k, 0, 1)) / c | w (K × width), b (K) |

`blended_learn` is `blended_variation` with α learned, one scalar per block: `alpha` is its initialisation (1.0
starts as the time clock, 0.5 midway), and `alpha_parametrisation` holds it in [0, 1] by clipping (the default) or
through a sigmoid. At any fixed α it equals `blended_variation`. It has no token cache, since a token's clock increment
is then not a constant.

The gated clocks make an exactly-zero increment reachable on an open set of parameters: a token with
w·Δx ≤ b contributes nothing to the clock, so "stay" tokens can stop the reweighting entirely. The
clamped gate lands increments on integer multiples of 1/c, which cyclic-group tasks need.

## Keys

| key | values | default | meaning |
|---|---|---|---|
| `mode` | see above | `time` | |
| `alpha` | float in [0, 1] | 1.0 | blend weight of the time component; α = 0 is a pure variation clock |
| `scaling` | `none`, `[0,1]`, `[-1,1]`, `steps` | `[0,1]` | each component is scaled to this range before blending; the gated modes require `none`; `steps` (mode `time` only) is the sample counter θ_i = i, not normalised by length, the basepoint step included |
| `normalize` | `per_sample`, `constant` | `per_sample` | divide the variation by its own total, which is not causal, or by `norm_constant`, which is; the gated modes require `constant` |
| `norm_constant` | float | null | the constant c |
| `time_denominator` | float | null | scale of the time component when it is not scaled per sample |
| `variation_eps` | float | 1e-8 | inside the variation square root, a gradient guard at zero increments |
| `stop_gradient` | bool | false | detach θ from the loss |
| `gate_init_scale` | float | null (width^−½) | scale of the gate weights at init |
| `gate_init_bias` | float or list | −2.0 | the threshold(s); a list gives the clamped gate staggered thresholds |
| `gate_K` | int | 4 | number of clamped terms when `gate_init_bias` is a scalar |
| `source` | `block_input`, `encoder_output`, `raw_input` | `block_input` | stacked models only: which signal the block hands to its clock |

`append_clock` takes the same keys and describes a second clock added to the path as a channel
(see `path.clock_channel`). In a stacked model it inherits the main clock's settings except
`mode`, `alpha` and `norm_constant`.

## Static or in the forward pass

A clock is precomputed once when it has no parameters and its driver is raw data, which is every
banked UEA config. Inside a block the driver is learnable, so every clock is evaluated per forward
pass. The token cache evaluates the per-token increments once per step instead; it supports the
gated modes and the pure variation modes with `constant` normalisation, and rejects the others.

## Numpy and jnp builders

`build_clock_np` is the float64 numpy builder used for precomputation, `build_clock_jnp` its
in-model twin. They agree to machine precision for the parameter-free modes; the jnp builder adds
mask gating with frozen pads, constant normalisation, the epsilon guard and the stop-gradient.
