# The EWS block and the stack (`ews/block.py`)

`EWSBlock` is a residual layer whose mixer is the EWS. With `norm_placement` `pre`, one block does:

1. θ = clock(t, driver, mask), where the driver is the block's pre-norm input by default
2. x_t = RMSNorm(x)
3. u = [θ_app | x_t] if an `append_clock` is configured, else x_t
4. p = B(u): the lift to m channels, forward-filled after the true end of a masked sequence
5. S = prefix EWS of p under θ with the increment representation and the basepoint as configured; level 0 dropped, one row per position
6. s = GradedRMSNorm(S)
7. r = C(s), a linear map back to the hidden width, optionally gated by a sigmoid of the branch input
8. x = x + Dropout(r), then the channel stage on RMSNorm(x) with a second residual

`slice_post` places the norm after the residual sum, `none` uses no norm. The channel stage is one
of `mlp`, `tanh`, `glu`, `gelu_glu`.

## Block keys

`m` (signature width), `depth` (signature depth), `norm_placement`, `second_norm`, `graded_norm`,
`graded_eps`, `graded_prescale`, `channel_stage`, `ff_mult`, `output_gate`, `dropout_rate`,
`norm_eps`, `angle_target`, and the sections `A`, `B`, `clock`, `append_clock`, `path`
(`representation`, `basepoint`), `compute`.

## GradedRMSNorm

Signature levels sit orders of magnitude apart, and the top level owns almost all coordinates, so
a flat norm would be level N's statistic. `GradedRMSNorm` treats each level block independently,
per sample and position:

| variant | block k becomes |
|---|---|
| `live_rms` | g_k · block / (RMS_k + eps_k), RMS over the level's coordinates |
| `static_gain` | g_k · block |
| `dilation` | λ^k · block, the dilation automorphism of the tensor algebra |

`graded_eps` is the per-level floor; `model.graded_eps_init` `calibrated` sets it from a training
batch at init. `graded_prescale` `binomial` divides level k at position t by C(t, k), turning
subsequence counts into densities for one-hot inputs.

## The stack

`StackedEWS`: an embedding encoder for tokens or a linear encoder for series, `num_blocks` blocks
built from `model.block` with `model.blocks[i]` overrides, an optional final norm, and a head that
is per-position (`seq2seq`) or applied to the masked mean of the positions (`mean_pool`).
Clock sources: `block_input` is the block's own input, `encoder_output` the embedding stream,
`raw_input` an external driver such as one-hot tokens; at block 1 the first two coincide.

For the rotation families the stack computes each block's `a_scale` = 2π / Δθ_max from the
embedding table, Δθ_max being the largest per-token clock increment over the input vocabulary
(`model.input_token_ids`; the pad id is excluded). Under a gated clock that reads the embeddings,
the increment is the gate's own.

## The token cache

With `model.token_cache` true, the first block is replaced by `TokenCachedEWSBlock`. When the path
is the increment stream of token embeddings and the clock's increment depends only on the current
token, every position's local segment is one of |V| cached ones, so the (L, m, m) associative scan
becomes a gather and a sequential Chen scan with a single running signature. It reproduces the
default path exactly and is what makes wide blocks affordable. Preconditions, enforced at
construction: token inputs, `path.representation` `increment` on block 1, the block-input clock
source, no `append_clock` on block 1, and a clock whose per-token increment is defined (the gated
modes, or the pure variation modes with constant normalisation).

## Notes

- Block defaults under `model.block` apply to every block; a key meant for one block, such as
  `angle_target`, belongs in that block's entry of `model.blocks`.
- `angle_target` rescales the block's A once at init to a median per-token angle; it is meant for
  the linear families (see params.md).
- Old checkpoints load into the current block layout: the clock module sits where the gate
  parameters were.
