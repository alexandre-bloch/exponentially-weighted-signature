# Parametrisations of A and B (`ews/params.py`, config sections `A` and `B`)

`A` is the m × m drift that discounts increments; `B` is the optional linear lift from the input
channels to the m channels the signature sees. Both are built from their config section by
`build_ab_from_config`.

## `A`

| key | values | default | meaning |
|---|---|---|---|
| `parametrization` | see table | `general` | the structural family |
| `init.type` | `sig`, or a parametrisation name | `sig` | `sig` is the zero matrix for the linear families (the classical signature) and the family's own random init for the rotation families; a family name draws that family's parameters and expresses them in the chosen parametrisation, which lets `general` start from a structured draw |
| `init.mean`, `init.variance` | floats | 0.0, 1.0 | the Gaussian init of the raw parameters |
| `init.sigma_range` | float | 0.0 | rotation families: frequencies start uniform in ±logit(0.5 + sigma_range), i.e. sigmoid(u) in 0.5 ± sigma_range |
| `omega_scale` | float | 1.0 | rotation families: multiplier of the frequency range, so the realised per-token angle can exceed one turn |
| `damp_init_mean`, `damp_init_std` | floats | −2.0, 1.0 | damped rotations: init of the damping logits |
| `learn` | bool | true | train A |
| `blocks`, `kinds` | list of int; list of `real` / `complex` | null | `block_diagonal` (blocks), `unconstrained_rotation` and `jordan_repeated` (both) |
| `epsilon` | float | 0.0 | added to the diagonal of the definite families |

| parametrisation | A is |
|---|---|
| `sig` | the zero matrix: the classical signature |
| `efm` | a positive diagonal, per-channel exponential decay; `positivity` selects `softplus` (default) or `square` of the parameters |
| `general` | unconstrained, dense |
| `skew_symmetric` | U − Uᵀ: pure rotations, oscillatory memory |
| `symmetric` | symmetric |
| `positive_definite` | L Lᵀ + εI |
| `definite_plus_skew` | a positive-definite part plus a skew part |
| `triangular_upper`, `triangular_lower` | triangular |
| `block_diagonal` | blocks of the given sizes, each unconstrained; sizes 1 give a free diagonal |
| `unconstrained_rotation` | block-diagonal 2 × 2 rotation planes `[[a, −b], [b, a]]` with both the scale and the frequency free, so about half the planes expand: fine on a short path (Heartbeat), not on a long one. Blocks are 2 × 2 |
| `jordan_repeated` | Jordan blocks: one eigenvalue per block, repeated down its diagonal, with ones (real) or 2 × 2 identities (complex) above, and the real part kept ≥ 0 by softplus; one (real) or two (complex) parameters per block. exp(−hA) is a decay times a polynomial in h, so one coordinate accumulates another's output. A real block needs size ≥ 2 and a complex one size ≥ 4; different eigenvalues come from different blocks |
| `bounded_rotation` | axis-aligned 2 × 2 planes with per-plane frequency omega_scale · sigmoid(u); the block rescales the matrix by 2π / Δθ_max so the largest per-token angle is one turn |
| `damped_rotation` | the same planes with a per-plane contraction rate softplus(v): exp(−δA_j) is a decay times a rotation |
| `relu_damped_rotation` | damping relu(v) instead of softplus, so an exactly undamped plane is reachable on an open set |
| `clamped_rotation` | damping softplus(v) with the frequency clipped to one period instead of squashed |
| `unbounded_rotation` | the bounded rotation with the frequency cap removed, softplus(u) |
| `stable_block` | the free 2 × 2 block restricted to the non-expanding region: a·I + b·J + s·S(t) with a = softplus(v), b = omega_scale · sigmoid(u), s = a · sigmoid(z) < a and S(t) the unit traceless symmetric matrix, so sym(A) has eigenvalues a ± s ≥ 0; `damped_rotation` is its zero-shear case. An unconstrained 2 × 2 block (`block_diagonal`) expands with probability ½ and overflows over a long path |

The rotation families are the ones the formal-language models use: their per-plane structure is
what makes a token's rotation angle an interpretable quantity. For these families the enclosing
block computes the 2π / Δθ_max rescale from the embedding table over the vocabulary ids listed in
`model.input_token_ids`, pad and label-only ids excluded.

## Init scale of the other families in a block: `angle_target`

A block key, not an `A` key. When set, the block's A is rescaled once at init so that the median per-token rotation
angle (Δθ times the median imaginary eigenvalue) equals the target in radians. It acts on the parameter tensors, so for
a logit-parametrised family it does not scale A linearly; it is used on the linear families. Under `model.block` it
applies to every block.

## `B`

| key | values | default | meaning |
|---|---|---|---|
| `enabled` | bool | false | apply the lift |
| `out_dim` | int | | the number of channels the signature sees; A is out_dim × out_dim |
| `parametrization` | `general` | `general` | |
| `init.type` | `general`, `identity_block`, `sig` | `general` | Gaussian, identity-like pass-through, or zeros |
| `init.mean`, `init.variance` | floats | 0.0, 1.0 | |
| `learn` | bool | true | |
| `_preserve_first_channel` | bool | true | channel 0 is passed through unmixed, for an appended clock channel |
