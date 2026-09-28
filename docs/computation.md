# Computing the signature (`ews/core/`, `ews/layer.py`, config section `compute`)

The signature is computed in two stages. The **leaf** produces, for every interval between
consecutive knots of the path, the local reweighted signature of that interval. The **combine**
composes the intervals into prefix signatures at every position or into the terminal signature,
using Chen's identity with the discount e^{-hA} carried along.

## Keys

| key | values | default | meaning |
|---|---|---|---|
| `leaf` | `van_loan`, `series`, `plane_closed`, `block_series`, `jordan_closed`, `jordan_closed_tables` | `van_loan` | how the local signatures are computed |
| `series_q` | int | null | order of the series leaves; fixed for a run because it sizes an array |
| `combine` | `assoc`, `lean`, `block_kernel`, `dense_kernel` | `assoc` | how the intervals are composed (see "The combine") |
| `combine_chunk` | int | 1024 | chunk length of the fused kernels (largest divisor of L below it); no effect on the result |
| `combine_tile` | int | 4 | block tile of the block kernel; no effect on the result |
| `van_loan_expm` | `exact`, `cayley`, `pade`, `euler` | `cayley` (layer), `exact` (block) | the matrix exponential inside the Van Loan leaf |
| `chen_expm` | `exact`, `cayley`, `pade`, `euler` | `exact` | the matrix exponential of the discount e^{-hA} in the combine |
| `M` | int | 20 (layer), 4 (block) | sub-steps per interval in the Van Loan leaf |
| `terminal_reduce` | `tree`, `assoc` | `tree` | the terminal signature by a padded tree reduction, or as the last element of the prefix scan |
| `reweighting_mode` | `displacement`, `displacement_uniform`, `displacement_uniform_basepoint` | `displacement` | the general kernel, one exponential per interval; or the fast path that reuses one shared exponential when every interval has the same Δθ, with or without a Δθ = 0 basepoint interval |

## The Van Loan leaf

Each interval's reweighted increment is the top-right block of the exponential of the augmented
matrix [[A·δ, ΔX / M], [0, 0]], evaluated at M sub-steps, which produces M + 1 reweighted knots;
their signature is then taken with signax. `M` trades accuracy for cost. The uniform reweighting
modes are only valid when the clock is uniform; the builders route a non-uniform clock to
`displacement` automatically.

## The series leaf

A closed form for the same local signature as a power series in the interval's regime
x = ‖A‖·Δθ, assembled directly from per-interval letters without a signax step. `series_q` is
its order. `ews.core.leaf.suggest_series_q` sizes q from x and a tolerance, and the trainer warns
at build time when a block's measured x is too large for its q. The series is faster and more
accurate than Van Loan at the same cost when x ≲ 0.5; its error bound grows with x, so the order
must be sized for the largest ‖A‖ seen during training, not the value at init.

## The structured leaves (`plane_closed`, `block_series`)

When A is block-diagonal the local signature factorises over the blocks, so no m × m exponential is formed.
`plane_closed` (the rotation-plane parametrisations `bounded_rotation`, `damped_rotation`, `relu_damped_rotation`,
`clamped_rotation`, `unconstrained_rotation` and `unbounded_rotation`) evaluates the depth-1 and depth-2 local terms of
each 2 × 2 plane [[d, w], [−w, d]] in closed form (φ(x) = (eˣ − 1)/x and its divided differences), exact in Δθ. It
reads only the (d, w) pair of each block, so it does not apply to `stable_block`, whose shear it would drop, or to
`jordan_repeated`, whose blocks are chains. `block_series` (`block_diagonal`, `stable_block`, `jordan_repeated` and the
rotation classes, any block sizes) runs the series leaf block by block and needs `series_q` at depth 2. Both cover
depths 1 and 2, ignore the Van Loan options, and live in `ews/core/kernels/`.

## The Jordan-chain leaf (`jordan_closed`)

For `jordan_repeated` with uniform chains (every block of the same kind and size: real chains of 2-4, complex chains of
size 4, 6 or 8), depths 1 and 2. A chain is λI + Nil with Nil nilpotent, so the rotation-plane closed forms carry over
with their first few derivatives, exact in Δθ and without a series order. Real chains of 2 and complex chains of 4 use
forward-mode derivatives in the layer's precision; longer chains use float64 tables (`jordan_closed_tables` uses the
tables for every chain, as a cross-check). For real chains of 2 and complex chains of 4 it is faster and more accurate
than `series`; for longer chains `series` is faster at depth 2, and above x = ‖A‖ Δθ ≈ 20 the closed form is the only
accurate leaf.

## Exponential methods

`exact` uses the true matrix exponential; `cayley`, `pade` and `euler` are the low-order
approximations kept for the UEA results, whose banked configs use `euler`. The formal-language
configs use `exact` in both roles.

## The combine

Intervals are composed with Chen's identity on packed signatures, carrying the discount matrix
e^{-hA} in the scan state, by `jax.lax.associative_scan`, which is O(log L) in sequential depth.
The terminal signature is reduced by a padded tree or read off the last prefix. That is
`combine: assoc`, the default. Three alternatives live in `ews/core/kernels/` and are
tested against it in `tests/test_engines.py`:

| `combine` | applies to | what it is | when to use it |
|---|---|---|---|
| `lean` | any A, any depth, f32 / f64 | the same associative scan (prefix) or pairwise tree (terminal) over the same packed Chen operator, but on precomputed per-interval discounts M_t = e^{−Δθ_t A}: no exponentials inside the scan | dense (non-block-diagonal) A: the fastest option there, with the memory of the scan tree |
| `block_kernel` | block-diagonal A (rotation planes, `unconstrained_rotation`, `stable_block`, `jordan_repeated`, `block_diagonal` of any sizes), depths 1–2, tested for blocks of size 2 and 4 | one fused Pallas/Triton kernel per chunk in block coordinates (the flow of level k costs K·m^k elementwise operations instead of an m^{k+1} matmul), with a custom VJP that stores only the prefix stream; under a uniform `reweighting_mode` the per-block flows e^{−Δθ A_j} are computed once and reused for every interval (the identity for the Δθ = 0 basepoint interval) | block-diagonal A on a GPU: 1.4× (m 32) to 9× (m 128) faster training steps than `lean` with 3–10× less memory |
| `dense_kernel` | any A, f32, m ≥ 16, prefix readouts | a buffered Pallas/Triton kernel with a custom VJP | the smallest memory footprint for dense A; slower than `lean` |

The kernels pad block counts and block sizes to powers of two with zero blocks (exactly: a zero
block has identity flow and zero local terms) and fail fast when a configuration is outside what
they support. The two stages are chosen independently: any leaf feeds any combine. `assoc` is never
faster than `lean` or `block_kernel`.

For a `jordan_repeated` A with uniform chains, `EWS(..., jordan_flows=True)` (also an `EWSBlock` argument) makes the
block kernel write each interval's flow e^{−Δθ J} in closed form, a finite sum over the chain's nilpotent part, instead
of computing a matrix exponential per interval and block. Under a gated clock every interval has its own Δθ, so on
long paths those exponentials are most of the cost; the PPG model uses it.

## Depth and output

`depth` (N) is the truncation level: the feature has 1 + Σ_{k=1..N} m^k coordinates including the
level-0 constant, laid out level by level. `ews/core/output_format.py` converts between the
list-of-tensors and flat layouts. The log-signature variant (`LogEWS`) takes the tensor log at the
readout and compresses to the Lyndon basis; the reweighting and the scan are unchanged.
