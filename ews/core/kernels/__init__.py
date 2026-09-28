"""Optional EWS engines: closed-form and block-diagonal leaves and
fused / lean combines, selected per layer with ``EWS(leaf=..., combine=...)``.

    leaf     "van_loan" (default) | "series" | "plane_closed" | "block_series" | "jordan_closed"   (ews.core.leaf)
    combine  "assoc"    (default) | "lean"   | "block_kernel" | "dense_kernel"   (ews.core.kernels.combine)

The defaults are the plain engine. The two choices are independent: any leaf feeds
any combine (the block kernel needs a block-diagonal A; the closed-form leaves need rotation planes
/ uniform Jordan chains). Which leaf suits which Jordan chain: ``ews.core.leaf.local_signatures``.

Modules:
    plane_engine      closed-form local terms for 2x2 rotation planes (depths 1-2)
    blockdiag_engine  blockwise series local terms for block-diagonal A (depths 1-2)
    jordan_engine     closed-form local terms for uniform Jordan chains, real and complex (depths 1-2)
    pallas_combine    the Pallas/Triton kernels (block / dense, prefix / terminal, custom VJPs) and the
                      lean JAX scans; ``sequential_ref`` is the gradient reference
    combine           the ``combine=`` dispatcher (padding, operators, packing)
"""
