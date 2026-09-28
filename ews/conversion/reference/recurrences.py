"""Minimal reference recurrences, transcribed from each family's published equations.

The tests compare against these, so the suite runs on a fresh clone with no third-party checkouts. Each was checked
against the vendored implementation in `vendored/`.
"""

import jax
import jax.numpy as jnp


def reference_linoss_im(layer, u, dtype=jnp.float64):
    """Transcription of the LinOSS-IM recurrence, state (z, y) complex.

    Returns (y_out (L, H), h (L, 4P)) with h realified in the EWS layout
    4k + 2*(re/im) + (z/y). float64 by default (validated == vendored layer);
    float32 serves as the f32 proxy of the source, since the vendored scan
    cannot run f32 under the global x64 flag.
    """
    cdtype = jnp.complex128 if dtype == jnp.float64 else jnp.complex64
    Acal = jax.nn.relu(jnp.asarray(layer.A_diag, dtype))
    Delta = jax.nn.sigmoid(jnp.asarray(layer.steps, dtype))
    S = 1.0 / (1.0 + Delta**2 * Acal)
    B_c = jnp.asarray(layer.B[..., 0] + 1j * layer.B[..., 1], cdtype)
    C_c = jnp.asarray(layer.C[..., 0] + 1j * layer.C[..., 1], cdtype)
    D = jnp.asarray(layer.D, dtype)
    u = jnp.asarray(u, dtype)
    P = Acal.shape[0]

    M11, M12, M21, M22 = S, -Delta * Acal * S, Delta * S, S
    Bu = u @ B_c.T  # (L, P) complex

    def step(carry, bu):
        z, y = carry
        z2 = M11 * z + M12 * y + Delta * S * bu  # F1 = Delta*S*Bu
        y2 = M21 * z + M22 * y + Delta**2 * S * bu  # F2 = Delta^2*S*Bu
        return (z2, y2), (z2, y2)

    init = (jnp.zeros(P, cdtype), jnp.zeros(P, cdtype))
    _, (zs, ys) = jax.lax.scan(step, init, Bu)

    h = jnp.zeros((u.shape[0], 4 * P), dtype)
    h = h.at[:, 0::4].set(zs.real).at[:, 1::4].set(ys.real)
    h = h.at[:, 2::4].set(zs.imag).at[:, 3::4].set(ys.imag)
    y_out = (ys @ C_c.T).real + u * D
    return y_out, h


def reference_s5(layer, u, dtype=jnp.float64):
    """Transcription of the vendored apply_ssm recurrence, state x complex.

    Returns (y_out (L, H), h (L, 2P)) with h realified in the EWS layout
    2j + (re/im).
    """
    cdtype = jnp.complex128 if dtype == jnp.float64 else jnp.complex64
    Lam_re = jnp.asarray(layer.Lambda_re, dtype)
    if layer.clip_eigs:
        Lam_re = jnp.clip(Lam_re, None, -1e-4)
    lam = (Lam_re + 1j * jnp.asarray(layer.Lambda_im, dtype)).astype(cdtype)
    B_t = jnp.asarray(layer.B[..., 0] + 1j * layer.B[..., 1], cdtype)
    C_t = jnp.asarray(layer.C[..., 0] + 1j * layer.C[..., 1], cdtype)
    D = jnp.asarray(layer.D, dtype)
    Delta = layer.step_rescale * jnp.exp(jnp.asarray(layer.log_step[:, 0], dtype))
    u = jnp.asarray(u, dtype)
    P = lam.shape[0]

    lam_bar = jnp.exp(lam * Delta)
    B_bar = ((lam_bar - 1.0) / lam)[:, None] * B_t
    Bu = u @ B_bar.T  # (L, P) complex

    def step(x, bu):
        x2 = lam_bar * x + bu
        return x2, x2

    _, xs = jax.lax.scan(step, jnp.zeros(P, cdtype), Bu)

    h = jnp.zeros((u.shape[0], 2 * P), dtype)
    h = h.at[:, 0::2].set(xs.real).at[:, 1::2].set(xs.imag)
    c = 2.0 if layer.conj_sym else 1.0
    y_out = c * (xs @ C_t.T).real + u * D
    return y_out, h


def reference_lru(layer, u, dtype=jnp.float64):
    """Transcription of the vendored LRU recurrence, state x complex.

    Returns (y_out (L, H), h (L, 2N)) with h realified in the EWS layout
    2j + (re/im).
    """
    cdtype = jnp.complex128 if dtype == jnp.float64 else jnp.complex64
    nu = jnp.exp(jnp.asarray(layer.nu_log, dtype))
    theta = jnp.exp(jnp.asarray(layer.theta_log, dtype))
    lam = jnp.exp(-nu + 1j * theta).astype(cdtype)
    B_norm = (
        jnp.asarray(layer.B_re + 1j * layer.B_im, cdtype)
        * jnp.exp(jnp.asarray(layer.gamma_log, dtype))[:, None]
    )
    C_c = jnp.asarray(layer.C_re + 1j * layer.C_im, cdtype)
    D = jnp.asarray(layer.D, dtype)
    u = jnp.asarray(u, dtype)
    N = lam.shape[0]

    Bu = u @ B_norm.T  # (L, N) complex

    def step(x, bu):
        x2 = lam * x + bu
        return x2, x2

    _, xs = jax.lax.scan(step, jnp.zeros(N, cdtype), Bu)

    h = jnp.zeros((u.shape[0], 2 * N), dtype)
    h = h.at[:, 0::2].set(xs.real).at[:, 1::2].set(xs.imag)
    y_out = (xs @ C_c.T).real + u * D
    return y_out, h
