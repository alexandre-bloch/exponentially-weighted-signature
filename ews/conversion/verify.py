"""SSM -> depth-one EWS, verified at the level-one state against each model's own recurrence.

Every row routes the converted parameters through the real EWS layer of the experiments (`ews.layer.EWS`, via the
wrapper `ews.conversion.apply.ews_ssm_apply`, depth one) and compares the layer's level-one state h_t, at every
step and every coordinate, with the state the model's own implementation produces, realified into the conversion's
layout. Nothing here converts: the routes are the family modules' (families/*.py), called unchanged.

Protocol (the same for every row):
  inputs     N(0, 1), one length T = 128;  seeds 0, 1, 2 (model initialisation and input);  the worst seed is reported
  parameters each model's own initialisation (its constructor or, for Mamba, its initialisation rules, cited)
  precision  float64 on both sides (JAX x64; torch .double(); Mamba-1's reference casts to float32 internally and is
             run with Tensor.float mapped to Tensor.double for the call: no source change)
  metrics    max |h_ews - h_ref| over all steps and coordinates, and the same divided by max |h_ref|
  device     CPU only (JAX_PLATFORMS=cpu, CUDA_VISIBLE_DEVICES="")

Reference side, per row (read-only clones named by environment variables; nothing is vendored or edited):
  S5                   Walker's log-neural-cdes S5Layer (reference/vendored/s5.py, commit 5612f000)
  LRU                  Walker's log-neural-cdes LRULayer (reference/vendored/lru.py, commit 5612f000)
  LinOSS-IM, -IMEX     tk-rusch/linoss LinOSSLayer (reference/vendored/linoss.py, commit 05a83535)
  D-LinOSS             jaredbmit/damped-linoss a396997 LinOSSLayer (DLINOSS_REPO)
  S4, S4D              state-spaces/s4 S4Block, step mode (S4_REPO)
  Mamba-1              mamba_ssm selective_scan_ref (MAMBA_SSM_DIR: an installed mamba_ssm package, 1.2.2)
  Mamba-2              ssd_minimal_discrete, mamba_ssm/modules/ssd_minimal.py (MAMBA_REPO: a state-spaces/mamba clone)
The JAX references' states are read from their associative scan (`_tap_scan`); the torch references are stepped (S4,
S4D) or read through one-hot readouts C = e_n (Mamba-1, Mamba-2).

    python -m ews.conversion.verify                     # the table (markdown), spot checks and the Mamba control
    python -m ews.conversion.verify --latex             # Model & Route & Parameters & max abs & max rel \\ + protocol
    python -m ews.conversion.verify --trained           # adds the trained LinOSS-IM Heartbeat rows
                                                        # (LINOSS_HEARTBEAT_CKPT, LNCDE_DATA_DIR; see trained_row.py)
"""

import argparse
import contextlib
import importlib.util
import math
import os
import sys
import types

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only, for torch as well
HERE = os.path.dirname(os.path.abspath(__file__))
EWS_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, EWS_ROOT)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402

T = 128
SEEDS = (0, 1, 2)
ROWS, NOTES = [], {}


# ----------------------------------------------------------------------------------------------------------- helpers
def f64(tree):
    return jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, tree
    )


@contextlib.contextmanager
def _tap_scan():
    """Record every jax.lax.associative_scan result while a reference layer runs (their code, unchanged)."""
    rec, orig = [], jax.lax.associative_scan

    def wrapped(fn, elems, *a, **k):
        out = orig(fn, elems, *a, **k)
        rec.append(out)
        return out

    jax.lax.associative_scan = wrapped
    try:
        yield rec
    finally:
        jax.lax.associative_scan = orig


def pairs(xs):
    """(L, P) complex -> (L, 2P) real, layout 2j + (re, im): S5, LRU, S4D."""
    xs = np.asarray(xs)
    h = np.zeros((xs.shape[0], 2 * xs.shape[1]))
    h[:, 0::2], h[:, 1::2] = xs.real, xs.imag
    return h


def linoss4(xs):
    """(L, 2P) complex [z; y] -> (L, 4P) real, layout 4k + 2*(re/im) + (z/y): LinOSS-IM/IMEX, D-LinOSS."""
    xs = np.asarray(xs)
    P = xs.shape[1] // 2
    z, y = xs[:, :P], xs[:, P:]
    h = np.zeros((xs.shape[0], 4 * P))
    h[:, 0::4], h[:, 1::4], h[:, 2::4], h[:, 3::4] = z.real, y.real, z.imag, y.imag
    return h


def ews_states(A, B, u, **kw):
    """The real EWS layer at depth one on converted (A, B): its level-one prefix states h_1..h_L."""
    from ews.conversion.apply import ews_ssm_apply

    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    u = np.asarray(u, np.float64)
    H = u.shape[1]
    w = dict(A=A, B=B, C=np.zeros((H, A.shape[0])), D=np.zeros(H))
    _y, h = ews_ssm_apply(w, u, N=1, **kw)
    return np.asarray(h, np.float64)


def err(h_ews, h_ref):
    d = np.abs(np.asarray(h_ews, np.float64) - np.asarray(h_ref, np.float64))
    return float(d.max()), float(d.max() / np.abs(h_ref).max())


def add_row(model, route, params, per_seed, ref, notes=""):
    worst = max(per_seed, key=lambda e: e[1])
    ROWS.append(
        dict(
            model=model,
            route=route,
            params=params,
            max_abs=max(e[0] for e in per_seed),
            max_rel=max(e[1] for e in per_seed),
            per_seed=per_seed,
            reference=ref,
            notes=notes,
        )
    )
    print(
        f"  {model:<12} {route:<44} {params:<28} max abs {max(e[0] for e in per_seed):.3e}  "
        f"max rel {max(e[1] for e in per_seed):.3e}  (worst seed {per_seed.index(worst)})",
        flush=True,
    )


def _require(var):
    p = os.environ.get(var, "")
    if not os.path.isdir(p):
        raise ImportError(f"set {var} (see the module docstring)")
    return p


# ------------------------------------------------------------------------------------------------- JAX references
def row_s5():
    from ews.conversion.families import s5 as fam
    from ews.conversion.reference.vendored.s5 import S5Layer

    out = []
    for seed in SEEDS:
        lay = f64(
            S5Layer(
                ssm_size=16,
                blocks=1,
                H=4,
                C_init="lecun_normal",
                conj_sym=True,
                clip_eigs=True,
                discretisation="zoh",
                dt_min=1e-3,
                dt_max=1e-1,
                step_rescale=1.0,
                key=jr.PRNGKey(seed),
            )
        )
        u = np.random.default_rng(seed).normal(size=(T, 4))
        conv = fam.to_ews(lay, method="A")  # read-off, raw ZOH generator
        with _tap_scan() as rec:
            lay(jnp.asarray(u))
        out.append(err(ews_states(conv.A, conv.B, u), pairs(rec[-1][1])))
    add_row(
        "S5",
        r"read-off: $-\Lambda_k\Delta t_k$, lift $\Delta t_k B_k$, unit clock",
        "P=8 pairs, H=4 (m=16)",
        out,
        "implementation: log-neural-cdes S5Layer",
    )


def row_lru():
    from ews.conversion.families import lru as fam
    from ews.conversion.reference.vendored.lru import LRULayer

    out = []
    for seed in SEEDS:
        lay = f64(LRULayer(N=16, H=4, key=jr.PRNGKey(seed)))
        u = np.random.default_rng(seed).normal(size=(T, 4))
        conv = fam.to_ews(lay, method="B")  # principal log, unit clock
        with _tap_scan() as rec:
            lay(jnp.asarray(u))
        out.append(err(ews_states(conv.A, conv.B, u), pairs(rec[-1][1])))
    add_row(
        "LRU",
        "eq. (8), unit clock, normaliser in $\\bar B$",
        "N=16, H=4 (m=32)",
        out,
        "implementation: log-neural-cdes LRULayer",
    )


def row_linoss(disc):
    from ews.conversion.families import linoss_im, linoss_imex
    from ews.conversion.reference.vendored.linoss import LinOSSLayer

    fam = linoss_im if disc == "IM" else linoss_imex
    out = []
    for seed in SEEDS:
        lay = f64(LinOSSLayer(ssm_size=8, H=4, discretization=disc, key=jr.PRNGKey(seed)))
        u = np.random.default_rng(seed).normal(size=(T, 4))
        conv = fam.to_ews(lay, method="B")
        with _tap_scan() as rec:
            lay(jnp.asarray(u))
        out.append(err(ews_states(conv.A, conv.B, u), linoss4(rec[-1][1])))
    add_row(
        f"LinOSS-{disc}",
        "eq. (8), per stored $2\\times2$ block",
        "P=8, H=4 (m=32)",
        out,
        "implementation: tk-rusch/linoss LinOSSLayer",
    )


def row_dlinoss():
    repo = _require("DLINOSS_REPO")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from linoss.models.LinOSS import LinOSSLayer as DL

    from ews.conversion.families import dlinoss as fam

    out, clamp = [], 0.0
    for seed in SEEDS:
        lay = f64(
            DL(
                ssm_size=8,
                H=4,
                discretization="IMEX",
                damping=True,
                r_min=0.9,
                theta_max=float(np.pi),
                key=jr.PRNGKey(seed),
            )
        )
        u = np.random.default_rng(seed).normal(size=(T, 4))
        conv = fam.to_ews(lay, method="B")
        with _tap_scan() as rec:
            lay(jnp.asarray(u))
        out.append(err(ews_states(conv.A, conv.B, u), linoss4(rec[-1][1])))
        NOTES["dlinoss_condG"] = max(
            NOTES.get("dlinoss_condG", 0.0), float(conv.diag["cond_G_max"])
        )
        # their __call__ clamps A_diag into the stability band before use (LinOSS.py:313-322); the conversion reads the
        # stored A_diag. Record how far apart the two are at this initialisation.
        st = jax.nn.sigmoid(lay.steps)
        G = jax.nn.relu(lay.G_diag)
        lo = (2 + st * G - 2 * jnp.sqrt(1 + st * G)) / st**2
        hi = (2 + st * G + 2 * jnp.sqrt(1 + st * G)) / st**2
        Ac = lo + jax.nn.relu(lay.A_diag - lo) - jax.nn.relu(lay.A_diag - hi)
        clamp = max(clamp, float(jnp.max(jnp.abs(Ac - lay.A_diag))))
    NOTES["dlinoss_clamp"] = clamp
    add_row(
        "D-LinOSS",
        "eq. (8), per stored $2\\times2$ block",
        "P=8, H=4 (m=32)",
        out,
        "implementation: damped-linoss a396997 LinOSSLayer",
        notes=f"max cond G(A) over seeds {NOTES['dlinoss_condG']:.2e}: the lift inversion amplifies float64 rounding",
    )


# ----------------------------------------------------------------------------------------------- torch references
def _s4_block(**kw):
    repo = _require("S4_REPO")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from models.s4.s4 import S4Block

    return S4Block(transposed=False, **kw).double()


def row_s4d(spot=False):
    import torch

    from ews.conversion.families import s4d as fam

    H, N = 4, 64
    out, spot_ok, spot_flip = [], [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        kern = _s4_block(
            d_model=H, d_state=N, l_max=T, mode="s4d", init="diag-lin", disc="zoh"
        ).layer.kernel
        kern._setup_step()
        with torch.no_grad():
            dt, A, B, C = kern._get_params()
        dt = dt.detach().numpy()
        A = A.detach().numpy()
        B = B.detach().numpy()[0]
        C = C.detach().numpy()[0]
        dt = dt if dt.ndim == 1 else dt[:, 0]
        conv = fam.to_ews_arrays(A, dt, C, None, B=B, method="A", disc="zoh")  # read-off
        u = np.random.default_rng(seed).normal(size=(T, H))
        st = kern.default_state(1)
        xs, ys = [], []
        with torch.no_grad():
            for t in range(T):
                y, st = kern.step(torch.as_tensor(u[t : t + 1]), st)
                xs.append(st[0].numpy().reshape(-1))
                ys.append(y[0, 0].numpy())
        h_ref = pairs(np.stack(xs))  # mode j = h*N2 + n -> 2j, 2j+1
        h = ews_states(conv.A, conv.B, u)
        out.append(err(h, h_ref))
        # spot check (a): the real rotation-block readout against their complex output 2 Re(C x), directly
        y_ref = np.stack(ys)  # (T, H)
        spot_ok.append(err(h @ conv.C.T, y_ref))
        Cf = conv.C.copy()
        Cf[:, 1::2] *= -1.0  # the other sign: (Re C, +Im C)
        spot_flip.append(err(h @ Cf.T, y_ref))
    add_row(
        "S4D",
        "read-off (ZOH), realified $2\\times2$ rotation blocks",
        f"H={H}, N={N} (m={conv.m})",
        out,
        "implementation: state-spaces/s4 S4Block(mode=s4d), step",
    )
    NOTES["s4d_spot"] = (
        max(e[0] for e in spot_ok),
        max(e[1] for e in spot_ok),
        max(e[0] for e in spot_flip),
        max(e[1] for e in spot_flip),
    )


def row_s4():
    import torch

    from ews.conversion.families import s4 as fam

    H, N = 2, 8
    out = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        kern = _s4_block(d_model=H, d_state=N, l_max=T).layer.kernel
        conv = fam.from_torch_kernel(kern, method="B")  # their dense (dA, dB, dC), eq. (8)
        u = np.random.default_rng(seed).normal(size=(T, H))
        st = kern.default_state(1)
        xs = []
        with torch.no_grad():
            for t in range(T):
                _y, st = kern.step(torch.as_tensor(u[t : t + 1]), st)
                xs.append(st[0].numpy())  # (H, n) complex, conj-extended
        xs = np.stack(xs)
        n = xs.shape[2]
        h_ref = np.zeros((T, H * 2 * n))
        for h in range(H):  # block layout [Re x; Im x] per channel
            h_ref[:, h * 2 * n : h * 2 * n + n] = xs[:, h].real
            h_ref[:, h * 2 * n + n : (h + 1) * 2 * n] = xs[:, h].imag
        out.append(err(ews_states(conv.A, conv.B, u), h_ref))
    add_row(
        "S4",
        "eq. (8) on their dense $(\\bar A,\\bar B)$",
        f"H={H}, N={N} (m={conv.m})",
        out,
        "implementation: state-spaces/s4 S4Block(mode=dplr), dense step",
    )


# -------------------------------------------------------------------------------------------------------- Mamba
def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules.setdefault(name, mod)


def _selective_scan_ref():
    """Their selective_scan_ref, loaded from an installed mamba_ssm; the CUDA extension it imports at module level is
    stubbed (the reference never calls it)."""
    d = _require("MAMBA_SSM_DIR")
    _stub("selective_scan_cuda")
    spec = importlib.util.spec_from_file_location(
        "_mamba_ssi", os.path.join(d, "ops", "selective_scan_interface.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.selective_scan_ref


def _ssd_minimal():
    """ssd_minimal_discrete from a state-spaces/mamba clone; the Triton kernel it imports at module level is stubbed."""
    repo = _require("MAMBA_REPO")
    for name in ("mamba_ssm", "mamba_ssm.ops", "mamba_ssm.ops.triton"):
        _stub(name)
    _stub("mamba_ssm.ops.triton.ssd_combined", mamba_chunk_scan_combined=None)
    path = os.path.join(repo, "mamba_ssm", "modules", "ssd_minimal.py")
    spec = importlib.util.spec_from_file_location("_ssd_minimal", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ssd_minimal_discrete


@contextlib.contextmanager
def _float_is_double():
    import torch

    orig = torch.Tensor.float
    torch.Tensor.float = lambda self, *a, **k: self.double()
    try:
        yield
    finally:
        torch.Tensor.float = orig


def _dt_bias(rng, n, dt_min=1e-3, dt_max=1e-1, floor=1e-4):
    """Their dt initialisation (mamba_simple.py:91-99 / mamba2_simple.py:94-101): softplus(bias) log-uniform."""
    dt = np.exp(rng.uniform(size=n) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
    dt = np.maximum(dt, floor)
    return dt + np.log(-np.expm1(-dt))


def _phi1_inv(z):
    return np.where(np.abs(z) < 1e-12, 1.0, z / (1.0 - np.exp(-z)))


def row_mamba1():
    """Per channel: clock theta_t = sum Delta_s, identity lift, increment phi1(-Delta_t A)^-1 Delta_t B_t u_t; the
    arguments of ClockedConversion.forward_ews (families/mamba.py:62-74), rebuilt here so the state can be read; the
    readout of that state is checked to equal forward_ews() exactly."""
    import torch
    import torch.nn.functional as F

    from ews.conversion.families.mamba import mamba1_to_ews

    ssr = _selective_scan_ref()
    Dch, N, dt_rank = 4, 16, 1
    out, neg, mirror = [], [], 0.0
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        A = -np.tile(
            np.arange(1, N + 1, dtype=np.float64), (Dch, 1)
        )  # mamba_simple.py:104-110, 143
        bias = _dt_bias(rng, Dch)
        W = rng.uniform(
            -(dt_rank**-0.5), dt_rank**-0.5, size=(Dch, dt_rank)
        )  # dt_proj init, :83-86
        delta = W @ rng.normal(size=(dt_rank, T))  # (D, L), input-dependent
        Bt, Ct, u = rng.normal(size=(N, T)), rng.normal(size=(N, T)), rng.normal(size=(Dch, T))
        tt = lambda x: torch.as_tensor(x, dtype=torch.float64)  # noqa: E731
        states = np.zeros((Dch, T, N))
        with _float_is_double(), torch.no_grad():
            for n in range(N):  # one-hot readout -> n-th coordinate
                e = np.zeros((N, T))
                e[n] = 1.0
                y = ssr(
                    tt(u)[None],
                    tt(delta)[None],
                    tt(A),
                    tt(Bt)[None],
                    tt(e)[None],
                    D=None,
                    delta_bias=tt(bias),
                    delta_softplus=True,
                )
                states[:, :, n] = y[0].numpy()
            Delta = F.softplus(
                tt(delta) + tt(bias)[:, None]
            ).numpy()  # the same Delta their ref uses
        worst = worst_neg = (0.0, 0.0)
        for d in range(Dch):
            cv = mamba1_to_ews(A[d], Delta[d], Bt.T, Ct.T, u[d])
            z = cv.dtheta[:, None] * cv.A[None, :]
            theta = np.concatenate([[0.0], np.cumsum(cv.dtheta)])
            h = ews_states(
                np.diag(cv.A),
                np.eye(N),
                cv.dX * _phi1_inv(z),
                clock=theta,
                reweighting_mode="displacement",
            )
            hn = ews_states(
                np.diag(cv.A), np.eye(N), cv.dX, clock=theta, reweighting_mode="displacement"
            )
            worst = max(worst, err(h, states[d]), key=lambda e: e[1])
            worst_neg = max(worst_neg, err(hn, states[d]), key=lambda e: e[1])
            mirror = max(
                mirror, float(np.max(np.abs(np.einsum("tm,tm->t", cv.C, h) - cv.forward_ews())))
            )
        out.append(worst)
        neg.append(worst_neg)
    NOTES["mamba1_neg"] = (max(e[0] for e in neg), max(e[1] for e in neg))
    NOTES["mamba1_mirror"] = mirror
    add_row(
        "Mamba-1",
        "per channel: clock $\\sum\\Delta$, identity lift, $\\Delta G_\\Delta(A)^{-1}\\Delta B x$",
        f"{Dch} channels, N={N}",
        out,
        "implementation: mamba_ssm selective_scan_ref",
    )


def row_mamba2():
    """Per head: the same clock and increment with the head's scalar a, Delta_t B_t x_t^T (N x P), one EWS of width
    N*P per head (layout p*N + n); each channel's readout is checked against mamba2_to_ews(...).forward_ews()."""
    import torch
    import torch.nn.functional as F

    from ews.conversion.families.mamba import mamba2_to_ews

    ssd = _ssd_minimal()
    Hh, P, N, blk = 2, 4, 16, 16
    out, mirror = [], 0.0
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        a = -rng.uniform(1.0, 16.0, size=Hh)  # mamba2_simple.py:107-110, 132
        bias = _dt_bias(rng, Hh)
        delta = rng.normal(size=(T, Hh))
        x, Bt, Ct = rng.normal(size=(T, Hh, P)), rng.normal(size=(T, N)), rng.normal(size=(T, N))
        tt = lambda v: torch.as_tensor(v, dtype=torch.float64)  # noqa: E731
        with torch.no_grad():
            Delta = F.softplus(tt(delta) + tt(bias)[None]).numpy()  # (T, H)
            X = tt(x * Delta[..., None])[None]  # their discrete form: X dt, A dt
            Adt = tt(a[None] * Delta)[None]
            Bh = tt(np.repeat(Bt[:, None, :], Hh, axis=1))[None]
            states = np.zeros((Hh, T, P, N))
            for n in range(N):
                e = np.zeros((T, Hh, N))
                e[:, :, n] = 1.0
                Y, _ = ssd(X, Adt, Bh, tt(e)[None], blk)
                states[:, :, :, n] = Y[0].numpy().transpose(1, 0, 2)  # (H, T, P)
        worst = (0.0, 0.0)
        for hh in range(Hh):
            dth = Delta[:, hh]
            m = P * N
            Aew = np.full(m, -a[hh])
            dX = (dth[:, None, None] * x[:, hh, :, None] * Bt[:, None, :]).reshape(T, m)
            theta = np.concatenate([[0.0], np.cumsum(dth)])
            h = ews_states(
                np.diag(Aew),
                np.eye(m),
                dX * _phi1_inv(dth[:, None] * Aew[None, :]),
                clock=theta,
                reweighting_mode="displacement",
            )
            worst = max(worst, err(h, states[hh].reshape(T, m)), key=lambda e: e[1])
            for p in range(P):
                cv = mamba2_to_ews(a[hh], dth, Bt, Ct, x[:, hh, p])
                own = np.einsum("tn,tn->t", Ct, h[:, p * N : (p + 1) * N])
                mirror = max(mirror, float(np.max(np.abs(own - cv.forward_ews()))))
        out.append(worst)
    NOTES["mamba2_mirror"] = mirror
    add_row(
        "Mamba-2",
        "per head: clock $\\sum\\Delta$, identity lift, $\\Delta G_\\Delta(A)^{-1}\\Delta B x^\\top$",
        f"{Hh} heads, P={P}, N={N}",
        out,
        "implementation: mamba ssd_minimal_discrete",
    )


# ------------------------------------------------------------------------------------------------- spot check (b)
def spot_s5():
    """S5 read-off: with the lift row Delta t_k B_k the EWS state is exact; with B_k alone it is not."""
    from ews.conversion.families import s5 as fam
    from ews.conversion.reference.vendored.s5 import S5Layer

    good, bad = [], []
    for seed in SEEDS:
        lay = f64(
            S5Layer(
                ssm_size=16,
                blocks=1,
                H=4,
                C_init="lecun_normal",
                conj_sym=True,
                clip_eigs=True,
                discretisation="zoh",
                dt_min=1e-3,
                dt_max=1e-1,
                step_rescale=1.0,
                key=jr.PRNGKey(seed),
            )
        )
        u = np.random.default_rng(seed).normal(size=(T, 4))
        conv = fam.to_ews(lay, method="A")
        Delta = np.repeat(np.asarray(conv.diag["Delta"]), 2)  # per real coordinate
        with _tap_scan() as rec:
            lay(jnp.asarray(u))
        h_ref = pairs(rec[-1][1])
        good.append(err(ews_states(conv.A, conv.B, u), h_ref))
        bad.append(
            err(ews_states(conv.A, conv.B / Delta[:, None], u), h_ref)
        )  # lift row B_k, no Delta t_k
    NOTES["s5_spot"] = (
        max(e[0] for e in good),
        max(e[1] for e in good),
        max(e[0] for e in bad),
        max(e[1] for e in bad),
    )


# --------------------------------------------------------------------------------------------------- trained rows
def trained_rows():
    from ews.conversion import trained_row as TR

    if TR.SKIP_REASON:
        NOTES["trained"] = f"SKIPPED: {TR.SKIP_REASON}"
        return []
    rows = []
    for s in (2345, 3456, 4567, 5678, 6789):
        datasetkey = jr.split(jr.PRNGKey(s), 4)[0]
        ds = TR.create_dataset(
            TR.DATA_DIR,
            "Heartbeat",
            stepsize=1,
            depth=1,
            include_time=True,
            T=1,
            use_idxs=False,
            use_presplit=False,
            key=datasetkey,
        )
        test = ds.raw_dataloaders["test"]
        model, st, best = TR.load(s)
        conv = TR.swap_blocks(model, TR.from_linoss_block, N=1)  # every block's LinOSS layer
        before, after = TR.accuracy(model, st, test), TR.accuracy(conv, st, test)
        lb, la = TR.logits(model, st, test), TR.logits(conv, st, test)
        rows.append(
            dict(
                seed=s,
                before=before,
                after=after,
                logit_abs=float(np.max(np.abs(la - lb))),
                logit_rel=float(np.max(np.abs(la - lb)) / np.max(np.abs(lb))),
                n_layers=len(model.blocks),
            )
        )
        print(
            f"  LinOSS-IM Heartbeat seed {s}: acc {before:.2f} -> {after:.2f}, max abs logit residual "
            f"{rows[-1]['logit_abs']:.3e} ({rows[-1]['n_layers']}/{rows[-1]['n_layers']} layers converted)",
            flush=True,
        )
    return rows


# ------------------------------------------------------------------------------------------------------- output
def _num(x):
    if x != x:
        return "---"
    m, e = f"{x:.1e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def latex(trained):
    out = [
        r"\begin{tabular}{lllrr}",
        r"\toprule",
        r"Model & Route & Parameters & max abs & max rel \\",
        r"\midrule",
    ]
    for r in ROWS:
        out.append(
            f"{r['model']} & {r['route']} & {r['params']} & {_num(r['max_abs'])} & {_num(r['max_rel'])} \\\\"
        )
    if trained:
        out.append(r"\midrule")
        for t in trained:
            out.append(
                f"LinOSS-IM (Heartbeat, seed {t['seed']}) & eq. (8), all {t['n_layers']} layers & "
                f"test acc. {t['before']:.2f}\\% $\\to$ {t['after']:.2f}\\% & {_num(t['logit_abs'])} & "
                f"{_num(t['logit_rel'])} \\\\"
            )
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def protocol(trained):
    impl = ", ".join(r["model"] for r in ROWS if r["reference"].startswith("implementation"))
    s = (
        f"Protocol. Inputs $N(0,1)$ of length $T={T}$; parameters from each model's own initialisation; three seeds "
        f"(0, 1, 2) for initialisation and input, worst seed reported; float64 on both sides; the error is the maximum "
        f"absolute difference between the EWS layer's level-one state and the model's own state over all steps and "
        f"coordinates, and the same divided by the largest state entry. Widths: "
        + "; ".join(f"{r['model']} {r['params']}" for r in ROWS)
        + f". Every row runs the converted parameters through the EWS layer used in the experiments at depth one. "
        f"Rows checked against the model's reference implementation: {impl}."
    )
    if trained:
        s += (
            f" Trained rows: the five LinOSS-IM Heartbeat checkpoints with every one of the "
            f"{trained[0]['n_layers']} SSM layers replaced by its converted EWS layer, the rest of the model "
            f"untouched; residual on the test logits."
        )
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    ap.add_argument(
        "--trained", action="store_true", help="add the trained LinOSS-IM Heartbeat rows"
    )
    ap.add_argument(
        "--only",
        default=None,
        help="comma list of rows: s5,lru,im,imex,dlinoss,s4d,s4,mamba1,mamba2",
    )
    a = ap.parse_args()
    plan = dict(
        s5=row_s5,
        lru=row_lru,
        im=lambda: row_linoss("IM"),
        imex=lambda: row_linoss("IMEX"),
        dlinoss=row_dlinoss,
        s4d=row_s4d,
        s4=row_s4,
        mamba1=row_mamba1,
        mamba2=row_mamba2,
    )
    want = a.only.split(",") if a.only else list(plan)
    print(f"State-level verification, T={T}, seeds {SEEDS}, float64, CPU", flush=True)
    for k in want:
        try:
            plan[k]()
        except Exception as e:  # noqa: BLE001
            ROWS.append(
                dict(
                    model=k,
                    route="NOT RUN",
                    params="",
                    max_abs=float("nan"),
                    max_rel=float("nan"),
                    per_seed=[],
                    reference="",
                    notes=f"{type(e).__name__}: {str(e)[:200]}",
                )
            )
            print(f"  {k}: NOT RUN -- {type(e).__name__}: {str(e)[:200]}", flush=True)
    if not a.only or "s5" in want:
        spot_s5()
    trained = trained_rows() if a.trained else []
    print("\nSpot checks and controls:")
    if "s4d_spot" in NOTES:
        g_a, g_r, b_a, b_r = NOTES["s4d_spot"]
        print(
            f"  (a) S4D output 2Re(Cx), readout (Re C, -Im C): max abs {g_a:.3e} rel {g_r:.3e}; "
            f"with (Re C, +Im C): max abs {b_a:.3e} rel {b_r:.3e}"
        )
    if "s5_spot" in NOTES:
        g_a, g_r, b_a, b_r = NOTES["s5_spot"]
        print(
            f"  (b) S5 lift row Delta t_k B_k: max abs {g_a:.3e} rel {g_r:.3e}; B_k alone: max abs {b_a:.3e} rel {b_r:.3e}"
        )
    if "mamba1_neg" in NOTES:
        print(
            f"  negative control, Mamba-1 without phi1: max abs {NOTES['mamba1_neg'][0]:.3e} rel {NOTES['mamba1_neg'][1]:.3e}"
        )
        print(
            f"  Mamba-1 readout of the rebuilt state vs forward_ews(): {NOTES['mamba1_mirror']:.3e}"
        )
    if "mamba2_mirror" in NOTES:
        print(
            f"  Mamba-2 per-channel readout vs mamba2_to_ews(...).forward_ews(): {NOTES['mamba2_mirror']:.3e}"
        )
    if "dlinoss_condG" in NOTES:
        print(f"  D-LinOSS: max cond G(A) over the three seeds {NOTES['dlinoss_condG']:.3e}")
    if "dlinoss_clamp" in NOTES:
        print(
            f"  D-LinOSS: max |clamped A - stored A| at init {NOTES['dlinoss_clamp']:.3e} (the conversion reads the stored A)"
        )
    if "trained" in NOTES:
        print(f"  trained rows: {NOTES['trained']}")
    if a.latex:
        print("\n" + latex(trained) + "\n\n" + protocol(trained))


if __name__ == "__main__":
    main()
