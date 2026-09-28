"""What every ablation run shares: the champion configuration per dataset, the edit that turns it into one cell
(depth, width and the drawn knobs), the norm-constant calibration of a gated clock, and the call to the trainer.

`uea_paired.py` (the 33-configuration population) and `ethanol_champion_row.py` (the Ethanol champion row) both go
through `run_one`, so every run is `experiments/uea/train.py` on the per-seed split `data/<stem>/splits/`, with
`--autotune off --deterministic-ops on`.
"""

import copy
import json
import math
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
EWS = os.path.dirname(
    os.path.dirname(HERE)
)  # the repository root (this file is uea_results/ablation/)
OUT = os.path.join(
    EWS, "outputs", "uea_ablation"
)  # run configs and logs, cells, claims (gitignored)
PY = sys.executable
STEM = {
    "EigenWorms": "eigenworms_walker",
    "EthanolConcentration": "ethanolconcentration_walker",
    "Heartbeat": "heartbeat_walker",
    "MotorImagery": "motorimagery_walker",
    "SelfRegulationSCP1": "selfregulationscp1_walker",
    "SelfRegulationSCP2": "selfregulationscp2_walker",
}
GATED = ("clamped_gate", "token_gated")
FAST_LEAVES = ("plane_closed", "block_series")  # leaves that exist for depths 1-2 only
DEPTH3_SERIES_Q = 14  # the series leaf's q at depth 3 (x = 0.5 with headroom)


def champion(ds):
    """The champion configuration the dataset's population was built on: `<Dataset>/champion_config.json` next to
    this file (see the README for where each one comes from). UEA_CHAMPION_CONFIG runs the ablation against another
    configuration instead; nothing else changes, the cells simply describe that model."""
    path = os.environ.get("UEA_CHAMPION_CONFIG") or os.path.join(HERE, ds, "champion_config.json")
    with open(path) as f:
        return json.load(f)


def clock_mode_of(c):
    """The clock mode, whichever schema the config uses: flat `preprocessing.clock_mode` or sectioned
    `preprocessing.clock.mode`. Reading only the flat key returns "" for a sectioned config, which silently drops
    the clock-alpha axis from a draw (found building the EigenWorms draw: its champion is `blended_learn`)."""
    pp = c["preprocessing"]
    return str(pp.get("clock_mode") or pp.get("clock", {}).get("mode") or "")


def blended(c):
    return clock_mode_of(c).startswith("blended")


def set_depth(m, depth):
    """Set the signature depth under the key the trainer reads. ews.config.upgrade_layer_model keeps `model.depth`
    and drops `model.ews_depth` when both are present, so writing ews_depth into a config that already has depth
    changes nothing (that is how ten Ethanol runs meant for depth 3 trained at depth 2; see the README). Flat
    configs keep ews_depth; sectioned ones get depth."""
    if "depth" in m:
        m["depth"] = int(depth)
    if "ews_depth" in m or "depth" not in m:
        m["ews_depth"] = int(depth)


def leaf_of(m):
    """The leaf the trainer will use: `model.compute.leaf` wins over the flat `model.leaf` (upgrade_layer_model
    moves a flat engine key into compute only when compute does not already set it)."""
    return m.get("compute", {}).get("leaf", m.get("leaf"))


def use_series_leaf(m, q=DEPTH3_SERIES_Q):
    """Switch to the series leaf, writing each key where the config spells it (nested compute, else flat)."""
    comp = m.get("compute", {})
    tgt = comp if "leaf" in comp else m
    tgt["leaf"] = "series"
    (comp if "series_q" in comp else tgt)["series_q"] = int(q)


def make_config(ds, depth, p, steps, eval_every=None):
    """The champion with one cell's settings: depth, width d, and whichever of the drawn knobs `p` carries."""
    c = copy.deepcopy(champion(ds))
    m, t = c["model"], c["training"]
    set_depth(m, depth)
    m["B"]["out_dim"] = int(p["d"])
    if m["A"].get("blocks"):  # block-structured A: regrow the block list for the new width
        bs = int(m["A"]["blocks"][0])
        kind = m["A"]["kinds"][0] if m["A"].get("kinds") else None
        nb = max(1, int(p["d"]) // bs)
        m["A"]["blocks"] = [bs] * nb
        if kind is not None:
            m["A"]["kinds"] = [kind] * nb
    if "damp" in p:
        # stable_block reads damp_init_* / shear_init_* / sigma_range at init and never init.variance, so a grid over
        # the init variance would repeat one model (two configs differing only in it build a bit-identical A). Its
        # live A-scale knob is the damping init mean.
        m["A"]["damp_init_mean"] = float(p["damp"])
    else:
        m["A"]["init"]["variance"] = float(p["a_var"])
    m["head_init_scale"] = float(p["head"])
    if blended(c) and "alpha" in p:
        if "clock" in c["preprocessing"] and "clock_mode" not in c["preprocessing"]:
            c["preprocessing"]["clock"]["alpha"] = float(p["alpha"])  # sectioned schema
        else:
            c["preprocessing"]["clock_alpha"] = float(p["alpha"])  # flat schema
    for k in ("lr_features", "lr_head", "min_lr_features", "min_lr_head"):
        t[k] = float(p["lr"])
    if "readout" in p:
        m["readout"] = p["readout"]
    if "lambda2" in p:
        t["lambda2"] = float(p["lambda2"])
    if "label_smoothing" in p:
        t["label_smoothing"] = float(p["label_smoothing"])
    # The plane-closed and block-series leaves exist for depths 1-2 only, so a depth-3 cell of a champion that uses
    # one runs the series leaf on the associative combine.
    if int(depth) > 2 and leaf_of(m) in FAST_LEAVES:
        use_series_leaf(m)
        m.setdefault("compute", {})["combine"] = "assoc"
    # The block-kernel combine is gated for depths 1-2 only (its per-program state does not fit at depth 3), so
    # depth-3 cells fall back to the dense `lean` combine, which computes the same thing more slowly.
    if int(depth) > 2 and m.get("compute", {}).get("combine") == "block_kernel":
        m["compute"]["combine"] = "lean"
    # A gated clock's norm constant is re-calibrated for the cell (mean increment 1 at init on the B-projected driver).
    if c["preprocessing"].get("clock", {}).get("mode") in GATED:
        c["preprocessing"]["clock"]["norm_constant"] = float(calibrate_norm_constant(ds, c))
    # UEA_ABL_BATCH: run the whole ablation at a batch other than the champion's (EigenWorms' depth 3 fits only at a
    # small batch). It applies to every depth: a pair that differs in batch as well as depth confounds the two. No
    # banked population sets it.
    if os.environ.get("UEA_ABL_BATCH"):
        t["batch_size"] = int(os.environ["UEA_ABL_BATCH"])
    t.update(num_steps=int(steps), log_every=max(int(steps), 1), early_stopping_patience=10**9)
    if eval_every:  # smoke runs only; cells keep the champion's eval cadence
        t["eval_every"] = int(eval_every)
    return c


_NORM_CACHE = {}


def calibrate_norm_constant(ds, cfg, n_paths=8):
    """The constant that puts a gated clock's mean increment at 1 at init: build the model with the constant at 1 and
    read the clock on the first `n_paths` training paths of split seed 0. Cached per (dataset, path section, clock
    mode, width, depth); it sets only the clock's overall scale."""
    pp, m = cfg["preprocessing"], cfg["model"]
    key = (
        ds,
        json.dumps(pp["path"], sort_keys=True),
        pp["clock"]["mode"],
        m["B"]["out_dim"],
        m.get("ews_depth", m.get("depth")),
    )
    if key in _NORM_CACHE:
        return _NORM_CACHE[key]
    if EWS not in sys.path:
        sys.path.insert(0, EWS)
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    import experiments.uea.train as T

    probe = copy.deepcopy(cfg)
    probe["preprocessing"]["clock"]["norm_constant"] = 1.0
    fd, cp = tempfile.mkstemp(prefix="uea_ablation_probe_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(probe, f)
        c_obj = T.load_config(cp)
    finally:
        os.remove(cp)
    split = os.path.join(EWS, "data", STEM[ds], "splits", f"{STEM[ds]}_seed0.npz")
    data = T.load_path_classification_dataset(split, c_obj.preprocessing)
    paths = jnp.stack(data["train"]["paths"][:n_paths])
    model = T.build_model(
        c_obj, data["metadata"]["d_path"], data["metadata"]["n_classes"], jr.PRNGKey(0)
    )
    # the model applies its clock to the path after the B projection, so the driver has width m, not d_path
    driver = jax.vmap(model.b_layer)(paths) if model.b_layer is not None else paths
    t = jnp.arange(driver.shape[1], dtype=driver.dtype)
    theta = jax.vmap(lambda pth: model.clock(t, pth, None))(driver)
    inc = float(jnp.mean(jnp.diff(theta, axis=1)))
    val = inc if inc > 1e-9 else 1.0
    _NORM_CACHE[key] = val
    return val


_EVAL_LOSS = re.compile(r"^\s+(?:val|test)\s+loss=(\S+)", re.M)


def diverged(log_text):
    """Why a run produced no result, or "" if it produced one.

    A run whose evaluation loss is nan or inf has not scored: the trainer keeps printing an accuracy, and that
    accuracy is the majority-class rate. Divergence must raise, never return a number; every run goes through
    run_one, so it is checked there.
    """
    vals = _EVAL_LOSS.findall(log_text)
    if not vals:
        return ""

    def bad(v):
        try:
            return not math.isfinite(float(v))
        except ValueError:
            return True

    n_bad = sum(bad(v) for v in vals)
    if bad(vals[-1]):
        return f"the final evaluation loss was {vals[-1]} ({n_bad} of {len(vals)} evaluations non-finite)"
    if n_bad >= max(1, len(vals) // 2):
        return f"{n_bad} of {len(vals)} evaluation losses were non-finite"
    return ""


def run_one(ds, depth, p, seed, steps, tag, gpu, cpu=False, eval_every=None, config=None):
    """One train.py run on split `seed`; returns (best val acc, test acc, best step) or raises.

    The run's configuration is make_config(ds, depth, p, steps, eval_every), or `config` when given (the champion
    row builds its own). It is written to <OUT>/runs/<ds>_n<depth>/<tag>_s<seed>.json next to the run's log."""
    rd = os.path.join(OUT, "runs", f"{ds}_n{depth}")
    os.makedirs(rd, exist_ok=True)
    cfg = os.path.join(rd, f"{tag}_s{seed}.json")
    with open(cfg, "w") as f:
        json.dump(
            config if config is not None else make_config(ds, depth, p, steps, eval_every),
            f,
            indent=1,
        )
    split = os.path.join(EWS, "data", STEM[ds], "splits", f"{STEM[ds]}_seed{seed}.npz")
    tmp = tempfile.gettempdir()
    ckpt = os.path.join(tmp, "uea_ablation_ckpt", f"{ds}_n{depth}_{tag}_s{seed}.eqx")
    log = os.path.join(rd, f"{tag}_s{seed}.log")
    env = dict(os.environ, PYTHONPATH=EWS, PYTHONUNBUFFERED="1", TMPDIR=tmp)
    if cpu:
        env.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
    cmd = [
        PY,
        os.path.join(EWS, "experiments", "uea", "train.py"),
        "--config",
        cfg,
        "--gpu",
        str(gpu),
        "--seed",
        str(seed),
        "--dataset_file",
        split,
        "--out_ckpt",
        ckpt,
        "--autotune",
        "off",
        "--deterministic-ops",
        "on",
    ]

    # The trainer must die with its worker, or a killed worker leaves train.py holding GPU memory for a result nobody
    # reads. PR_SET_PDEATHSIG makes the kernel send the child SIGTERM the moment its parent exits.
    def _die_with_parent():
        try:
            import ctypes
            import signal as _sig

            ctypes.CDLL("libc.so.6").prctl(1, _sig.SIGTERM)  # 1 = PR_SET_PDEATHSIG
        except Exception:
            pass

    with open(log, "w") as f:
        rc = subprocess.run(
            cmd, cwd=EWS, env=env, stdout=f, stderr=subprocess.STDOUT, preexec_fn=_die_with_parent
        ).returncode
    try:
        os.remove(ckpt)
    except OSError:
        pass
    with open(log) as f:
        s = f.read()
    why = diverged(s)
    if why:
        raise RuntimeError(f"diverged: {why}")
    mv = re.search(r"Best\s+step=(\d+)\s+val_accuracy=([0-9.]+)", s)
    mt = re.search(r"Test metrics: \{'accuracy': ([0-9.]+)", s)
    if not mv or not mt:
        # a non-zero rc with both metrics printed is the interpreter's teardown abort (JAX on CPU + OpenBLAS), not a
        # failed run, so the result is kept and only a missing metric counts as a failure
        tail = s.strip().splitlines()[-1][:200] if s.strip() else ""
        raise RuntimeError(f"run failed rc={rc}: {tail}")
    return 100 * float(mv.group(2)), 100 * float(mt.group(1)), int(mv.group(1))
