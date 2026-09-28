"""The two XLA switches that decide whether a GPU run is reproducible.

    autotuning         XLA picks kernels by timing them; faster, but the choice can differ between
                       runs, so results are not bit-reproducible.
    deterministic ops  forces deterministic reductions and scatters; bit-reproducible, slower.

Both switches are exposed the same way in every trainer: on the command line as
``--autotune {on,off}`` and ``--deterministic-ops {on,off}``, and in the config as
``training.xla_autotune`` and ``training.deterministic_ops``. Precedence per switch: the command
line, then the config, then a value already present in ``XLA_FLAGS`` (set by a launcher), then
the trainer's default. Each trainer's default is the regime its banked results were produced in.
The flags must be set before JAX is imported, which is why trainers call this module first.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

AUTOTUNE_FLAG = "--xla_gpu_autotune_level"
DETERMINISTIC_FLAG = "--xla_gpu_deterministic_ops"

NOTE = (
    "Deterministic ops make a run bit-reproducible and slower; XLA autotuning makes it faster and "
    "not reproducible. Switches: --autotune {on,off}, --deterministic-ops {on,off} "
    "(config: training.xla_autotune, training.deterministic_ops)."
)


def configure_xla(
    autotune: Optional[bool] = None,
    deterministic_ops: Optional[bool] = None,
    *,
    default_autotune: bool,
    default_deterministic_ops: bool,
) -> str:
    """Write the two switches into ``XLA_FLAGS`` and return it.

    An explicit argument replaces whatever the environment holds; ``None`` keeps a flag already
    present in the environment and otherwise applies the default. Other flags are preserved."""
    tokens = [t for t in os.environ.get("XLA_FLAGS", "").split() if t]

    def present(name: str) -> bool:
        return any(t.startswith(name + "=") for t in tokens)

    def set_flag(name: str, value: str) -> None:
        nonlocal tokens
        tokens = [t for t in tokens if not t.startswith(name + "=")]
        tokens.append(f"{name}={value}")

    if autotune is not None:
        set_flag(AUTOTUNE_FLAG, "4" if autotune else "0")
    elif not present(AUTOTUNE_FLAG) and not default_autotune:
        set_flag(AUTOTUNE_FLAG, "0")
    if deterministic_ops is not None:
        set_flag(DETERMINISTIC_FLAG, "true" if deterministic_ops else "false")
    elif not present(DETERMINISTIC_FLAG) and default_deterministic_ops:
        set_flag(DETERMINISTIC_FLAG, "true")
    os.environ["XLA_FLAGS"] = " ".join(tokens)
    return os.environ["XLA_FLAGS"]


def add_switches(parser: argparse.ArgumentParser) -> None:
    """Add the two switches to an argument parser (values default to "not given")."""
    parser.add_argument(
        "--autotune", choices=["on", "off"], default=None, help="XLA kernel autotuning"
    )
    parser.add_argument(
        "--deterministic-ops",
        choices=["on", "off"],
        default=None,
        help="deterministic reductions/scatters (bit-reproducible, slower)",
    )


def _from_config(path: Optional[str]):
    if not path or not os.path.exists(path):
        return None, None
    try:
        tc = json.load(open(path)).get("training", {})
    except Exception:
        return None, None
    a = tc.get("xla_autotune")
    d = tc.get("deterministic_ops")
    return (None if a is None else bool(a)), (None if d is None else bool(d))


def configure_from_cli(*, default_autotune: bool, default_deterministic_ops: bool) -> str:
    """Read the switches from the command line and the config file, then configure XLA.

    Meant to run at the top of a trainer, before any JAX import. Unknown arguments are ignored."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    add_switches(pre)
    args, _ = pre.parse_known_args()
    cfg_autotune, cfg_det = _from_config(args.config)
    autotune = (args.autotune == "on") if args.autotune is not None else cfg_autotune
    det = (args.deterministic_ops == "on") if args.deterministic_ops is not None else cfg_det
    return configure_xla(
        autotune,
        det,
        default_autotune=default_autotune,
        default_deterministic_ops=default_deterministic_ops,
    )
