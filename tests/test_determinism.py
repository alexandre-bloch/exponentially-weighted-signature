"""XLA determinism switches: precedence rules, and every trainer sets them before importing JAX."""

import os
import subprocess
import sys

import pytest

from ews.determinism import AUTOTUNE_FLAG, DETERMINISTIC_FLAG, configure_xla

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINERS = {
    "experiments/fl/train.py": (
        False,
        False,
    ),  # banked FL regime: autotune off, deterministic ops off
    "experiments/uea/train.py": (
        True,
        False,
    ),  # UEA trainer default: autotune on, deterministic ops off
    "experiments/sde/train.py": (True, False),
}


def _flags(s):
    return {t.split("=")[0]: t.split("=")[1] for t in s.split()}


def test_defaults_and_explicit_values(monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", "")
    f = _flags(configure_xla(default_autotune=True, default_deterministic_ops=False))
    assert AUTOTUNE_FLAG not in f and DETERMINISTIC_FLAG not in f  # published regime adds nothing
    f = _flags(configure_xla(default_autotune=False, default_deterministic_ops=False))
    assert f[AUTOTUNE_FLAG] == "0" and DETERMINISTIC_FLAG not in f
    monkeypatch.setenv("XLA_FLAGS", "")
    f = _flags(configure_xla(True, True, default_autotune=False, default_deterministic_ops=False))
    assert (
        f[AUTOTUNE_FLAG] == "4" and f[DETERMINISTIC_FLAG] == "true"
    )  # explicit values win over defaults


def test_environment_precedence(monkeypatch):
    monkeypatch.setenv(
        "XLA_FLAGS", "--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true --xla_other=1"
    )
    f = _flags(configure_xla(default_autotune=True, default_deterministic_ops=False))
    assert (
        f[AUTOTUNE_FLAG] == "0" and f[DETERMINISTIC_FLAG] == "true" and f["--xla_other"] == "1"
    )  # env kept
    f = _flags(
        configure_xla(
            deterministic_ops=False, default_autotune=True, default_deterministic_ops=False
        )
    )
    assert (
        f[DETERMINISTIC_FLAG] == "false" and f[AUTOTUNE_FLAG] == "0"
    )  # explicit beats env, other flag untouched


def _import_flags(trainer, extra):
    """Import a trainer in a subprocess with the given CLI arguments and report XLA_FLAGS after the import."""
    mod = trainer[:-3].replace("/", ".")
    code = (
        f"import sys, os; sys.argv = ['x'] + {extra!r}; os.environ['JAX_PLATFORMS'] = 'cpu'; "
        f"import importlib; importlib.import_module({mod!r}); print('FLAGS=' + os.environ.get('XLA_FLAGS', ''))"
    )
    env = dict(os.environ, PYTHONPATH=ROOT, JAX_PLATFORMS="cpu")
    env.pop("XLA_FLAGS", None)
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600
    )
    lines = [x for x in out.stdout.splitlines() if x.startswith("FLAGS=")]
    assert lines, out.stderr[-2000:]
    return _flags(lines[-1][len("FLAGS=") :])


@pytest.mark.parametrize("trainer", sorted(TRAINERS))
def test_trainer_defaults(trainer):
    autotune, det = TRAINERS[trainer]
    f = _import_flags(trainer, ["--gpu", ""])
    assert (f.get(AUTOTUNE_FLAG, "4") == "4") == autotune
    assert (f.get(DETERMINISTIC_FLAG, "false") == "true") == det


def test_cli_switches_reach_the_environment():
    f = _import_flags(
        "experiments/uea/train.py", ["--gpu", "", "--autotune", "off", "--deterministic-ops", "on"]
    )
    assert f[AUTOTUNE_FLAG] == "0" and f[DETERMINISTIC_FLAG] == "true"
