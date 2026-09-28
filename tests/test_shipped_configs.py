"""Every shipped config validates, and both templates build a model."""

import glob
import json
import os
import sys

import pytest

from ews.config import upgrade, validate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIPPED = sorted(
    glob.glob(f"{ROOT}/uea_results/*/config.json")
    + glob.glob(f"{ROOT}/formal_language_results/*/config.json")
    + glob.glob(f"{ROOT}/sde_results/*/*/config.json")
    + glob.glob(f"{ROOT}/experiments/*/configs/template.json")
    + glob.glob(f"{ROOT}/experiments/*/*/config_*.json")
)


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: os.path.relpath(p, ROOT))
def test_shipped_config_validates(path):
    cfg = json.load(open(path))
    validate(upgrade(cfg))


def _trainer(module):
    sys.argv = [sys.argv[0], "--gpu", ""]
    os.environ["JAX_PLATFORMS"] = "cpu"
    import importlib

    return importlib.import_module(module)


def test_uea_template_builds():
    import jax.random as jr

    UT = _trainer("experiments.uea.train")
    cfg = UT.load_config(f"{ROOT}/experiments/uea/configs/template.json")
    model = UT.build_model(cfg, d_in=6, n_classes=2, key=jr.PRNGKey(0))
    assert model.ewms.d == int(cfg.model["B"]["out_dim"])  # the lift maps d_in channels to out_dim


def test_fl_template_builds():
    import jax.random as jr
    import numpy as np

    FT = _trainer("experiments.fl.train")
    FD = _trainer("experiments.fl.dataloading")
    cfg = FT.load_config(f"{ROOT}/experiments/fl/configs/template.json")
    td = FD.make_task_loaders(
        cfg.dataset["task"],
        8,
        1234,
        length=cfg.dataset.get("length"),
        train_samples=None,
        val_samples=32,
        val_batch_size=cfg.training.get("val_batch_size"),
    )
    first = next(td.train_iter)
    X0, m0 = np.asarray(first[0]), np.asarray(first[2])
    model, _ = FT.calibrate_and_build(
        cfg,
        td.data_dim,
        td.label_dim,
        jr.PRNGKey(0),
        X0,
        FD.validity_mask(m0, td.loss_mask_is_dense),
        None,
        td.val_pad,
    )
    assert len(model.blocks) == int(cfg.model["num_blocks"])
