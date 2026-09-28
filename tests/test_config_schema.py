"""Config schema: the upgrade of flat layouts is idempotent and the banked configs validate."""

import glob
import json
import os

import pytest

from ews.config import upgrade, upgrade_block, validate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANKED = sorted(
    glob.glob(f"{ROOT}/uea_results/*/config.json")
    + glob.glob(f"{ROOT}/formal_language_results/*/config.json")
)


@pytest.mark.parametrize("path", BANKED, ids=lambda p: os.path.basename(os.path.dirname(p)))
def test_banked_configs_validate_and_are_fixed_points(path):
    cfg = json.load(open(path))
    validate(cfg)
    assert upgrade(cfg) == cfg


def test_flat_block_upgrades():
    flat = {
        "m": 8,
        "n": 1,
        "path_mode": "increment",
        "basepoint": True,
        "leaf": "series",
        "series_q": 12,
        "M": 4,
        "van_loan_expm": "exact",
        "chen_expm": "exact",
        "combine": "assoc.build",
        "clock": {"mode": "time"},
    }
    up = upgrade_block(flat)
    assert up["path"] == {"representation": "increment", "basepoint": True}
    assert up["compute"] == {
        "leaf": "series",
        "series_q": 12,
        "van_loan_expm": "exact",
        "chen_expm": "exact",
        "M": 4,
    }
    assert up["depth"] == 1 and "combine" not in up and "n" not in up
    assert upgrade_block(up) == up
    with pytest.raises(ValueError):
        upgrade_block({"combine": "seq.direct.remat"})


def test_flat_preprocessing_upgrades():
    cfg = {
        "preprocessing": {
            "dtype": "float32",
            "clock_mode": "blended_variation",
            "clock_alpha": 0.2,
            "add_basepoint": True,
            "normalize_paths": True,
            "normalize_range": "[-1,1]",
            "path_derivative": 2,
        },
        "model": {
            "A": {"parametrization": "skew_symmetric"},
            "ews_depth": 2,
            "M": 5,
            "van_loan_mode": "displacement_uniform",
        },
    }
    up = upgrade(cfg)
    assert (
        up["preprocessing"]["path"]["representation"] == "derivative"
        and up["preprocessing"]["path"]["basepoint"]
    )
    assert up["preprocessing"]["clock"]["mode"] == "blended_variation"
    assert (
        up["model"]["depth"] == 2
        and up["model"]["compute"]["reweighting_mode"] == "displacement_uniform"
    )
    validate(up)
    assert upgrade(up) == up


def test_validate_rejects_unknown():
    with pytest.raises(ValueError):
        validate({"model": {"block": {"clock": {"mode": "time"}, "bogus": 1}}})
    with pytest.raises(ValueError):
        validate({"preprocessing": {"clock_mode": "time"}})
