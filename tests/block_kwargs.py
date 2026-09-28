"""Test helper: build the block's `clock` / `append_clock` sections from flat clock_* keywords."""

from dataclasses import replace

from ews.clocks import ClockSpec

_CLOCK = (
    "mode",
    "alpha",
    "scaling",
    "normalize",
    "norm_constant",
    "time_denominator",
    "source",
    "stop_gradient",
    "variation_eps",
    "gate_init_scale",
    "gate_init_bias",
    "gate_K",
)


def specs(kw):
    kw = dict(kw)
    clock = {k: kw.pop("clock_" + k) for k in _CLOCK if "clock_" + k in kw}
    spec = ClockSpec.from_config(clock)
    mode = kw.pop("append_clock_mode", None)
    alpha = kw.pop("append_clock_alpha", 1.0)
    const = kw.pop("append_clock_norm_constant", None)
    kw["clock"] = spec
    kw["append_clock"] = (
        None if mode is None else replace(spec, mode=mode, alpha=float(alpha), norm_constant=const)
    )
    return kw
