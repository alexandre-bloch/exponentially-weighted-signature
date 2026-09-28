"""Data loader for path-to-path regression: an NPZ of paths, clocks and per-step targets, and its preprocessing.

NPZ keys: {train,val,test}_paths (n, T, d), {train,val,test}_clocks (n, T) and {train,val,test}_targets (n, T).

Preprocessing keys: augment_with_clock (default False), clock_normalize ("[0,1]" or "none"), normalize_paths (default
True), normalize_mode ("train_only" or "all_data"), normalize_range ("[0,1]" or "[-1,1]"), add_basepoint (default
True), normalize_targets (default False), target_range (default normalize_range) and dtype (default "float32").

Returns {"train", "val", "test"}, each with "paths", "clocks" and "targets" (lists of arrays), and "metadata" (the
path width, the path and target scalers and the split sizes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import jax.numpy as jnp
import numpy as np

from ews.paths import prepare_paths_np, specs_from_config


@dataclass
class TargetScaler:
    data_min: float
    data_max: float


def _apply_target_scaler(
    targets: List[np.ndarray],
    scaler: TargetScaler,
    target_range: str,
) -> List[np.ndarray]:
    denom = (scaler.data_max - scaler.data_min) + 1e-12
    out = []
    for t in targets:
        t = np.asarray(t).astype(np.float32)
        if target_range == "[0,1]":
            out.append((t - scaler.data_min) / denom)
        elif target_range == "[-1,1]":
            out.append(2.0 * (t - scaler.data_min) / denom - 1.0)
        else:
            raise ValueError(f"target_range must be '[0,1]' or '[-1,1]', got {target_range}")
    return out


def load_path_regression_dataset(
    npz_file: str,
    preprocessing_config: Dict,
) -> Dict:
    cfg = preprocessing_config
    dtype_str = cfg.get("dtype", "float32")
    dtype = jnp.float64 if dtype_str == "float64" else jnp.float32

    raw = np.load(npz_file)

    def _to_list(arr: np.ndarray) -> List[np.ndarray]:
        if arr.dtype == object:
            return [arr[i] for i in range(len(arr))]
        return [arr[i] for i in range(arr.shape[0])]

    # Convert stacked arrays (n, T, d) → lists of (T, d)
    splits: Dict[str, Dict] = {}
    for split in ("train", "val", "test"):
        paths_arr = raw[f"{split}_paths"]  # (n, T, d)
        clocks_arr = raw[f"{split}_clocks"]  # (n, T)
        targets_arr = raw[f"{split}_targets"]  # (n, T)
        splits[split] = {
            "paths": _to_list(paths_arr),
            "clocks": _to_list(clocks_arr),
            "targets": _to_list(targets_arr),
        }

    path_spec, clock_spec, append_spec = specs_from_config(cfg)
    clock_normalize = clock_spec.scaling
    augment_with_clock = path_spec.clock_channel == "main"
    normalize_mode = path_spec.normalize_mode
    normalize_range = path_spec.normalize_range

    def _preprocess(paths, clocks, scaler, fit):
        return prepare_paths_np(
            paths,
            clocks,
            path_spec,
            clock_spec,
            append_spec,
            scaler=scaler,
            fit_scaler=fit,
            dtype=dtype,
        )

    train_paths_raw = splits["train"]["paths"]
    train_clocks_raw = splits["train"]["clocks"]
    val_paths_raw = splits["val"]["paths"]
    val_clocks_raw = splits["val"]["clocks"]
    test_paths_raw = splits["test"]["paths"]
    test_clocks_raw = splits["test"]["clocks"]

    if normalize_mode == "all_data":
        all_paths = train_paths_raw + val_paths_raw + test_paths_raw
        all_clocks = train_clocks_raw + val_clocks_raw + test_clocks_raw
        _, _, scaler, d_path = _preprocess(all_paths, all_clocks, None, True)
        train_paths_jax, train_clocks_jax, _, _ = _preprocess(
            train_paths_raw, train_clocks_raw, scaler, False
        )
        val_paths_jax, val_clocks_jax, _, _ = _preprocess(
            val_paths_raw, val_clocks_raw, scaler, False
        )
        test_paths_jax, test_clocks_jax, _, _ = _preprocess(
            test_paths_raw, test_clocks_raw, scaler, False
        )
    else:
        train_paths_jax, train_clocks_jax, scaler, d_path = _preprocess(
            train_paths_raw, train_clocks_raw, None, True
        )
        val_paths_jax, val_clocks_jax, _, _ = _preprocess(
            val_paths_raw, val_clocks_raw, scaler, False
        )
        test_paths_jax, test_clocks_jax, _, _ = _preprocess(
            test_paths_raw, test_clocks_raw, scaler, False
        )

    # Target normalization
    normalize_targets = bool(cfg.get("normalize_targets", False))
    target_range = cfg.get("target_range", normalize_range)
    target_scaler: Optional[TargetScaler] = None

    train_targets = splits["train"]["targets"]
    val_targets = splits["val"]["targets"]
    test_targets = splits["test"]["targets"]

    if normalize_targets:
        all_train = np.concatenate(
            [np.asarray(t).reshape(-1).astype(np.float32) for t in train_targets]
        )
        target_scaler = TargetScaler(
            data_min=float(all_train.min()),
            data_max=float(all_train.max()),
        )
        train_targets = _apply_target_scaler(train_targets, target_scaler, target_range)
        val_targets = _apply_target_scaler(val_targets, target_scaler, target_range)
        test_targets = _apply_target_scaler(test_targets, target_scaler, target_range)

    train_targets_jax = [jnp.array(t, dtype=dtype) for t in train_targets]
    val_targets_jax = [jnp.array(t, dtype=dtype) for t in val_targets]
    test_targets_jax = [jnp.array(t, dtype=dtype) for t in test_targets]

    return {
        "train": {
            "paths": train_paths_jax,
            "clocks": train_clocks_jax,
            "targets": train_targets_jax,
        },
        "val": {"paths": val_paths_jax, "clocks": val_clocks_jax, "targets": val_targets_jax},
        "test": {"paths": test_paths_jax, "clocks": test_clocks_jax, "targets": test_targets_jax},
        "metadata": {
            "d_path": d_path,
            "path_scaler": scaler,
            "target_scaler": target_scaler,
            "n_train": len(train_paths_jax),
            "n_val": len(val_paths_jax),
            "n_test": len(test_paths_jax),
            "clock_normalize": clock_normalize,
            "augment_with_clock": augment_with_clock,
        },
    }
