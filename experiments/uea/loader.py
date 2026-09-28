"""Data loader for path-to-label classification: an NPZ of paths, clocks and labels, and its preprocessing.

NPZ keys: {train,val,test}_paths (n, T, d), {train,val,test}_clocks (n, T) and {train,val,test}_labels, either (n,)
integers (multiclass) or (n, C) binary (multilabel).

Preprocessing keys: augment_with_clock (prepend the clock as channel 0, default False), clock_normalize ("[0,1]" or
"none"), normalize_paths (default True), normalize_mode ("train_only" or "all_data"), normalize_range ("[0,1]" or
"[-1,1]"), add_basepoint (default False) and dtype (default "float32").

Returns {"train", "val", "test"}, each with "paths" and "clocks" (lists of arrays) and "labels", and "metadata" (the
path width, the number of classes, the task type, the path scaler and the split sizes).
"""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp
import numpy as np

from ews.paths import prepare_paths_np, specs_from_config


def load_path_classification_dataset(
    npz_file: str,
    preprocessing_config: Dict,
) -> Dict:
    cfg = preprocessing_config
    dtype_str = cfg.get("dtype", "float32")
    dtype = jnp.float64 if dtype_str == "float64" else jnp.float32

    raw = np.load(npz_file)

    def _to_list(arr: np.ndarray):
        if arr.dtype == object:
            return [arr[i] for i in range(len(arr))]
        return [arr[i] for i in range(arr.shape[0])]

    # Load paths as lists of (T, d)
    splits: Dict[str, Dict] = {}
    for split in ("train", "val", "test"):
        paths_arr = raw[f"{split}_paths"]  # (n, T, d)
        clocks_arr = raw[f"{split}_clocks"]  # (n, T)
        labels_arr = raw[f"{split}_labels"]  # (n,) or (n, C)
        splits[split] = {
            "paths": _to_list(paths_arr),
            "clocks": _to_list(clocks_arr),
            "labels": labels_arr,
        }

    # Infer task type from label shape
    sample_labels = splits["train"]["labels"]
    if sample_labels.ndim == 1:
        task_type = "multiclass"
        n_classes = int(sample_labels.max()) + 1
    else:
        task_type = "multilabel"
        n_classes = sample_labels.shape[1]

    path_spec, clock_spec, append_spec = specs_from_config(cfg)
    clock_normalize = clock_spec.scaling
    augment_with_clock = path_spec.clock_channel == "main"
    normalize_mode = path_spec.normalize_mode

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

    # Labels as jnp arrays (integer for multiclass, float32 for multilabel)
    if task_type == "multiclass":
        train_labels = jnp.array(splits["train"]["labels"], dtype=jnp.int32)
        val_labels = jnp.array(splits["val"]["labels"], dtype=jnp.int32)
        test_labels = jnp.array(splits["test"]["labels"], dtype=jnp.int32)
    else:
        train_labels = jnp.array(splits["train"]["labels"], dtype=jnp.float32)
        val_labels = jnp.array(splits["val"]["labels"], dtype=jnp.float32)
        test_labels = jnp.array(splits["test"]["labels"], dtype=jnp.float32)

    return {
        "train": {"paths": train_paths_jax, "clocks": train_clocks_jax, "labels": train_labels},
        "val": {"paths": val_paths_jax, "clocks": val_clocks_jax, "labels": val_labels},
        "test": {"paths": test_paths_jax, "clocks": test_clocks_jax, "labels": test_labels},
        "metadata": {
            "d_path": d_path,
            "n_classes": n_classes,
            "task_type": task_type,
            "path_scaler": scaler,
            "n_train": len(train_paths_jax),
            "n_val": len(val_paths_jax),
            "n_test": len(test_paths_jax),
            "clock_normalize": clock_normalize,
            "augment_with_clock": augment_with_clock,
        },
    }
