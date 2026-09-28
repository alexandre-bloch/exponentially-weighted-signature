"""PPG-DaLiA with Walker's protocol (github.com/Benjamin-Walker/log-neural-cdes: data_dir/process_ppg.py +
datasets.create_ppg_dataset + dataloaders.Dataloader), transcribed verbatim except that the per-subject split-variant
draw is seeded
(Walker's script uses an unseeded random.randint(0, 5), so his exact splits are not reproducible).

Raw:       data/ppg/raw/PPG_FieldStudy/S{i}/S{i}.pkl   (UCI archive id 495)
Processed: data/ppg/processed/{X,y}_{train,val,test}.npy (variant "ours", this script)
           data/ppg/processed_walker/                   (variant "walker": Walker's own process_ppg.py output, the banked runs)
Windows: inputs (n, 49920, 6) [ACC x3, BVP, EDA, TEMP at 64 Hz, each min-max scaled to [-1, 1] per subject],
targets (n, 390) heart rate (0.5 Hz, scaled to [-1, 1] per subject) = one target every 128 input steps.
Loader: Walker's Dataloader (random permutation batches, the last partial batch dropped; loop_epoch in order).
"""

import os
import pickle
import random

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(REPO, "data", "ppg", "raw", "PPG_FieldStudy")
PROCESSED = os.path.join(REPO, "data", "ppg", "processed")
PROCESSED_WALKER = os.path.join(REPO, "data", "ppg", "processed_walker")


def process(split_seed=0):
    random.seed(split_seed)
    parts = {
        k: []
        for k in (
            "train_input",
            "val_input",
            "test_input",
            "train_output",
            "val_output",
            "test_output",
        )
    }
    variants = []
    for i in range(1, 16):
        with open(os.path.join(RAW, f"S{i}", f"S{i}.pkl"), "rb") as f:
            data = pickle.load(f, encoding="latin1")
        ACC = np.repeat(data["signal"]["wrist"]["ACC"], 2, axis=0)
        BVP = data["signal"]["wrist"]["BVP"]
        EDA = np.repeat(data["signal"]["wrist"]["EDA"], 16, axis=0)
        TEMP = np.repeat(data["signal"]["wrist"]["TEMP"], 16, axis=0)
        ACC = 2 * (ACC - np.min(ACC)) / (np.max(ACC) - np.min(ACC)) - 1
        BVP = 2 * (BVP - np.min(BVP)) / (np.max(BVP) - np.min(BVP)) - 1
        EDA = 2 * (EDA - np.min(EDA)) / (np.max(EDA) - np.min(EDA)) - 1
        TEMP = 2 * (TEMP - np.min(TEMP)) / (np.max(TEMP) - np.min(TEMP)) - 1
        inp = np.concatenate([ACC, BVP, EDA, TEMP], axis=1)
        out = data["label"]
        out = np.concatenate([[out[0]], [out[0]], [out[0]], out], axis=0)
        out = 2 * (out - np.min(out)) / (np.max(out) - np.min(out)) - 1
        variant = random.randint(0, 5)
        variants.append(variant)
        n, no = len(inp), len(out)
        cuts = {
            0: (
                slice(0, int(0.7 * n)),
                slice(int(0.7 * n), int(0.85 * n)),
                slice(int(0.85 * n), None),
            ),
            1: (
                slice(0, int(0.7 * n)),
                slice(int(0.85 * n), None),
                slice(int(0.7 * n), int(0.85 * n)),
            ),
            2: (
                slice(int(0.15 * n), int(0.85 * n)),
                slice(0, int(0.15 * n)),
                slice(int(0.85 * n), None),
            ),
            3: (
                slice(int(0.15 * n), int(0.85 * n)),
                slice(int(0.85 * n), None),
                slice(0, int(0.15 * n)),
            ),
            4: (
                slice(int(0.30 * n), None),
                slice(0, int(0.15 * n)),
                slice(int(0.15 * n), int(0.30 * n)),
            ),
            5: (
                slice(int(0.30 * n), None),
                slice(int(0.15 * n), int(0.30 * n)),
                slice(0, int(0.15 * n)),
            ),
        }[variant]
        cuts_o = {
            0: (
                slice(0, int(0.7 * no)),
                slice(int(0.7 * no), int(0.85 * no)),
                slice(int(0.85 * no), None),
            ),
            1: (
                slice(0, int(0.7 * no)),
                slice(int(0.85 * no), None),
                slice(int(0.7 * no), int(0.85 * no)),
            ),
            2: (
                slice(int(0.15 * no), int(0.85 * no)),
                slice(0, int(0.15 * no)),
                slice(int(0.85 * no), None),
            ),
            3: (
                slice(int(0.15 * no), int(0.85 * no)),
                slice(int(0.85 * no), None),
                slice(0, int(0.15 * no)),
            ),
            4: (
                slice(int(0.30 * no), None),
                slice(0, int(0.15 * no)),
                slice(int(0.15 * no), int(0.30 * no)),
            ),
            5: (
                slice(int(0.30 * no), None),
                slice(int(0.15 * no), int(0.30 * no)),
                slice(0, int(0.15 * no)),
            ),
        }[variant]
        for name, ci, co in zip(("train", "val", "test"), cuts, cuts_o):
            parts[f"{name}_input"].append(np.swapaxes(swv(inp[ci], 49920, 0)[::4992], 1, 2))
            parts[f"{name}_output"].append(swv(out[co], 390, 0)[::39])
    os.makedirs(PROCESSED, exist_ok=True)
    for name in ("train", "val", "test"):
        X = np.concatenate(parts[f"{name}_input"], 0).astype(np.float32)
        y = np.concatenate(parts[f"{name}_output"], 0).astype(np.float32)
        np.save(os.path.join(PROCESSED, f"X_{name}.npy"), X)
        np.save(os.path.join(PROCESSED, f"y_{name}.npy"), y)
        print(name, X.shape, y.shape)
    with open(os.path.join(PROCESSED, "split_variants.txt"), "w") as f:
        f.write(f"split_seed {split_seed}\nvariants per subject S1..S15: {variants}\n")
    print("variants", variants)


class Dataloader:
    """Walker's Dataloader semantics (numpy arrays in memory; batches yielded as numpy)."""

    def __init__(self, data, labels):
        self.data, self.labels, self.size = data, labels, len(data)

    def loop(self, batch_size, *, key):
        import jax.random as jr

        if batch_size > self.size:
            raise ValueError("Batch size larger than dataset size")
        if batch_size == self.size:
            while True:
                yield self.data, self.labels
        while True:
            subkey, key = jr.split(key)
            perm = np.asarray(jr.permutation(subkey, self.size))
            start, end = 0, batch_size
            while end < self.size:  # Walker: the last (partial or exact) batch is dropped
                b = perm[start:end]
                yield self.data[b], self.labels[b]
                start, end = end, end + batch_size

    def loop_epoch(self, batch_size):
        if batch_size >= self.size:
            yield self.data, self.labels
            return
        start, end = 0, batch_size
        while end < self.size:
            yield self.data[start:end], self.labels[start:end]
            start, end = end, end + batch_size


def create_ppg(include_time=False, T=1.0, variant="ours"):
    """-> dict of Dataloaders (train/val/test), info (d_in, L, n_out). variant 'ours' (seeded split variants) or
    'walker' (his own process_ppg.py output, converted; data/ppg/processed_walker)."""
    out = {}
    pdir = PROCESSED if variant == "ours" else PROCESSED_WALKER
    for name in ("train", "val", "test"):
        X = np.load(os.path.join(pdir, f"X_{name}.npy"))
        y = np.load(os.path.join(pdir, f"y_{name}.npy"))
        if include_time:
            ts = (T / X.shape[1]) * np.arange(X.shape[1], dtype=np.float32)
            X = np.concatenate(
                [np.broadcast_to(ts[None, :, None], (X.shape[0], X.shape[1], 1)), X], axis=2
            )
        out[name] = Dataloader(X, y)
    info = dict(
        d_in=out["train"].data.shape[2],
        L=out["train"].data.shape[1],
        n_out=out["train"].labels.shape[1],
        output_step=out["train"].data.shape[1] // out["train"].labels.shape[1],
    )
    return out, info


if __name__ == "__main__":
    import sys

    process(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
