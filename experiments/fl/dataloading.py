"""Task data for the formal-language experiments, on the vendored loaders of experiments/fl/dataloaders.py.

The tasks (cycle_nav, parity, even_pairs, mod_arith_no_brack, ...) are generated on the fly: training lengths 3-40
padded to 40, validation lengths 40-256 padded to 256, seeds (seed, 2 * seed); the loss mask marks each sample's
final position. The model also needs a validity mask (real positions against padding, prefix-structured) for the clock
and the padding, which `validity_mask(loss_mask, dense)` derives. Batches are converted to numpy here.

    python -m experiments.fl.dataloading      # a smoke test
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import torch

from experiments.fl import dataloaders as dl

# Published protocol constants (lcde rules of the legacy train.py).
FL_TRAIN = dict(num_samples=25_600_000, min_length=3, max_length=40, padding_length=40)
FL_VAL = dict(num_samples=8192, min_length=40, max_length=256, padding_length=256)


def to_numpy(batch):
    X, y, mask = batch
    if isinstance(X, tuple):
        return (
            tuple(t.cpu().numpy() for t in X),
            tuple(t.cpu().numpy() for t in y),
            tuple(t.cpu().numpy() for t in mask),
        )
    return X.cpu().numpy(), y.cpu().numpy(), mask.cpu().numpy()


def validity_mask(loss_mask: np.ndarray, dense: bool) -> np.ndarray:
    """Derive the real-vs-pad validity mask from the loss mask.

    dense=False (FL): the loss mask marks only the true final position; every
    position up to and including it is real (prefix-structured by collate).
    """
    if dense:
        return loss_mask.astype(bool)
    last = loss_mask.argmax(axis=-1)  # exactly one True per row
    idx = np.arange(loss_mask.shape[-1])
    return idx[None, :] <= last[:, None]


def infinite_batches(loader) -> Iterator:
    """Re-iterate a DataLoader forever, yielding numpy triples."""
    while True:
        for batch in loader:
            yield to_numpy(batch)


def eval_batches(loader) -> Iterator:
    """One epoch over a DataLoader, yielding numpy triples."""
    for batch in loader:
        yield to_numpy(batch)


@dataclass
class TaskData:
    task: str
    train_iter: Iterator
    val_loader: object  # torch DataLoader; iterate via eval_batches()
    data_dim: int
    label_dim: int
    train_pad: int
    val_pad: int
    loss_mask_is_dense: bool
    length: Optional[int] = None


def make_task_loaders(
    task: str,
    batch_size: int,
    seed: int,
    *,
    length: Optional[int] = None,
    train_samples: Optional[int] = None,
    val_samples: Optional[int] = None,
    val_batch_size: Optional[int] = None,
) -> TaskData:
    """Build the loaders for one task, replicating the legacy dispatch.

    Args:
            (e.g. "cycle_nav", "parity", "even_pairs", "mod_arith_no_brack").
        batch_size: Training batch size (val uses the same).
        seed: Protocol seed; val generation uses 2*seed as in the legacy code.
        train_samples / val_samples: Override the protocol sample counts
            (25.6M / 8192): for smoke tests only; defaults match the paper.
        val_batch_size: Eval-loader batch size override (memory control for
            large m at padding 256; protocol-neutral: eval only).
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    tr = dict(FL_TRAIN)
    va = dict(FL_VAL)
    if train_samples is not None:
        tr["num_samples"] = int(train_samples)
    if val_samples is not None:
        va["num_samples"] = int(val_samples)
    train_loader, _, data_dim, label_dim = dl.create_fl_dataloaders(
        task, batch_size=batch_size, train_split=1.0, seed=seed, **tr
    )
    val_loader, _, _, _ = dl.create_fl_dataloaders(
        task, batch_size=(val_batch_size or batch_size), train_split=1.0, seed=2 * seed, **va
    )
    return TaskData(
        task,
        infinite_batches(train_loader),
        val_loader,
        data_dim,
        label_dim,
        train_pad=tr["padding_length"],
        val_pad=va["padding_length"],
        loss_mask_is_dense=False,
    )


# ---------------------------------------------------------------------------
# Smoke: python -m experiments.fl.dataloading
# ---------------------------------------------------------------------------


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def _smoke():
    print("== cycle_nav (FL) ==")
    td = make_task_loaders("cycle_nav", 32, 1234, train_samples=1024, val_samples=256)
    X, y, m = next(td.train_iter)
    _check(X.shape == (32, 40) and X.dtype == np.int64, f"train X {X.shape} {X.dtype}")
    _check(m.dtype == np.bool_ and (m.sum(axis=1) == 1).all(), "loss mask: one True/row")
    _check((X[m] != 0).all(), "masked position holds a real token")
    v = validity_mask(m, dense=False)
    lens = v.sum(axis=1)
    _check((v == (np.arange(40)[None, :] < lens[:, None])).all(), "validity prefix-structured")
    _check(
        lens.min() >= 3 and lens.max() <= 40, f"train lengths in [3,40] ({lens.min()}-{lens.max()})"
    )
    Xv, yv, mv = next(eval_batches(td.val_loader))
    _check(Xv.shape[1] == 256, f"val padded to 256 ({Xv.shape})")
    vlens = validity_mask(mv, dense=False).sum(axis=1)
    _check(
        vlens.min() >= 40 and vlens.max() <= 256,
        f"val lengths in [40,256] ({vlens.min()}-{vlens.max()})",
    )
    td_b = make_task_loaders("cycle_nav", 32, 1234, train_samples=1024, val_samples=256)
    Xb, yb, mb = next(td_b.train_iter)
    _check(
        (X == Xb).all() and (y == yb).all() and (m == mb).all(),
        "same seed -> identical first batch",
    )

    (X, X2), (y, y2), (m, m2) = next(td.train_iter)
    _check(td.data_dim == 60 and td.label_dim == 60, "vocab 60/60")
    _check(X.shape == (32, 10) and X2.shape[1] == 10, f"main {X.shape}, len2 {X2.shape}")
    _check(X2.shape[0] == 3, f"len2 batch = batch//10 ({X2.shape[0]})")
    _check(m.all(), "main mask dense (all real)")
    _check((m2.sum(axis=1) == 2).all() and m2[:, :2].all(), "len2 mask: first 2 positions")
    Xv, yv, mv = next(eval_batches(td.val_loader))
    _check(Xv.shape[1] == 10 and mv.all(), f"val fixed length ({Xv.shape})")

    X, y, m = next(td.train_iter)
    _check(X.shape == (32, 40), f"train padded to 40 ({X.shape})")
    v = validity_mask(m, dense=True)
    lens = v.sum(axis=1)
    _check((v == (np.arange(40)[None, :] < lens[:, None])).all(), "dense mask prefix-structured")
    _check(lens.min() >= 3 and lens.max() <= 40, f"lengths in [3,40] ({lens.min()}-{lens.max()})")
    _check((y[m] < 60).all() and (y[m] >= 0).all(), "labels are group elements")
    Xv, yv, mv = next(eval_batches(td.val_loader))
    _check(Xv.shape[1] == 128, f"val padded to 128 ({Xv.shape})")

    print("SMOKE PASSED")
