"""Informer weather (WTH.csv), a numpy transcription of state-spaces/s4's `InformerDataset` / `_Dataset_Weather` (the
loader S4 and LinOSS use), multivariate:

  * the 12 climate channels with the target 'WetBulbCelsius' last, 'date' dropped;
  * borders num_train = int(0.7 N) and num_test = int(0.2 N), the rest validation: train [0, num_train),
    val [num_train - seq_len, num_train + num_vali), test [N - num_test - seq_len, N);
  * a StandardScaler fitted on the train slice and applied to everything; the metrics are in this scaled space;
  * window i: x = [data[i : i + seq_len]; zeros(pred_len)] (L = 1440), y = data[i + seq_len : i + seq_len + pred_len].
Added here: an optional time channel t = linspace(0, 1, L) as channel 0, and the loop(batch, key) / loop_epoch(batch)
interface of the Walker dataloaders.

Data file: data/weather/WTH.csv (35,064 hourly rows of 12 channels and the date, 2010-2013; sha256 prefix
153fead3ea2a83dc).
"""

import csv
import os

import jax.random as jr
import numpy as np

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "weather", "WTH.csv"
)
TARGET = "WetBulbCelsius"


def load_raw(path=DATA_PATH):
    rows = list(csv.reader(open(path)))
    header, body = rows[0], rows[1:]
    cols = [c for c in header if c not in ("date", TARGET)] + [TARGET]
    idx = [header.index(c) for c in cols]
    data = np.array([[float(r[j]) for j in idx] for r in body], dtype=np.float64)
    return data, cols


def load_marks(path=DATA_PATH):
    """Informer timestamp features, timeenc=0 / freq 'h': [month, day, weekday, hour]
    (the S4 pipeline embeds these as an extra 'time' encoder input). Returned
    scaled to [0, 1): month/12, day/31, weekday/7, hour/24."""
    from datetime import datetime

    rows = list(csv.reader(open(path)))
    di = rows[0].index("date")
    out = []
    for r in rows[1:]:
        d = datetime.strptime(r[di], "%m/%d/%Y %H:%M")
        out.append([(d.month - 1) / 12, (d.day - 1) / 31, d.weekday() / 7, d.hour / 24])
    return np.array(out, dtype=np.float64)


class Split:
    def __init__(self, data_scaled, seq_len, pred_len, include_time, marks=None, marks_mode="none"):
        self.data = data_scaled.astype(np.float32)
        self.marks = None if marks is None else marks.astype(np.float32)
        self.marks_mode = marks_mode  # none | obs (zeros in the masked half, S4 default) | full
        self.seq_len, self.pred_len = seq_len, pred_len
        self.L = seq_len + pred_len
        self.n = len(self.data) - seq_len - pred_len + 1
        self.include_time = include_time
        self.d_out = self.data.shape[1]
        self.d_in = (
            self.d_out
            + (1 if include_time else 0)
            + (self.marks.shape[1] if self.marks is not None else 0)
        )
        self._time = np.linspace(0.0, 1.0, self.L, dtype=np.float32)[:, None]

    def window(self, idx):
        idx = np.asarray(idx)
        s = idx[:, None] + np.arange(self.seq_len)[None, :]
        x = np.concatenate(
            [self.data[s], np.zeros((len(idx), self.pred_len, self.d_out), np.float32)], axis=1
        )
        if self.marks is not None:
            full = idx[:, None] + np.arange(self.L)[None, :]
            mk = self.marks[full]
            if self.marks_mode == "obs":
                mk = mk.copy()
                mk[:, self.seq_len :] = 0.0
            x = np.concatenate([x, mk], axis=2)
        if self.include_time:
            x = np.concatenate([np.broadcast_to(self._time, (len(idx), self.L, 1)), x], axis=2)
        t = idx[:, None] + self.seq_len + np.arange(self.pred_len)[None, :]
        y = self.data[t]
        return x, y

    def loop(self, batch_size, *, key):
        """Random batches forever (with replacement across epochs, shuffled within)."""
        while True:
            key, sub = jr.split(key)
            perm = np.asarray(jr.permutation(sub, self.n))
            for i in range(0, self.n - batch_size + 1, batch_size):
                yield self.window(perm[i : i + batch_size])

    def loop_epoch(self, batch_size):
        for i in range(0, self.n, batch_size):
            yield self.window(np.arange(i, min(i + batch_size, self.n)))


def create_weather(seq_len=720, pred_len=720, include_time=True, path=DATA_PATH, marks="none"):
    data, cols = load_raw(path)
    mk = load_marks(path) if marks != "none" else None
    N = len(data)
    num_train = int(N * 0.7)
    num_test = int(N * 0.2)
    num_vali = N - num_train - num_test
    border1s = [0, num_train - seq_len, N - num_test - seq_len]
    border2s = [num_train, num_train + num_vali, N]
    train_slice = data[border1s[0] : border2s[0]]
    mean, std = train_slice.mean(0), train_slice.std(0)
    scaled = (data - mean) / std
    splits = {
        name: Split(
            scaled[b1:b2], seq_len, pred_len, include_time, None if mk is None else mk[b1:b2], marks
        )
        for name, b1, b2 in zip(("train", "val", "test"), border1s, border2s)
    }
    info = dict(
        columns=cols,
        N=N,
        num_train=num_train,
        num_vali=num_vali,
        num_test=num_test,
        borders=list(zip(border1s, border2s)),
        mean=mean.tolist(),
        std=std.tolist(),
        windows={k: v.n for k, v in splits.items()},
        seq_len=seq_len,
        pred_len=pred_len,
        include_time=include_time,
        marks=marks,
        d_in=splits["train"].d_in,
        d_out=splits["train"].d_out,
    )
    return splits, info


if __name__ == "__main__":
    import json

    splits, info = create_weather()
    print(json.dumps({k: v for k, v in info.items() if k not in ("mean", "std")}, indent=1))
    x, y = splits["train"].window(np.array([0, 1]))
    print(
        "x",
        x.shape,
        x.dtype,
        "y",
        y.shape,
        "masked half all zero:",
        bool((x[:, 720:, 1:] == 0).all()),
    )
