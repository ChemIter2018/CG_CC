"""Time-aware data splitting for the furnace soft-sensor.

The furnace data is a continuous time series whose adjacent rows are nearly
identical. A random train/test split therefore leaks almost-duplicate rows
across the boundary and inflates the score. These helpers split strictly by
time order, with an optional ``embargo`` gap between segments to further
reduce boundary leakage.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import TimeSeriesSplit


def chronological_split(n, ratios=(0.7, 0.15, 0.15), embargo=0):
    """Split ``n`` time-ordered rows into train/val/test index arrays.

    Segments are contiguous and strictly time-ordered (all train indices
    precede all val indices, which precede all test indices). With
    ``embargo > 0``, that many rows are dropped at the start of the val and
    test segments, creating gaps that nobody uses.
    """
    if len(ratios) != 3:
        raise ValueError("ratios must have exactly 3 elements (train, val, test)")
    if any(r <= 0 for r in ratios):
        raise ValueError("ratios must all be positive")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios must sum to 1.0")
    if embargo < 0:
        raise ValueError("embargo must be >= 0")

    r_train, r_val, _ = ratios
    cut1 = int(n * r_train)
    cut2 = int(n * (r_train + r_val))

    if cut1 + embargo >= cut2 or cut2 + embargo >= n:
        raise ValueError("embargo too large for the given ratios and n")

    train = np.arange(0, cut1)
    val = np.arange(cut1 + embargo, cut2)
    test = np.arange(cut2 + embargo, n)
    return train, val, test


def walk_forward_split(n, init_frac=0.5, interval=250, embargo=0, window="expanding"):
    """Rolling-origin (walk-forward) folds for the ONLINE recalibration scenario.

    Returns a list of ``(train_idx, test_idx)`` arrays. Starting after an initial
    ``init_frac`` of the rows, contiguous test blocks of ``interval`` rows tile
    the remainder of the series (the last block may be shorter). For each block
    the model trains on everything up to ``embargo`` rows before that block:

    - ``window="expanding"`` — train is ``[0, block_start - embargo)`` and grows.
    - ``window="sliding"``   — train is a fixed window of ``int(n*init_frac)``
      rows ending at ``block_start - embargo``.

    Every fold is strictly time-ordered (``max(train) < min(test)``) with an
    ``embargo`` gap, so no future or near-duplicate boundary rows leak. This
    mimics periodically refitting a deployed soft sensor and predicting the next
    ``interval`` rows until the next recalibration.
    """
    if not 0 < init_frac < 1:
        raise ValueError("init_frac must be in (0, 1)")
    if interval <= 0:
        raise ValueError("interval must be positive")
    if embargo < 0:
        raise ValueError("embargo must be >= 0")
    if window not in ("expanding", "sliding"):
        raise ValueError("window must be 'expanding' or 'sliding'")

    init = int(n * init_frac)
    start = init + embargo  # first test row
    if start >= n:
        raise ValueError("init_frac/embargo leave no test rows")

    folds = []
    k = 0
    while True:
        te_lo = start + k * interval
        if te_lo >= n:
            break
        te_hi = min(te_lo + interval, n)
        tr_hi = te_lo - embargo
        tr_lo = 0 if window == "expanding" else max(0, tr_hi - init)
        folds.append((np.arange(tr_lo, tr_hi), np.arange(te_lo, te_hi)))
        k += 1
    return folds


def time_series_cv(n_splits=5):
    """Return an sklearn TimeSeriesSplit for time-aware hyperparameter search."""
    return TimeSeriesSplit(n_splits=n_splits)
