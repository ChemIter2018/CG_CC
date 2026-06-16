"""Time-aware feature engineering for the soft-sensor.

Adds backward-looking lag and rolling-window (mean/std) features of the process
variables. These use only the current and past rows, so under a chronological
split they introduce no future leakage — they mimic what a deployed soft sensor
would have at inference time. Rows with NaNs from the initial window are dropped
(y is realigned).
"""
from __future__ import annotations

import pandas as pd


def add_time_features(X: pd.DataFrame, y: pd.Series, lags=(1, 2, 3), windows=(5, 15)):
    """Return (X_with_features, y_aligned) with lag/rolling columns added."""
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    parts = [X]
    for lag in lags:
        parts.append(X.shift(lag).add_suffix(f"_lag{lag}"))
    for w in windows:
        parts.append(X.rolling(w).mean().add_suffix(f"_rmean{w}"))
        parts.append(X.rolling(w).std().add_suffix(f"_rstd{w}"))

    Xf = pd.concat(parts, axis=1)
    valid = Xf.notna().all(axis=1)
    Xf = Xf[valid].reset_index(drop=True)
    yf = y[valid].reset_index(drop=True)
    return Xf, yf
