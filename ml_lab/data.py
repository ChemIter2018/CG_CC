"""Dataset loading and feature/target selection for the furnace soft-sensor.

Features are the 19 process variables (CSV columns 1..19). The product
composition columns (H2..C3H8) are deliberately NOT used as features: they
are downstream outputs and would leak information about the target PE.
"""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_dataframe(path) -> pd.DataFrame:
    """Read the furnace CSV into a DataFrame."""
    return pd.read_csv(path)


def select_xy(df: pd.DataFrame, feature_slice=(1, 20), target="PE"):
    """Select feature matrix X (positional slice) and target y (by name)."""
    start, stop = feature_slice
    X = df.iloc[:, start:stop].copy()
    if target in X.columns:
        raise ValueError(f"target '{target}' must not be inside the feature slice")
    y = df[target].copy()
    return X, y


def fit_scaler(X_train) -> StandardScaler:
    """Fit a StandardScaler on the TRAINING rows only (avoids leakage)."""
    return StandardScaler().fit(X_train)
