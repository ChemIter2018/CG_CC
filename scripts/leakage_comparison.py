"""Quantify the data leakage in the original random train/test split.

The furnace data is a continuous time series with near-duplicate adjacent rows.
A random split scatters those near-duplicates across train and test, so the test
score measures memorization, not generalization. This script trains the SAME
model (CatBoost, default params) under both a random split (the original
approach) and an honest chronological split, and prints both scores.

Run from the repo root:
    python scripts/leakage_comparison.py
"""
from __future__ import annotations

from sklearn.model_selection import train_test_split

from ml_lab.data import load_dataframe, select_xy
from ml_lab.metrics import regression_metrics
from ml_lab.models.gbdt import CatBoostRegressorModel
from ml_lab.splits import chronological_split

DATA = "data/00FurnaceCleanData.csv"


def main():
    df = load_dataframe(DATA)
    X, y = select_xy(df, (1, 20), "PE")
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Honest chronological split (what ml_lab uses)
    tr, va, te = chronological_split(len(X), (0.7, 0.15, 0.15), embargo=50)
    m = CatBoostRegressorModel().fit(X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va])
    chrono = regression_metrics(y.iloc[te], m.predict(X.iloc[te]))

    # Original-style random split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    X_tr2, X_va, y_tr2, y_va = train_test_split(X_tr, y_tr, test_size=0.15, random_state=0)
    m2 = CatBoostRegressorModel().fit(X_tr2, y_tr2, X_va, y_va)
    rand = regression_metrics(y_te, m2.predict(X_te))

    print("CatBoost (default params), target = PE")
    print(f"  random split (leaky, original): R2={rand['r2']:.4f}  RMSE={rand['rmse']:.5f}")
    print(f"  chronological split (honest):   R2={chrono['r2']:.4f}  RMSE={chrono['rmse']:.5f}")
    print(f"  => random split overstates R2 by {rand['r2'] - chrono['r2']:.3f}")


if __name__ == "__main__":
    main()
