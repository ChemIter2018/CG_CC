import numpy as np
import pandas as pd
import pytest

from ml_lab.data import select_xy, fit_scaler


def _toy_df():
    # columns: idx, a, b, c, PE  -> feature slice (1,4) selects a,b,c; target PE
    return pd.DataFrame(
        {
            "idx": [0, 1, 2, 3],
            "a": [0.0, 2.0, 4.0, 6.0],
            "b": [1.0, 1.0, 1.0, 1.0],
            "c": [10.0, 20.0, 30.0, 40.0],
            "PE": [0.5, 0.6, 0.7, 0.8],
        }
    )


def test_select_xy_picks_feature_slice_and_target():
    X, y = select_xy(_toy_df(), feature_slice=(1, 4), target="PE")
    assert list(X.columns) == ["a", "b", "c"]
    assert list(y) == [0.5, 0.6, 0.7, 0.8]
    # target must not appear among features
    assert "PE" not in X.columns


def test_fit_scaler_uses_training_rows_only():
    X, _ = select_xy(_toy_df(), feature_slice=(1, 4), target="PE")
    X_train = X.iloc[:2]  # rows a=[0,2] -> mean 1.0 ; full-data mean would be 3.0
    scaler = fit_scaler(X_train)
    assert scaler.mean_[0] == pytest.approx(1.0)  # NOT 3.0 (would mean fit on all rows)


def test_scaler_transform_standardizes_training_data():
    X, _ = select_xy(_toy_df(), feature_slice=(1, 4), target="PE")
    X_train = X.iloc[:2]
    scaler = fit_scaler(X_train)
    Xt = scaler.transform(X_train)
    assert np.allclose(Xt.mean(axis=0), 0.0)
