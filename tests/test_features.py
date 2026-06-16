import pandas as pd

from ml_lab.features import add_time_features


def test_lag_and_rolling_columns_and_alignment():
    X = pd.DataFrame({"a": [0., 1., 2., 3., 4., 5.],
                      "b": [10., 11., 12., 13., 14., 15.]})
    y = pd.Series([100., 101., 102., 103., 104., 105.])
    Xf, yf = add_time_features(X, y, lags=[1], windows=[2])

    for col in ["a", "a_lag1", "a_rmean2", "a_rstd2", "b_lag1"]:
        assert col in Xf.columns
    assert not Xf.isna().any().any()          # NaN rows dropped
    assert len(Xf) == len(yf)                 # X and y stay aligned
    # first surviving row corresponds to original index 1
    assert Xf["a"].iloc[0] == 1.0
    assert Xf["a_lag1"].iloc[0] == 0.0        # lag = previous row's value
    assert yf.iloc[0] == 101.0


def test_rolling_is_backward_looking_no_future_leak():
    X = pd.DataFrame({"a": [1., 2., 3., 4., 5.]})
    y = pd.Series([1., 2., 3., 4., 5.])
    Xf, _ = add_time_features(X, y, lags=[], windows=[2])
    # rolling mean at first surviving row = mean(current, previous) = mean(1,2) = 1.5
    # (never includes future rows)
    assert Xf["a_rmean2"].iloc[0] == 1.5


def test_no_op_when_empty():
    X = pd.DataFrame({"a": [1., 2., 3.]})
    y = pd.Series([1., 2., 3.])
    Xf, yf = add_time_features(X, y, lags=[], windows=[])
    assert list(Xf.columns) == ["a"]
    assert len(Xf) == 3
