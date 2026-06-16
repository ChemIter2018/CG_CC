import numpy as np
import pytest

from ml_lab.metrics import regression_metrics


def test_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = regression_metrics(y, y.copy())
    assert m["r2"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["mae"] == pytest.approx(0.0)


def test_known_values_constant_offset():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])  # +0.5 everywhere
    m = regression_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx(0.5)
    assert m["rmse"] == pytest.approx(0.5)
    assert m["r2"] == pytest.approx(0.625)


def test_accepts_list_and_pandas_like_input():
    m = regression_metrics([0.0, 10.0], [0.0, 10.0])
    assert m["r2"] == pytest.approx(1.0)
