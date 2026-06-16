import numpy as np
import pandas as pd

from ml_lab.models.base import BaseRegressor


class DummyMeanRegressor(BaseRegressor):
    """Minimal concrete model used to exercise the BaseRegressor contract."""

    name = "dummy"

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.mean_ = float(np.mean(y_train))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def _xy():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([10.0, 20.0, 30.0, 40.0])
    return X, y


def test_fit_returns_self_and_predict_shape():
    X, y = _xy()
    model = DummyMeanRegressor().fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(X)
    assert np.allclose(preds, 25.0)  # mean of y


def test_save_and_load_roundtrip(tmp_path):
    X, y = _xy()
    model = DummyMeanRegressor().fit(X, y)
    path = tmp_path / "model.joblib"
    model.save(path)

    loaded = DummyMeanRegressor.load(path)
    assert np.allclose(loaded.predict(X), model.predict(X))


def test_default_search_space_is_empty():
    # models without tuning should return an empty param dict
    assert DummyMeanRegressor().suggest_params(trial=None) == {}
