import numpy as np
import pandas as pd

from ml_lab.models.gbdt import LightGBMRegressorModel
from ml_lab.models.base import BaseRegressor
from ml_lab.tuning import tune


def _synth(n=300):
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)])
    y = pd.Series(2.0 * X["f0"] - X["f1"] + 0.1 * rng.normal(size=n))
    return X, y


def test_tune_returns_usable_params():
    X, y = _synth()
    best = tune(LightGBMRegressorModel, X, y, n_trials=3, cv_splits=3, seed=0)
    assert isinstance(best, dict) and len(best) > 0
    model = LightGBMRegressorModel(**best).fit(X.iloc[:250], y.iloc[:250])
    assert len(model.predict(X.iloc[250:])) == 50


def test_tune_no_search_space_returns_empty():
    class NoTune(BaseRegressor):
        name = "notune"

        def fit(self, X_train, y_train, X_val=None, y_val=None):
            return self

        def predict(self, X):
            return np.zeros(len(X))

    X, y = _synth(120)
    assert tune(NoTune, X, y, n_trials=5, cv_splits=3) == {}
