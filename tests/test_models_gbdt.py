import numpy as np
import pandas as pd
import pytest

from ml_lab.metrics import regression_metrics
from ml_lab.models.gbdt import (
    CatBoostRegressorModel,
    LightGBMRegressorModel,
    XGBoostRegressorModel,
)

CLASSES = [XGBoostRegressorModel, LightGBMRegressorModel, CatBoostRegressorModel]


def _synth():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(2.0 * X["f0"] - 3.0 * X["f1"] + 0.1 * rng.normal(size=400))
    return (
        X.iloc[:300], y.iloc[:300],
        X.iloc[300:350], y.iloc[300:350],
        X.iloc[350:], y.iloc[350:],
    )


@pytest.mark.parametrize("cls", CLASSES)
def test_gbdt_learns_linear_signal(cls):
    Xtr, ytr, Xva, yva, Xte, yte = _synth()
    model = cls(n_estimators=200).fit(Xtr, ytr, Xva, yva)
    preds = model.predict(Xte)
    assert len(preds) == len(Xte)
    assert regression_metrics(yte, preds)["r2"] > 0.9


@pytest.mark.parametrize("cls", CLASSES)
def test_suggest_params_returns_dict(cls):
    import optuna

    study = optuna.create_study()
    trial = study.ask()
    params = cls().suggest_params(trial)
    assert isinstance(params, dict) and len(params) > 0
