"""AutoGluon Tabular wrapper — a strong auto-ensemble baseline.

AutoGluon trains and stacks many models (GBDTs, NNs, ...) with its own internal
tuning, so it does not use our Optuna search (empty search space). It serves as
a "can your hand-tuned SOTA models beat the auto-ensemble?" yardstick.
"""
from __future__ import annotations

import numpy as np

from .base import BaseRegressor

_LABEL = "__pe_target__"


class AutoGluonModel(BaseRegressor):
    name = "autogluon"
    needs_scaling = False

    def __init__(self, time_limit=600, presets="best_quality", **_):
        self.time_limit = time_limit
        self.presets = presets
        self.predictor = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import pandas as pd
        from autogluon.tabular import TabularPredictor

        # AutoGluon manages its own validation (CV/bagging), so give it the full
        # train+val portion and let it split internally. We do NOT pass
        # tuning_data: combining it with bagging/stacking presets triggers a
        # DyStack sub-fit conflict ("Learner is already fit").
        if X_val is not None:
            X_all = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
            y_all = pd.concat([pd.Series(np.asarray(y_train)),
                               pd.Series(np.asarray(y_val))], axis=0).reset_index(drop=True)
        else:
            X_all, y_all = X_train.reset_index(drop=True), pd.Series(np.asarray(y_train))

        train_df = X_all.copy()
        train_df[_LABEL] = np.asarray(y_all, dtype=float)

        self.predictor = TabularPredictor(
            label=_LABEL, problem_type="regression", eval_metric="r2", verbosity=0
        )
        self.predictor.fit(train_df, time_limit=self.time_limit,
                           presets=self.presets, dynamic_stacking=False)
        return self

    def predict(self, X):
        return np.asarray(self.predictor.predict(X)).ravel()

    def suggest_params(self, trial) -> dict:
        return {}  # AutoGluon self-tunes
