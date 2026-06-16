"""Gradient-boosted decision tree wrappers (XGBoost / LightGBM / CatBoost).

On industrial tabular data these are typically the strongest models. Each
wrapper exposes the same BaseRegressor interface and an Optuna search space,
and uses the validation set for early stopping when one is provided.
"""
from __future__ import annotations

import numpy as np

from .base import BaseRegressor

EARLY_STOPPING_ROUNDS = 50
SEED = 42


class XGBoostRegressorModel(BaseRegressor):
    name = "xgboost"
    needs_scaling = False
    tunable = True

    def __init__(self, n_estimators=400, max_depth=6, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                 min_child_weight=1, **_):
        self.params = dict(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            colsample_bytree=colsample_bytree, reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
        )
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import xgboost as xgb

        kwargs = dict(self.params, random_state=SEED, n_jobs=-1, eval_metric="rmse",
                      tree_method="hist")
        if X_val is not None:
            kwargs["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
            self.model = xgb.XGBRegressor(**kwargs)
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.model = xgb.XGBRegressor(**kwargs)
            self.model.fit(X_train, y_train, verbose=False)
        return self

    def predict(self, X):
        return np.asarray(self.model.predict(X)).ravel()

    def suggest_params(self, trial) -> dict:
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1200, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        )


class LightGBMRegressorModel(BaseRegressor):
    name = "lightgbm"
    needs_scaling = False
    tunable = True

    def __init__(self, n_estimators=600, num_leaves=31, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                 min_child_samples=20, **_):
        self.params = dict(
            n_estimators=n_estimators, num_leaves=num_leaves,
            learning_rate=learning_rate, subsample=subsample, subsample_freq=1,
            colsample_bytree=colsample_bytree, reg_lambda=reg_lambda,
            min_child_samples=min_child_samples,
        )
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import lightgbm as lgb

        self.model = lgb.LGBMRegressor(**self.params, random_state=SEED,
                                       n_jobs=-1, verbosity=-1)
        if X_val is not None:
            self.model.fit(
                X_train, y_train, eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                           lgb.log_evaluation(0)],
            )
        else:
            self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return np.asarray(self.model.predict(X)).ravel()

    def suggest_params(self, trial) -> dict:
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1500, step=100),
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        )


class CatBoostRegressorModel(BaseRegressor):
    name = "catboost"
    needs_scaling = False
    tunable = True

    def __init__(self, n_estimators=600, depth=6, learning_rate=0.05,
                 l2_leaf_reg=3.0, **_):
        self.params = dict(n_estimators=n_estimators, depth=depth,
                           learning_rate=learning_rate, l2_leaf_reg=l2_leaf_reg)
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from catboost import CatBoostRegressor

        self.model = CatBoostRegressor(
            iterations=self.params["n_estimators"], depth=self.params["depth"],
            learning_rate=self.params["learning_rate"],
            l2_leaf_reg=self.params["l2_leaf_reg"], random_seed=SEED,
            loss_function="RMSE", verbose=False,
        )
        if X_val is not None:
            self.model.fit(X_train, y_train, eval_set=(X_val, y_val),
                           early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
        else:
            self.model.fit(X_train, y_train, verbose=False)
        return self

    def predict(self, X):
        return np.asarray(self.model.predict(X)).ravel()

    def suggest_params(self, trial) -> dict:
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1500, step=100),
            depth=trial.suggest_int("depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        )
