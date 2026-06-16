"""Uniform interface every model wrapper implements.

A single interface lets the experiment loop treat XGBoost, a torch
FT-Transformer and AutoGluon identically: fit, predict, save, load, and
(optionally) declare a hyperparameter search space for Optuna.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import joblib


class BaseRegressor(ABC):
    name = "base"
    tunable = False  # True if the model declares an Optuna search space

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model; return self."""

    @abstractmethod
    def predict(self, X):
        """Return a 1-D array of predictions for X."""

    def save(self, path):
        """Persist the fitted model. Default uses joblib; override if needed."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        """Load a model persisted with ``save``."""
        return joblib.load(path)

    def suggest_params(self, trial) -> dict:
        """Sample hyperparameters from an Optuna ``trial``.

        Default: no tuning (empty dict). Models override to declare a search
        space.
        """
        return {}
