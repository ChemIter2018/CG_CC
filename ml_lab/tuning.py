"""Optuna hyperparameter search with time-aware cross-validation.

``optuna`` is imported lazily inside ``tune`` so that environments running only
non-tunable models (e.g. the AutoGluon env) don't need it installed.
"""
from __future__ import annotations

import numpy as np

from .metrics import regression_metrics
from .splits import time_series_cv


def tune(model_cls, X, y, n_trials=30, cv_splits=4, timeout=None, seed=42) -> dict:
    """Return the best hyperparameters for ``model_cls`` via TimeSeriesSplit CV.

    Non-tunable models (``tunable = False``) return ``{}`` without importing
    optuna.
    """
    if not getattr(model_cls, "tunable", False):
        return {}

    import optuna

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    cv = time_series_cv(cv_splits)

    def objective(trial):
        params = model_cls().suggest_params(trial)
        scores = []
        for tr_idx, va_idx in cv.split(X):
            model = model_cls(**params)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[va_idx], y.iloc[va_idx])
            preds = model.predict(X.iloc[va_idx])
            scores.append(regression_metrics(y.iloc[va_idx], preds)["r2"])
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return study.best_params
