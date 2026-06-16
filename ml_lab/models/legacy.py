"""Legacy models (SVR / KNR / DTR / MLP) re-evaluated under the corrected split.

These mirror the original 2021 scripts but are run with the honest time-aware
split and train-only scaling, so the leaderboard shows a fair baseline. The
target is standardized internally (the original scripts multiplied PE by 100/
1000 for the same numerical reason).
"""
from __future__ import annotations

import numpy as np

from .base import BaseRegressor


class _SklearnLegacy(BaseRegressor):
    needs_scaling = True

    def _make(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def __init__(self, **_):
        self.model = self._make()

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        y = np.asarray(y_train, dtype=float).ravel()
        self.y_mean = float(y.mean())
        self.y_std = float(y.std()) + 1e-8
        self.model.fit(X_train, (y - self.y_mean) / self.y_std)
        return self

    def predict(self, X):
        return np.asarray(self.model.predict(X)).ravel() * self.y_std + self.y_mean


class LegacySVR(_SklearnLegacy):
    name = "legacy_svr"

    def _make(self):
        from sklearn.svm import SVR

        return SVR(kernel="rbf", C=100.0, epsilon=0.1, gamma="scale")


class LegacyKNR(_SklearnLegacy):
    name = "legacy_knr"

    def _make(self):
        from sklearn.neighbors import KNeighborsRegressor

        return KNeighborsRegressor(n_neighbors=5, weights="distance")


class LegacyDTR(_SklearnLegacy):
    name = "legacy_dtr"
    needs_scaling = False

    def _make(self):
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.tree import DecisionTreeRegressor

        return AdaBoostRegressor(
            DecisionTreeRegressor(max_depth=10, random_state=42),
            n_estimators=100, random_state=42,
        )


class LegacyMLP(_SklearnLegacy):
    name = "legacy_mlp"

    def _make(self):
        from sklearn.neural_network import MLPRegressor

        return MLPRegressor(hidden_layer_sizes=(30, 15), activation="relu",
                            solver="adam", max_iter=500, random_state=42)


_LEGACY = {c.name: c for c in (LegacySVR, LegacyKNR, LegacyDTR, LegacyMLP)}


def get_legacy_class(name: str):
    return _LEGACY[name.lower()]
