"""Walk-forward (rolling-origin) evaluation orchestrator."""
import json

import numpy as np
import pandas as pd

from ml_lab.config import ExperimentConfig
from ml_lab.experiment import run_walk_forward
from ml_lab.models.base import BaseRegressor
from ml_lab.splits import walk_forward_split


def _toy_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = pd.Series(0.5 + 0.01 * (X["a"] - X["b"]) + 0.002 * rng.normal(size=n), name="PE")
    return X, y


def _cfg(tmp_path, **kw):
    base = dict(
        data_path="x.csv", embargo=5, n_trials=0, cv_splits=3, output_dir=str(tmp_path),
        eval="walk_forward",
        walk_forward={"init_frac": 0.5, "interval": 50, "window": "expanding"},
    )
    base.update(kw)
    return ExperimentConfig(**base)


def test_run_walk_forward_returns_ok_record(tmp_path):
    X, y = _toy_data()
    rec = run_walk_forward("legacy_knr", X, y, _cfg(tmp_path),
                           tmp_path / "runs" / "legacy_knr")
    assert rec["model"] == "legacy_knr"
    assert rec["status"] == "ok"
    assert set(rec) >= {"model", "r2", "rmse", "mae", "train_time_s", "status"}
    assert np.isfinite(rec["r2"]) and np.isfinite(rec["rmse"])


def test_run_walk_forward_pools_all_origins(tmp_path):
    X, y = _toy_data()
    outdir = tmp_path / "runs" / "legacy_knr"
    run_walk_forward("legacy_knr", X, y, _cfg(tmp_path), outdir)
    res = json.load(open(outdir / "result.json"))
    folds = walk_forward_split(len(X), init_frac=0.5, interval=50, embargo=5)
    assert res["n_origins"] == len(folds) > 1
    assert res["n_test"] == sum(len(te) for _, te in folds)  # pooled over all blocks


def test_run_walk_forward_appends_to_leaderboard(tmp_path):
    X, y = _toy_data()
    board = []
    run_walk_forward("legacy_knr", X, y, _cfg(tmp_path),
                     tmp_path / "runs" / "legacy_knr", leaderboard=board)
    assert len(board) == 1 and board[0]["model"] == "legacy_knr"


def test_run_walk_forward_fits_each_fold_to_completion(tmp_path, monkeypatch):
    """Each fold must fit on its FULL training window (no early-stopping val).

    Carving a recent tail for early stopping underfits badly: on this slowly
    drifting target the tail is a different regime, so early stopping halts after
    a few trees and predictions collapse to ~constant. Walk-forward therefore
    fits to completion, i.e. calls ``fit`` with ``X_val=None`` on every fold.
    """
    seen_val = []

    class SpyModel(BaseRegressor):
        needs_scaling = False
        tunable = False

        def __init__(self, **_):
            self._mean = 0.0

        def fit(self, X_train, y_train, X_val=None, y_val=None):
            seen_val.append(X_val)
            self._mean = float(np.asarray(y_train, dtype=float).mean())
            return self

        def predict(self, X):
            return np.full(len(X), self._mean)

    monkeypatch.setattr("ml_lab.experiment.get_model_class", lambda name: SpyModel)
    X, y = _toy_data()
    run_walk_forward("spy", X, y, _cfg(tmp_path), tmp_path / "runs" / "spy")

    assert len(seen_val) > 1                       # several origins were fit
    assert all(v is None for v in seen_val)        # no early-stopping validation set
