"""Single-model experiment: split -> (scale) -> tune -> fit -> evaluate -> log."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import fit_scaler
from .metrics import regression_metrics
from .plots import measured_vs_predicted
from .registry import get_model_class
from .splits import chronological_split, walk_forward_split
from .tuning import tune


def _scale(scaler, X):
    return pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)


def run_experiment(model_name, X, y, cfg, outdir, leaderboard=None) -> dict:
    """Train and evaluate one model on a time-aware split; return its record."""
    model_cls = get_model_class(model_name)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    tr, va, te = chronological_split(len(X), cfg.ratios, cfg.embargo)
    Xtr, ytr = X.iloc[tr], y.iloc[tr]
    Xva, yva = X.iloc[va], y.iloc[va]
    Xte, yte = X.iloc[te], y.iloc[te]

    if getattr(model_cls, "needs_scaling", False):
        scaler = fit_scaler(Xtr)
        Xtr, Xva, Xte = _scale(scaler, Xtr), _scale(scaler, Xva), _scale(scaler, Xte)

    fixed_kwargs = (getattr(cfg, "model_kwargs", {}) or {}).get(model_name, {})

    t0 = time.time()
    best = tune(model_cls, Xtr, ytr, n_trials=cfg.n_trials, cv_splits=cfg.cv_splits,
                timeout=cfg.tuning_timeout, seed=cfg.seed)
    model = model_cls(**{**fixed_kwargs, **best})
    model.fit(Xtr, ytr, Xva, yva)
    preds = model.predict(Xte)
    metrics = regression_metrics(yte, preds)
    train_time = time.time() - t0

    run_dir = Path(outdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    json.dump(
        {"model": model_name, "params": best, "metrics": metrics,
         "train_time_s": train_time, "n_test": int(len(yte))},
        open(run_dir / "result.json", "w"), indent=2, default=str,
    )
    measured_vs_predicted(
        yte, preds, f"{model_name}  (R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f})",
        run_dir / "pred_vs_actual.png",
    )

    record = {"model": model_name, "r2": metrics["r2"], "rmse": metrics["rmse"],
              "mae": metrics["mae"], "train_time_s": round(train_time, 1),
              "n_trials": cfg.n_trials, "status": "ok"}
    if leaderboard is not None:
        leaderboard.append(record)
    return record


def run_walk_forward(model_name, X, y, cfg, outdir, leaderboard=None) -> dict:
    """Rolling-origin evaluation for the online (periodic-recalibration) scenario.

    Tunes once on the initial training window, then walks forward: at each origin
    it refits the model on all history up to an embargo gap and predicts the next
    ``interval`` rows. Predictions from every block are pooled and scored once, in
    the target's original units. The returned record matches ``run_experiment`` so
    it drops straight into the leaderboard.

    Each fold is fit to completion with the tuned params and NO early-stopping
    validation set: under the slow PE drift a held-out recent tail is a different
    regime, so early stopping halts after a few trees and collapses accuracy
    (R² 0.61 -> 0.09 in testing). The single tuning pass already sized the model.
    """
    model_cls = get_model_class(model_name)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    wf = cfg.walk_forward or {}
    init_frac = wf.get("init_frac", 0.5)
    interval = wf.get("interval", 250)
    window = wf.get("window", "expanding")
    folds = walk_forward_split(len(X), init_frac=init_frac, interval=interval,
                               embargo=cfg.embargo, window=window)
    if not folds:
        raise ValueError("walk_forward_split produced no folds")

    needs_scaling = getattr(model_cls, "needs_scaling", False)
    fixed_kwargs = (getattr(cfg, "model_kwargs", {}) or {}).get(model_name, {})

    t0 = time.time()
    # Tune ONCE on the initial training window (occasional tuning, frequent refits).
    tr0 = folds[0][0]
    Xtune, ytune = X.iloc[tr0], y.iloc[tr0]
    if needs_scaling:
        Xtune = _scale(fit_scaler(Xtune), Xtune)
    best = tune(model_cls, Xtune, ytune, n_trials=cfg.n_trials, cv_splits=cfg.cv_splits,
                timeout=cfg.tuning_timeout, seed=cfg.seed)

    # Refit at each origin with the chosen params; pool predictions across blocks.
    preds, trues = [], []
    for tr, te in folds:
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        Xte, yte = X.iloc[te], y.iloc[te]
        if needs_scaling:
            scaler = fit_scaler(Xtr)
            Xtr, Xte = _scale(scaler, Xtr), _scale(scaler, Xte)
        model = model_cls(**{**fixed_kwargs, **best})
        model.fit(Xtr, ytr)  # fit to completion (no early-stopping val; see docstring)
        preds.append(np.asarray(model.predict(Xte)).ravel())
        trues.append(np.asarray(yte).ravel())
    train_time = time.time() - t0

    y_true, y_pred = np.concatenate(trues), np.concatenate(preds)
    metrics = regression_metrics(y_true, y_pred)

    run_dir = Path(outdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    json.dump(
        {"model": model_name, "params": best, "metrics": metrics,
         "train_time_s": train_time, "n_test": int(len(y_true)),
         "n_origins": len(folds), "eval": "walk_forward",
         "walk_forward": {"init_frac": init_frac, "interval": interval,
                          "window": window, "embargo": cfg.embargo}},
        open(run_dir / "result.json", "w"), indent=2, default=str,
    )
    measured_vs_predicted(
        y_true, y_pred,
        f"{model_name}  walk-forward  (R²={metrics['r2']:.4f}, "
        f"RMSE={metrics['rmse']:.4f}, {len(folds)} folds)",
        run_dir / "pred_vs_actual.png",
    )

    record = {"model": model_name, "r2": metrics["r2"], "rmse": metrics["rmse"],
              "mae": metrics["mae"], "train_time_s": round(train_time, 1),
              "n_trials": cfg.n_trials, "status": "ok"}
    if leaderboard is not None:
        leaderboard.append(record)
    return record
