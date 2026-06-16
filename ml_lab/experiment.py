"""Single-model experiment: split -> (scale) -> tune -> fit -> evaluate -> log."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from .data import fit_scaler
from .metrics import regression_metrics
from .plots import measured_vs_predicted
from .registry import get_model_class
from .splits import chronological_split
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
