"""Run ONE model in an isolated process.

Invoked by ``ml_lab.run`` as a subprocess so each model gets its own native
runtime (avoiding the torch/GBDT duplicate-OpenMP crash) and so a crash or OOM
in one model cannot abort the whole sweep. Writes ``result.json`` into the run
directory; the parent reads it to update the leaderboard.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig
from .data import load_dataframe, select_xy
from .experiment import run_experiment, run_walk_forward


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run a single model in isolation.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--n-trials", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.subsample is not None:
        cfg.subsample = args.subsample
    if args.n_trials is not None:
        cfg.n_trials = args.n_trials

    df = load_dataframe(cfg.data_path)
    if cfg.subsample:
        df = df.iloc[: cfg.subsample].reset_index(drop=True)
    X, y = select_xy(df, cfg.feature_slice, cfg.target)

    if cfg.features:
        from .features import add_time_features
        X, y = add_time_features(X, y, **cfg.features)

    outdir = Path(cfg.output_dir) / "runs" / args.model
    if getattr(cfg, "eval", "single") == "walk_forward":
        rec = run_walk_forward(args.model, X, y, cfg, outdir, leaderboard=None)
    else:
        rec = run_experiment(args.model, X, y, cfg, outdir, leaderboard=None)
    print(f"RESULT {args.model}: r2={rec['r2']:.6f} rmse={rec['rmse']:.6f} "
          f"mae={rec['mae']:.6f} ({rec['train_time_s']}s)")


if __name__ == "__main__":
    main()
