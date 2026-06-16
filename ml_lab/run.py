"""CLI entry point: run a model sweep and build a leaderboard.

Each model runs in its own subprocess (``ml_lab.run_one``) for two reasons:
  1. Native isolation — torch (FT-Transformer) and the GBDT libs ship conflicting
     OpenMP runtimes that segfault when loaded together in one process.
  2. Failure isolation — a crash/OOM in one model returns a non-zero exit code
     that the parent records, instead of killing the whole sweep (a segfault
     cannot be caught by try/except).

Usage:
    python -m ml_lab.run --config configs/pe_soft_sensor.yaml --models all
    python -m ml_lab.run --config configs/pe_soft_sensor.yaml --models xgboost,lightgbm
    python -m ml_lab.run --config configs/pe_soft_sensor.yaml --subsample 3000 --n-trials 2
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from .config import ExperimentConfig
from .leaderboard import Leaderboard


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run furnace PE soft-sensor experiments.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", default="all", help="'all' or comma-separated names")
    ap.add_argument("--subsample", type=int, default=None, help="cap rows (smoke test)")
    ap.add_argument("--n-trials", type=int, default=None, help="override tuning trials")
    ap.add_argument("--model-python", action="append", default=[],
                    help="per-model interpreter, e.g. autogluon=/path/to/python")
    args = ap.parse_args(argv)

    cfg = ExperimentConfig.from_yaml(args.config)
    model_python = dict(cfg.model_python or {})
    for kv in args.model_python:
        key, _, val = kv.partition("=")
        model_python[key] = val
    models = cfg.models if args.models == "all" else args.models.split(",")
    out_root = Path(cfg.output_dir)
    leaderboard = Leaderboard(out_root / "leaderboard.csv")

    print(f"Config: {args.config}")
    print(f"Models: {models}")
    print(f"Each model runs in an isolated subprocess.\n")

    for name in models:
        name = name.strip()
        run_dir = out_root / "runs" / name
        result_path = run_dir / "result.json"
        if result_path.exists():
            result_path.unlink()  # avoid reading a stale result if this run crashes

        cmd = [model_python.get(name, sys.executable), "-m", "ml_lab.run_one",
               "--config", args.config, "--model", name]
        if args.subsample is not None:
            cmd += ["--subsample", str(args.subsample)]
        if args.n_trials is not None:
            cmd += ["--n-trials", str(args.n_trials)]

        print(f">>> {name}")
        proc = subprocess.run(cmd)

        if proc.returncode == 0 and result_path.exists():
            res = json.load(open(result_path))
            m = res["metrics"]
            leaderboard.append({
                "model": name, "r2": m["r2"], "rmse": m["rmse"], "mae": m["mae"],
                "train_time_s": round(res.get("train_time_s", 0.0), 1), "status": "ok",
            })
            print(f"[OK]   {name:14s} R2={m['r2']:.4f}  RMSE={m['rmse']:.4f}  "
                  f"MAE={m['mae']:.4f}\n")
        else:
            print(f"[FAIL] {name:14s} returncode={proc.returncode}\n")
            leaderboard.append({"model": name, "r2": math.nan, "rmse": math.nan,
                                "mae": math.nan, "train_time_s": math.nan,
                                "status": f"failed(rc={proc.returncode})"})

    print("=== Leaderboard ===")
    print(leaderboard.to_table().to_string(index=False))


if __name__ == "__main__":
    main()
