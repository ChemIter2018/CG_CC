"""Experiment configuration (dataclass + YAML loader)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class ExperimentConfig:
    data_path: str
    target: str = "PE"
    feature_slice: tuple = (1, 20)
    ratios: tuple = (0.7, 0.15, 0.15)
    embargo: int = 0
    models: list = field(
        default_factory=lambda: ["xgboost", "lightgbm", "catboost", "ft_transformer"]
    )
    n_trials: int = 30
    tuning_timeout: Optional[int] = None
    cv_splits: int = 4
    seed: int = 42
    subsample: Optional[int] = None  # cap rows (smoke test)
    output_dir: str = "experiments"
    model_kwargs: dict = field(default_factory=dict)  # per-model fixed kwargs
    model_python: dict = field(default_factory=dict)  # per-model python interpreter
    features: dict = field(default_factory=dict)  # lag/rolling feature engineering
    eval: str = "single"  # "single" (chronological split) or "walk_forward"
    walk_forward: dict = field(default_factory=dict)  # init_frac, interval, window

    @classmethod
    def from_yaml(cls, path) -> "ExperimentConfig":
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        if "feature_slice" in d:
            d["feature_slice"] = tuple(d["feature_slice"])
        if "ratios" in d:
            d["ratios"] = tuple(d["ratios"])
        return cls(**d)
