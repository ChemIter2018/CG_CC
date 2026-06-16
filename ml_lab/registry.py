"""Map model names to classes with lazy imports.

Heavy backends (torch FT-Transformer, AutoGluon) are imported only when the
model is actually requested, so a GBDT-only run never pays their import cost.
"""
from __future__ import annotations


def get_model_class(name: str):
    key = name.lower()
    if key in ("xgboost", "xgb"):
        from .models.gbdt import XGBoostRegressorModel
        return XGBoostRegressorModel
    if key in ("lightgbm", "lgbm"):
        from .models.gbdt import LightGBMRegressorModel
        return LightGBMRegressorModel
    if key in ("catboost", "cat"):
        from .models.gbdt import CatBoostRegressorModel
        return CatBoostRegressorModel
    if key in ("ft_transformer", "ft", "fttransformer"):
        from .models.ft_transformer import FTTransformerModel
        return FTTransformerModel
    if key == "autogluon":
        from .models.autogluon_model import AutoGluonModel
        return AutoGluonModel
    if key.startswith("legacy_"):
        from .models.legacy import get_legacy_class
        return get_legacy_class(key)
    raise KeyError(f"unknown model: {name}")
