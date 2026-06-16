from ml_lab.config import ExperimentConfig


def test_defaults():
    cfg = ExperimentConfig(data_path="data/00FurnaceCleanData.csv")
    assert cfg.target == "PE"
    assert cfg.feature_slice == (1, 20)
    assert cfg.ratios == (0.7, 0.15, 0.15)
    assert cfg.embargo == 0


def test_from_yaml(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "data_path: data/00FurnaceCleanData.csv\n"
        "n_trials: 5\n"
        "ratios: [0.6, 0.2, 0.2]\n"
        "feature_slice: [1, 20]\n"
        "models: [xgboost, lightgbm]\n"
    )
    cfg = ExperimentConfig.from_yaml(p)
    assert cfg.n_trials == 5
    assert cfg.ratios == (0.6, 0.2, 0.2)          # list coerced to tuple
    assert cfg.feature_slice == (1, 20)
    assert cfg.models == ["xgboost", "lightgbm"]
