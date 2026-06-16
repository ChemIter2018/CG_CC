import numpy as np
import pandas as pd

from ml_lab.metrics import regression_metrics
from ml_lab.models.ft_transformer import FTTransformerModel


def _synth(n=600, d=6):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])
    y = pd.Series(2.0 * X["f0"] - 3.0 * X["f1"] + 0.5 * X["f2"] + 0.1 * rng.normal(size=n))
    return X, y


def test_ft_transformer_learns_signal():
    X, y = _synth()
    Xtr, ytr = X.iloc[:450], y.iloc[:450]
    Xva, yva = X.iloc[450:525], y.iloc[450:525]
    Xte, yte = X.iloc[525:], y.iloc[525:]
    model = FTTransformerModel(n_blocks=2, d_block=96, max_epochs=120, patience=20).fit(
        Xtr, ytr, Xva, yva
    )
    preds = model.predict(Xte)
    assert len(preds) == len(Xte)
    assert regression_metrics(yte, preds)["r2"] > 0.7


def test_ft_transformer_save_load(tmp_path):
    X, y = _synth(300)
    Xtr, ytr = X.iloc[:240], y.iloc[:240]
    Xva, yva = X.iloc[240:], y.iloc[240:]
    model = FTTransformerModel(n_blocks=1, d_block=64, max_epochs=20).fit(Xtr, ytr, Xva, yva)
    path = tmp_path / "ft.joblib"
    model.save(path)
    loaded = FTTransformerModel.load(path)
    assert np.allclose(loaded.predict(Xva), model.predict(Xva), atol=1e-4)
