"""FT-Transformer wrapper (rtdl_revisiting_models) with a compact torch loop.

A modern attention-based deep tabular model. Inputs must be scaled (handled by
the experiment layer); the target is standardized internally. Training uses
AdamW with validation-RMSE early stopping. Runs on Apple-Silicon MPS when
available, otherwise CPU.
"""
from __future__ import annotations

import numpy as np

from .base import BaseRegressor
from ..metrics import regression_metrics


def _device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class FTTransformerModel(BaseRegressor):
    name = "ft_transformer"
    needs_scaling = True
    tunable = True

    def __init__(self, n_blocks=3, d_block=192, attention_dropout=0.2,
                 ffn_dropout=0.1, residual_dropout=0.0, lr=1e-3,
                 weight_decay=1e-5, batch_size=256, max_epochs=100, patience=16, **_):
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.model = None
        self.device = "cpu"

    def _build(self, n_features):
        from rtdl_revisiting_models import FTTransformer

        kw = FTTransformer.get_default_kwargs()
        kw.pop("_is_default", None)
        kw.update(
            n_blocks=self.n_blocks, d_block=self.d_block,
            attention_dropout=self.attention_dropout,
            ffn_dropout=self.ffn_dropout, residual_dropout=self.residual_dropout,
        )
        return FTTransformer(n_cont_features=n_features, cat_cardinalities=[],
                             d_out=1, **kw)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(42)
        self.device = _device()
        Xtr = torch.tensor(np.asarray(X_train, dtype=np.float32))
        ytr = torch.tensor(np.asarray(y_train, dtype=np.float32)).view(-1, 1)
        self.y_mean = float(ytr.mean())
        self.y_std = float(ytr.std()) + 1e-8
        ytr_n = (ytr - self.y_mean) / self.y_std

        self.model = self._build(Xtr.shape[1]).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        loss_fn = torch.nn.MSELoss()
        loader = DataLoader(TensorDataset(Xtr, ytr_n), batch_size=self.batch_size,
                            shuffle=True)

        has_val = X_val is not None
        if has_val:
            Xva = torch.tensor(np.asarray(X_val, dtype=np.float32)).to(self.device)
            y_val_np = np.asarray(y_val, dtype=float).ravel()

        best_rmse, best_state, waited = np.inf, None, 0
        for _ in range(self.max_epochs):
            self.model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.model(xb, None), yb)
                loss.backward()
                opt.step()

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    pred = self.model(Xva, None).cpu().numpy().ravel()
                pred = pred * self.y_std + self.y_mean
                rmse = regression_metrics(y_val_np, pred)["rmse"]
                if rmse < best_rmse - 1e-9:
                    best_rmse = rmse
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                    waited = 0
                else:
                    waited += 1
                    if waited >= self.patience:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X):
        import torch

        self.model.eval()
        Xt = torch.tensor(np.asarray(X, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            out = self.model(Xt, None).cpu().numpy().ravel()
        return out * self.y_std + self.y_mean

    def save(self, path):
        import joblib

        self.model.to("cpu")
        self.device = "cpu"
        joblib.dump(self, path)

    def suggest_params(self, trial) -> dict:
        return dict(
            n_blocks=trial.suggest_int("n_blocks", 1, 4),
            d_block=trial.suggest_categorical("d_block", [96, 128, 192, 256]),
            attention_dropout=trial.suggest_float("attention_dropout", 0.0, 0.4),
            ffn_dropout=trial.suggest_float("ffn_dropout", 0.0, 0.4),
            lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        )
