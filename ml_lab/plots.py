"""Measured-vs-predicted diagnostic plot."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def measured_vs_predicted(y_true, y_pred, title, out_path, dpi=150):
    """Scatter of measured vs predicted with an identity reference line."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, s=8, alpha=0.3, edgecolor="none")
    ax.plot([lo, hi], [lo, hi], color="navy", ls="--", lw=2)
    ax.set_xlabel("Measured PE")
    ax.set_ylabel("Predicted PE")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
