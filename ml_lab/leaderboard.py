"""A small persistent leaderboard for comparing model runs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class Leaderboard:
    """Append model results to a CSV and rank them by R^2 (descending).

    Appending a record for a model name that already exists replaces the old
    row, so re-running a model updates its entry instead of duplicating it.
    """

    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists():
            self._df = pd.read_csv(self.path)
        else:
            self._df = pd.DataFrame()

    def append(self, record: dict) -> None:
        row = pd.DataFrame([record])
        if self._df.empty:
            self._df = row
        else:
            if "model" in self._df.columns and "model" in record:
                self._df = self._df[self._df["model"] != record["model"]]
            self._df = pd.concat([self._df, row], ignore_index=True)
        self._save()

    def to_table(self) -> pd.DataFrame:
        if self._df.empty or "r2" not in self._df.columns:
            return self._df.reset_index(drop=True)
        return self._df.sort_values("r2", ascending=False).reset_index(drop=True)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_csv(self.path, index=False)
