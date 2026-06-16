from ml_lab.leaderboard import Leaderboard


def test_append_and_rank_by_r2_desc(tmp_path):
    lb = Leaderboard(tmp_path / "lb.csv")
    lb.append({"model": "low", "r2": 0.50, "rmse": 0.2, "mae": 0.1})
    lb.append({"model": "high", "r2": 0.90, "rmse": 0.1, "mae": 0.05})
    lb.append({"model": "mid", "r2": 0.70, "rmse": 0.15, "mae": 0.08})

    table = lb.to_table()
    assert list(table["model"]) == ["high", "mid", "low"]


def test_persists_to_disk_and_reloads(tmp_path):
    path = tmp_path / "lb.csv"
    lb = Leaderboard(path)
    lb.append({"model": "a", "r2": 0.6, "rmse": 0.2, "mae": 0.1})
    assert path.exists()

    lb2 = Leaderboard(path)  # fresh instance reads existing file
    table = lb2.to_table()
    assert "a" in list(table["model"])


def test_append_updates_existing_model_row(tmp_path):
    lb = Leaderboard(tmp_path / "lb.csv")
    lb.append({"model": "a", "r2": 0.5, "rmse": 0.2, "mae": 0.1})
    lb.append({"model": "a", "r2": 0.8, "rmse": 0.1, "mae": 0.05})  # rerun, better
    table = lb.to_table()
    # only one row for model 'a', keeping the latest values
    assert list(table["model"]).count("a") == 1
    assert float(table.loc[table["model"] == "a", "r2"].iloc[0]) == 0.8
