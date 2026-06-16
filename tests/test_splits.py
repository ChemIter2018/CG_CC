import numpy as np
import pytest

from ml_lab.splits import chronological_split, walk_forward_split


def test_split_sizes_and_order_no_embargo():
    train, val, test = chronological_split(n=100, ratios=(0.7, 0.15, 0.15), embargo=0)
    assert list(train) == list(range(0, 70))
    assert list(val) == list(range(70, 85))
    assert list(test) == list(range(85, 100))


def test_split_segments_are_disjoint_and_time_ordered():
    train, val, test = chronological_split(n=1000, ratios=(0.6, 0.2, 0.2), embargo=0)
    s_train, s_val, s_test = set(train), set(val), set(test)
    # disjoint
    assert s_train.isdisjoint(s_val)
    assert s_val.isdisjoint(s_test)
    assert s_train.isdisjoint(s_test)
    # strictly time-ordered: every train index before every val index before every test index
    assert max(train) < min(val) < max(val) < min(test)


def test_split_no_embargo_covers_all_rows():
    n = 137
    train, val, test = chronological_split(n=n, ratios=(0.7, 0.15, 0.15), embargo=0)
    covered = list(train) + list(val) + list(test)
    assert covered == list(range(n))


def test_embargo_creates_gaps_between_segments():
    train, val, test = chronological_split(n=100, ratios=(0.7, 0.15, 0.15), embargo=5)
    # train still ends at 69; val starts 5 rows later
    assert max(train) == 69
    assert min(val) == 75
    # val block [85 cut] -> test starts 5 after cut2 (=85)
    assert min(test) == 90
    # no overlap still holds
    assert set(train).isdisjoint(val)
    assert set(val).isdisjoint(test)


def test_invalid_ratios_raise():
    with pytest.raises(ValueError):
        chronological_split(n=100, ratios=(0.5, 0.4, 0.4), embargo=0)


def test_embargo_too_large_raises():
    with pytest.raises(ValueError):
        chronological_split(n=20, ratios=(0.7, 0.15, 0.15), embargo=100)


# --- walk_forward_split (rolling-origin evaluation) ---------------------------


def test_walk_forward_blocks_tile_back_and_are_time_ordered():
    folds = walk_forward_split(n=1000, init_frac=0.5, interval=100, embargo=0)
    # test blocks tile [500, 1000) contiguously
    test_concat = np.concatenate([te for _, te in folds])
    assert list(test_concat) == list(range(500, 1000))
    # each fold: every train index precedes every test index (no future leakage)
    for tr, te in folds:
        assert tr.max() < te.min()
    # expanding window: train starts at 0 and grows monotonically
    assert all(tr[0] == 0 for tr, _ in folds)
    assert [len(tr) for tr, _ in folds] == sorted(len(tr) for tr, _ in folds)


def test_walk_forward_embargo_creates_gap_each_fold():
    folds = walk_forward_split(n=1000, init_frac=0.5, interval=100, embargo=10)
    for tr, te in folds:
        assert te.min() - tr.max() - 1 == 10  # exactly `embargo` rows between
    assert folds[0][1].min() == 510  # first test row = init + embargo


def test_walk_forward_no_future_leakage_off_grid():
    for tr, te in walk_forward_split(n=777, init_frac=0.6, interval=37, embargo=5):
        assert tr.max() < te.min()
        assert set(tr.tolist()).isdisjoint(te.tolist())


def test_walk_forward_sliding_keeps_train_size_constant():
    folds = walk_forward_split(n=1000, init_frac=0.5, interval=100, embargo=0,
                               window="sliding")
    assert {len(tr) for tr, _ in folds} == {500}  # == int(n * init_frac)


def test_walk_forward_last_block_may_be_shorter():
    folds = walk_forward_split(n=1050, init_frac=0.5, interval=100, embargo=0)
    last_tr, last_te = folds[-1]
    assert len(last_te) < 100
    assert last_te.max() == 1049  # coverage reaches the final row


def test_walk_forward_invalid_args_raise():
    with pytest.raises(ValueError):
        walk_forward_split(n=1000, init_frac=1.5)
    with pytest.raises(ValueError):
        walk_forward_split(n=1000, interval=0)
    with pytest.raises(ValueError):
        walk_forward_split(n=1000, embargo=-1)
    with pytest.raises(ValueError):
        walk_forward_split(n=1000, window="nope")
    with pytest.raises(ValueError):
        walk_forward_split(n=100, init_frac=0.99, embargo=50)  # leaves no test rows
