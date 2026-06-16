import numpy as np
import pytest

from ml_lab.splits import chronological_split


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
