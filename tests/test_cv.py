import numpy as np
import pandas as pd
import pytest

from meta_labeling.cv import PurgedKFold


@pytest.fixture
def t1():
    """Her 2 barda bir başlayan, 5 bar süren örtüşen etiketler."""
    idx = pd.bdate_range("2020-01-01", periods=400)
    t0 = idx[:390:2]
    return pd.Series(idx[idx.get_indexer(t0) + 5], index=t0)


def _overlaps(t1, i, j):
    return t1.index[i] <= t1.iloc[j] and t1.index[j] <= t1.iloc[i]


def test_no_train_label_overlaps_test_labels(t1):
    for train, test in PurgedKFold(t1, n_splits=5, embargo_pct=0.0).split():
        test_start, test_end = t1.index[test[0]], t1.iloc[test].max()
        for i in train:
            assert t1.iloc[i] < test_start or t1.index[i] > test_end


def test_purging_removes_only_overlapping_neighbours(t1):
    train, test = list(PurgedKFold(t1, n_splits=5, embargo_pct=0.0).split())[2]
    removed = np.setdiff1d(np.arange(len(t1)), np.concatenate([train, test]))
    assert len(removed) > 0
    assert all(any(_overlaps(t1, r, k) for k in test) for r in removed)


def test_embargo_removes_observations_after_test(t1):
    base = list(PurgedKFold(t1, n_splits=5, embargo_pct=0.0).split())
    emb = list(PurgedKFold(t1, n_splits=5, embargo_pct=0.05).split())
    embargo = (t1.index[-1] - t1.index[0]) * 0.05
    for (train0, test), (train1, _) in zip(base, emb):
        dropped = np.setdiff1d(train0, train1)
        test_end = t1.iloc[test].max()
        # Embargo yalnızca testten SONRA ve embargo süresi içinde başlayan gözlemleri atar
        assert all(test_end < t1.index[d] <= test_end + embargo for d in dropped)
    assert sum(len(t) for t, _ in base) > sum(len(t) for t, _ in emb)


def test_walk_forward_trains_only_on_closed_past_labels(t1):
    splits = list(PurgedKFold(t1, n_splits=5, embargo_pct=0.01, walk_forward=True).split())
    assert len(splits[0][0]) == 0  # ilk fold için geçmiş yok
    for train, test in splits[1:]:
        assert (t1.iloc[train] < t1.index[test[0]]).all()


def test_test_folds_partition_the_sample(t1):
    tests = [test for _, test in PurgedKFold(t1, n_splits=4).split()]
    assert np.array_equal(np.concatenate(tests), np.arange(len(t1)))


def test_rejects_misaligned_features(t1):
    X = pd.DataFrame({"f": 0.0}, index=t1.index + pd.Timedelta(days=1))
    with pytest.raises(ValueError):
        next(PurgedKFold(t1, n_splits=4).split(X))
