import numpy as np
import pandas as pd
import pytest

from meta_labeling.backtest import average_active_positions, probabilistic_sharpe_ratio
from meta_labeling.config import SimulationConfig
from meta_labeling.data import simulate_ohlcv
from meta_labeling.features import build_feature_matrix
from meta_labeling.sample_weights import average_uniqueness, num_concurrent_events
from meta_labeling.sizing import bet_size_from_proba


@pytest.fixture(scope="module")
def ohlcv():
    return simulate_ohlcv(SimulationConfig(n_bars=600, seed=1))


def test_features_have_no_look_ahead(ohlcv):
    """t anındaki öznitelik, t sonrasındaki veriler silindiğinde değişmemeli."""
    full = build_feature_matrix(ohlcv)
    for cut in (150, 320, 599):
        truncated = build_feature_matrix(ohlcv.iloc[:cut])
        pd.testing.assert_series_equal(
            truncated.iloc[-1], full.iloc[cut - 1], check_names=False, rtol=1e-9
        )


def test_simulated_ohlc_is_consistent(ohlcv):
    assert (ohlcv["High"] >= ohlcv[["Open", "Close"]].max(axis=1)).all()
    assert (ohlcv["Low"] <= ohlcv[["Open", "Close"]].min(axis=1)).all()
    assert (ohlcv["Volume"] > 0).all()


def test_concurrency_and_uniqueness_match_hand_calculation():
    idx = pd.bdate_range("2020-01-01", periods=6)
    # e0: [0,2], e1: [1,3], e2: [4,5]
    t1 = pd.Series(idx[[2, 3, 5]], index=idx[[0, 1, 4]])
    co = num_concurrent_events(idx, t1)
    assert co.tolist() == [1, 2, 2, 1, 1, 1]
    u = average_uniqueness(idx, t1, co)
    assert u.iloc[0] == pytest.approx((1 + 0.5 + 0.5) / 3)
    assert u.iloc[1] == pytest.approx((0.5 + 0.5 + 1) / 3)
    assert u.iloc[2] == pytest.approx(1.0)


def test_bet_size_is_monotonic_and_thresholded():
    p = pd.Series([0.3, 0.5, 0.55, 0.56, 0.7, 0.9])
    size = bet_size_from_proba(p, threshold=0.55)
    assert size.iloc[:3].eq(0).all()
    assert size.iloc[3] > 0
    assert size.is_monotonic_increasing
    assert size.max() < 1.0


def test_average_active_positions_starts_after_entry_bar():
    idx = pd.bdate_range("2020-01-01", periods=6)
    t1 = pd.Series(idx[[3, 4]], index=idx[[1, 2]])
    pos = average_active_positions(idx, t1, pd.Series([1.0, -0.5], index=t1.index))
    # e0 barlar (1,3], e1 barlar (2,4]; ortak barlarda ortalama alınır
    assert pos.tolist() == pytest.approx([0.0, 0.0, 1.0, 0.25, -0.5, 0.0])


def test_psr_is_high_for_clearly_positive_returns():
    rng = np.random.default_rng(0)
    good = pd.Series(rng.normal(0.002, 0.01, 1000))
    bad = pd.Series(rng.normal(-0.002, 0.01, 1000))
    assert probabilistic_sharpe_ratio(good) > 0.95
    assert probabilistic_sharpe_ratio(bad) < 0.05
