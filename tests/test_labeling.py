import numpy as np
import pandas as pd
import pytest

from meta_labeling.labeling import apply_triple_barrier, get_meta_labels, get_vertical_barriers
from meta_labeling.sampling import cusum_filter, sample_events


def _close(values):
    idx = pd.bdate_range("2020-01-01", periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def _label(close, t0_pos, side, pt=1.0, sl=1.0, horizon=5, trgt=0.05):
    t_events = close.index[[t0_pos]]
    target = pd.Series(trgt, index=close.index)
    sides = pd.Series(side, index=close.index)
    vertical = get_vertical_barriers(t_events, close.index, horizon)
    return apply_triple_barrier(close, t_events, target, sides, vertical, pt, sl).iloc[0]


def test_long_hits_profit_take_first():
    close = _close([100, 101, 103, 106, 90, 90, 90])
    row = _label(close, 0, side=1)
    assert row["barrier"] == "pt"
    assert row["t1"] == close.index[3]  # ln(106/100) = 0.058 > 0.05
    assert row["ret"] == pytest.approx(np.log(1.06))


def test_long_hits_stop_loss_first():
    close = _close([100, 99, 94, 120, 120, 120, 120])
    row = _label(close, 0, side=1)
    assert row["barrier"] == "sl"
    assert row["t1"] == close.index[2]
    assert row["gross_ret"] == pytest.approx(-0.06)


def test_short_side_flips_barriers():
    # Short pozisyonda fiyat düşüşü kâr bariyerine dokunur
    close = _close([100, 99, 94, 120, 120, 120, 120])
    row = _label(close, 0, side=-1)
    assert row["barrier"] == "pt"
    assert row["gross_ret"] == pytest.approx(0.06)


def test_vertical_barrier_when_no_touch():
    close = _close([100, 101, 100, 102, 101, 101.5, 130])
    row = _label(close, 0, side=1, horizon=5)
    assert row["barrier"] == "vertical"
    assert row["t1"] == close.index[5]


def test_disabled_barrier_is_ignored():
    close = _close([100, 110, 120, 130, 140, 150, 160])
    row = _label(close, 0, side=1, pt=0.0, sl=1.0, horizon=5)
    assert row["barrier"] == "vertical"


def test_events_without_full_horizon_are_dropped():
    close = _close(np.linspace(100, 110, 10))
    vertical = get_vertical_barriers(close.index[[2, 8]], close.index, num_bars=3)
    assert list(vertical.index) == [close.index[2]]


def test_meta_labels_account_for_costs():
    events = pd.DataFrame({"gross_ret": [0.02, -0.01, 0.0005]})
    out = get_meta_labels(events, cost_per_side=0.0005)
    assert out["bin"].tolist() == [1, 0, 0]  # 5 bps kâr, 10 bps gidiş-dönüş maliyeti karşılamaz


def test_cusum_filter_triggers_on_cumulative_drift():
    close = _close(np.exp(np.cumsum([0.0] + [0.004] * 10 + [-0.004] * 10)))
    events = cusum_filter(close, threshold=0.01)
    # Her ~3 bar'lık 0.004 birikimi eşiği (0.01) aşar; yukarı ve aşağı yönde olay üretilir
    assert len(events) >= 4
    assert close.index[3] in events


def test_sample_events_signal_mode_detects_entries_and_flips():
    close = _close(np.arange(8) + 100.0)
    side = pd.Series([0, 1, 1, 0, -1, -1, 1, 1], index=close.index)
    events = sample_events(side, close, mode="signal")
    assert list(events) == list(close.index[[1, 4, 6]])
