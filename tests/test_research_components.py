"""Araştırma katmanı bileşen testleri: execution, maliyet, risk, CPCV, leakage, kalibrasyon, metrikler."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import meta_labeling.features as features_mod
from meta_labeling.research import ConfigError, LeakageError, ResearchConfig, config_from_dict, run_leakage_audit
from meta_labeling.research.cpcv import CombinatorialPurgedKFold
from meta_labeling.research.execution import cost_rate, events_to_target, simulate_portfolio
from meta_labeling.research.leakage import calibration_checks, split_checks
from meta_labeling.research.metrics import deflated_sharpe, drawdown_episodes, psr
from meta_labeling.research.modeling import OOSPrediction, calibrate_walk_forward, position_sizes
from meta_labeling.research.settings import CostSettings, ExecutionSettings, RiskSettings

IDX = pd.bdate_range("2021-01-01", periods=6)


def _ohlcv(opens, closes):
    return pd.DataFrame({"Open": opens, "High": np.maximum(opens, closes), "Low": np.minimum(opens, closes),
                         "Close": closes, "Volume": 1e6}, index=IDX[: len(closes)], dtype=float)


# ------------------------------------------------------------------ execution
def test_next_open_execution_uses_next_bar_open():
    ohlcv = _ohlcv([100, 102, 110, 111, 111, 111], [100, 104, 112, 111, 111, 111])
    target = pd.Series([1.0, 0, 0, 0, 0, 0], index=IDX)  # t0 kapanışında karar, t1'de çık
    port = simulate_portfolio(target, ohlcv, ExecutionSettings(mode="next_open"), CostSettings(commission_bps=0),
                              RiskSettings())
    # Bar 1: gece 100->102 pozisyonsuz, açılışta long, gün içi 102->104
    assert port.returns.iloc[1] == pytest.approx(104 / 102 - 1)
    # Bar 2: gece 104->110 long tutuluyor, açılışta çıkış
    assert port.returns.iloc[2] == pytest.approx(110 / 104 - 1)
    total = (1 + port.returns).prod() - 1
    assert total == pytest.approx(110 / 102 - 1)  # giriş open(t+1)=102, çıkış open=110


def test_close_execution_is_same_bar_close():
    ohlcv = _ohlcv([100] * 6, [100, 104, 112, 111, 111, 111])
    target = pd.Series([1.0, 0, 0, 0, 0, 0], index=IDX)
    port = simulate_portfolio(target, ohlcv, ExecutionSettings(mode="close"), CostSettings(commission_bps=0),
                              RiskSettings())
    assert (1 + port.returns).prod() - 1 == pytest.approx(104 / 100 - 1)


def test_costs_charged_on_turnover():
    ohlcv = _ohlcv([100] * 6, [100] * 6)
    target = pd.Series([1.0, 1.0, 0, 0, 0, 0], index=IDX)
    port = simulate_portfolio(target, ohlcv, ExecutionSettings(mode="close"), CostSettings(commission_bps=10),
                              RiskSettings())
    assert port.costs.sum() == pytest.approx(2 * 0.001)
    assert port.turnover.sum() == pytest.approx(2.0)


def test_market_impact_grows_with_order_size():
    c = CostSettings(commission_bps=5, impact_k=0.1)
    small, large = cost_rate(np.array([0.1, 1.0]), np.array([1e8, 1e8]), c, capital=1e7)
    assert small < large
    assert large == pytest.approx(5e-4 + 0.1 * np.sqrt(1e7 / 1e8))


def test_position_is_capped_and_drawdown_stop_halts():
    ohlcv = _ohlcv([100] * 6, [100, 80, 60, 70, 80, 90])
    target = pd.Series(3.0, index=IDX)  # kaldıraç isteği
    port = simulate_portfolio(target, ohlcv, ExecutionSettings(mode="close"), CostSettings(commission_bps=0),
                              RiskSettings(max_position=1.0, max_drawdown_stop=0.3))
    assert port.positions.abs().max() <= 1.0
    assert port.halted_at == IDX[2]
    assert (port.positions.iloc[3:] == 0).all()


def test_events_to_target_averages_concurrent_signals():
    t0 = IDX[[0, 1]]
    t1 = pd.Series(IDX[[3, 2]], index=t0)
    target = events_to_target(IDX, t0, t1, pd.Series([1.0, -0.5], index=t0))
    assert target.tolist() == pytest.approx([1.0, 0.25, 1.0, 0.0, 0.0, 0.0])


def test_kelly_is_fractional_and_capped():
    cfg = ResearchConfig()
    ev = pd.DataFrame({"trgt": 0.01, "side": 1}, index=IDX)
    p = pd.Series([0.4, 0.56, 0.7, 0.9, 0.99, 1.0], index=IDX)
    size = position_sizes("kelly", p, ev, cfg, kelly_fraction=0.25)
    assert size.iloc[0] == 0
    assert size.max() <= 0.25 + 1e-12
    assert size.is_monotonic_increasing


# ------------------------------------------------------------------ CPCV
def test_cpcv_paths_cover_every_group_once():
    idx = pd.bdate_range("2020-01-01", periods=300)
    t0 = idx[:290:2]
    t1 = pd.Series(idx[idx.get_indexer(t0) + 4], index=t0)
    cp = CombinatorialPurgedKFold(t1, n_groups=6, n_test_groups=2, embargo_pct=0.01)
    splits = list(cp.split())
    assert len(splits) == 15 and cp.n_paths == 5
    paths = cp.paths([c for c, _, _ in splits])
    for path in paths:
        assert sorted(path) == list(range(6))
        for g, s in path.items():
            assert g in splits[s][0]
    report = split_checks("cpcv", t1, [(tr, te) for _, tr, te in splits], False, 0.01)
    assert report[0]["status"] == "PASS"


# ------------------------------------------------------------------ leakage
@pytest.fixture(scope="module")
def small_cfg():
    cfg = ResearchConfig()
    return replace(cfg, data=replace(cfg.data, simulation=replace(cfg.data.simulation, n_bars=800)),
                   analysis=replace(cfg.analysis, leakage_samples=8),
                   experiment=replace(cfg.experiment, log_level="ERROR"))


def test_leakage_audit_passes_clean_pipeline(small_cfg):
    from meta_labeling.data import simulate_ohlcv

    report = run_leakage_audit(simulate_ohlcv(small_cfg.data.simulation), small_cfg)
    assert report.status == "PASS"
    assert {"feature_lookahead", "volatility_lookahead", "regime_lookahead"} <= set(report.findings["check"])


def test_leakage_audit_flags_future_feature(small_cfg, monkeypatch):
    from meta_labeling.data import simulate_ohlcv

    original = features_mod.build_feature_matrix

    def leaky(ohlcv, cfg=None):
        X = original(ohlcv, cfg)
        X["future_ret"] = ohlcv["Close"].pct_change().shift(-1)
        return X

    monkeypatch.setattr(features_mod, "build_feature_matrix", leaky)
    ohlcv = simulate_ohlcv(small_cfg.data.simulation)
    with pytest.raises(LeakageError):
        run_leakage_audit(ohlcv, small_cfg, strict=True)
    report = run_leakage_audit(ohlcv, small_cfg, strict=False)
    bad = report.failures
    assert list(bad["item"]) == ["future_ret"]
    assert bad["timestamp"].notna().all()


def test_split_check_detects_training_on_future():
    idx = pd.bdate_range("2020-01-01", periods=50)
    t1 = pd.Series(idx[5:45], index=idx[:40])
    leaky_split = [(np.arange(0, 40), np.arange(20, 30))]  # eğitim test'i içeriyor
    assert split_checks("wf", t1, leaky_split, True, 0.0)[0]["status"] == "FAIL"


def test_calibration_uses_only_closed_past_labels():
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(0)
    proba = pd.Series(rng.uniform(0.3, 0.8, 400), index=idx)
    y = pd.Series((rng.uniform(size=400) < proba).astype(int), index=idx)
    fold = pd.Series(np.repeat([0, 1, 2, 3], 100), index=idx)
    label_end = pd.Series(idx[np.minimum(np.arange(400) + 5, 399)], index=idx)
    res = calibrate_walk_forward(OOSPrediction(proba, fold), y, label_end, "isotonic", min_history=50)
    assert res.proba.iloc[:100].isna().all()  # ilk fold: geçmiş yok
    assert all(last < start for _, last, start, _ in res.fit_windows)
    assert calibration_checks({"isotonic": res.fit_windows})[0]["status"] == "PASS"
    assert calibration_checks({"bad": [(1, idx[200], idx[100], 10)]})[0]["status"] == "FAIL"


# ------------------------------------------------------------------ metrikler / config
def test_dsr_penalizes_many_trials():
    r = pd.Series(np.random.default_rng(1).normal(0.0006, 0.01, 2000))
    single = psr(r)
    many = deflated_sharpe(r, np.random.default_rng(2).normal(0, 0.02, 50), 50)
    assert many < single


def test_drawdown_episode_hand_example():
    r = pd.Series([0.1, -0.5, 0.5, 0.5, 0.1], index=IDX[:5])
    ep = drawdown_episodes(r)
    assert len(ep) == 1
    assert ep.loc[0, "depth"] == pytest.approx(-0.5)
    assert ep.loc[0, "duration_bars"] == 2


def test_config_rejects_unknown_keys():
    with pytest.raises(ConfigError):
        config_from_dict({"meta_model": {"treshold": 0.6}})
    with pytest.raises(ConfigError):
        config_from_dict({"primary": {"event_mode": "cusum"}})
    cfg = config_from_dict({"meta_model": {"threshold": 0.6}, "analysis": {"threshold_grid": [0.5, 0.6]}})
    assert cfg.meta_model.threshold == 0.6 and cfg.analysis.threshold_grid == (0.5, 0.6)
