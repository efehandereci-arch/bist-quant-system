from dataclasses import replace

import pytest

from meta_labeling import MetaLabelingPipeline, PipelineConfig, load_ohlcv_csv, simulate_ohlcv


@pytest.fixture(scope="module")
def small_config():
    cfg = PipelineConfig()
    return replace(
        cfg,
        sim=replace(cfg.sim, n_bars=1500, seed=3),
        cv=replace(cfg.cv, n_splits=4, min_train_events=50),
        model=replace(cfg.model, n_estimators=80),
    )


@pytest.fixture(scope="module")
def result(small_config):
    return MetaLabelingPipeline(small_config).run()


def test_pipeline_produces_meta_labels_and_oos_probabilities(result):
    ev = result.events
    assert set(ev["bin"].unique()) <= {0, 1}
    assert (ev["t1"] > ev.index).all()
    assert ev["proba"].notna().sum() > 0
    assert ev["proba"].dropna().between(0, 1).all()
    assert result.X.index.equals(ev.index)
    assert not result.X.isna().any().any()


def test_comparison_uses_same_oos_events(result):
    cmp_ = result.comparison
    assert len(cmp_) == 3
    n_oos = int(result.oos_metrics["n_oos"])
    assert cmp_["n_trades"].iloc[0] == n_oos
    assert cmp_["n_trades"].iloc[1] == cmp_["n_trades"].iloc[2] <= n_oos


def test_summary_contains_all_sections(result):
    text = result.summary()
    for token in ("Purged K-Fold", "walk-forward", "Win", "Sharpe"):
        assert token in text


def test_live_scoring_returns_latest_signals(small_config, result):
    ohlcv = simulate_ohlcv(small_config.sim)
    live = MetaLabelingPipeline(small_config).score_events(ohlcv, result.final_model, last_n=3)
    assert len(live) == 3
    assert live["size"].between(0, 1).all()
    assert (live["signal"] == live["side"] * live["size"]).all()


def test_random_forest_variant_runs(small_config):
    cfg = replace(small_config, model=replace(small_config.model, kind="rf", rf_n_estimators=50))
    res = MetaLabelingPipeline(cfg).run()
    assert res.comparison["n_trades"].iloc[0] > 0


def test_csv_loader_roundtrip(tmp_path, small_config):
    ohlcv = simulate_ohlcv(small_config.sim).iloc[:50]
    path = tmp_path / "bist.csv"
    ohlcv.rename(columns=str.lower).reset_index().rename(columns={"Date": "date"}).to_csv(path, index=False)
    loaded = load_ohlcv_csv(path, date_col="date")
    assert list(loaded.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert loaded.index.equals(ohlcv.index)
