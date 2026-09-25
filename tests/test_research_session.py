"""Uçtan uca araştırma oturumu (küçük, hızlı konfigürasyon)."""

from dataclasses import replace
from pathlib import Path

import pytest

from meta_labeling import MetaLabelingPipeline
from meta_labeling.research import ResearchConfig, ResearchSession, load_research_config, reproduce
from meta_labeling.research.report import (
    INCONCLUSIVE,
    VERDICT_FAILED,
    VERDICT_INVALID,
    VERDICT_MIXED,
    VERDICT_SURVIVED,
)

ROOT = Path(__file__).resolve().parents[1]


def fast_config(tmp: Path) -> ResearchConfig:
    cfg = ResearchConfig()
    a = cfg.analysis
    return replace(
        cfg,
        experiment=replace(cfg.experiment, output_dir=str(tmp), log_level="WARNING", save_figures=False),
        data=replace(cfg.data, simulation=replace(cfg.data.simulation, n_bars=1500, seed=3)),
        meta_model=replace(cfg.meta_model, n_estimators=60),
        cv=replace(cfg.cv, n_splits=4, min_train_events=50),
        cpcv=replace(cfg.cpcv, n_groups=4, n_test_groups=2),
        benchmarks=replace(cfg.benchmarks, random_entry_reps=5),
        analysis=replace(a, bootstrap_n=50, placebo_label_reps=2, placebo_feature_reps=2, placebo_return_reps=1,
                         placebo_timestamp_reps=1, permutation_repeats=1, leakage_samples=6,
                         periods=(("2010-2012", "2010-01-01", "2012-12-31"), ("2013-2016", "2013-01-01", "2016-12-31"))),
    )


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    s = ResearchSession(fast_config(tmp_path_factory.mktemp("research")))
    s.run_all()
    return s


def test_default_yaml_matches_dataclass_defaults():
    assert load_research_config(ROOT / "config.yaml").fingerprint() == ResearchConfig().fingerprint()


def test_close_mode_reproduces_core_pipeline(tmp_path):
    """Araştırma katmanı (close execution) çekirdek pipeline ile birebir aynı OOS olasılıkları üretmeli."""
    cfg = fast_config(tmp_path)
    cfg = replace(cfg, execution=replace(cfg.execution, mode="close"))
    s = ResearchSession(cfg)
    for step in (s.build_features, s.sample_events, s.triple_barrier, s.walk_forward):
        step()
    core = MetaLabelingPipeline(cfg.pipeline_config()).run()
    ev = s.oos_events
    common = ev.index.intersection(core.events.index)
    assert len(common) == len(ev)
    assert (ev["proba"] - core.events.loc[common, "proba"]).abs().max() < 1e-12
    assert (ev["bin"] == core.events.loc[common, "bin"]).all()


def test_report_has_all_sections_and_final_tables(session):
    report = session.results["report"]
    for i in range(1, 26):
        assert f"## {i}. " in report.markdown
    assert list(report.final_table.columns) == ["Primary", "Meta", "Meta+Sizing", "Buy&Hold"]
    assert list(report.robustness_table.columns) == ["Robustness Test", "Result", "Pass/Fail", "Interpretation"]
    assert report.verdict in {VERDICT_FAILED, VERDICT_SURVIVED, VERDICT_MIXED, VERDICT_INVALID}
    for key in ("Strongest evidence", "Weakest evidence", "Major risks", "Unresolved questions", "Next experiments"):
        assert report.evidence[key]
    assert report.path.exists() and report.experiment_path.exists()


def test_few_placebo_reps_are_inconclusive_not_fail(session):
    table = session.results["report"].robustness_table
    placebo = table[table["Robustness Test"].str.startswith("Placebo")]
    assert (placebo["Pass/Fail"] == INCONCLUSIVE).all()  # 2 tekrar ile p < 0.05 imkansız


def test_all_strategies_share_window_and_costs(session):
    bench = session.results["benchmarks"]
    assert len(bench) == 9
    start, end = session.window
    for name in ("Primary", "Meta", "Meta+Sizing", "Buy&Hold"):
        r = session._runs[name].port.returns
        assert r.index[0] == start and r.index[-1] == end


def test_full_leakage_audit_passes(session):
    report = session.leakage["full"]
    assert report.status == "PASS"
    assert {"walk_forward", "purged_kfold", "cpcv", "calibration"} <= set(report.findings["check"])


def test_threshold_grid_is_reported_not_selected(session):
    thr = session.results["thresholds"]
    assert list(thr.index) == list(session.cfg.analysis.threshold_grid)
    assert session.cfg.meta_model.threshold == 0.55  # eşik değişmedi
    assert thr["Trades"].is_monotonic_decreasing


def test_experiment_is_reproducible(session):
    s2 = reproduce(session.results["report"].experiment_path)
    assert s2.results["report"].final_table.equals(session.results["report"].final_table)
