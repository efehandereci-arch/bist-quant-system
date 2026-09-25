"""Jupyter çıktılarının paket kaynağıyla senkron olduğunu doğrular."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_notebook", ROOT / "scripts" / "build_notebook.py")
build_notebook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_notebook)


def test_notebook_is_up_to_date():
    expected = build_notebook.build_notebook()
    actual = build_notebook.NOTEBOOK_PATH.read_text(encoding="utf-8")
    assert actual == expected, "Notebook eski: python scripts/build_notebook.py çalıştırın"


def test_single_cell_script_is_up_to_date_and_valid():
    expected = build_notebook.build_script()
    actual = build_notebook.SCRIPT_PATH.read_text(encoding="utf-8")
    assert actual == expected, "Script eski: python scripts/build_notebook.py çalıştırın"
    compile(actual, str(build_notebook.SCRIPT_PATH), "exec")
    assert "from ." not in actual  # paket içi göreli import kalmamalı


def test_research_notebook_is_up_to_date():
    expected = build_notebook.build_research_notebook()
    actual = build_notebook.RESEARCH_NOTEBOOK_PATH.read_text(encoding="utf-8")
    assert actual == expected, "Notebook eski: python scripts/build_notebook.py çalıştırın"
    names = [n for n, _, _ in build_notebook.RESEARCH_CELLS]
    assert names == [f"{i:02d}_{s}" for i, s in enumerate(
        ["config", "imports", "data_validation", "data_loading", "feature_engineering", "leakage_audit",
         "event_sampling", "triple_barrier", "primary_model", "meta_labels", "cv", "walk_forward",
         "probability_calibration", "benchmark_strategies", "transaction_costs", "position_sizing", "backtest",
         "regime_analysis", "feature_importance", "statistical_tests", "CPCV", "randomization_tests",
         "visualizations", "final_report"], start=1)]
