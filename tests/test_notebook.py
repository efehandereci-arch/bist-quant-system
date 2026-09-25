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
